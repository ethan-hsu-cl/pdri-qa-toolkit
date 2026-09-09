#!/usr/bin/env python3
"""Fetch and summarize the fixed bugs in the newest TR detected by a refresh.

The regular eBug refresh marks status transitions with ``_status_changed``.
This script selects the newest product version's ``TRCreated`` cohort (the bugs
from which the Testing Request is created), derives its dominant Close Build#
for display, fetches their full status logs through n8n, and
generates the QA test-focus report. Successful analyses are published as
immutable evidence snapshots; latest.json is switched only after validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:  # Direct script execution (scripts/ is first on sys.path)
    from fetch_from_n8n import PRODUCT_MAP
    from fetch_tr_bug_details import (
        DEFAULT_WEBHOOK_URL,
        FetchError,
        normalize_response,
        post_json,
        sanitize_label,
    )
    from tr_prediction_reviews import (
        FEATURE_MATCHER_POLICY,
        build_target_alias_policy,
        build_analysis_guardrail_policy,
        calibrate_historical_analog_policy,
        calibrate_threshold_policy,
        compact_analysis_guardrail_policy,
        compact_target_alias_policy,
        load_grounding_review_ledger,
        load_review_ledger,
        resolve_historical_analog_threshold_for_category,
        resolve_threshold_for_category,
    )
except ModuleNotFoundError:  # Imported as scripts.process_newest_tr in tests/tools
    from scripts.fetch_from_n8n import PRODUCT_MAP
    from scripts.fetch_tr_bug_details import (
        DEFAULT_WEBHOOK_URL,
        FetchError,
        normalize_response,
        post_json,
        sanitize_label,
    )
    from scripts.tr_prediction_reviews import (
        FEATURE_MATCHER_POLICY,
        build_target_alias_policy,
        build_analysis_guardrail_policy,
        calibrate_historical_analog_policy,
        calibrate_threshold_policy,
        compact_analysis_guardrail_policy,
        compact_target_alias_policy,
        load_grounding_review_ledger,
        load_review_ledger,
        resolve_historical_analog_threshold_for_category,
        resolve_threshold_for_category,
    )

MANIFEST_SCHEMA_VERSION = 2
STATUS_SCHEMA_VERSION = 1
DEFAULT_FEATURE_MATCH_THRESHOLD = 0.24
FEATURE_MATCH_AUDIT_MIN_SCORE = 0.12
FEATURE_MATCH_AUDIT_BAND = 0.12
HISTORICAL_ANALOG_MIN_SCORE = 0.18
HISTORICAL_ANALOG_MAX_SAME_MODULE = 1
HISTORICAL_ANALOG_MAX_ADJACENT_MODULE = 2

def casefold_order(value: str) -> tuple[str, str]:
    """Total order for case-insensitive sorts of hashed analysis input.

    ``str.casefold`` alone leaves distinct case variants ("PIP" / "PiP") tied,
    so their relative order falls back to set iteration order, which Python
    randomizes per process. The parent hashes the analysis source in-process
    while the summarizer re-hashes it in a subprocess, so a tie that resolves
    differently between them breaks the source-hash check.
    """
    return (value.casefold(), value)


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def mmdd_date_value(token: str, reference: datetime | None = None) -> int | None:
    """Resolve MMDD around New Year without interpreting a future December."""
    if not re.fullmatch(r"\d{4}", token):
        return None
    mm, dd = int(token[:2]), int(token[2:])
    now = reference or datetime.now()
    try:
        candidate = datetime(now.year, mm, dd)
    except ValueError:
        return None
    if candidate > now + timedelta(days=45):
        candidate = candidate.replace(year=now.year - 1)
    return candidate.year * 10_000 + mm * 100 + dd


def close_build_sort_key(value: str) -> tuple[int, int, int]:
    """Return a chronology-oriented key for the build formats used by eCL."""
    # Full YYYYMMDD[HHMM] timestamps (mobile/RC build formats).
    full_dates = re.findall(r"(?<!\d)(20\d{6}(?:\d{4})?)(?!\d)", value)
    if full_dates:
        return 2, max(int(v.ljust(12, "0")) for v in full_dates), 0

    # YYMMDD + revision, e.g. PDRA's 2607013.
    dated_revisions = []
    for token in re.findall(r"(?<!\d)(\d{7})(?!\d)", value):
        yy, mm, dd, revision = int(token[:2]), int(token[2:4]), int(token[4:6]), int(token[6])
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            dated_revisions.append(((2000 + yy) * 10_000 + mm * 100 + dd, revision))
    if dated_revisions:
        date_value, revision = max(dated_revisions)
        return 2, date_value * 10_000, revision

    # MMDD embedded in desktop build names, e.g. 0702_25.0.7 or .0703.1204.
    month_days = []
    mmdd_tokens = re.findall(r"(?<!\d)(\d{4})(?!\d)", value)
    # Only the first standalone 4-digit token can be MMDD. Later tokens are
    # normally HHMM; if the first is invalid (e.g. legacy build 1826), do not
    # reinterpret its following time as a date.
    for token in mmdd_tokens[:1]:
        date_value = mmdd_date_value(token)
        if date_value is not None:
            month_days.append((date_value, token))
    if month_days:
        # Build names put MMDD before HHMM (e.g. .0703.1204); choosing the
        # numerically largest token would misread 1204 as December 4.
        date_value, chosen = month_days[0]
        following = re.search(re.escape(chosen) + r"[._](\d{3,4})", value)
        time_or_revision = int(following.group(1)) if following else 0
        return 2, date_value * 10_000 + time_or_revision, 0

    # Sequential build IDs (PDR-i and Promeo).
    numbers = [int(v) for v in re.findall(r"\d+", value)]
    return 1, max(numbers, default=-1), numbers[-1] if numbers else -1


def close_build_confidence(value: str) -> int:
    """Prefer complete build formats over ambiguous manual shorthand."""
    if re.search(r"(?<!\d)20\d{6}(?:\d{4})?(?!\d)", value):
        return 3
    if re.search(r"(?<!\d)\d{7}(?!\d)", value):
        return 3
    # MMDD plus a version/hash/time component is a full desktop build name.
    mmdd_tokens = re.findall(r"(?<!\d)(\d{4})(?!\d)", value)
    if mmdd_tokens:
        mm, dd = int(mmdd_tokens[0][:2]), int(mmdd_tokens[0][2:])
        if 1 <= mm <= 12 and 1 <= dd <= 31 and len(re.findall(r"\d+", value)) >= 2:
            return 3
    # Dotted version followed by a sequential build (PDR-i style).
    if re.search(r"\d+\.\d+\.\d+\.\d{5,}", value):
        return 3
    # Bare numeric IDs are legitimate for some products, but are also common
    # manual shorthand, so use them only when no complete format is present.
    return 1


def close_build_family(value: str) -> str:
    """Group product-specific build formats so stray manual values lose."""
    value = value.strip()
    if re.match(r"^\d+\.\d+\.\d+\.", value):
        return "version_dot_build"
    if re.match(r"^\d{4}_\d+\.\d+\.\d+_", value):
        return "mmdd_version_build"
    if re.match(r"^RC:\s*20\d{6}_\d+\.\d+\.\d+", value, flags=re.I):
        return "rc_date_version"
    if re.match(r"^v\d{4}\s*\[", value, flags=re.I):
        return "manual_hash_build"
    if re.fullmatch(r"\d+", value):
        return "numeric_build"
    return "other"


def select_newest_tr_cohort(parsed_csv: Path) -> tuple[str, list[str]]:
    """Return (close_build, bug_codes) for the newest Testing Request.

    eCL marks the authoritative "RD Resolved Bug" list as TRCreated. The report
    certifies one **Close Build**, so the cohort is every TRCreated bug closed
    to the newest build — not every bug of the newest reported version.

    Those two differ in practice and the difference is not cosmetic: a bug
    filed against 16.8.0 can be fixed in an older build than bugs filed against
    16.7.x, so selecting by reported version picks a stale build carrying a
    fraction of its bugs (one bug instead of four, in the case that exposed
    this). Builds are ordered by the existing confidence/chronology helpers, so
    complete build names beat manual shorthand.
    """
    with parsed_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    tr_rows = [
        row for row in rows
        if str(row.get("Status") or "").strip().lower() == "trcreated"
        and str(row.get("BugCode") or "").strip()
    ]
    if not tr_rows:
        return "", []

    tr_code_set = {str(row.get("BugCode") or "").strip() for row in tr_rows}

    def usable_build(value: str) -> bool:
        looks_like_bug_code = bool(re.fullmatch(r"[A-Za-z]{2,}\d{6,}-?\d+", value))
        return bool(
            value and re.search(r"\d", value)
            and value not in tr_code_set and not looks_like_bug_code
        )

    build_rows: dict[str, list[dict]] = defaultdict(list)
    for row in tr_rows:
        build = str(row.get("Close Build#") or "").strip()
        if usable_build(build):
            build_rows[build].append(row)
    if not build_rows:
        # No usable Close Build#: fall back to the newest reported version so a
        # malformed build column cannot silence the report entirely.
        return _newest_version_cohort(tr_rows)

    # A stray manual build string ("v0716 [abc1234]") can look newer than the
    # real one, so pick the format family the product actually uses first.
    family_rows: dict[str, int] = defaultdict(int)
    for build, rows_for_build in build_rows.items():
        family_rows[close_build_family(build)] += len(rows_for_build)
    dominant_family = max(
        family_rows,
        key=lambda family: (
            family_rows[family],
            max(
                close_build_confidence(build) for build in build_rows
                if close_build_family(build) == family
            ),
            family,
        ),
    )
    close_build = max(
        (
            build for build in build_rows
            if close_build_family(build) == dominant_family
        ),
        key=lambda build: (
            close_build_confidence(build), close_build_sort_key(build),
            build.casefold(),
        ),
    )

    # Component-build outliers: bugs of the same reported version whose build
    # was written in another format still belong to this Testing Request.
    version_counts: dict[str, int] = defaultdict(int)
    for row in build_rows[close_build]:
        version = str(row.get("parsed_version") or "").strip()
        if version:
            version_counts[version] += 1
    cohort_version = max(
        version_counts, key=lambda value: (version_counts[value], value)
    ) if version_counts else ""

    cohort_codes = []
    for row in tr_rows:
        build = str(row.get("Close Build#") or "").strip()
        version = str(row.get("parsed_version") or "").strip()
        in_build = build == close_build
        is_outlier = (
            bool(cohort_version)
            and version == cohort_version
            and close_build_family(build) != dominant_family
        )
        if in_build or is_outlier:
            cohort_codes.append(str(row.get("BugCode") or "").strip())
    return close_build, list(dict.fromkeys(cohort_codes))


def _newest_version_cohort(tr_rows: list[dict]) -> tuple[str, list[str]]:
    """Legacy fallback: the newest reported version's TRCreated bugs."""
    versioned = [
        row for row in tr_rows if str(row.get("parsed_version") or "").strip()
    ]
    if not versioned:
        return "", list(dict.fromkeys(
            str(row.get("BugCode") or "").strip() for row in tr_rows
        ))

    def version_key(value: str) -> tuple[int, ...]:
        return tuple(int(part) for part in re.findall(r"\d+", value))

    newest_version = max(
        {str(row.get("parsed_version") or "").strip() for row in versioned},
        key=lambda value: (version_key(value), value.casefold()),
    )
    return newest_version, list(dict.fromkeys(
        str(row.get("BugCode") or "").strip() for row in versioned
        if str(row.get("parsed_version") or "").strip() == newest_version
    ))


def _is_empty(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def detail_quality_score(bug: dict) -> int:
    """Score whether a returned BugCode carries useful full-detail evidence."""
    score = 0
    if bug.get("Short Description"):
        score += 1
    if bug.get("ReproSteps") or bug.get("Description"):
        score += 2
    if bug.get("Resolutions"):
        score += 3
    raw = bug.get("raw")
    if isinstance(raw, dict) and len(raw) > 3:
        score += 1
    if bug.get("VerifyComments"):
        score += 1
    return score


def has_usable_full_detail(bug: dict) -> bool:
    return not bug.get("DetailFetchError") and detail_quality_score(bug) >= 2


def merge_bug_records(existing: dict, incoming: dict) -> dict:
    """Prefer fresh non-empty fields without discarding richer cached evidence."""
    merged = dict(existing)
    existing_is_richer = detail_quality_score(existing) > detail_quality_score(incoming)
    for key, value in incoming.items():
        if _is_empty(value):
            continue
        if key == "DetailFetchError" and existing_is_richer and has_usable_full_detail(existing):
            continue
        if key == "raw" and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    if has_usable_full_detail(incoming):
        merged.pop("DetailFetchError", None)
    return merged


def merge_detail_data(existing: dict | None, incoming: dict) -> dict:
    """Merge organized responses by product/BugCode without evidence loss."""
    if not existing:
        return incoming

    by_code: dict[tuple[str, str], dict] = {}
    for source in (existing, incoming):
        for product, bugs in source.get("products", {}).items():
            for bug in bugs:
                code = str(bug.get("BugCode", "")).strip()
                if not code:
                    continue
                key = (product, code)
                if key in by_code:
                    by_code[key] = merge_bug_records(by_code[key], bug)
                else:
                    by_code[key] = dict(bug)

    merged_products: dict[str, list[dict]] = {}
    for (product, _code), bug in by_code.items():
        merged_products.setdefault(product, []).append(bug)

    meta = {**existing.get("meta", {}), **incoming.get("meta", {})}
    all_bugs = [bug for bugs in merged_products.values() for bug in bugs]
    meta["bug_count"] = len(all_bugs)
    meta["products"] = sorted(merged_products)
    meta["tr_codes"] = sorted({
        str(bug.get("ParentTRCode")).strip()
        for bug in all_bugs if bug.get("ParentTRCode")
    })
    meta["errors"] = list(dict.fromkeys(incoming.get("meta", {}).get("errors", [])))
    meta.pop("missing_detail_bug_codes", None)
    meta.pop("recovered_errors", None)
    return {"meta": meta, "products": merged_products}


def filter_detail_data(data: dict, bug_codes: list[str],
                       expected_product: str | None = None) -> dict:
    """Prune, globally deduplicate, and order details by the authoritative cohort."""
    wanted = set(bug_codes)
    by_code: dict[str, tuple[str, dict]] = {}
    unexpected_products = []
    for product, bugs in data.get("products", {}).items():
        for bug in bugs:
            code = str(bug.get("BugCode") or "").strip()
            if code not in wanted:
                continue
            if expected_product and product != expected_product:
                unexpected_products.append({"bug_code": code, "product": product})
                continue
            if code in by_code:
                old_product, old_bug = by_code[code]
                by_code[code] = (old_product, merge_bug_records(old_bug, bug))
            else:
                by_code[code] = (product, bug)
    products: dict[str, list[dict]] = {}
    for code in bug_codes:
        if code not in by_code:
            continue
        product, bug = by_code[code]
        products.setdefault(product, []).append(bug)
    all_bugs = [bug for bugs in products.values() for bug in bugs]
    meta = dict(data.get("meta", {}))
    meta["bug_count"] = len(all_bugs)
    meta["products"] = sorted(products)
    meta["tr_codes"] = sorted({
        str(bug.get("ParentTRCode")).strip()
        for bug in all_bugs if bug.get("ParentTRCode")
    })
    meta["missing_detail_bug_codes"] = [
        str(bug.get("BugCode")) for bug in all_bugs if bug.get("DetailFetchError")
    ]
    meta["unexpected_product_records"] = unexpected_products
    return {"meta": meta, "products": products}


def local_fallback_bugs(parsed_csv: Path, missing_codes: list[str]) -> list[dict]:
    """Build transparent summary-only records when getOldeBug is unavailable."""
    wanted = set(missing_codes)
    best_rows: dict[str, dict] = {}
    with parsed_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            code = str(row.get("BugCode") or "").strip()
            if code not in wanted:
                continue
            current = best_rows.get(code)
            def row_key(value: dict) -> tuple:
                version = tuple(int(part) for part in re.findall(
                    r"\d+", str(value.get("parsed_version") or "")
                ))
                return (
                    str(value.get("Status") or "").strip().lower() == "trcreated",
                    version,
                    close_build_sort_key(str(value.get("Close Build#") or "")),
                )
            if current is None or row_key(row) > row_key(current):
                best_rows[code] = row
    fallbacks = []
    for code in missing_codes:
        row = best_rows.get(code)
        if row:
            fallbacks.append({
                "BugCode": code,
                "ProductName": row.get("ProductName") or None,
                "ParentTRCode": None,
                "Status": row.get("Status") or None,
                "Priority": row.get("Priority") or None,
                "Severity": row.get("Severity") or None,
                "PriorityLabel": row.get("priority_label") or None,
                "SeverityLabel": row.get("severity_label") or None,
                "Short Description": row.get("Short Description") or None,
                "ModuleOwner": row.get("BugBelong") or None,
                "ParsedModule": row.get("parsed_module") or None,
                "ModuleCategory": row.get("module_category") or None,
                "Create Date": row.get("Create Date") or None,
                "Closed Date": row.get("Closed Date") or None,
                "Build#": row.get("Build#") or None,
                "Close Build#": row.get("Close Build#") or None,
                "Creator": row.get("Creator") or None,
                "Handler": row.get("Handler") or None,
                "Resolutions": [],
                "StatusTimeline": [],
                "VerifyComments": [],
                "DetailFetchError": "Full eBug status log unavailable after 3 n8n attempts",
            })
    return fallbacks


def _history_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _resolved_history_row(row: dict) -> bool:
    status = re.sub(r"[^a-z]", "", str(row.get("Status") or "").casefold())
    return bool(str(row.get("Closed Date") or "").strip()) or status in {
        "close", "closed", "resolved", "buildresolved", "trcreated", "verified",
    }


def _historical_analogs_for_bug(bug: dict, history_rows: list[dict],
                                current_codes: set[str], *,
                                minimum_similarity_score: float = (
                                    HISTORICAL_ANALOG_MIN_SCORE
                                )) -> list[dict]:
    """Select leakage-safe historical analogs as corroboration, never proof."""
    source_code = str(bug.get("BugCode") or "").strip()
    source_module = str(bug.get("ParsedModule") or "").strip()
    source_category = str(bug.get("ModuleCategory") or "").strip()
    source_created = _history_datetime(bug.get("Create Date"))
    if not source_code or not source_category or source_created is None:
        return []

    # The short description is the most stable statement of the observed
    # failure. Repro and resolution text often mention incidental setup or
    # shared implementation vocabulary that creates plausible-looking but
    # unrelated matches.
    source_tokens = feature_match_tokens(bug.get("Short Description"))
    source_module_tokens = feature_match_tokens(source_module)
    if len(source_tokens) < 2:
        return []

    candidates = []
    for row in history_rows:
        code = str(row.get("BugCode") or "").strip()
        category = str(row.get("module_category") or "").strip()
        module = str(row.get("parsed_module") or "").strip()
        created = _history_datetime(row.get("Create Date"))
        summary = str(
            row.get("parsed_description") or row.get("Short Description") or ""
        ).strip()
        if (
            not code or code == source_code or code in current_codes
            or category != source_category or created is None
            or created >= source_created or not summary
            or not _resolved_history_row(row)
        ):
            continue
        candidate_tokens = feature_match_tokens(summary, module)
        overlap = source_tokens & candidate_tokens
        module_tokens = source_module_tokens | feature_match_tokens(module)
        distinctive_overlap = sorted(overlap - module_tokens)
        if (
            len(overlap) < 2 or len(distinctive_overlap) < 2
            or not candidate_tokens
        ):
            continue
        score = len(overlap) / math.sqrt(
            len(source_tokens) * len(candidate_tokens)
        )
        if score < minimum_similarity_score:
            continue
        candidates.append({
            "bug_code": code,
            "module": module,
            "module_category": category,
            "summary": re.sub(r"\s+", " ", summary)[:300],
            "created_at": str(row.get("Create Date") or "").strip(),
            "closed_at": str(row.get("Closed Date") or "").strip(),
            "relationship": (
                "same_module" if module and module == source_module
                else "same_category_adjacent_module"
            ),
            "similarity_score": round(score, 6),
            "selection_threshold": round(minimum_similarity_score, 3),
            "overlap_terms": sorted(overlap),
            "distinctive_overlap_terms": distinctive_overlap,
            "_created_sort": created,
        })

    candidates.sort(key=lambda item: (
        -float(item["similarity_score"]),
        -item["_created_sort"].timestamp(),
        item["bug_code"],
    ))
    same_module = [
        item for item in candidates if item["relationship"] == "same_module"
    ][:HISTORICAL_ANALOG_MAX_SAME_MODULE]
    adjacent = [
        item for item in candidates
        if item["relationship"] == "same_category_adjacent_module"
    ][:HISTORICAL_ANALOG_MAX_ADJACENT_MODULE]
    selected = same_module + adjacent
    selected.sort(key=lambda item: (
        -float(item["similarity_score"]), item["bug_code"]
    ))
    for item in selected:
        item.pop("_created_sort", None)
    return selected


def attach_module_category_context(
    details: dict, parsed_csv: Path, *,
    historical_analog_threshold_policy: dict | None = None,
) -> dict:
    """Attach normalized bug labels and the stable product category taxonomy."""
    category_counts: Counter[str] = Counter()
    category_modules: defaultdict[str, set[str]] = defaultdict(set)
    bug_context: dict[str, tuple[str, str]] = {}
    parsed_rows: list[dict] = []
    with parsed_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            parsed_rows.append(row)
            category = str(row.get("module_category") or "").strip()
            module = str(row.get("parsed_module") or "").strip()
            code = str(row.get("BugCode") or "").strip()
            if category:
                category_counts[category] += 1
                if module:
                    category_modules[category].add(module)
            if code:
                bug_context[code] = (module, category)

    current_codes = {
        str(bug.get("BugCode") or "").strip()
        for bugs in details.get("products", {}).values() for bug in bugs
        if str(bug.get("BugCode") or "").strip()
    }
    history_by_code: dict[str, dict] = {}
    for row in parsed_rows:
        code = str(row.get("BugCode") or "").strip()
        if not code:
            continue
        existing = history_by_code.get(code)
        row_key = (
            _resolved_history_row(row),
            _history_datetime(row.get("Create Date")) or datetime.min,
            _history_datetime(row.get("Closed Date")) or datetime.min,
        )
        existing_key = (
            _resolved_history_row(existing),
            _history_datetime(existing.get("Create Date")) or datetime.min,
            _history_datetime(existing.get("Closed Date")) or datetime.min,
        ) if existing else None
        if existing_key is None or row_key > existing_key:
            history_by_code[code] = row
    history_rows = list(history_by_code.values())

    enriched = {
        **details,
        "meta": dict(details.get("meta") or {}),
        "products": {},
    }
    applied_historical_thresholds = {}
    for product, bugs in details.get("products", {}).items():
        enriched_bugs = []
        for bug in bugs:
            item = dict(bug)
            module, category = bug_context.get(str(item.get("BugCode") or ""), ("", ""))
            if module:
                item["ParsedModule"] = module
            if category:
                item["ModuleCategory"] = category
            context_row = history_by_code.get(str(item.get("BugCode") or "")) or {}
            if not item.get("Create Date") and context_row.get("Create Date"):
                item["Create Date"] = context_row["Create Date"]
            analog_threshold = resolve_historical_analog_threshold_for_category(
                historical_analog_threshold_policy, category
            )
            if category:
                applied_historical_thresholds[category] = analog_threshold
            item["HistoricalAnalogs"] = _historical_analogs_for_bug(
                item, history_rows, current_codes,
                minimum_similarity_score=float(analog_threshold["threshold"]),
            )
            enriched_bugs.append(item)
        enriched["products"][product] = enriched_bugs

    enriched["meta"]["module_category_taxonomy"] = [
        {
            "category": category,
            "historical_bug_count": category_counts[category],
            "module_count": len(category_modules[category]),
            "modules": sorted(category_modules[category], key=casefold_order),
        }
        for category in sorted(category_counts, key=casefold_order)
    ]
    enriched["meta"]["historical_analog_policy"] = {
        "schema_version": 1,
        "minimum_similarity_score": HISTORICAL_ANALOG_MIN_SCORE,
        "maximum_same_module_per_bug": HISTORICAL_ANALOG_MAX_SAME_MODULE,
        "maximum_adjacent_module_per_bug": (
            HISTORICAL_ANALOG_MAX_ADJACENT_MODULE
        ),
        "requires_older_create_date": True,
        "requires_resolved_record": True,
        "excludes_current_cohort": True,
        "requires_meaningful_summary_overlap_terms": 2,
        "threshold_calibration": historical_analog_threshold_policy or {
            "schema_version": 1,
            "global": {
                "status": "insufficient_reviews",
                "eligible": False,
                "threshold": HISTORICAL_ANALOG_MIN_SCORE,
            },
            "categories": {},
        },
        "applied_thresholds_by_category": applied_historical_thresholds,
    }
    enriched["meta"]["historical_analog_count"] = sum(
        len(bug.get("HistoricalAnalogs") or [])
        for bugs in enriched["products"].values() for bug in bugs
    )
    return enriched


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_product_artifact(product_dir: Path, value: object) -> Path | None:
    """Resolve legacy absolute or v2 relative paths without escaping product data."""
    if not value:
        return None
    base = product_dir.resolve()
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


def resolve_prior_scoreable_analysis(
    report_root: Path, product_dir: Path, prior_manifest: dict,
    close_build: str,
) -> tuple[Path | None, str]:
    """Return the newest stored analysis from a *different* Close Build.

    The retrospective asks whether the previous build's predictions produced
    bugs in this build, so a prior run from the same Close Build can never
    answer it — it would score a build against itself. Re-running one build
    (a re-fetch, or a cohort that grew during the day) is routine, so fall
    back to the newest stored run from another build instead of silently
    producing an unscoreable comparison.
    """
    manifest_build = str(prior_manifest.get("close_build") or "")
    manifest_path = resolve_product_artifact(
        product_dir, prior_manifest.get("analysis_path")
    )
    if (
        manifest_path and manifest_path.is_file()
        and manifest_build and manifest_build != close_build
    ):
        return manifest_path, manifest_build
    candidates = []
    for candidate in report_root.glob("*/runs/*/analysis.json"):
        build = candidate.parent.parent.parent.name
        if not build or build == close_build or not candidate.is_file():
            continue
        candidates.append((candidate.parent.name, build, candidate))
    if not candidates:
        return None, ""
    _, build, path = max(candidates, key=lambda item: item[0])
    return path, build


def write_attempt_status(report_root: Path, product: str, outcome: str,
                         *, close_build: str = "", bug_codes: list[str] | None = None,
                         message: str = "", latest_success: str = "") -> None:
    write_json(report_root / "status.json", {
        "schema_version": STATUS_SCHEMA_VERSION,
        "product": product,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "selected_close_build": close_build or None,
        "selected_bug_codes": bug_codes or [],
        "message": _status_message(message),
        "latest_success_generated_at": latest_success or None,
    })


def _status_message(message: object) -> str:
    return re.sub(r"\s+", " ", str(message or "")).strip()[:500]


def analysis_source_hash(details: dict) -> tuple[str, int]:
    try:
        from summarize_tr_bugs import (
            ANALYSIS_SCHEMA_VERSION, structured_bug_evidence,
            target_category_support,
        )
    except ModuleNotFoundError:
        from scripts.summarize_tr_bugs import (
            ANALYSIS_SCHEMA_VERSION, structured_bug_evidence,
            target_category_support,
        )
    evidence = [
        structured_bug_evidence(bug)
        for bugs in details.get("products", {}).values() for bug in bugs
    ]
    source = {
        "bug_evidence": evidence,
        "module_category_taxonomy": (
            details.get("meta", {}).get("module_category_taxonomy") or []
        ),
        "historical_analog_policy": (
            details.get("meta", {}).get("historical_analog_policy") or {}
        ),
        "analysis_guardrail_policy": (
            details.get("meta", {}).get("analysis_guardrail_policy") or {}
        ),
        "target_alias_policy": (
            details.get("meta", {}).get("target_alias_policy")
            or compact_target_alias_policy(None)
        ),
        "target_category_support": target_category_support(details, evidence),
    }
    encoded = json.dumps(
        source, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), ANALYSIS_SCHEMA_VERSION


def analysis_engine_sha256() -> str:
    """Fingerprint the summarizer so prompt/policy edits cannot hit stale cache."""
    return sha256_file(Path(__file__).with_name("summarize_tr_bugs.py"))


def manifest_cache_hit(product_dir: Path, manifest: dict, *, source_hash: str,
                       engine_sha256: str, model: str, analysis_schema: int,
                       close_build: str, bug_codes: list[str]) -> bool:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return False
    if (
        manifest.get("source_sha256") != source_hash
        or manifest.get("engine_sha256") != engine_sha256
        or manifest.get("model") != model
        or manifest.get("analysis_schema_version") != analysis_schema
        or str(manifest.get("close_build") or "") != close_build
        or [str(code) for code in manifest.get("bug_codes", [])] != bug_codes
    ):
        return False
    for path_key, hash_key in (
        ("details_path", "details_sha256"),
        ("analysis_path", "analysis_sha256"),
        ("report_path", "report_sha256"),
    ):
        path = resolve_product_artifact(product_dir, manifest.get(path_key))
        if not path or not path.is_file() or sha256_file(path) != manifest.get(hash_key):
            return False
    return True


def structured_prediction_errors(analysis: dict) -> list[str]:
    """Validate scoreable prediction identity fields in schema-v4 analyses."""
    schema_version = int(analysis.get("schema_version") or 0)
    if schema_version < 4:
        return []
    taxonomy = {
        str(value) for value in (analysis.get("target_category_taxonomy") or [])
    }
    errors = []
    hypothesis_ids = []
    prediction_boundary_errors = None
    prediction_specificity_errors = None
    deterministic_confidence_basis = None
    deterministic_risk_basis = None
    deterministic_recommendation_risk_basis = None
    deterministic_recommendation_priority_basis = None
    deterministic_recommendation_priority_policy = None
    deterministic_recommendation_portfolio = None
    deterministic_synthesis_backfill = None
    deterministic_adjacency_support_basis = None
    deterministic_shared_path_support_basis = None
    deterministic_scenario_specificity_basis = None
    deterministic_evidence_ref_completion = None
    published_recommendation_limit = 8
    deterministic_target_category_basis = None
    deterministic_checklist_grounding = None
    confidence_evidence_map = {}
    category_support = {}
    target_alias_policy = {}
    checklist_evidence_map = {}
    discovery_origins_by_id = {}
    if schema_version >= 7:
        try:
            from summarize_tr_bugs import (
                prediction_boundary_errors as _prediction_boundary_errors,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                prediction_boundary_errors as _prediction_boundary_errors,
            )
        prediction_boundary_errors = _prediction_boundary_errors
    if schema_version >= 12:
        try:
            from summarize_tr_bugs import (
                prediction_specificity_errors as _prediction_specificity_errors,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                prediction_specificity_errors as _prediction_specificity_errors,
            )
        prediction_specificity_errors = _prediction_specificity_errors
    if schema_version >= 15:
        try:
            from summarize_tr_bugs import (
                deterministic_confidence_basis as _deterministic_confidence_basis,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                deterministic_confidence_basis as _deterministic_confidence_basis,
            )
        deterministic_confidence_basis = _deterministic_confidence_basis
        quality_by_bug = analysis.get("evidence_quality_by_bug")
        expected_bug_codes = {
            str(code) for code in (analysis.get("bug_codes") or [])
        }
        if (
            not isinstance(quality_by_bug, dict)
            or set(quality_by_bug) != expected_bug_codes
            or any(
                not isinstance(code, str)
                or quality not in {"high", "medium", "low"}
                for code, quality in (
                    quality_by_bug.items()
                    if isinstance(quality_by_bug, dict) else []
                )
            )
        ):
            errors.append("evidence quality by BugCode is missing or invalid")
        else:
            confidence_evidence_map = {
                code: {"evidence_quality": quality}
                for code, quality in quality_by_bug.items()
            }
        if schema_version >= 18:
            independence_by_bug = analysis.get(
                "evidence_independence_by_bug"
            )
            if (
                not isinstance(independence_by_bug, dict)
                or set(independence_by_bug) != expected_bug_codes
                or any(
                    not isinstance(value, dict)
                    or set(value) != {
                        "evidence_quality", "parsed_module",
                        "adjacency_signal_quotes",
                    }
                    or value.get("evidence_quality")
                    not in {"high", "medium", "low"}
                    or value.get("evidence_quality")
                    != (quality_by_bug or {}).get(code)
                    or not isinstance(value.get("parsed_module"), str)
                    or len(value.get("parsed_module")) > 120
                    or not isinstance(
                        value.get("adjacency_signal_quotes"), list
                    )
                    or len(value.get("adjacency_signal_quotes")) > 3
                    or any(
                        not isinstance(quote, str) or not quote
                        or len(quote) > 300
                        for quote in value.get("adjacency_signal_quotes", [])
                    )
                    for code, value in (
                        independence_by_bug.items()
                        if isinstance(independence_by_bug, dict) else []
                    )
                )
            ):
                errors.append(
                    "evidence independence by BugCode is missing or invalid"
                )
            else:
                confidence_evidence_map = independence_by_bug
    if schema_version >= 16:
        try:
            from summarize_tr_bugs import (
                deterministic_risk_basis as _deterministic_risk_basis,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                deterministic_risk_basis as _deterministic_risk_basis,
            )
        deterministic_risk_basis = _deterministic_risk_basis
    if schema_version >= 30:
        try:
            from summarize_tr_bugs import (
                deterministic_recommendation_risk_basis as _recommendation_risk_basis,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                deterministic_recommendation_risk_basis as _recommendation_risk_basis,
            )
        deterministic_recommendation_risk_basis = _recommendation_risk_basis
    if schema_version >= 17:
        try:
            from summarize_tr_bugs import (
                deterministic_target_category_basis as _deterministic_target_category_basis,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                deterministic_target_category_basis as _deterministic_target_category_basis,
            )
        deterministic_target_category_basis = _deterministic_target_category_basis
        raw_support = analysis.get("target_category_support")
        support_taxonomy = (
            raw_support.get("taxonomy") if isinstance(raw_support, dict) else None
        )
        source_category_by_bug = (
            raw_support.get("source_category_by_bug")
            if isinstance(raw_support, dict) else None
        )
        expected_bug_codes = {
            str(code) for code in (analysis.get("bug_codes") or [])
        }
        support_valid = (
            isinstance(raw_support, dict)
            and raw_support.get("schema_version") == 1
            and raw_support.get("policy")
            == "known_target_module_then_unique_current_source_category_v1"
            and isinstance(support_taxonomy, list)
            and isinstance(source_category_by_bug, dict)
            and set(source_category_by_bug) == expected_bug_codes
            and all(
                isinstance(code, str) and isinstance(category, str)
                and (not category or category in taxonomy)
                for code, category in source_category_by_bug.items()
            )
            and all(
                isinstance(row, dict)
                and isinstance(row.get("category"), str)
                and row.get("category") in taxonomy
                and isinstance(row.get("modules"), list)
                and all(
                    isinstance(module, str) and bool(module.strip())
                    for module in row.get("modules", [])
                )
                and len(row.get("modules", []))
                == len(set(row.get("modules", [])))
                for row in support_taxonomy
            )
        )
        if not support_valid:
            errors.append("target category support is missing or invalid")
        else:
            category_support = raw_support
    if schema_version >= 20:
        try:
            from summarize_tr_bugs import (
                deterministic_checklist_grounding as _deterministic_checklist_grounding,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                deterministic_checklist_grounding as _deterministic_checklist_grounding,
            )
        deterministic_checklist_grounding = _deterministic_checklist_grounding
        raw_checklist_evidence = analysis.get("checklist_evidence_by_bug")
        expected_bug_codes = {
            str(code) for code in (analysis.get("bug_codes") or [])
        }
        allowed_gaps = {
            "full_detail_unavailable", "missing_issue_summary",
            "missing_reproduction_context", "missing_root_cause",
            "missing_fix_description",
        }
        checklist_evidence_valid = (
            isinstance(raw_checklist_evidence, dict)
            and set(raw_checklist_evidence) == expected_bug_codes
            and all(
                isinstance(value, dict)
                and set(value) == {
                    "summary", "repro_or_description", "locale_or_os",
                    "changes", "qa_verify_notes", "evidence_quality",
                    "evidence_gaps",
                }
                and all(
                    isinstance(value.get(field), str)
                    for field in (
                        "summary", "repro_or_description", "locale_or_os",
                    )
                )
                and len(value.get("summary")) <= 500
                and len(value.get("repro_or_description")) <= 900
                and len(value.get("locale_or_os")) <= 180
                and isinstance(value.get("changes"), list)
                and len(value.get("changes")) <= 3
                and all(
                    isinstance(change, dict)
                    and set(change) == {
                        "root_cause", "solution", "stated_risk",
                    }
                    and all(isinstance(text, str) for text in change.values())
                    for change in value.get("changes", [])
                )
                and isinstance(value.get("qa_verify_notes"), list)
                and len(value.get("qa_verify_notes")) <= 3
                and all(
                    isinstance(note, str) and len(note) <= 350
                    for note in value.get("qa_verify_notes", [])
                )
                and value.get("evidence_quality")
                in {"high", "medium", "low"}
                and isinstance(value.get("evidence_gaps"), list)
                and all(
                    gap in allowed_gaps
                    for gap in value.get("evidence_gaps", [])
                )
                for value in (
                    raw_checklist_evidence.values()
                    if isinstance(raw_checklist_evidence, dict) else []
                )
            )
        )
        if not checklist_evidence_valid:
            errors.append("checklist evidence by BugCode is missing or invalid")
        else:
            checklist_evidence_map = raw_checklist_evidence
    if schema_version >= 22:
        try:
            from summarize_tr_bugs import (
                deterministic_recommendation_priority_basis as _priority_basis,
                deterministic_recommendation_priority_policy as _priority_policy,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                deterministic_recommendation_priority_basis as _priority_basis,
                deterministic_recommendation_priority_policy as _priority_policy,
            )
        deterministic_recommendation_priority_basis = _priority_basis
        deterministic_recommendation_priority_policy = _priority_policy
    if schema_version >= 23:
        try:
            from summarize_tr_bugs import (
                PUBLISHED_RECOMMENDATION_LIMIT as _published_limit,
                deterministic_synthesis_backfill as _synthesis_backfill,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                PUBLISHED_RECOMMENDATION_LIMIT as _published_limit,
                deterministic_synthesis_backfill as _synthesis_backfill,
            )
        published_recommendation_limit = _published_limit
        deterministic_synthesis_backfill = _synthesis_backfill
    if schema_version >= 31:
        try:
            from summarize_tr_bugs import (
                deterministic_recommendation_portfolio as _recommendation_portfolio,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                deterministic_recommendation_portfolio as _recommendation_portfolio,
            )
        deterministic_recommendation_portfolio = _recommendation_portfolio
    if schema_version >= 32:
        try:
            from summarize_tr_bugs import (
                deterministic_scenario_specificity_basis as _scenario_specificity_basis,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                deterministic_scenario_specificity_basis as _scenario_specificity_basis,
            )
        deterministic_scenario_specificity_basis = _scenario_specificity_basis
    if schema_version >= 25:
        try:
            from summarize_tr_bugs import (
                deterministic_adjacency_support_basis as _adjacency_support,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                deterministic_adjacency_support_basis as _adjacency_support,
            )
        deterministic_adjacency_support_basis = _adjacency_support
    if schema_version >= 26:
        try:
            from summarize_tr_bugs import (
                deterministic_evidence_ref_completion as _ref_completion,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                deterministic_evidence_ref_completion as _ref_completion,
            )
        deterministic_evidence_ref_completion = _ref_completion
    if schema_version >= 27:
        raw_target_alias_policy = analysis.get("target_alias_policy")
        expected_target_alias_policy = compact_target_alias_policy(
            raw_target_alias_policy
            if isinstance(raw_target_alias_policy, dict) else None
        )
        if raw_target_alias_policy != expected_target_alias_policy:
            errors.append("reviewed target alias policy is missing or invalid")
        else:
            target_alias_policy = raw_target_alias_policy
    if schema_version >= 29:
        try:
            from summarize_tr_bugs import (
                deterministic_shared_path_support_basis as _shared_path_support,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                deterministic_shared_path_support_basis as _shared_path_support,
            )
        deterministic_shared_path_support_basis = _shared_path_support

    def priority_basis_types_valid(basis: object) -> bool:
        if not isinstance(basis, dict):
            return False
        integer_fields = (
            "target_anchored_independent_group_count",
            "high_quality_independent_group_count",
            "medium_or_better_independent_group_count",
            "independent_source_group_count",
        )
        sort_key = basis.get("sort_key")
        return (
            all(
                isinstance(basis.get(field), int)
                and not isinstance(basis.get(field), bool)
                and basis.get(field) >= 0
                for field in integer_fields
            )
            and isinstance(sort_key, list)
            and len(sort_key) == 7
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in sort_key[:6]
            )
            and isinstance(sort_key[6], str)
            and all(
                isinstance(basis.get(field), bool)
                for field in (
                    "model_order_influences_rank",
                    "historical_analogs_influence_rank",
                    "source_ticket_severity_influences_rank",
                )
            )
        )

    def validate_category_basis(item: dict, label: str) -> None:
        if (
            schema_version < 17
            or deterministic_target_category_basis is None
            or not category_support
        ):
            return
        basis = item.get("target_category_basis")
        requested = (
            str(basis.get("requested_category") or "")
            if isinstance(basis, dict) else ""
        )
        expected = deterministic_target_category_basis(
            requested,
            str(item.get("area") or ""),
            str(item.get("trigger") or ""),
            str(item.get("action") or ""),
            [str(code) for code in (item.get("source_bug_codes") or [])],
            category_support,
        )
        if basis != expected:
            errors.append(f"{label} target category basis mismatch")
        if item.get("target_category") != expected["label"]:
            errors.append(f"{label} target category label mismatch")

    def validate_confidence_refs(item: dict, label: str) -> None:
        if schema_version < 18 or not confidence_evidence_map:
            return
        refs = item.get("evidence_refs")
        if not isinstance(refs, list):
            errors.append(f"{label} confidence reference provenance mismatch")
            return
        for ref in refs:
            if not isinstance(ref, dict):
                errors.append(
                    f"{label} confidence reference provenance mismatch"
                )
                return
            code = str(ref.get("bug_code") or "")
            signal_index = ref.get("signal_index")
            quotes = list(
                (confidence_evidence_map.get(code) or {}).get(
                    "adjacency_signal_quotes"
                ) or []
            )
            if (
                isinstance(signal_index, bool)
                or not isinstance(signal_index, int)
                or not 0 <= signal_index < len(quotes)
                or ref.get("quote") != quotes[signal_index]
            ):
                errors.append(
                    f"{label} confidence reference provenance mismatch"
                )
                return

    def validate_adjacency_support(
        item: dict, label: str, *, expected_supported: bool = True,
        require_ref_completion: bool = False,
    ) -> None:
        if (
            schema_version < 25
            or deterministic_adjacency_support_basis is None
        ):
            return
        refs = item.get("evidence_refs")
        if not isinstance(refs, list):
            errors.append(f"{label} adjacency support provenance mismatch")
            return
        expected = deterministic_adjacency_support_basis(
            str(item.get("area") or ""), refs, target_alias_policy,
        )
        if item.get("adjacency_support_basis") != expected:
            errors.append(f"{label} adjacency support provenance mismatch")
        combined_supported = expected.get("supported") is True
        if schema_version >= 29:
            if deterministic_shared_path_support_basis is None:
                errors.append(f"{label} shared-path support validator is unavailable")
                return
            expected_shared_path = deterministic_shared_path_support_basis(
                str(item.get("area") or ""),
                str(item.get("rationale") or ""), refs, expected,
            )
            if item.get("shared_path_support_basis") != expected_shared_path:
                errors.append(f"{label} shared-path support provenance mismatch")
            combined_supported = (
                combined_supported
                and expected_shared_path.get("supported") is True
            )
        if schema_version >= 32:
            if deterministic_scenario_specificity_basis is None:
                errors.append(
                    f"{label} scenario specificity validator is unavailable"
                )
                return
            expected_scenario = deterministic_scenario_specificity_basis(
                str(item.get("trigger") or ""),
                str(item.get("expected_failure_mode") or ""),
                [str(code) for code in item.get("source_bug_codes") or []],
                refs,
                checklist_evidence_map,
                **({
                    "action": str(item.get("action") or ""),
                    "disconfirming_observation": str(item.get("disconfirming_observation") or ""),
                } if schema_version >= 37 else {}),
            )
            if item.get("scenario_specificity_basis") != expected_scenario:
                errors.append(
                    f"{label} scenario specificity provenance mismatch"
                )
            if expected_scenario.get("supported") is not True:
                errors.append(
                    f"{label} contains unsupported scenario specificity"
                )
        if combined_supported is not expected_supported:
            errors.append(f"{label} has invalid deterministic adjacency support")
        if not require_ref_completion or schema_version < 26:
            return
        completion = item.get("evidence_ref_completion_basis")
        if (
            not isinstance(completion, dict)
            or deterministic_evidence_ref_completion is None
        ):
            errors.append(f"{label} evidence reference completion is missing")
            return
        original_refs = completion.get("original_evidence_refs")
        source_codes = item.get("source_bug_codes")
        if not isinstance(original_refs, list) or not isinstance(source_codes, list):
            errors.append(
                f"{label} evidence reference completion provenance mismatch"
            )
            return
        expected_refs, expected_completion = (
            deterministic_evidence_ref_completion(
                str(item.get("area") or ""),
                [str(code) for code in source_codes],
                original_refs,
                confidence_evidence_map,
                max_refs=4,
            )
        )
        if completion != expected_completion or refs != expected_refs:
            errors.append(
                f"{label} evidence reference completion provenance mismatch"
            )
    observed_model_selected_ids = set()
    observed_backfill_ids = set()

    def validate_generation_origins(
        item: dict, label: str, expected: list[str] | None = None,
    ) -> list[str]:
        if schema_version < 24:
            return []
        origins = item.get("candidate_generation_origins")
        if (
            not isinstance(origins, list)
            or not origins
            or len(origins) != len(set(origins))
            or any(
                origin not in {"primary_discovery", "coverage_challenge"}
                for origin in (origins or [])
            )
        ):
            errors.append(f"{label} has invalid candidate generation origins")
            return []
        if expected is not None and origins != expected:
            errors.append(f"{label} candidate generation origin mismatch")
        return list(origins)

    def observe_selection_origin(item: dict, label: str) -> None:
        if schema_version < 23:
            return
        origin = item.get("selection_origin")
        lineage_ids = item.get("synthesis_source_candidate_ids")
        if origin not in {
            "model_synthesis",
            "deterministic_synthesis_omission_backfill",
        }:
            errors.append(f"{label} has invalid selection origin")
            return
        if (
            not isinstance(lineage_ids, list) or not lineage_ids
            or len(set(lineage_ids)) != len(lineage_ids)
            or any(
                not isinstance(candidate_id, str)
                or not re.fullmatch(r"TRC-[A-F0-9]{12}", candidate_id)
                for candidate_id in lineage_ids
            )
        ):
            errors.append(f"{label} has invalid selection lineage")
            return
        if origin == "deterministic_synthesis_omission_backfill":
            if len(lineage_ids) != 1:
                errors.append(f"{label} backfill must cite one candidate")
            observed_backfill_ids.update(lineage_ids)
        else:
            observed_model_selected_ids.update(lineage_ids)

    expected_priority_rows = []
    for index, item in enumerate(analysis.get("recommendations") or [], start=1):
        if not isinstance(item, dict):
            errors.append(f"recommendation {index} is not an object")
            continue
        observe_selection_origin(item, f"recommendation {index}")
        validate_adjacency_support(item, f"recommendation {index}")
        hypothesis_id = str(item.get("hypothesis_id") or "")
        target_category = str(item.get("target_category") or "")
        if not re.fullmatch(r"TRH-[A-F0-9]{12}", hypothesis_id):
            errors.append(f"recommendation {index} has invalid hypothesis_id")
        else:
            hypothesis_ids.append(hypothesis_id)
        if not target_category or target_category not in taxonomy:
            errors.append(f"recommendation {index} has invalid target_category")
        if schema_version >= 5:
            invalid_feature_fields = [
                field for field in ("trigger", "expected_failure_mode")
                if not isinstance(item.get(field), str)
                or not str(item.get(field)).strip()
                or len(str(item.get(field))) > 320
            ]
            if invalid_feature_fields:
                errors.append(
                    f"recommendation {index} has invalid feature field(s): "
                    + ", ".join(invalid_feature_fields)
                )
            elif (
                schema_version >= 12
                and prediction_specificity_errors is not None
            ):
                errors.extend(prediction_specificity_errors(
                    str(item.get("area") or "").strip(),
                    str(item.get("trigger") or "").strip(),
                    str(item.get("expected_failure_mode") or "").strip(),
                    f"recommendation {index}",
                ))
        if schema_version >= 6:
            analog_refs = item.get("historical_analog_refs") or []
            source_codes = {
                str(code) for code in (item.get("source_bug_codes") or [])
            }
            if not isinstance(analog_refs, list):
                errors.append(
                    f"recommendation {index} historical_analog_refs is not an array"
                )
            else:
                seen_analogs = set()
                for ref_index, ref in enumerate(analog_refs, start=1):
                    if not isinstance(ref, dict):
                        errors.append(
                            f"recommendation {index} historical analog "
                            f"{ref_index} is not an object"
                        )
                        continue
                    source_code = str(ref.get("source_bug_code") or "")
                    analog_code = str(ref.get("analog_bug_code") or "")
                    if source_code not in source_codes:
                        errors.append(
                            f"recommendation {index} historical analog "
                            f"{ref_index} has invalid source BugCode"
                        )
                    if not re.fullmatch(r"[A-Za-z0-9._-]+", analog_code):
                        errors.append(
                            f"recommendation {index} historical analog "
                            f"{ref_index} has invalid analog BugCode"
                        )
                    analog_score = ref.get("similarity_score")
                    if (
                        isinstance(analog_score, bool)
                        or not isinstance(analog_score, (int, float))
                        or not math.isfinite(float(analog_score))
                        or not 0 <= float(analog_score) <= 1
                    ):
                        errors.append(
                            f"recommendation {index} historical analog "
                            f"{ref_index} has invalid similarity score"
                        )
                    key = (source_code, analog_code)
                    if key in seen_analogs:
                        errors.append(
                            f"recommendation {index} has duplicate historical analog"
                        )
                    seen_analogs.add(key)
        if schema_version >= 15 and deterministic_confidence_basis is not None:
            validate_confidence_refs(item, f"recommendation {index}")
            source_codes_for_confidence = [
                str(code) for code in (item.get("source_bug_codes") or [])
            ]
            evidence_refs_for_confidence = item.get("evidence_refs")
            historical_refs_for_confidence = item.get(
                "historical_analog_refs"
            )
            if (
                not isinstance(evidence_refs_for_confidence, list)
                or not isinstance(historical_refs_for_confidence, list)
            ):
                errors.append(
                    f"recommendation {index} has invalid confidence inputs"
                )
            else:
                expected_confidence_basis = deterministic_confidence_basis(
                    source_codes_for_confidence,
                    evidence_refs_for_confidence,
                    historical_refs_for_confidence,
                    confidence_evidence_map,
                    correlate_sources=schema_version >= 18,
                    target_area=(
                        str(item.get("area") or "")
                        if schema_version >= 19 else None
                    ),
                )
                if item.get("confidence_basis") != expected_confidence_basis:
                    errors.append(
                        f"recommendation {index} confidence basis mismatch"
                    )
                if item.get("confidence") != expected_confidence_basis["label"]:
                    errors.append(
                        f"recommendation {index} confidence label mismatch"
                    )
        if schema_version >= 16 and deterministic_risk_basis is not None:
            expected_risk_basis = (
                deterministic_recommendation_risk_basis(
                    item, checklist_evidence_map,
                )
                if schema_version >= 30
                and deterministic_recommendation_risk_basis is not None
                else deterministic_risk_basis(
                    str(item.get("expected_failure_mode") or ""),
                    str(item.get("confidence") or ""),
                    negation_aware=schema_version >= 21,
                )
            )
            if item.get("risk_basis") != expected_risk_basis:
                errors.append(f"recommendation {index} risk basis mismatch")
            if item.get("risk") != expected_risk_basis["label"]:
                errors.append(f"recommendation {index} risk label mismatch")
            if item.get("impact_basis") != expected_risk_basis["impact_basis"]:
                errors.append(f"recommendation {index} impact basis mismatch")
        if (
            schema_version >= 22
            and deterministic_recommendation_priority_basis is not None
        ):
            expected_priority_basis = (
                deterministic_recommendation_priority_basis(
                    item, confidence_evidence_map,
                )
            )
            expected_priority_rows.append((expected_priority_basis, item))
            if (
                item.get("priority_basis") != expected_priority_basis
                or not priority_basis_types_valid(item.get("priority_basis"))
            ):
                errors.append(f"recommendation {index} priority basis mismatch")
        validate_category_basis(item, f"recommendation {index}")
        if schema_version >= 7:
            invalid_boundary_fields = [
                field for field in (
                    "scope_boundary", "disconfirming_observation",
                )
                if not isinstance(item.get(field), str)
                or not str(item.get(field)).strip()
                or len(str(item.get(field))) > 320
            ]
            if invalid_boundary_fields:
                errors.append(
                    f"recommendation {index} has invalid boundary field(s): "
                    + ", ".join(invalid_boundary_fields)
                )
            elif prediction_boundary_errors is not None:
                errors.extend(prediction_boundary_errors(
                    str(item.get("area") or ""),
                    str(item.get("scope_boundary") or ""),
                    str(item.get("disconfirming_observation") or ""),
                    f"recommendation {index}",
                ))
    if (
        schema_version >= 22
        and deterministic_recommendation_priority_policy is not None
    ):
        expected_priority_rows.sort(
            key=lambda row: tuple(row[0]["sort_key"])
        )
        expected_priority_items = [row[1] for row in expected_priority_rows]
        actual_priority_items = [
            item for item in (analysis.get("recommendations") or [])
            if isinstance(item, dict)
        ]
        if actual_priority_items != expected_priority_items:
            errors.append("recommendation priority order mismatch")
        for expected_rank, item in enumerate(actual_priority_items, start=1):
            priority_rank = item.get("priority_rank")
            if (
                isinstance(priority_rank, bool)
                or not isinstance(priority_rank, int)
                or priority_rank != expected_rank
            ):
                errors.append(
                    f"recommendation {expected_rank} priority rank mismatch"
                )
        expected_priority_policy = (
            deterministic_recommendation_priority_policy(
                expected_priority_items
            )
        )
        raw_priority_policy = analysis.get("recommendation_priority_policy")
        policy_type_valid = (
            isinstance(raw_priority_policy, dict)
            and isinstance(
                raw_priority_policy.get("recommendation_count"), int
            )
            and not isinstance(
                raw_priority_policy.get("recommendation_count"), bool
            )
            and all(
                isinstance(raw_priority_policy.get(field), bool)
                for field in (
                    "model_order_influences_rank",
                    "historical_analogs_influence_rank",
                    "source_ticket_severity_influences_rank",
                )
            )
        )
        if (
            raw_priority_policy != expected_priority_policy
            or not policy_type_valid
        ):
            errors.append("recommendation priority policy mismatch")
    duplicates = [
        value for value in dict.fromkeys(hypothesis_ids)
        if hypothesis_ids.count(value) > 1
    ]
    if duplicates:
        errors.append("duplicate hypothesis_id values: " + ", ".join(duplicates))
    if schema_version >= 6:
        expected_supported = sum(
            bool(item.get("historical_analog_refs"))
            for item in (analysis.get("recommendations") or [])
            if isinstance(item, dict)
        )
        reported_supported = analysis.get(
            "recommendations_with_historical_support_count"
        )
        if (
            reported_supported is not None
            and reported_supported != expected_supported
        ):
            errors.append("historical support recommendation count mismatch")
    if schema_version >= 8:
        try:
            from summarize_tr_bugs import (
                GROUNDING_AUDIT_CHECKS,
                GROUNDING_AUDIT_REASON_BY_CHECK,
                GROUNDING_AUDIT_WITHHOLD_REASONS,
                GROUNDING_HUMAN_REVIEW_SAMPLE_LIMIT,
                GROUNDING_REPAIRABLE_CHECKS,
                GROUNDING_REPAIR_TRACKED_FIELDS,
                _select_grounding_human_review_samples,
                recommendation_coverage_signature,
                stable_candidate_audit_id,
                stable_hypothesis_id,
            )
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import (
                GROUNDING_AUDIT_CHECKS,
                GROUNDING_AUDIT_REASON_BY_CHECK,
                GROUNDING_AUDIT_WITHHOLD_REASONS,
                GROUNDING_HUMAN_REVIEW_SAMPLE_LIMIT,
                GROUNDING_REPAIRABLE_CHECKS,
                GROUNDING_REPAIR_TRACKED_FIELDS,
                _select_grounding_human_review_samples,
                recommendation_coverage_signature,
                stable_candidate_audit_id,
                stable_hypothesis_id,
            )
        audit = analysis.get("candidate_grounding_audit")
        if not isinstance(audit, dict) or audit.get("schema_version") != 1:
            errors.append("candidate grounding audit is missing or invalid")
        else:
            gate_withheld_records = []
            count_fields = (
                "generated_candidate_count", "duplicate_candidate_count",
                "audited_candidate_count", "supported_candidate_count",
                "withheld_candidate_count",
                "synthesized_recommendation_count",
                "duplicate_final_recommendation_count",
                "final_recommendation_audited_count",
                "final_recommendation_supported_count",
                "final_recommendation_withheld_count",
            )
            counts = {}
            for field in count_fields:
                value = audit.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    errors.append(f"candidate grounding audit has invalid {field}")
                else:
                    counts[field] = value
            if schema_version >= 24:
                discovery = audit.get("candidate_discovery_audit")
                discovery_count_fields = (
                    "primary_candidate_count",
                    "coverage_challenge_candidate_count",
                    "coverage_challenge_distinct_count",
                    "coverage_overlap_filtered_count",
                    "grounding_input_candidate_count",
                    "unique_grounding_input_count",
                    "exact_duplicate_candidate_count",
                ) + (
                    (
                        "adjacency_gate_evaluated_count",
                        "adjacency_gate_supported_count",
                        "adjacency_gate_withheld_count",
                    ) if schema_version >= 25 else ()
                ) + (
                    ("evidence_ref_completion_applied_count",)
                    if schema_version >= 26 else ()
                ) + (
                    ("reviewed_alias_supported_count",)
                    if schema_version >= 27 else ()
                ) + (
                    (
                        "shared_path_gate_supported_count",
                        "shared_path_gate_withheld_count",
                    ) if schema_version >= 29 else ()
                )
                discovery_counts = {}
                discovery_lists = {}
                if (
                    not isinstance(discovery, dict)
                    or discovery.get("schema_version") != 1
                    or discovery.get("policy")
                    != "primary_plus_distinct_coverage_challenge_v1"
                ):
                    errors.append("candidate discovery audit is missing or invalid")
                    discovery = {}
                for field in discovery_count_fields:
                    value = discovery.get(field)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        errors.append(
                            f"candidate discovery audit has invalid {field}"
                        )
                    else:
                        discovery_counts[field] = value
                discovery_id_list_fields = [
                    ("primary_candidate_ids", "primary_candidate_count"),
                    (
                        "coverage_challenge_candidate_ids",
                        "coverage_challenge_candidate_count",
                    ),
                    (
                        "coverage_challenge_distinct_candidate_ids",
                        "coverage_challenge_distinct_count",
                    ),
                ]
                if schema_version >= 25:
                    discovery_id_list_fields.append((
                        "adjacency_gate_supported_candidate_ids",
                        "adjacency_gate_supported_count",
                    ))
                if schema_version >= 26:
                    discovery_id_list_fields.append((
                        "evidence_ref_completion_candidate_ids",
                        "evidence_ref_completion_applied_count",
                    ))
                if schema_version >= 27:
                    discovery_id_list_fields.append((
                        "reviewed_alias_supported_candidate_ids",
                        "reviewed_alias_supported_count",
                    ))
                if schema_version >= 29:
                    discovery_id_list_fields.append((
                        "shared_path_gate_withheld_candidate_ids",
                        "shared_path_gate_withheld_count",
                    ))
                for field, count_field in discovery_id_list_fields:
                    values = discovery.get(field)
                    expected_count = discovery_counts.get(count_field)
                    if (
                        not isinstance(values, list)
                        or any(
                            not isinstance(candidate_id, str)
                            or not re.fullmatch(
                                r"TRC-[A-F0-9]{12}", candidate_id
                            )
                            for candidate_id in (values or [])
                        )
                        or (
                            expected_count is not None
                            and len(values) != expected_count
                        )
                    ):
                        errors.append(
                            f"candidate discovery audit has invalid {field}"
                        )
                        values = []
                    discovery_lists[field] = values
                if schema_version >= 38:
                    malformed_dropped = discovery.get(
                        "malformed_candidate_dropped_count"
                    )
                    malformed_records = discovery.get(
                        "malformed_candidate_drop_records"
                    )
                    if (
                        isinstance(malformed_dropped, bool)
                        or not isinstance(malformed_dropped, int)
                        or malformed_dropped < 0
                    ):
                        errors.append(
                            "candidate discovery audit has invalid "
                            "malformed_candidate_dropped_count"
                        )
                    elif (
                        not isinstance(malformed_records, list)
                        or len(malformed_records) != malformed_dropped
                        or not all(
                            isinstance(record, dict)
                            and isinstance(
                                record.get("rejection_reasons"), list
                            )
                            and record["rejection_reasons"]
                            and all(
                                isinstance(reason, str)
                                for reason in record["rejection_reasons"]
                            )
                            for record in malformed_records
                        )
                    ):
                        errors.append(
                            "candidate discovery audit has invalid "
                            "malformed_candidate_drop_records"
                        )
                def discovery_overlap_record_valid(record: object) -> bool:
                    if (
                        not isinstance(record, dict)
                        or set(record) != {
                            "candidate_id", "coverage_signature",
                            "equivalent_to_candidate_id", "candidate",
                        }
                        or not re.fullmatch(
                            r"TRC-[A-F0-9]{12}",
                            str(record.get("candidate_id") or ""),
                        )
                        or not re.fullmatch(
                            r"TRD-[A-F0-9]{12}",
                            str(record.get("coverage_signature") or ""),
                        )
                        or not re.fullmatch(
                            r"TRC-[A-F0-9]{12}",
                            str(record.get("equivalent_to_candidate_id") or ""),
                        )
                        or not isinstance(record.get("candidate"), dict)
                    ):
                        return False
                    try:
                        return (
                            stable_candidate_audit_id(record["candidate"])
                            == record["candidate_id"]
                            and recommendation_coverage_signature(
                                record["candidate"]
                            ) == record["coverage_signature"]
                        )
                    except (TypeError, ValueError):
                        return False

                overlap_records = discovery.get(
                    "coverage_overlap_filtered_candidates"
                )
                if (
                    not isinstance(overlap_records, list)
                    or any(
                        not discovery_overlap_record_valid(record)
                        for record in (overlap_records or [])
                    )
                    or (
                        discovery_counts.get("coverage_overlap_filtered_count")
                        is not None
                        and len(overlap_records)
                        != discovery_counts["coverage_overlap_filtered_count"]
                    )
                ):
                    errors.append(
                        "candidate discovery coverage-overlap audit is invalid"
                    )
                    overlap_records = []
                gate_supported_records = []
                gate_withheld_records = []
                gate_supported_ids_by_origin = defaultdict(list)
                gate_withheld_ids_by_origin = defaultdict(list)
                if schema_version >= 25:
                    gate_supported_records = discovery.get(
                        "adjacency_gate_supported_candidates"
                    )
                    gate_withheld_records = discovery.get(
                        "adjacency_gate_withheld_candidates"
                    )
                    if not isinstance(gate_supported_records, list):
                        errors.append(
                            "candidate discovery adjacency gate supported audit "
                            "is invalid"
                        )
                        gate_supported_records = []
                    if not isinstance(gate_withheld_records, list):
                        errors.append(
                            "candidate discovery adjacency gate withheld audit "
                            "is invalid"
                        )
                        gate_withheld_records = []
                    if len(gate_supported_records) != discovery_counts.get(
                        "adjacency_gate_supported_count"
                    ):
                        errors.append(
                            "candidate discovery adjacency gate supported count "
                            "mismatch"
                        )
                    if len(gate_withheld_records) != discovery_counts.get(
                        "adjacency_gate_withheld_count"
                    ):
                        errors.append(
                            "candidate discovery adjacency gate withheld count "
                            "mismatch"
                        )
                    for expected_supported, records, record_type in (
                        (True, gate_supported_records, "supported"),
                        (False, gate_withheld_records, "withheld"),
                    ):
                        expected_keys = {
                            "candidate_id", "discovery_origin", "candidate",
                            "adjacency_support_basis",
                        } | ({
                            "reason", "target_category", "area",
                            "source_bug_codes", "evidence_refs",
                            "failed_checks", "audit_stage",
                            "human_review_sample",
                        } if not expected_supported else set())
                        # Schema 35 records why a direct-fix restatement was
                        # withheld alongside the adjacency basis.
                        direct_fix_keys = (
                            expected_keys | {"direct_fix_basis"}
                            if schema_version >= 35 and not expected_supported
                            else expected_keys
                        )
                        for index, record in enumerate(records, start=1):
                            label = (
                                f"adjacency gate {record_type} candidate {index}"
                            )
                            if schema_version >= 29:
                                allowed_reasons = {
                                    "adjacent_target_absent_from_current_signal",
                                    "shared_path_not_concrete",
                                }
                            else:
                                allowed_reasons = {
                                    "adjacent_target_absent_from_current_signal"
                                }
                            if schema_version >= 35:
                                allowed_reasons = allowed_reasons | {
                                    "direct_fix_restatement"
                                }
                            record_is_direct_fix = (
                                not expected_supported
                                and record.get("reason") == "direct_fix_restatement"
                            ) if isinstance(record, dict) else False
                            if (
                                not isinstance(record, dict)
                                or set(record) != (
                                    direct_fix_keys if record_is_direct_fix
                                    else expected_keys
                                )
                                or record.get("discovery_origin") not in {
                                    "primary_discovery", "coverage_challenge",
                                }
                                or not isinstance(record.get("candidate"), dict)
                                or (
                                    not expected_supported
                                    and record.get("reason") not in allowed_reasons
                                )
                            ):
                                errors.append(f"{label} is invalid")
                                continue
                            candidate = record["candidate"]
                            candidate_id = str(record.get("candidate_id") or "")
                            try:
                                identity_valid = (
                                    re.fullmatch(
                                        r"TRC-[A-F0-9]{12}", candidate_id
                                    ) is not None
                                    and stable_candidate_audit_id(candidate)
                                    == candidate_id
                                )
                            except (TypeError, ValueError):
                                identity_valid = False
                            if not identity_valid:
                                errors.append(f"{label} identity mismatch")
                            origin = str(record["discovery_origin"])
                            validate_generation_origins(
                                candidate, label, [origin],
                            )
                            if record.get(
                                "adjacency_support_basis"
                            ) != candidate.get("adjacency_support_basis"):
                                errors.append(
                                    f"{label} adjacency support record mismatch"
                                )
                            validate_confidence_refs(candidate, label)
                            validate_adjacency_support(
                                candidate, label,
                                # A direct-fix restatement is withheld even
                                # though its target and shared path are both
                                # grounded, so it keeps a supported basis.
                                expected_supported=(
                                    True if record_is_direct_fix
                                    else expected_supported
                                ),
                                require_ref_completion=True,
                            )
                            if not expected_supported and (
                                record.get("target_category")
                                != candidate.get("target_category")
                                or record.get("area") != candidate.get("area")
                                or record.get("source_bug_codes")
                                != candidate.get("source_bug_codes")
                                or record.get("evidence_refs")
                                != candidate.get("evidence_refs")
                                or record.get("failed_checks") != {
                                    "adjacent_target_absent_from_current_signal":
                                        ["target_supported"],
                                    "direct_fix_restatement":
                                        ["adjacent_not_direct_fix"],
                                }.get(
                                    str(record.get("reason") or ""),
                                    ["shared_path_concrete"],
                                )
                                or record.get("audit_stage")
                                != "deterministic_adjacency_gate"
                                or not isinstance(
                                    record.get("human_review_sample"), bool
                                )
                            ):
                                errors.append(
                                    f"{label} review provenance is invalid"
                                )
                            destination = (
                                gate_supported_ids_by_origin
                                if expected_supported
                                else gate_withheld_ids_by_origin
                            )
                            destination[origin].append(candidate_id)
                    gate_supported_ids = discovery_lists.get(
                        "adjacency_gate_supported_candidate_ids", []
                    )
                    if gate_supported_ids != [
                        str(record.get("candidate_id") or "")
                        for record in gate_supported_records
                        if isinstance(record, dict)
                    ]:
                        errors.append(
                            "candidate discovery adjacency gate supported identity "
                            "mismatch"
                        )
                    if discovery_counts.get(
                        "adjacency_gate_evaluated_count"
                    ) != (
                        discovery_counts.get("primary_candidate_count", 0)
                        + discovery_counts.get(
                            "coverage_challenge_candidate_count", 0
                        )
                    ):
                        errors.append(
                            "candidate discovery adjacency gate evaluation count "
                            "mismatch"
                        )
                    if schema_version >= 29:
                        shared_path_withheld_ids = [
                            str(record.get("candidate_id") or "")
                            for record in gate_withheld_records
                            if isinstance(record, dict)
                            and record.get("reason") == "shared_path_not_concrete"
                        ]
                        if discovery_counts.get(
                            "shared_path_gate_supported_count"
                        ) != len(gate_supported_records):
                            errors.append(
                                "candidate discovery shared-path supported count mismatch"
                            )
                        if discovery_counts.get(
                            "shared_path_gate_withheld_count"
                        ) != len(shared_path_withheld_ids):
                            errors.append(
                                "candidate discovery shared-path withheld count mismatch"
                            )
                        if discovery_lists.get(
                            "shared_path_gate_withheld_candidate_ids", []
                        ) != shared_path_withheld_ids:
                            errors.append(
                                "candidate discovery shared-path withheld identity mismatch"
                            )
                    if discovery_counts.get(
                        "adjacency_gate_evaluated_count"
                    ) != (
                        len(gate_supported_records) + len(gate_withheld_records)
                    ):
                        errors.append(
                            "candidate discovery adjacency gate decision count "
                            "mismatch"
                        )
                    if schema_version >= 26:
                        completion_records = discovery.get(
                            "evidence_ref_completion_candidates"
                        )
                        if not isinstance(completion_records, list):
                            errors.append(
                                "candidate discovery evidence reference completion "
                                "audit is invalid"
                            )
                            completion_records = []
                        if len(completion_records) != discovery_counts.get(
                            "evidence_ref_completion_applied_count"
                        ):
                            errors.append(
                                "candidate discovery evidence reference completion "
                                "count mismatch"
                            )
                        completion_ids = discovery_lists.get(
                            "evidence_ref_completion_candidate_ids", []
                        )
                        if completion_ids != [
                            str(record.get("candidate_id") or "")
                            for record in completion_records
                            if isinstance(record, dict)
                        ]:
                            errors.append(
                                "candidate discovery evidence reference completion "
                                "identity mismatch"
                            )
                        gate_candidates = [
                            record
                            for record in (
                                gate_supported_records + gate_withheld_records
                            )
                            if isinstance(record, dict)
                            and isinstance(record.get("candidate"), dict)
                        ]
                        applied_gate_candidates = [
                            record for record in gate_candidates
                            if (
                                record["candidate"].get(
                                    "evidence_ref_completion_basis"
                                ) or {}
                            ).get("applied") is True
                        ]
                        if len(applied_gate_candidates) != len(completion_records):
                            errors.append(
                                "candidate discovery completed reference partition "
                                "mismatch"
                            )
                        for index, record in enumerate(
                            completion_records, start=1
                        ):
                            label = (
                                f"evidence reference completion candidate {index}"
                            )
                            if not isinstance(record, dict) or set(record) != {
                                "candidate_id", "discovery_origin", "area",
                                "source_bug_codes", "original_evidence_refs",
                                "added_evidence_refs", "final_evidence_refs",
                            }:
                                errors.append(f"{label} is invalid")
                                continue
                            matches = [
                                row["candidate"] for row in gate_candidates
                                if row.get("candidate_id")
                                == record.get("candidate_id")
                                and row.get("discovery_origin")
                                == record.get("discovery_origin")
                            ]
                            if not matches:
                                errors.append(f"{label} lacks gate provenance")
                                continue
                            candidate = matches[0]
                            completion = candidate.get(
                                "evidence_ref_completion_basis"
                            ) or {}
                            if (
                                completion.get("applied") is not True
                                or record.get("area") != candidate.get("area")
                                or record.get("source_bug_codes")
                                != candidate.get("source_bug_codes")
                                or record.get("original_evidence_refs")
                                != completion.get("original_evidence_refs")
                                or record.get("added_evidence_refs")
                                != completion.get("added_evidence_refs")
                                or record.get("final_evidence_refs")
                                != completion.get("final_evidence_refs")
                            ):
                                errors.append(f"{label} provenance mismatch")
                        for field in (
                            "evidence_ref_completion_scans_uncited_bugcodes",
                            "evidence_ref_completion_changes_source_bugcodes",
                            "model_critic_can_override_evidence_ref_completion",
                        ):
                            if discovery.get(field) is not False:
                                errors.append(
                                    f"candidate discovery {field} must be false"
                                )
                    if schema_version >= 27:
                        alias_records = discovery.get(
                            "reviewed_alias_supported_candidates"
                        )
                        if not isinstance(alias_records, list):
                            errors.append(
                                "candidate discovery reviewed alias audit is invalid"
                            )
                            alias_records = []
                        if len(alias_records) != discovery_counts.get(
                            "reviewed_alias_supported_count"
                        ):
                            errors.append(
                                "candidate discovery reviewed alias count mismatch"
                            )
                        alias_ids = discovery_lists.get(
                            "reviewed_alias_supported_candidate_ids", []
                        )
                        if alias_ids != [
                            str(record.get("candidate_id") or "")
                            for record in alias_records
                            if isinstance(record, dict)
                        ]:
                            errors.append(
                                "candidate discovery reviewed alias identity mismatch"
                            )
                        for index, record in enumerate(alias_records, start=1):
                            label = f"reviewed alias candidate {index}"
                            if not isinstance(record, dict) or set(record) != {
                                "candidate_id", "discovery_origin", "area",
                                "reviewed_alias_matches",
                            }:
                                errors.append(f"{label} is invalid")
                                continue
                            matches = [
                                row["candidate"] for row in gate_supported_records
                                if isinstance(row, dict)
                                and isinstance(row.get("candidate"), dict)
                                and row.get("candidate_id")
                                == record.get("candidate_id")
                                and row.get("discovery_origin")
                                == record.get("discovery_origin")
                            ]
                            if not matches:
                                errors.append(f"{label} lacks supported gate provenance")
                                continue
                            candidate = matches[0]
                            basis = candidate.get("adjacency_support_basis") or {}
                            if (
                                basis.get("reviewed_aliases_influence_support")
                                is not True
                                or record.get("area") != candidate.get("area")
                                or record.get("reviewed_alias_matches")
                                != basis.get("reviewed_alias_matches")
                            ):
                                errors.append(f"{label} provenance mismatch")
                        for field in (
                            "reviewed_aliases_influence_confidence",
                            "reviewed_aliases_influence_risk",
                            "reviewed_aliases_influence_priority",
                        ):
                            if discovery.get(field) is not False:
                                errors.append(
                                    f"candidate discovery {field} must be false"
                                )
                origin_records = discovery.get("unique_candidate_origins")
                allowed_generation_origins = {
                    "primary_discovery", "coverage_challenge",
                }
                origin_record_ids = []
                if not isinstance(origin_records, list):
                    errors.append("candidate discovery origin registry is invalid")
                    origin_records = []
                for index, record in enumerate(origin_records, start=1):
                    if not isinstance(record, dict) or set(record) != {
                        "candidate_id", "origins",
                    }:
                        errors.append(
                            f"candidate discovery origin {index} is invalid"
                        )
                        continue
                    candidate_id = str(record.get("candidate_id") or "")
                    origins = record.get("origins")
                    if (
                        not re.fullmatch(r"TRC-[A-F0-9]{12}", candidate_id)
                        or not isinstance(origins, list)
                        or not origins
                        or len(origins) != len(set(origins))
                        or any(
                            origin not in allowed_generation_origins
                            for origin in (origins or [])
                        )
                    ):
                        errors.append(
                            f"candidate discovery origin {index} is invalid"
                        )
                        continue
                    if candidate_id in discovery_origins_by_id:
                        errors.append(
                            "candidate discovery origin registry has duplicate identities"
                        )
                        continue
                    origin_record_ids.append(candidate_id)
                    discovery_origins_by_id[candidate_id] = list(origins)
                if (
                    discovery_counts.get("unique_grounding_input_count")
                    is not None
                    and len(origin_records)
                    != discovery_counts["unique_grounding_input_count"]
                ):
                    errors.append(
                        "candidate discovery origin registry count mismatch"
                    )
                discovery_false_fields = (
                    "challenge_coverage_overlap_can_reach_grounding",
                    "generation_origin_influences_confidence",
                    "generation_origin_influences_risk",
                    "generation_origin_influences_priority",
                ) + (
                    (
                        "adjacency_gate_withheld_can_reach_grounding",
                        "model_critic_can_override_adjacency_gate",
                    ) if schema_version >= 25 else ()
                ) + (
                    ("model_critic_can_override_shared_path_gate",)
                    if schema_version >= 29 else ()
                )
                for field in discovery_false_fields:
                    if discovery.get(field) is not False:
                        errors.append(
                            f"candidate discovery audit has invalid {field}"
                        )
                primary_ids = discovery_lists.get("primary_candidate_ids", [])
                challenge_ids = discovery_lists.get(
                    "coverage_challenge_candidate_ids", []
                )
                distinct_ids = discovery_lists.get(
                    "coverage_challenge_distinct_candidate_ids", []
                )
                overlap_ids = [
                    str(record.get("candidate_id") or "")
                    for record in overlap_records
                ]
                if schema_version >= 25:
                    primary_grounding_ids = gate_supported_ids_by_origin[
                        "primary_discovery"
                    ]
                    challenge_supported_ids = gate_supported_ids_by_origin[
                        "coverage_challenge"
                    ]
                    if Counter(primary_ids) != (
                        Counter(primary_grounding_ids)
                        + Counter(gate_withheld_ids_by_origin["primary_discovery"])
                    ):
                        errors.append(
                            "candidate discovery primary adjacency gate partition "
                            "mismatch"
                        )
                    if Counter(challenge_ids) != (
                        Counter(challenge_supported_ids)
                        + Counter(gate_withheld_ids_by_origin["coverage_challenge"])
                    ):
                        errors.append(
                            "candidate discovery challenge adjacency gate partition "
                            "mismatch"
                        )
                    challenge_partition_source = challenge_supported_ids
                else:
                    primary_grounding_ids = primary_ids
                    challenge_partition_source = challenge_ids
                if Counter(challenge_partition_source) != (
                    Counter(distinct_ids) + Counter(overlap_ids)
                ):
                    errors.append(
                        "candidate discovery challenge partition mismatch"
                    )
                grounding_ids = primary_grounding_ids + distinct_ids
                if (
                    discovery_counts.get("grounding_input_candidate_count")
                    is not None
                    and len(grounding_ids)
                    != discovery_counts["grounding_input_candidate_count"]
                ):
                    errors.append(
                        "candidate discovery grounding input count mismatch"
                    )
                if set(grounding_ids) != set(origin_record_ids):
                    errors.append(
                        "candidate discovery unique identity registry mismatch"
                    )
                if any(
                    str(record.get("equivalent_to_candidate_id") or "")
                    not in set(origin_record_ids)
                    for record in overlap_records
                ):
                    errors.append(
                        "candidate discovery overlap owner exceeds grounding registry"
                    )
                if (
                    discovery_counts.get("exact_duplicate_candidate_count")
                    is not None
                    and len(grounding_ids) - len(set(grounding_ids))
                    != discovery_counts["exact_duplicate_candidate_count"]
                ):
                    errors.append(
                        "candidate discovery exact duplicate count mismatch"
                    )
                for candidate_id, origins in discovery_origins_by_id.items():
                    expected_origins = set()
                    if candidate_id in primary_grounding_ids:
                        expected_origins.add("primary_discovery")
                    if candidate_id in distinct_ids:
                        expected_origins.add("coverage_challenge")
                    if set(origins) != expected_origins:
                        errors.append(
                            "candidate discovery generation origin mismatch"
                        )
                if counts:
                    if discovery_counts.get(
                        "grounding_input_candidate_count"
                    ) != counts.get("generated_candidate_count"):
                        errors.append(
                            "candidate discovery generated count mismatch"
                        )
                    if discovery_counts.get(
                        "unique_grounding_input_count"
                    ) != counts.get("audited_candidate_count"):
                        errors.append(
                            "candidate discovery audited count mismatch"
                        )
                    if discovery_counts.get(
                        "exact_duplicate_candidate_count"
                    ) != counts.get("duplicate_candidate_count"):
                        errors.append(
                            "candidate discovery duplicate count mismatch"
                        )
            coverage_equivalent_count = 0
            final_grounded_count = 0
            synthesis_backfill_count = 0
            published_recommendation_count = len(
                analysis.get("recommendations") or []
            )
            portfolio_excluded_count = 0
            if schema_version >= 13:
                value = audit.get("coverage_equivalent_recommendation_count")
                if (
                    isinstance(value, bool) or not isinstance(value, int)
                    or value < 0
                ):
                    errors.append(
                        "candidate grounding audit has invalid "
                        "coverage_equivalent_recommendation_count"
                    )
                else:
                    coverage_equivalent_count = value
            if schema_version >= 14:
                value = audit.get("final_recommendation_grounded_count")
                if (
                    isinstance(value, bool) or not isinstance(value, int)
                    or value < 0
                ):
                    errors.append(
                        "candidate grounding audit has invalid "
                        "final_recommendation_grounded_count"
                    )
                else:
                    final_grounded_count = value
            if schema_version >= 23:
                for field, destination in (
                    ("synthesis_backfill_candidate_count", "backfill"),
                    ("published_recommendation_count", "published"),
                    ("portfolio_excluded_recommendation_count", "excluded"),
                ):
                    value = audit.get(field)
                    if (
                        isinstance(value, bool) or not isinstance(value, int)
                        or value < 0
                    ):
                        errors.append(
                            f"candidate grounding audit has invalid {field}"
                        )
                        continue
                    if destination == "backfill":
                        synthesis_backfill_count = value
                    elif destination == "published":
                        published_recommendation_count = value
                    else:
                        portfolio_excluded_count = value
            if len(counts) == len(count_fields):
                if counts["generated_candidate_count"] != (
                    counts["audited_candidate_count"]
                    + counts["duplicate_candidate_count"]
                ):
                    errors.append("candidate grounding audit generation count mismatch")
                if counts["audited_candidate_count"] != (
                    counts["supported_candidate_count"]
                    + counts["withheld_candidate_count"]
                ):
                    errors.append("candidate grounding audit decision count mismatch")
                if len(analysis.get("recommendations") or []) > counts[
                    "supported_candidate_count"
                ]:
                    errors.append(
                        "published recommendations exceed supported candidates"
                    )
                expected_final_decisions = (
                    final_grounded_count
                    if schema_version >= 14 else
                    counts["final_recommendation_supported_count"]
                ) + counts["final_recommendation_withheld_count"]
                if counts["final_recommendation_audited_count"] != (
                    expected_final_decisions
                ):
                    errors.append(
                        "final recommendation grounding decision count mismatch"
                    )
                expected_synthesized = (
                    counts["final_recommendation_audited_count"]
                    + counts["duplicate_final_recommendation_count"]
                    - (
                        synthesis_backfill_count
                        if schema_version >= 23 else 0
                    )
                    + (
                        0 if schema_version >= 14
                        else coverage_equivalent_count
                    )
                )
                if counts["synthesized_recommendation_count"] != (
                    expected_synthesized
                ):
                    errors.append(
                        "final recommendation grounding generation count mismatch"
                    )
                if schema_version >= 23:
                    if len(analysis.get("recommendations") or []) != (
                        published_recommendation_count
                    ):
                        errors.append(
                            "published recommendation count mismatch"
                        )
                    if counts["final_recommendation_supported_count"] != (
                        published_recommendation_count
                        + portfolio_excluded_count
                    ):
                        errors.append(
                            "supported recommendation portfolio count mismatch"
                        )
                elif len(analysis.get("recommendations") or []) != counts[
                    "final_recommendation_supported_count"
                ]:
                    errors.append(
                        "published recommendation grounding count mismatch"
                    )
                if schema_version >= 14 and final_grounded_count != (
                    counts["final_recommendation_supported_count"]
                    + coverage_equivalent_count
                ):
                    errors.append(
                        "grounded recommendation coverage count mismatch"
                    )
            if schema_version >= 13:
                coverage_records = audit.get(
                    "coverage_equivalent_recommendations"
                )
                if not isinstance(coverage_records, list):
                    errors.append(
                        "coverage-equivalent recommendation audit is invalid"
                    )
                    coverage_records = []
                elif len(coverage_records) != coverage_equivalent_count:
                    errors.append(
                        "coverage-equivalent recommendation count mismatch"
                    )
                portfolio_records = []
                portfolio_recommendations = []
                if schema_version >= 23:
                    portfolio_records = audit.get(
                        "portfolio_excluded_recommendations"
                    )
                    if not isinstance(portfolio_records, list):
                        errors.append(
                            "portfolio-excluded recommendation audit is invalid"
                        )
                        portfolio_records = []
                    elif len(portfolio_records) != portfolio_excluded_count:
                        errors.append(
                            "portfolio-excluded recommendation count mismatch"
                        )
                    if audit.get(
                        "published_recommendation_limit"
                    ) != published_recommendation_limit:
                        errors.append("published recommendation limit mismatch")
                    if published_recommendation_count > published_recommendation_limit:
                        errors.append("published recommendation limit exceeded")
                    for index, record in enumerate(portfolio_records, start=1):
                        label = f"portfolio-excluded recommendation {index}"
                        if not isinstance(record, dict):
                            errors.append(f"{label} is not an object")
                            continue
                        item = record.get("recommendation")
                        if not isinstance(item, dict):
                            errors.append(f"{label} lacks claim provenance")
                            continue
                        observe_selection_origin(item, label)
                        validate_adjacency_support(item, label)
                        expected_position = (
                            published_recommendation_count + index
                            if schema_version < 31 else None
                        )
                        if (
                            isinstance(record.get("portfolio_position"), bool)
                            or not isinstance(
                                record.get("portfolio_position"), int
                            )
                            or record.get("portfolio_position") <= 0
                            or (
                                expected_position is not None
                                and record.get("portfolio_position")
                                != expected_position
                            )
                        ):
                            errors.append(f"{label} position mismatch")
                        expected_reasons = (
                            {
                                "equal_risk_confidence_tier_diversity_cutoff",
                                "below_risk_confidence_portfolio_cutoff",
                            }
                            if schema_version >= 31 else {
                                "below_deterministic_published_portfolio_limit"
                            }
                        )
                        if record.get("reason") not in expected_reasons:
                            errors.append(f"{label} has invalid reason")
                        if item.get("priority_rank") is not None:
                            errors.append(f"{label} must not have a visible rank")
                        if (
                            deterministic_confidence_basis is not None
                            and confidence_evidence_map
                        ):
                            validate_confidence_refs(item, label)
                            expected_basis = deterministic_confidence_basis(
                                [
                                    str(code) for code in (
                                        item.get("source_bug_codes") or []
                                    )
                                ],
                                list(item.get("evidence_refs") or []),
                                list(item.get("historical_analog_refs") or []),
                                confidence_evidence_map,
                                correlate_sources=True,
                                target_area=str(item.get("area") or ""),
                            )
                            if (
                                item.get("confidence_basis") != expected_basis
                                or item.get("confidence") != expected_basis["label"]
                            ):
                                errors.append(f"{label} confidence mismatch")
                        if deterministic_risk_basis is not None:
                            expected_risk = (
                                deterministic_recommendation_risk_basis(
                                    item, checklist_evidence_map,
                                )
                                if schema_version >= 30
                                and deterministic_recommendation_risk_basis
                                is not None
                                else deterministic_risk_basis(
                                    str(item.get("expected_failure_mode") or ""),
                                    str(item.get("confidence") or ""),
                                    negation_aware=True,
                                )
                            )
                            if (
                                item.get("risk_basis") != expected_risk
                                or item.get("risk") != expected_risk["label"]
                                or item.get("impact_basis")
                                != expected_risk["impact_basis"]
                            ):
                                errors.append(f"{label} risk mismatch")
                        if (
                            deterministic_recommendation_priority_basis
                            is not None
                        ):
                            expected_priority = (
                                deterministic_recommendation_priority_basis(
                                    item, confidence_evidence_map,
                                )
                            )
                            if (
                                item.get("priority_basis") != expected_priority
                                or not priority_basis_types_valid(
                                    item.get("priority_basis")
                                )
                            ):
                                errors.append(f"{label} priority mismatch")
                        validate_category_basis(item, label)
                        portfolio_recommendations.append(item)
                retained_by_signature = {}
                retained_items = [
                    item for item in (
                        analysis.get("recommendations") or []
                    ) if isinstance(item, dict)
                ] + portfolio_recommendations
                for item in retained_items:
                    if not isinstance(item, dict):
                        continue
                    signature = recommendation_coverage_signature(item)
                    if signature in retained_by_signature:
                        errors.append(
                            "supported recommendations contain duplicate coverage"
                        )
                    retained_by_signature[signature] = item
                if schema_version >= 31:
                    if deterministic_recommendation_portfolio is None:
                        errors.append(
                            "recommendation portfolio validator is unavailable"
                        )
                    else:
                        (
                            expected_selected,
                            expected_excluded,
                            expected_portfolio_policy,
                        ) = deterministic_recommendation_portfolio(
                            retained_items,
                            limit=published_recommendation_limit,
                        )
                        displayed_items = [
                            item for item in (
                                analysis.get("recommendations") or []
                            ) if isinstance(item, dict)
                        ]
                        if displayed_items != expected_selected:
                            errors.append(
                                "supported recommendation portfolio selection mismatch"
                            )
                        if portfolio_recommendations != expected_excluded:
                            errors.append(
                                "portfolio-excluded recommendation order mismatch"
                            )
                        if analysis.get(
                            "recommendation_portfolio_policy"
                        ) != expected_portfolio_policy:
                            errors.append(
                                "recommendation portfolio policy mismatch"
                            )
                        expected_exclusion_by_id = {
                            row["hypothesis_id"]: row
                            for row in expected_portfolio_policy[
                                "exclusion_records"
                            ]
                        }
                        for record in portfolio_records:
                            item = record.get("recommendation") or {}
                            item_id = stable_hypothesis_id(item)
                            expected_exclusion = expected_exclusion_by_id.get(
                                item_id
                            )
                            if (
                                expected_exclusion is None
                                or record.get("portfolio_position")
                                != expected_exclusion[
                                    "original_priority_position"
                                ]
                                or record.get("reason")
                                != expected_exclusion["reason"]
                            ):
                                errors.append(
                                    "portfolio-excluded recommendation selection "
                                    "provenance mismatch"
                                )
                elif schema_version >= 23:
                    expected_portfolio = sorted(
                        retained_items,
                        key=lambda item: tuple(
                            deterministic_recommendation_priority_basis(
                                item, confidence_evidence_map,
                            )["sort_key"]
                        ),
                    )
                    if retained_items != expected_portfolio:
                        errors.append("supported recommendation portfolio order mismatch")
                suppressed_ids = []
                for index, record in enumerate(coverage_records, start=1):
                    label = f"coverage-equivalent recommendation {index}"
                    if not isinstance(record, dict):
                        errors.append(f"{label} is not an object")
                        continue
                    signature = str(record.get("coverage_signature") or "")
                    suppressed = record.get("suppressed_recommendation")
                    if not re.fullmatch(r"TRD-[A-F0-9]{12}", signature):
                        errors.append(f"{label} has invalid signature")
                    if not isinstance(suppressed, dict):
                        errors.append(f"{label} lacks suppressed claim provenance")
                        continue
                    observe_selection_origin(suppressed, label)
                    validate_adjacency_support(suppressed, label)
                    if (
                        schema_version >= 15
                        and deterministic_confidence_basis is not None
                    ):
                        validate_confidence_refs(suppressed, label)
                        expected_suppressed_basis = (
                            deterministic_confidence_basis(
                                [
                                    str(code) for code in (
                                        suppressed.get("source_bug_codes") or []
                                    )
                                ],
                                list(suppressed.get("evidence_refs") or []),
                                list(
                                    suppressed.get("historical_analog_refs")
                                    or []
                                ),
                                confidence_evidence_map,
                                correlate_sources=schema_version >= 18,
                                target_area=(
                                    str(suppressed.get("area") or "")
                                    if schema_version >= 19 else None
                                ),
                            )
                        )
                        if suppressed.get(
                            "confidence_basis"
                        ) != expected_suppressed_basis or suppressed.get(
                            "confidence"
                        ) != expected_suppressed_basis["label"]:
                            errors.append(
                                f"{label} confidence provenance mismatch"
                            )
                    if (
                        schema_version >= 16
                        and deterministic_risk_basis is not None
                    ):
                        expected_suppressed_risk = (
                            deterministic_recommendation_risk_basis(
                                suppressed, checklist_evidence_map,
                            )
                            if schema_version >= 30
                            and deterministic_recommendation_risk_basis is not None
                            else deterministic_risk_basis(
                                str(suppressed.get("expected_failure_mode") or ""),
                                str(suppressed.get("confidence") or ""),
                                negation_aware=schema_version >= 21,
                            )
                        )
                        if (
                            suppressed.get("risk_basis")
                            != expected_suppressed_risk
                            or suppressed.get("risk")
                            != expected_suppressed_risk["label"]
                            or suppressed.get("impact_basis")
                            != expected_suppressed_risk["impact_basis"]
                        ):
                            errors.append(f"{label} risk provenance mismatch")
                    if (
                        schema_version >= 22
                        and deterministic_recommendation_priority_basis
                        is not None
                    ):
                        expected_suppressed_priority = (
                            deterministic_recommendation_priority_basis(
                                suppressed, confidence_evidence_map,
                            )
                        )
                        if (
                            suppressed.get("priority_basis")
                            != expected_suppressed_priority
                            or not priority_basis_types_valid(
                                suppressed.get("priority_basis")
                            )
                        ):
                            errors.append(f"{label} priority provenance mismatch")
                    validate_category_basis(suppressed, label)
                    suppressed_id = stable_candidate_audit_id(suppressed)
                    suppressed_ids.append(suppressed_id)
                    kept = retained_by_signature.get(signature)
                    if recommendation_coverage_signature(suppressed) != signature:
                        errors.append(f"{label} suppressed signature mismatch")
                    if kept is None:
                        errors.append(f"{label} lacks a retained equivalent")
                    elif stable_candidate_audit_id(kept) != str(
                        record.get("kept_recommendation_id") or ""
                    ):
                        errors.append(f"{label} retained identity mismatch")
                    elif (
                        schema_version >= 22
                        and deterministic_recommendation_priority_basis
                        is not None
                        and tuple(
                            deterministic_recommendation_priority_basis(
                                kept, confidence_evidence_map,
                            )["sort_key"]
                        ) > tuple(expected_suppressed_priority["sort_key"])
                    ):
                        errors.append(
                            f"{label} retained claim is not highest priority"
                        )
                    if suppressed_id != str(
                        record.get("suppressed_recommendation_id") or ""
                    ):
                        errors.append(f"{label} suppressed identity mismatch")
                    if kept is not None and suppressed_id == stable_candidate_audit_id(
                        kept
                    ):
                        errors.append(f"{label} should be an exact duplicate")
                    expected_selection_basis = (
                        "highest-priority final-grounding-supported equivalent by "
                        "schema-v22 deterministic impact and independent target evidence"
                        if schema_version >= 22 else
                        "highest-ranked final-grounding-supported equivalent by "
                        "qualitative risk, confidence, category, and area"
                        if schema_version >= 14 else
                        "higher qualitative risk, then confidence, then "
                        "deterministic category and area order"
                    )
                    if record.get("selection_basis") != expected_selection_basis:
                        errors.append(f"{label} has invalid selection basis")
                    if schema_version >= 14 and (
                        record.get("suppression_stage")
                        != "after_final_grounding_audit"
                        or record.get("suppressed_grounding_status")
                        != "supported"
                    ):
                        errors.append(
                            f"{label} lacks final-grounding support provenance"
                        )
                if len(suppressed_ids) != len(set(suppressed_ids)):
                    errors.append(
                        "coverage-equivalent recommendation audit has duplicate "
                        "suppressed identities"
                    )
            withheld = audit.get("withheld_candidates")
            if not isinstance(withheld, list):
                errors.append("candidate grounding audit withheld list is invalid")
                withheld = []
            elif (
                "withheld_candidate_count" in counts
                and len(withheld) != counts["withheld_candidate_count"]
            ):
                errors.append("candidate grounding audit withheld count mismatch")
            withheld_ids = []
            actual_reasons = Counter()
            for index, item in enumerate(withheld, start=1):
                if not isinstance(item, dict):
                    errors.append(f"withheld candidate {index} is not an object")
                    continue
                validate_adjacency_support(
                    item, f"withheld candidate {index}",
                    require_ref_completion=True,
                )
                discovery_original_id = str(
                    item.get("original_candidate_id")
                    or item.get("candidate_id") or ""
                )
                if schema_version >= 24:
                    if discovery_original_id not in discovery_origins_by_id:
                        errors.append(
                            f"withheld candidate {index} exceeds discovery registry"
                        )
                    validate_generation_origins(
                        item, f"withheld candidate {index}",
                        discovery_origins_by_id.get(discovery_original_id),
                    )
                candidate_id = str(item.get("candidate_id") or "")
                reason = str(item.get("reason") or "")
                failed_checks = item.get("failed_checks")
                if not re.fullmatch(r"TRC-[A-F0-9]{12}", candidate_id):
                    errors.append(f"withheld candidate {index} has invalid identity")
                else:
                    withheld_ids.append(candidate_id)
                if reason not in GROUNDING_AUDIT_WITHHOLD_REASONS:
                    errors.append(f"withheld candidate {index} has invalid reason")
                else:
                    actual_reasons[reason] += 1
                if (
                    not isinstance(failed_checks, list) or not failed_checks
                    or any(check not in GROUNDING_AUDIT_CHECKS for check in failed_checks)
                    or len(set(failed_checks)) != len(failed_checks)
                ):
                    errors.append(f"withheld candidate {index} has invalid failed checks")
                elif reason in GROUNDING_AUDIT_WITHHOLD_REASONS and reason not in {
                    GROUNDING_AUDIT_REASON_BY_CHECK[check]
                    for check in failed_checks
                }:
                    errors.append(
                        f"withheld candidate {index} reason contradicts failed checks"
                    )
                if str(item.get("target_category") or "") not in taxonomy:
                    errors.append(
                        f"withheld candidate {index} has invalid target category"
                    )
            if len(withheld_ids) != len(set(withheld_ids)):
                errors.append("candidate grounding audit has duplicate identities")
            reason_counts = audit.get("withheld_reason_counts")
            if not isinstance(reason_counts, dict) or any(
                reason not in GROUNDING_AUDIT_WITHHOLD_REASONS
                or isinstance(count, bool) or not isinstance(count, int) or count < 0
                for reason, count in (
                    reason_counts.items() if isinstance(reason_counts, dict) else []
                )
            ):
                errors.append("candidate grounding audit reason counts are invalid")
            elif dict(sorted(actual_reasons.items())) != dict(sorted(reason_counts.items())):
                errors.append("candidate grounding audit reason count mismatch")
            final_withheld = audit.get("final_withheld_recommendations")
            if not isinstance(final_withheld, list):
                errors.append(
                    "final recommendation grounding withheld list is invalid"
                )
                final_withheld = []
            elif (
                "final_recommendation_withheld_count" in counts
                and len(final_withheld)
                != counts["final_recommendation_withheld_count"]
            ):
                errors.append(
                    "final recommendation grounding withheld count mismatch"
                )
            final_ids = []
            final_reasons = Counter()
            for index, item in enumerate(final_withheld, start=1):
                if not isinstance(item, dict):
                    errors.append(
                        f"withheld final recommendation {index} is not an object"
                    )
                    continue
                observe_selection_origin(
                    item, f"withheld final recommendation {index}"
                )
                validate_adjacency_support(
                    item, f"withheld final recommendation {index}"
                )
                candidate_id = str(item.get("candidate_id") or "")
                reason = str(item.get("reason") or "")
                failed_checks = item.get("failed_checks")
                if not re.fullmatch(r"TRC-[A-F0-9]{12}", candidate_id):
                    errors.append(
                        f"withheld final recommendation {index} has invalid identity"
                    )
                else:
                    final_ids.append(candidate_id)
                if reason not in GROUNDING_AUDIT_WITHHOLD_REASONS:
                    errors.append(
                        f"withheld final recommendation {index} has invalid reason"
                    )
                else:
                    final_reasons[reason] += 1
                if (
                    not isinstance(failed_checks, list) or not failed_checks
                    or any(check not in GROUNDING_AUDIT_CHECKS for check in failed_checks)
                    or len(set(failed_checks)) != len(failed_checks)
                ):
                    errors.append(
                        f"withheld final recommendation {index} has invalid failed checks"
                    )
                elif reason in GROUNDING_AUDIT_WITHHOLD_REASONS and reason not in {
                    GROUNDING_AUDIT_REASON_BY_CHECK[check]
                    for check in failed_checks
                }:
                    errors.append(
                        f"withheld final recommendation {index} reason contradicts "
                        "failed checks"
                    )
                if str(item.get("target_category") or "") not in taxonomy:
                    errors.append(
                        f"withheld final recommendation {index} has invalid target category"
                    )
            if len(final_ids) != len(set(final_ids)):
                errors.append(
                    "final recommendation grounding has duplicate identities"
                )
            final_reason_counts = audit.get("final_withheld_reason_counts")
            if not isinstance(final_reason_counts, dict) or any(
                reason not in GROUNDING_AUDIT_WITHHOLD_REASONS
                or isinstance(count, bool) or not isinstance(count, int) or count < 0
                for reason, count in (
                    final_reason_counts.items()
                    if isinstance(final_reason_counts, dict) else []
                )
            ):
                errors.append(
                    "final recommendation grounding reason counts are invalid"
                )
            elif dict(sorted(final_reasons.items())) != dict(sorted(
                final_reason_counts.items()
            )):
                errors.append(
                    "final recommendation grounding reason count mismatch"
                )
            if schema_version >= 9:
                repair_count_fields = (
                    "initial_supported_candidate_count",
                    "initial_withheld_candidate_count",
                    "repair_eligible_candidate_count",
                    "repair_ineligible_candidate_count",
                    "repair_attempted_candidate_count",
                    "repair_proposed_candidate_count",
                    "repair_declined_candidate_count",
                    "repair_unavailable_candidate_count",
                    "repair_reaudited_candidate_count",
                    "repair_reaudit_unavailable_candidate_count",
                    "repair_supported_candidate_count",
                    "repair_withheld_candidate_count",
                    "recovered_candidate_count",
                )
                repair_counts = {}
                for field in repair_count_fields:
                    value = audit.get(field)
                    if (
                        isinstance(value, bool) or not isinstance(value, int)
                        or value < 0
                    ):
                        errors.append(
                            f"candidate grounding audit has invalid {field}"
                        )
                    else:
                        repair_counts[field] = value
                if len(repair_counts) == len(repair_count_fields) and counts:
                    if counts.get("audited_candidate_count") != (
                        repair_counts["initial_supported_candidate_count"]
                        + repair_counts["initial_withheld_candidate_count"]
                    ):
                        errors.append("initial grounding decision count mismatch")
                    if repair_counts["initial_withheld_candidate_count"] != (
                        repair_counts["repair_eligible_candidate_count"]
                        + repair_counts["repair_ineligible_candidate_count"]
                    ):
                        errors.append("grounding repair eligibility count mismatch")
                    if repair_counts["repair_attempted_candidate_count"] != (
                        repair_counts["repair_eligible_candidate_count"]
                    ):
                        errors.append("grounding repair attempted count mismatch")
                    if repair_counts["repair_attempted_candidate_count"] != (
                        repair_counts["repair_proposed_candidate_count"]
                        + repair_counts["repair_declined_candidate_count"]
                        + repair_counts["repair_unavailable_candidate_count"]
                    ):
                        errors.append("grounding repair decision count mismatch")
                    if repair_counts["repair_proposed_candidate_count"] != (
                        repair_counts["repair_reaudited_candidate_count"]
                        + repair_counts[
                            "repair_reaudit_unavailable_candidate_count"
                        ]
                    ):
                        errors.append("grounding repair reaudit count mismatch")
                    if repair_counts["repair_reaudited_candidate_count"] != (
                        repair_counts["repair_supported_candidate_count"]
                        + repair_counts["repair_withheld_candidate_count"]
                    ):
                        errors.append("grounding repair outcome count mismatch")
                    if repair_counts["recovered_candidate_count"] != (
                        repair_counts["repair_supported_candidate_count"]
                    ):
                        errors.append("grounding recovered candidate count mismatch")
                    if counts.get("supported_candidate_count") != (
                        repair_counts["initial_supported_candidate_count"]
                        + repair_counts["recovered_candidate_count"]
                    ):
                        errors.append("post-repair supported candidate count mismatch")
                    if counts.get("withheld_candidate_count") != (
                        repair_counts["initial_withheld_candidate_count"]
                        - repair_counts["recovered_candidate_count"]
                    ):
                        errors.append("post-repair withheld candidate count mismatch")
                initial_reason_counts = audit.get("initial_withheld_reason_counts")
                if not isinstance(initial_reason_counts, dict) or any(
                    reason not in GROUNDING_AUDIT_WITHHOLD_REASONS
                    or isinstance(count, bool) or not isinstance(count, int)
                    or count < 0
                    for reason, count in (
                        initial_reason_counts.items()
                        if isinstance(initial_reason_counts, dict) else []
                    )
                ):
                    errors.append("initial grounding reason counts are invalid")
                elif sum(initial_reason_counts.values()) != repair_counts.get(
                    "initial_withheld_candidate_count"
                ):
                    errors.append("initial grounding reason count mismatch")
                recovered = audit.get("recovered_candidates")
                if not isinstance(recovered, list):
                    errors.append("grounding recovered candidate list is invalid")
                    recovered = []
                elif len(recovered) != repair_counts.get(
                    "recovered_candidate_count"
                ):
                    errors.append("grounding recovered candidate list count mismatch")
                recovered_original_ids = []
                recovered_repaired_ids = []
                for index, item in enumerate(recovered, start=1):
                    if not isinstance(item, dict):
                        errors.append(f"recovered candidate {index} is not an object")
                        continue
                    validate_adjacency_support(
                        item, f"recovered candidate {index}",
                        require_ref_completion=True,
                    )
                    original_id = str(item.get("original_candidate_id") or "")
                    repaired_id = str(item.get("repaired_candidate_id") or "")
                    if schema_version >= 24:
                        if original_id not in discovery_origins_by_id:
                            errors.append(
                                f"recovered candidate {index} exceeds discovery registry"
                            )
                        validate_generation_origins(
                            item, f"recovered candidate {index}",
                            discovery_origins_by_id.get(original_id),
                        )
                    if (
                        not re.fullmatch(r"TRC-[A-F0-9]{12}", original_id)
                        or not re.fullmatch(r"TRC-[A-F0-9]{12}", repaired_id)
                        or original_id == repaired_id
                    ):
                        errors.append(f"recovered candidate {index} has invalid identity")
                    else:
                        recovered_original_ids.append(original_id)
                        recovered_repaired_ids.append(repaired_id)
                    failed = item.get("initial_failed_checks")
                    initial_reason = str(item.get("initial_reason") or "")
                    if (
                        not isinstance(failed, list) or not 1 <= len(failed) <= 2
                        or any(check not in GROUNDING_REPAIRABLE_CHECKS for check in failed)
                    ):
                        errors.append(
                            f"recovered candidate {index} has invalid initial checks"
                        )
                    elif (
                        initial_reason not in GROUNDING_AUDIT_WITHHOLD_REASONS
                        or initial_reason not in {
                            GROUNDING_AUDIT_REASON_BY_CHECK[check]
                            for check in failed
                        }
                    ):
                        errors.append(
                            f"recovered candidate {index} has inconsistent initial reason"
                        )
                    changed = item.get("changed_fields")
                    if (
                        not isinstance(changed, list) or not changed
                        or len(set(changed)) != len(changed)
                        or any(
                            field not in GROUNDING_REPAIR_TRACKED_FIELDS
                            for field in changed
                        )
                    ):
                        errors.append(
                            f"recovered candidate {index} has invalid changed fields"
                        )
                    if str(item.get("target_category") or "") not in taxonomy:
                        errors.append(
                            f"recovered candidate {index} has invalid target category"
                        )
                    if not str(item.get("area") or "").strip():
                        errors.append(f"recovered candidate {index} has empty area")
                    source_codes = item.get("source_bug_codes")
                    evidence_refs = item.get("evidence_refs")
                    if (
                        not isinstance(source_codes, list) or not source_codes
                        or any(
                            not isinstance(code, str)
                            or not re.fullmatch(r"[A-Za-z0-9._-]+", code)
                            for code in source_codes
                        )
                        or len(set(source_codes)) != len(source_codes)
                    ):
                        errors.append(
                            f"recovered candidate {index} has invalid source BugCodes"
                        )
                        source_codes = []
                    if not isinstance(evidence_refs, list) or not evidence_refs or any(
                        not isinstance(ref, dict)
                        or ref.get("bug_code") not in source_codes
                        or isinstance(ref.get("signal_index"), bool)
                        or not isinstance(ref.get("signal_index"), int)
                        or ref.get("signal_index") < 0
                        for ref in (evidence_refs or [])
                    ):
                        errors.append(
                            f"recovered candidate {index} has invalid evidence refs"
                        )
                    if item.get("repair_status") != "recovered_after_reaudit":
                        errors.append(
                            f"recovered candidate {index} has invalid repair status"
                        )
                if len(recovered_original_ids) != len(set(recovered_original_ids)):
                    errors.append("grounding recovered originals are duplicated")
                if len(recovered_repaired_ids) != len(set(recovered_repaired_ids)):
                    errors.append("grounding recovered candidates are duplicated")
                withheld_outcome_ids = {
                    str(item.get("original_candidate_id") or item.get("candidate_id") or "")
                    for item in withheld if isinstance(item, dict)
                }
                if set(recovered_original_ids) & withheld_outcome_ids:
                    errors.append(
                        "grounding recovered and withheld outcomes overlap"
                    )
                if set(recovered_repaired_ids) & set(withheld_ids):
                    errors.append(
                        "grounding repaired identities overlap withheld candidates"
                    )
                allowed_repair_statuses = {
                    "ineligible_direct_fix", "ineligible_evidence_gap",
                    "ineligible_broad_failure",
                    "no_valid_repair", "repair_unavailable",
                    "reaudit_unavailable", "repair_withheld",
                }
                repair_status_counts = Counter()
                for index, item in enumerate(withheld, start=1):
                    if isinstance(item, dict) and item.get("repair_status") not in (
                        allowed_repair_statuses
                    ):
                        errors.append(
                            f"withheld candidate {index} has invalid repair status"
                        )
                    elif isinstance(item, dict):
                        repair_status_counts[str(item.get("repair_status"))] += 1
                repair_status_counts["recovered_after_reaudit"] = len(recovered)
                derived_ineligible = sum(
                    repair_status_counts[status] for status in (
                        "ineligible_direct_fix", "ineligible_evidence_gap",
                        "ineligible_broad_failure",
                    )
                )
                derived_eligible = sum(repair_status_counts.values()) - derived_ineligible
                if derived_ineligible != repair_counts.get(
                    "repair_ineligible_candidate_count"
                ):
                    errors.append("grounding repair ineligible outcomes mismatch")
                if derived_eligible != repair_counts.get(
                    "repair_eligible_candidate_count"
                ):
                    errors.append("grounding repair eligible outcomes mismatch")
                if repair_status_counts["no_valid_repair"] != repair_counts.get(
                    "repair_declined_candidate_count"
                ):
                    errors.append("grounding repair declined outcomes mismatch")
                if repair_status_counts["repair_unavailable"] != repair_counts.get(
                    "repair_unavailable_candidate_count"
                ):
                    errors.append("grounding repair unavailable outcomes mismatch")
                if repair_status_counts["reaudit_unavailable"] != repair_counts.get(
                    "repair_reaudit_unavailable_candidate_count"
                ):
                    errors.append("grounding repair reaudit unavailable outcomes mismatch")
                if repair_status_counts["repair_withheld"] != repair_counts.get(
                    "repair_withheld_candidate_count"
                ):
                    errors.append("grounding repair withheld outcomes mismatch")
            if schema_version >= 10:
                accepted_count = audit.get("accepted_candidate_lineage_count")
                accepted = audit.get("accepted_candidate_lineage")
                if (
                    isinstance(accepted_count, bool)
                    or not isinstance(accepted_count, int)
                    or accepted_count < 0
                ):
                    errors.append("accepted candidate lineage count is invalid")
                if not isinstance(accepted, list):
                    errors.append("accepted candidate lineage is invalid")
                    accepted = []
                elif isinstance(accepted_count, int) and len(accepted) != accepted_count:
                    errors.append("accepted candidate lineage count mismatch")
                if (
                    isinstance(accepted_count, int)
                    and accepted_count != counts.get("supported_candidate_count")
                ):
                    errors.append("accepted lineage and supported count mismatch")
                recovered_pairs = {
                    str(item.get("repaired_candidate_id") or ""): str(
                        item.get("original_candidate_id") or ""
                    )
                    for item in (recovered or []) if isinstance(item, dict)
                }
                accepted_by_id = {}
                for index, item in enumerate(accepted, start=1):
                    if not isinstance(item, dict):
                        errors.append(
                            f"accepted candidate lineage {index} is not an object"
                        )
                        continue
                    validate_adjacency_support(
                        item, f"accepted candidate lineage {index}",
                        require_ref_completion=True,
                    )
                    candidate_id = str(item.get("candidate_id") or "")
                    original_id = str(item.get("original_candidate_id") or "")
                    origin = str(item.get("origin") or "")
                    if schema_version >= 24:
                        if original_id not in discovery_origins_by_id:
                            errors.append(
                                f"accepted candidate lineage {index} exceeds "
                                "discovery registry"
                            )
                        generation_origins = validate_generation_origins(
                            item, f"accepted candidate lineage {index}",
                            discovery_origins_by_id.get(original_id),
                        )
                    else:
                        generation_origins = []
                    if not re.fullmatch(r"TRC-[A-F0-9]{12}", candidate_id):
                        errors.append(
                            f"accepted candidate lineage {index} has invalid identity"
                        )
                        continue
                    if candidate_id in accepted_by_id:
                        errors.append("accepted candidate lineage has duplicate identities")
                        continue
                    if origin not in {
                        "initial_audit_pass", "recovered_after_reaudit",
                    }:
                        errors.append(
                            f"accepted candidate lineage {index} has invalid origin"
                        )
                    elif origin == "initial_audit_pass" and original_id != candidate_id:
                        errors.append(
                            f"accepted candidate lineage {index} initial identity mismatch"
                        )
                    elif origin == "recovered_after_reaudit" and (
                        recovered_pairs.get(candidate_id) != original_id
                    ):
                        errors.append(
                            f"accepted candidate lineage {index} recovery mismatch"
                        )
                    if str(item.get("target_category") or "") not in taxonomy:
                        errors.append(
                            f"accepted candidate lineage {index} has invalid category"
                        )
                    source_codes = item.get("source_bug_codes")
                    evidence_refs = item.get("evidence_refs")
                    historical_refs = item.get("historical_analog_refs")
                    area = str(item.get("area") or "")
                    trigger = str(item.get("trigger") or "")
                    failure_mode = str(item.get("expected_failure_mode") or "")
                    scope_boundary = str(item.get("scope_boundary") or "")
                    disconfirming_observation = str(
                        item.get("disconfirming_observation") or ""
                    )
                    action = str(item.get("action") or "")
                    if (
                        not isinstance(source_codes, list) or not source_codes
                        or any(
                            not isinstance(code, str)
                            or not re.fullmatch(r"[A-Za-z0-9._-]+", code)
                            for code in source_codes
                        )
                        or len(set(source_codes)) != len(source_codes)
                    ):
                        errors.append(
                            f"accepted candidate lineage {index} has invalid BugCodes"
                        )
                        source_codes = []
                    ref_keys = set()
                    if not isinstance(evidence_refs, list) or not evidence_refs:
                        errors.append(
                            f"accepted candidate lineage {index} has invalid evidence refs"
                        )
                    else:
                        for ref in evidence_refs:
                            if (
                                not isinstance(ref, dict)
                                or ref.get("bug_code") not in source_codes
                                or isinstance(ref.get("signal_index"), bool)
                                or not isinstance(ref.get("signal_index"), int)
                                or ref.get("signal_index") < 0
                            ):
                                errors.append(
                                    f"accepted candidate lineage {index} has invalid "
                                    "evidence refs"
                                )
                                continue
                            ref_keys.add((ref["bug_code"], ref["signal_index"]))
                        if len(ref_keys) != len(evidence_refs):
                            errors.append(
                                f"accepted candidate lineage {index} has duplicate "
                                "evidence refs"
                            )
                    historical_ref_keys = set()
                    if not isinstance(historical_refs, list):
                        errors.append(
                            f"accepted candidate lineage {index} has invalid "
                            "historical refs"
                        )
                    else:
                        for ref in historical_refs:
                            if (
                                not isinstance(ref, dict)
                                or ref.get("source_bug_code") not in source_codes
                                or not re.fullmatch(
                                    r"[A-Za-z0-9._-]+",
                                    str(ref.get("analog_bug_code") or ""),
                                )
                            ):
                                errors.append(
                                    f"accepted candidate lineage {index} has invalid "
                                    "historical refs"
                                )
                                continue
                            historical_ref_keys.add((
                                str(ref["source_bug_code"]),
                                str(ref["analog_bug_code"]),
                            ))
                        if len(historical_ref_keys) != len(historical_refs):
                            errors.append(
                                f"accepted candidate lineage {index} has duplicate "
                                "historical refs"
                            )
                    if (
                        not trigger or len(trigger) > 320
                        or not failure_mode or len(failure_mode) > 320
                    ):
                        errors.append(
                            f"accepted candidate lineage {index} has invalid "
                            "trigger/failure pair"
                        )
                    elif (
                        schema_version >= 12
                        and prediction_specificity_errors is not None
                    ):
                        errors.extend(prediction_specificity_errors(
                            area, trigger, failure_mode,
                            f"accepted candidate lineage {index}",
                        ))
                    if (
                        not scope_boundary or len(scope_boundary) > 320
                        or not disconfirming_observation
                        or len(disconfirming_observation) > 320
                    ):
                        errors.append(
                            f"accepted candidate lineage {index} has invalid boundary"
                        )
                    elif prediction_boundary_errors is not None:
                        errors.extend(prediction_boundary_errors(
                            str(item.get("area") or ""),
                            scope_boundary,
                            disconfirming_observation,
                            f"accepted candidate lineage {index}",
                        ))
                    if schema_version >= 11:
                        if not action or len(action) > 500:
                            errors.append(
                                f"accepted candidate lineage {index} has invalid action"
                            )
                        elif stable_candidate_audit_id(item) != candidate_id:
                            errors.append(
                                f"accepted candidate lineage {index} identity hash "
                                "mismatch"
                            )
                    if (
                        schema_version >= 15
                        and deterministic_confidence_basis is not None
                    ):
                        validate_confidence_refs(
                            item, f"accepted candidate lineage {index}"
                        )
                        expected_lineage_confidence = (
                            deterministic_confidence_basis(
                                list(source_codes),
                                list(evidence_refs or []),
                                list(historical_refs or []),
                                confidence_evidence_map,
                                correlate_sources=schema_version >= 18,
                                target_area=(
                                    area if schema_version >= 19 else None
                                ),
                            )
                        )
                        if item.get(
                            "confidence_basis"
                        ) != expected_lineage_confidence or item.get(
                            "confidence"
                        ) != expected_lineage_confidence["label"]:
                            errors.append(
                                f"accepted candidate lineage {index} confidence "
                                "provenance mismatch"
                            )
                    validate_category_basis(
                        item, f"accepted candidate lineage {index}"
                    )
                    accepted_by_id[candidate_id] = {
                        "origin": origin,
                        "original_candidate_id": original_id,
                        "candidate_generation_origins": generation_origins,
                        "target_category": str(item.get("target_category") or ""),
                        "source_bug_codes": set(source_codes),
                        "evidence_refs": ref_keys,
                        "historical_analog_refs": historical_ref_keys,
                        "trigger_failure_pair": (trigger, failure_mode),
                        "boundary_pair": (
                            scope_boundary, disconfirming_observation,
                        ),
                        "claim_surface": (
                            str(item.get("area") or ""),
                            str(item.get("target_category") or ""),
                            trigger,
                            failure_mode,
                            scope_boundary,
                            disconfirming_observation,
                            action,
                        ),
                    }
                if schema_version >= 24:
                    accepted_original_ids = {
                        row["original_candidate_id"]
                        for row in accepted_by_id.values()
                    }
                    discovered_outcome_ids = (
                        accepted_original_ids | withheld_outcome_ids
                    )
                    if accepted_original_ids & withheld_outcome_ids:
                        errors.append(
                            "candidate discovery accepted and withheld outcomes overlap"
                        )
                    if discovered_outcome_ids != set(discovery_origins_by_id):
                        errors.append(
                            "candidate discovery outcome registry mismatch"
                        )
                    synthesis_origin_items = [
                        (
                            item,
                            f"recommendation {index}",
                        )
                        for index, item in enumerate(
                            analysis.get("recommendations") or [], start=1
                        )
                        if isinstance(item, dict)
                    ]
                    synthesis_origin_items.extend(
                        (
                            record["recommendation"],
                            f"portfolio-excluded recommendation {index}",
                        )
                        for index, record in enumerate(
                            portfolio_records, start=1
                        )
                        if isinstance(record, dict)
                        and isinstance(record.get("recommendation"), dict)
                    )
                    synthesis_origin_items.extend(
                        (
                            record["suppressed_recommendation"],
                            f"coverage-equivalent recommendation {index}",
                        )
                        for index, record in enumerate(
                            coverage_records, start=1
                        )
                        if isinstance(record, dict)
                        and isinstance(
                            record.get("suppressed_recommendation"), dict
                        )
                    )
                    synthesis_origin_items.extend(
                        (
                            item,
                            f"withheld final recommendation {index}",
                        )
                        for index, item in enumerate(final_withheld, start=1)
                        if isinstance(item, dict)
                    )
                    for item, label in synthesis_origin_items:
                        lineage_ids = item.get(
                            "synthesis_source_candidate_ids"
                        )
                        if (
                            not isinstance(lineage_ids, list)
                            or not lineage_ids
                            or any(
                                candidate_id not in accepted_by_id
                                for candidate_id in (lineage_ids or [])
                            )
                        ):
                            continue
                        expected_origins = list(dict.fromkeys(
                            generation_origin
                            for candidate_id in lineage_ids
                            for generation_origin in accepted_by_id[
                                candidate_id
                            ]["candidate_generation_origins"]
                        ))
                        validate_generation_origins(
                            item, label, expected_origins,
                        )
                if schema_version >= 14:
                    for index, record in enumerate(
                        coverage_records, start=1
                    ):
                        if not isinstance(record, dict) or not isinstance(
                            record.get("suppressed_recommendation"), dict
                        ):
                            continue
                        suppressed = record["suppressed_recommendation"]
                        label = f"coverage-equivalent recommendation {index}"
                        lineage_ids = suppressed.get(
                            "synthesis_source_candidate_ids"
                        )
                        if (
                            not isinstance(lineage_ids, list) or not lineage_ids
                            or any(
                                not isinstance(candidate_id, str)
                                or not re.fullmatch(
                                    r"TRC-[A-F0-9]{12}", candidate_id
                                )
                                or candidate_id not in accepted_by_id
                                for candidate_id in (lineage_ids or [])
                            )
                            or len(set(lineage_ids)) != len(lineage_ids)
                        ):
                            errors.append(
                                f"{label} has invalid synthesis lineage"
                            )
                            continue
                        selected = [
                            accepted_by_id[candidate_id]
                            for candidate_id in lineage_ids
                        ]
                        claim_surface = (
                            str(suppressed.get("area") or ""),
                            str(suppressed.get("target_category") or ""),
                            str(suppressed.get("trigger") or ""),
                            str(suppressed.get("expected_failure_mode") or ""),
                            str(suppressed.get("scope_boundary") or ""),
                            str(suppressed.get(
                                "disconfirming_observation"
                            ) or ""),
                            str(suppressed.get("action") or ""),
                        )
                        if any(
                            row["claim_surface"] != claim_surface
                            for row in selected
                        ):
                            errors.append(
                                f"{label} has incoherent synthesis claim lineage"
                            )
                        source_codes = {
                            str(code) for code in (
                                suppressed.get("source_bug_codes") or []
                            )
                        }
                        evidence_refs = {
                            (
                                str(ref.get("bug_code") or ""),
                                ref.get("signal_index"),
                            )
                            for ref in (suppressed.get("evidence_refs") or [])
                            if isinstance(ref, dict)
                        }
                        historical_refs = {
                            (
                                str(ref.get("source_bug_code") or ""),
                                str(ref.get("analog_bug_code") or ""),
                            )
                            for ref in (
                                suppressed.get("historical_analog_refs") or []
                            )
                            if isinstance(ref, dict)
                        }
                        selected_codes = set().union(*(
                            row["source_bug_codes"] for row in selected
                        ))
                        selected_refs = set().union(*(
                            row["evidence_refs"] for row in selected
                        ))
                        selected_historical_refs = set().union(*(
                            row["historical_analog_refs"] for row in selected
                        ))
                        if not source_codes.issubset(selected_codes):
                            errors.append(f"{label} BugCodes exceed lineage")
                        if not evidence_refs.issubset(selected_refs):
                            errors.append(f"{label} evidence exceeds lineage")
                        if not historical_refs.issubset(
                            selected_historical_refs
                        ):
                            errors.append(
                                f"{label} historical refs exceed lineage"
                            )
                        if any(
                            not (row["evidence_refs"] & evidence_refs)
                            for row in selected
                        ):
                            errors.append(
                                f"{label} has noncontributing synthesis lineage"
                            )
                if (
                    schema_version >= 23
                    and deterministic_synthesis_backfill is not None
                ):
                    if not observed_model_selected_ids.issubset(accepted_by_id):
                        errors.append(
                            "model synthesis selection exceeds accepted lineage"
                        )
                    if not observed_backfill_ids.issubset(accepted_by_id):
                        errors.append(
                            "synthesis backfill selection exceeds accepted lineage"
                        )
                    _expected_backfills, expected_safeguard = (
                        deterministic_synthesis_backfill(
                            [
                                item for item in accepted
                                if isinstance(item, dict)
                            ],
                            [{
                                "synthesis_source_candidate_ids": sorted(
                                    observed_model_selected_ids
                                ),
                            }],
                            confidence_evidence_map,
                        )
                    )
                    raw_safeguard = audit.get(
                        "synthesis_selection_safeguard"
                    )
                    if raw_safeguard != expected_safeguard:
                        errors.append(
                            "synthesis selection safeguard provenance mismatch"
                        )
                    raw_backfill_ids = (
                        raw_safeguard.get("backfill_candidate_ids")
                        if isinstance(raw_safeguard, dict) else None
                    )
                    if (
                        not isinstance(raw_backfill_ids, list)
                        or observed_backfill_ids != set(raw_backfill_ids)
                    ):
                        errors.append(
                            "synthesis backfill selection origin mismatch"
                        )
                    if observed_backfill_ids != set(
                        expected_safeguard["backfill_candidate_ids"]
                    ):
                        errors.append(
                            "synthesis backfill selection origin mismatch"
                        )
                    for index, record in enumerate(
                        portfolio_records, start=1
                    ):
                        if not isinstance(record, dict) or not isinstance(
                            record.get("recommendation"), dict
                        ):
                            continue
                        item = record["recommendation"]
                        label = f"portfolio-excluded recommendation {index}"
                        lineage_ids = item.get(
                            "synthesis_source_candidate_ids"
                        )
                        if (
                            not isinstance(lineage_ids, list) or not lineage_ids
                            or len(set(lineage_ids)) != len(lineage_ids)
                            or any(
                                candidate_id not in accepted_by_id
                                for candidate_id in (lineage_ids or [])
                            )
                        ):
                            errors.append(f"{label} has invalid synthesis lineage")
                            continue
                        selected = [
                            accepted_by_id[candidate_id]
                            for candidate_id in lineage_ids
                        ]
                        claim_surface = (
                            str(item.get("area") or ""),
                            str(item.get("target_category") or ""),
                            str(item.get("trigger") or ""),
                            str(item.get("expected_failure_mode") or ""),
                            str(item.get("scope_boundary") or ""),
                            str(item.get("disconfirming_observation") or ""),
                            str(item.get("action") or ""),
                        )
                        if any(
                            row["claim_surface"] != claim_surface
                            for row in selected
                        ):
                            errors.append(f"{label} has incoherent claim lineage")
                        item_codes = {
                            str(code) for code in (
                                item.get("source_bug_codes") or []
                            )
                        }
                        item_refs = {
                            (
                                str(ref.get("bug_code") or ""),
                                ref.get("signal_index"),
                            )
                            for ref in (item.get("evidence_refs") or [])
                            if isinstance(ref, dict)
                        }
                        item_historical_refs = {
                            (
                                str(ref.get("source_bug_code") or ""),
                                str(ref.get("analog_bug_code") or ""),
                            )
                            for ref in (
                                item.get("historical_analog_refs") or []
                            )
                            if isinstance(ref, dict)
                        }
                        selected_codes = set().union(*(
                            row["source_bug_codes"] for row in selected
                        ))
                        selected_refs = set().union(*(
                            row["evidence_refs"] for row in selected
                        ))
                        selected_historical_refs = set().union(*(
                            row["historical_analog_refs"] for row in selected
                        ))
                        if not item_codes.issubset(selected_codes):
                            errors.append(f"{label} BugCodes exceed lineage")
                        if not item_refs.issubset(selected_refs):
                            errors.append(f"{label} evidence exceeds lineage")
                        if not item_historical_refs.issubset(
                            selected_historical_refs
                        ):
                            errors.append(f"{label} historical refs exceed lineage")
                        if any(
                            not (row["evidence_refs"] & item_refs)
                            for row in selected
                        ):
                            errors.append(f"{label} has noncontributing lineage")
                        expected_lineage = [{
                            "candidate_id": candidate_id,
                            "origin": accepted_by_id[candidate_id]["origin"],
                            "original_candidate_id": accepted_by_id[
                                candidate_id
                            ]["original_candidate_id"],
                        } for candidate_id in lineage_ids]
                        if item.get("synthesis_lineage") != expected_lineage:
                            errors.append(f"{label} synthesis lineage mismatch")
                        recovered_ids = [
                            candidate_id for candidate_id in lineage_ids
                            if accepted_by_id[candidate_id]["origin"]
                            == "recovered_after_reaudit"
                        ]
                        expected_origin = (
                            "recovered_draft_only"
                            if len(recovered_ids) == len(lineage_ids) else
                            "mixed_initial_and_recovered"
                            if recovered_ids else "initial_audit_pass_only"
                        )
                        if item.get("prediction_origin") != expected_origin:
                            errors.append(f"{label} prediction origin mismatch")
                        if item.get(
                            "recovered_source_candidate_ids"
                        ) != recovered_ids:
                            errors.append(f"{label} recovered source IDs mismatch")
                        if item.get("hypothesis_id") != stable_hypothesis_id(item):
                            errors.append(f"{label} hypothesis identity mismatch")
                recovered_accepted = sum(
                    row["origin"] == "recovered_after_reaudit"
                    for row in accepted_by_id.values()
                )
                if recovered_accepted != repair_counts.get("recovered_candidate_count"):
                    errors.append("accepted recovered lineage count mismatch")
                if set(accepted_by_id) & set(withheld_ids):
                    errors.append("accepted and withheld candidate lineage overlap")
                recovered_recommendation_count = 0
                for index, item in enumerate(
                    analysis.get("recommendations") or [], start=1
                ):
                    if not isinstance(item, dict):
                        continue
                    lineage_ids = item.get("synthesis_source_candidate_ids")
                    if (
                        not isinstance(lineage_ids, list) or not lineage_ids
                        or any(
                            not isinstance(candidate_id, str)
                            or not re.fullmatch(
                                r"TRC-[A-F0-9]{12}", candidate_id
                            )
                            for candidate_id in (lineage_ids or [])
                        )
                        or len(set(lineage_ids)) != len(lineage_ids)
                        or any(
                            candidate_id not in accepted_by_id
                            for candidate_id in lineage_ids
                        )
                    ):
                        errors.append(
                            f"recommendation {index} has invalid synthesis lineage IDs"
                        )
                        continue
                    expected_lineage = [{
                        "candidate_id": candidate_id,
                        "origin": accepted_by_id[candidate_id]["origin"],
                        "original_candidate_id": accepted_by_id[candidate_id][
                            "original_candidate_id"
                        ],
                    } for candidate_id in lineage_ids]
                    if item.get("synthesis_lineage") != expected_lineage:
                        errors.append(
                            f"recommendation {index} synthesis lineage mismatch"
                        )
                    recovered_ids = [
                        candidate_id for candidate_id in lineage_ids
                        if accepted_by_id[candidate_id]["origin"]
                        == "recovered_after_reaudit"
                    ]
                    expected_origin = (
                        "recovered_draft_only"
                        if len(recovered_ids) == len(lineage_ids) else
                        "mixed_initial_and_recovered"
                        if recovered_ids else "initial_audit_pass_only"
                    )
                    if item.get("prediction_origin") != expected_origin:
                        errors.append(
                            f"recommendation {index} prediction origin mismatch"
                        )
                    if item.get("recovered_source_candidate_ids") != recovered_ids:
                        errors.append(
                            f"recommendation {index} recovered source IDs mismatch"
                        )
                    recovered_recommendation_count += bool(recovered_ids)
                    selected = [accepted_by_id[candidate_id] for candidate_id in lineage_ids]
                    selected_codes = set().union(*(
                        row["source_bug_codes"] for row in selected
                    ))
                    selected_refs = set().union(*(
                        row["evidence_refs"] for row in selected
                    ))
                    raw_item_codes = item.get("source_bug_codes")
                    if (
                        not isinstance(raw_item_codes, list)
                        or any(
                            not isinstance(code, str) for code in raw_item_codes
                        )
                    ):
                        errors.append(
                            f"recommendation {index} has invalid lineage BugCodes"
                        )
                        raw_item_codes = []
                    item_codes = set(raw_item_codes)
                    raw_item_refs = item.get("evidence_refs")
                    if not isinstance(raw_item_refs, list):
                        errors.append(
                            f"recommendation {index} has invalid lineage evidence"
                        )
                        raw_item_refs = []
                    item_refs = {
                        (ref.get("bug_code"), ref.get("signal_index"))
                        for ref in raw_item_refs
                        if isinstance(ref, dict)
                    }
                    raw_item_historical_refs = item.get(
                        "historical_analog_refs"
                    )
                    if not isinstance(raw_item_historical_refs, list):
                        errors.append(
                            f"recommendation {index} has invalid lineage "
                            "historical refs"
                        )
                        raw_item_historical_refs = []
                    item_historical_refs = {
                        (
                            str(ref.get("source_bug_code") or ""),
                            str(ref.get("analog_bug_code") or ""),
                        )
                        for ref in raw_item_historical_refs
                        if isinstance(ref, dict)
                    }
                    if not item_codes.issubset(selected_codes):
                        errors.append(
                            f"recommendation {index} BugCodes exceed synthesis lineage"
                        )
                    if not item_refs.issubset(selected_refs):
                        errors.append(
                            f"recommendation {index} evidence exceeds synthesis lineage"
                        )
                    selected_historical_refs = set().union(*(
                        row["historical_analog_refs"] for row in selected
                    ))
                    if not item_historical_refs.issubset(
                        selected_historical_refs
                    ):
                        errors.append(
                            f"recommendation {index} historical refs exceed "
                            "synthesis lineage"
                        )
                    if str(item.get("target_category") or "") not in {
                        row["target_category"] for row in selected
                    }:
                        errors.append(
                            f"recommendation {index} category exceeds synthesis lineage"
                        )
                    if (
                        str(item.get("trigger") or ""),
                        str(item.get("expected_failure_mode") or ""),
                    ) not in {
                        row["trigger_failure_pair"] for row in selected
                    }:
                        errors.append(
                            f"recommendation {index} trigger/failure exceeds "
                            "synthesis lineage"
                        )
                    if (
                        str(item.get("scope_boundary") or ""),
                        str(item.get("disconfirming_observation") or ""),
                    ) not in {row["boundary_pair"] for row in selected}:
                        errors.append(
                            f"recommendation {index} boundary exceeds synthesis lineage"
                        )
                    if schema_version >= 11:
                        recommendation_claim_surface = (
                            str(item.get("area") or ""),
                            str(item.get("target_category") or ""),
                            str(item.get("trigger") or ""),
                            str(item.get("expected_failure_mode") or ""),
                            str(item.get("scope_boundary") or ""),
                            str(item.get("disconfirming_observation") or ""),
                            str(item.get("action") or ""),
                        )
                        mismatched_claim_ids = [
                            candidate_id for candidate_id in lineage_ids
                            if accepted_by_id[candidate_id]["claim_surface"]
                            != recommendation_claim_surface
                        ]
                        if mismatched_claim_ids:
                            errors.append(
                                f"recommendation {index} has incoherent synthesis "
                                "claim lineage: " + ", ".join(
                                    mismatched_claim_ids
                                )
                            )
                    noncontributing = [
                        candidate_id for candidate_id in lineage_ids
                        if not (accepted_by_id[candidate_id]["evidence_refs"] & item_refs)
                    ]
                    if noncontributing:
                        errors.append(
                            f"recommendation {index} has noncontributing synthesis lineage"
                        )
                if analysis.get(
                    "recommendations_with_recovered_source_count"
                ) != recovered_recommendation_count:
                    errors.append("recovered-source recommendation count mismatch")
                for index, item in enumerate(final_withheld, start=1):
                    if not isinstance(item, dict):
                        continue
                    lineage_ids = item.get("synthesis_source_candidate_ids")
                    if (
                        not isinstance(lineage_ids, list) or not lineage_ids
                        or any(
                            not isinstance(candidate_id, str)
                            or not re.fullmatch(
                                r"TRC-[A-F0-9]{12}", candidate_id
                            )
                            for candidate_id in (lineage_ids or [])
                        )
                        or any(candidate_id not in accepted_by_id for candidate_id in lineage_ids)
                    ):
                        errors.append(
                            f"withheld final recommendation {index} has invalid lineage"
                        )
                        continue
                    if schema_version >= 11:
                        withheld_claim_surface = (
                            str(item.get("area") or ""),
                            str(item.get("target_category") or ""),
                            str(item.get("trigger") or ""),
                            str(item.get("expected_failure_mode") or ""),
                            str(item.get("scope_boundary") or ""),
                            str(item.get("disconfirming_observation") or ""),
                            str(item.get("action") or ""),
                        )
                        if any(
                            accepted_by_id[candidate_id]["claim_surface"]
                            != withheld_claim_surface
                            for candidate_id in lineage_ids
                        ):
                            errors.append(
                                f"withheld final recommendation {index} has "
                                "incoherent claim lineage"
                            )
            actual_sampled_refs = []
            for collection, expected_stage, label in (
                (
                    gate_withheld_records,
                    "deterministic_adjacency_gate",
                    "adjacency gate withheld candidate",
                ),
                (withheld, "draft_before_synthesis", "withheld candidate"),
                (
                    final_withheld, "final_after_synthesis",
                    "withheld final recommendation",
                ),
            ):
                for index, item in enumerate(collection, start=1):
                    if not isinstance(item, dict):
                        continue
                    if item.get("audit_stage") != expected_stage:
                        errors.append(f"{label} {index} has invalid audit stage")
                    sampled = item.get("human_review_sample")
                    if not isinstance(sampled, bool):
                        errors.append(f"{label} {index} has invalid review sample flag")
                    elif sampled:
                        actual_sampled_refs.append({
                            "audit_stage": expected_stage,
                            "candidate_id": str(item.get("candidate_id") or ""),
                        })
            sampling = audit.get("human_review_sampling")
            if not isinstance(sampling, dict):
                errors.append("grounding human-review sampling is missing or invalid")
            else:
                max_samples = sampling.get("max_samples_per_run")
                sampled_count = sampling.get("sampled_count")
                sampled_refs = sampling.get("sampled_refs")
                if max_samples != GROUNDING_HUMAN_REVIEW_SAMPLE_LIMIT:
                    errors.append("grounding human-review sample limit mismatch")
                if (
                    isinstance(sampled_count, bool)
                    or not isinstance(sampled_count, int)
                    or sampled_count < 0
                    or sampled_count > GROUNDING_HUMAN_REVIEW_SAMPLE_LIMIT
                ):
                    errors.append("grounding human-review sampled count is invalid")
                if not isinstance(sampled_refs, list):
                    errors.append("grounding human-review sampled refs are invalid")
                elif sampled_refs != actual_sampled_refs:
                    errors.append("grounding human-review sampled refs mismatch")
                if isinstance(sampled_count, int) and (
                    sampled_count != len(actual_sampled_refs)
                ):
                    errors.append("grounding human-review sampled count mismatch")
                source_sha = str(analysis.get("source_sha256") or "")
                if re.fullmatch(r"[a-f0-9]{64}", source_sha):
                    expected_samples = _select_grounding_human_review_samples(
                        withheld, final_withheld, source_sha,
                        gate_withheld=gate_withheld_records,
                        limit=GROUNDING_HUMAN_REVIEW_SAMPLE_LIMIT,
                    )
                    actual_samples = {
                        (item["audit_stage"], item["candidate_id"])
                        for item in actual_sampled_refs
                    }
                    if actual_samples != expected_samples:
                        errors.append(
                            "grounding human-review sample is not deterministic"
                        )
        invalid_audit_provenance = [
            index for index, item in enumerate(
                analysis.get("recommendations") or [], start=1
            )
            if not isinstance(item, dict)
            or item.get("grounding_audit_provenance")
            != "draft_and_final_adversarial_passes"
        ]
        if invalid_audit_provenance:
            errors.append(
                "recommendation(s) lack grounding-audit provenance: "
                + ", ".join(str(index) for index in invalid_audit_provenance)
            )
    if schema_version >= 20 and deterministic_checklist_grounding is not None:
        raw_checklists = analysis.get("checklists")
        if not isinstance(raw_checklists, list):
            errors.append("checklists are missing or invalid")
            raw_checklists = []
        seen_checklist_codes = []
        for index, checklist in enumerate(raw_checklists, start=1):
            label = f"checklist {index}"
            if not isinstance(checklist, dict):
                errors.append(f"{label} is not an object")
                continue
            code = str(checklist.get("bug_code") or "")
            evidence = checklist_evidence_map.get(code)
            steps = checklist.get("steps")
            if code not in checklist_evidence_map:
                errors.append(f"{label} has invalid BugCode")
                continue
            seen_checklist_codes.append(code)
            if (
                not isinstance(steps, list) or not 2 <= len(steps) <= 4
                or any(
                    not isinstance(step, str) or not step or len(step) > 600
                    for step in (steps or [])
                )
            ):
                errors.append(f"{label} has invalid steps")
                continue
            if checklist.get("evidence_quality") != evidence.get(
                "evidence_quality"
            ) or checklist.get("evidence_gaps") != evidence.get(
                "evidence_gaps"
            ):
                errors.append(f"{label} evidence metadata mismatch")
            expected = deterministic_checklist_grounding({
                "bug_code": code,
                "steps": list(steps),
                "evidence_quality": checklist.get("evidence_quality"),
                "evidence_gaps": checklist.get("evidence_gaps"),
            }, evidence)
            if expected["steps"] != steps:
                errors.append(f"{label} evidence anchors are missing")
            if checklist.get("grounding_basis") != expected[
                "grounding_basis"
            ]:
                errors.append(f"{label} grounding basis mismatch")
            if expected["grounding_basis"][
                "unsupported_specificity_terms"
            ]:
                errors.append(f"{label} contains unsupported specificity")
            if expected["grounding_basis"][
                "non_executable_step_indexes"
            ]:
                errors.append(f"{label} contains non-executable steps")
        expected_checklist_codes = [
            str(code) for code in (analysis.get("bug_codes") or [])
        ]
        if seen_checklist_codes != expected_checklist_codes:
            errors.append("checklist BugCodes/order do not match analysis")
    if schema_version >= 5 and not errors:
        try:
            from summarize_tr_bugs import stable_hypothesis_id
        except ModuleNotFoundError:
            from scripts.summarize_tr_bugs import stable_hypothesis_id
        mismatched = [
            str(item.get("hypothesis_id") or "")
            for item in (analysis.get("recommendations") or [])
            if stable_hypothesis_id(item) != item.get("hypothesis_id")
        ]
        if mismatched:
            errors.append(
                "hypothesis_id does not match structured feature identity: "
                + ", ".join(mismatched)
            )
    return errors


def feature_match_tokens(*values: object) -> set[str]:
    """Extract explainable content tokens for conservative same-category matching."""
    try:
        from summarize_tr_bugs import prediction_specificity_tokens
    except ModuleNotFoundError:
        from scripts.summarize_tr_bugs import prediction_specificity_tokens
    return prediction_specificity_tokens(*values)


def scenario_evaluation_reasons(hypothesis: dict) -> list[str]:
    """Explain why a prior claim can or cannot be matched at scenario level."""
    try:
        from summarize_tr_bugs import prediction_specificity_reasons
    except ModuleNotFoundError:
        from scripts.summarize_tr_bugs import prediction_specificity_reasons
    return prediction_specificity_reasons(
        str(hypothesis.get("area") or "").strip(),
        str(hypothesis.get("trigger") or "").strip(),
        str(hypothesis.get("expected_failure_mode") or "").strip(),
    )


def ranked_feature_matches(hypothesis: dict,
                           current_bugs: list[dict]) -> list[dict]:
    """Rank same-category matches that cover the area and a scenario facet."""
    trigger = str(hypothesis.get("trigger") or "").strip()
    failure_mode = str(hypothesis.get("expected_failure_mode") or "").strip()
    if not trigger or not failure_mode:
        return []
    area_tokens = feature_match_tokens(hypothesis.get("area"))
    facet_tokens = {
        "area": area_tokens,
        # Target words commonly repeat in the prose for both scenario facets.
        # Remove them so "preview" cannot masquerade as trigger/failure proof.
        "trigger": feature_match_tokens(trigger) - area_tokens,
        "failure_mode": feature_match_tokens(failure_mode) - area_tokens,
    }
    hypothesis_tokens = set().union(*facet_tokens.values())
    if len(hypothesis_tokens) < 2:
        return []
    candidates = []
    for bug in current_bugs:
        bug_tokens = feature_match_tokens(
            bug.get("Short Description"), bug.get("ReproSteps"),
            bug.get("Description"), bug.get("ParsedModule"),
        )
        overlap_facets = {
            facet: sorted(tokens & bug_tokens)
            for facet, tokens in facet_tokens.items()
        }
        overlap = sorted(hypothesis_tokens & bug_tokens)
        # Area-only overlap is not scenario evidence. Require the observed bug
        # to name the predicted target plus at least one trigger/failure term.
        if (
            len(overlap) < 2 or not bug_tokens
            or not overlap_facets["area"]
            or not (
                overlap_facets["trigger"]
                or overlap_facets["failure_mode"]
            )
        ):
            continue
        cosine_score = len(overlap) / math.sqrt(
            len(hypothesis_tokens) * len(bug_tokens)
        )
        covered_facets = sum(bool(values) for values in overlap_facets.values())
        # Preserve the explainable cosine score when all three facets match;
        # discount partial scenario coverage so it must clear the same cutoff
        # with less evidence from generic area language.
        score = cosine_score * (0.5 + covered_facets / 6)
        candidates.append({
            "bug_code": str(bug.get("BugCode") or ""),
            "summary": re.sub(
                r"\s+", " ", str(bug.get("Short Description") or "")
            ).strip()[:300],
            "match_score": round(score, 3),
            "_raw_match_score": score,
            "overlap_terms": overlap,
            "overlap_facets": overlap_facets,
            "matcher_policy": FEATURE_MATCHER_POLICY,
            "review_status": "unreviewed",
        })
    candidates.sort(key=lambda item: (-item["_raw_match_score"], item["bug_code"]))
    return candidates


def feature_match_candidates(
    hypothesis: dict, current_bugs: list[dict], *,
    threshold: float = DEFAULT_FEATURE_MATCH_THRESHOLD,
) -> list[dict]:
    """Return above-threshold review candidates; never auto-confirm a prediction."""
    candidates = [
        {
            **{key: value for key, value in item.items()
               if key != "_raw_match_score"},
            "match_score": max(
                round(float(item["_raw_match_score"]), 6), float(threshold)
            ),
            "candidate_kind": "feature_match",
            "applied_threshold": round(float(threshold), 3),
        }
        for item in ranked_feature_matches(hypothesis, current_bugs)
        if float(item["_raw_match_score"]) >= threshold
    ]
    return candidates[:3]


def threshold_audit_candidates(
    hypothesis: dict, current_bugs: list[dict], *,
    threshold: float = DEFAULT_FEATURE_MATCH_THRESHOLD,
) -> list[dict]:
    """Sample one near miss to measure false negatives without calling it a prediction."""
    audit_floor = max(
        FEATURE_MATCH_AUDIT_MIN_SCORE,
        float(threshold) - FEATURE_MATCH_AUDIT_BAND,
    )
    candidates = [
        {
            **{key: value for key, value in item.items()
               if key != "_raw_match_score"},
            "match_score": min(
                round(float(item["_raw_match_score"]), 6),
                max(0.0, float(threshold) - 0.000001),
            ),
            "candidate_kind": "threshold_audit",
            "applied_threshold": round(float(threshold), 3),
            "audit_reason": "highest-scoring candidate just below the applied threshold",
        }
        for item in ranked_feature_matches(hypothesis, current_bugs)
        if audit_floor <= float(item["_raw_match_score"]) < threshold
    ]
    return candidates[:1]


def evaluate_prior_hypotheses(prior_analysis: dict, current_details: dict, *,
                              prior_close_build: str,
                              current_close_build: str,
                              feature_match_threshold: float = DEFAULT_FEATURE_MATCH_THRESHOLD,
                              threshold_calibration: dict | None = None,
                              category_thresholds: dict | None = None) -> dict:
    """Measure coarse category overlap without claiming scenario-level accuracy."""
    observed: defaultdict[str, list[str]] = defaultdict(list)
    observed_bugs: defaultdict[str, list[dict]] = defaultdict(list)
    for bugs in current_details.get("products", {}).values():
        for bug in bugs:
            category = str(bug.get("ModuleCategory") or "").strip()
            code = str(bug.get("BugCode") or "").strip()
            if category and code and code not in observed[category]:
                observed[category].append(code)
                observed_bugs[category].append(bug)

    hypotheses = []
    overlap_count = 0
    feature_candidate_count = 0
    threshold_audit_hypothesis_count = 0
    threshold_audit_candidate_count = 0
    category_evaluable_count = 0
    scenario_evaluable_count = 0
    scenario_evaluable_overlap_count = 0
    insufficient_prediction_detail_count = 0
    historical_support_hypothesis_count = 0
    historical_support_provenance_count = 0
    historical_provenance_available = (
        int(prior_analysis.get("schema_version") or 0) >= 6
    )
    boundary_provenance_available = (
        int(prior_analysis.get("schema_version") or 0) >= 7
    )
    synthesis_lineage_schema_available = (
        int(prior_analysis.get("schema_version") or 0) >= 10
    )
    raw_target_alias_policy = prior_analysis.get("target_alias_policy")
    target_alias_provenance_available = (
        int(prior_analysis.get("schema_version") or 0) >= 27
        and isinstance(raw_target_alias_policy, dict)
        and raw_target_alias_policy.get("schema_version") == 1
        and isinstance(raw_target_alias_policy.get("active_aliases"), dict)
    )
    active_target_aliases = (
        raw_target_alias_policy.get("active_aliases") or {}
        if target_alias_provenance_available else {}
    )
    raw_guardrail_policy = prior_analysis.get("analysis_guardrail_policy")
    prior_analysis_model = str(prior_analysis.get("model") or "").strip()
    prior_analysis_engine = str(
        prior_analysis.get("engine_sha256") or ""
    ).strip().lower()
    lineage_run_provenance_available = (
        synthesis_lineage_schema_available
        and bool(prior_analysis_model)
        and len(prior_analysis_model) <= 120
        and bool(re.fullmatch(r"[a-f0-9]{64}", prior_analysis_engine))
    )
    guardrail_provenance_available = (
        isinstance(raw_guardrail_policy, dict)
        and raw_guardrail_policy.get("schema_version") == 1
        and isinstance(raw_guardrail_policy.get("active_guardrails"), dict)
        and bool(prior_analysis_model)
        and len(prior_analysis_model) <= 120
        and bool(re.fullmatch(r"[a-f0-9]{64}", prior_analysis_engine))
    )
    prior_guardrails = (
        compact_analysis_guardrail_policy(raw_guardrail_policy).get(
            "active_guardrails"
        ) or {}
        if guardrail_provenance_available else {}
    )
    applied_guardrail_hypothesis_count = 0
    synthesis_lineage_provenance_count = 0
    recovered_origin_hypothesis_count = 0
    target_alias_provenance_hypothesis_count = 0
    alias_supported_hypothesis_count = 0
    for item in prior_analysis.get("recommendations") or []:
        if not isinstance(item, dict):
            continue
        historical_refs = [
            ref for ref in (item.get("historical_analog_refs") or [])
            if isinstance(ref, dict)
            and str(ref.get("analog_bug_code") or "").strip()
        ]
        historical_codes = list(dict.fromkeys(
            str(ref.get("analog_bug_code") or "").strip()
            for ref in historical_refs
        ))[:3]
        historical_scores = [
            float(ref["similarity_score"])
            for ref in historical_refs
            if isinstance(ref.get("similarity_score"), (int, float))
            and not isinstance(ref.get("similarity_score"), bool)
            and math.isfinite(float(ref["similarity_score"]))
            and 0 <= float(ref["similarity_score"]) <= 1
        ]
        if historical_provenance_available:
            historical_support_provenance_count += 1
            if historical_codes:
                historical_support_hypothesis_count += 1
        target = str(item.get("target_category") or "").strip()
        applied_guardrail = prior_guardrails.get(target)
        if not isinstance(applied_guardrail, dict):
            applied_guardrail = {}
        if applied_guardrail:
            applied_guardrail_hypothesis_count += 1
        prediction_origin = str(item.get("prediction_origin") or "")
        synthesis_source_ids = item.get("synthesis_source_candidate_ids")
        recovered_source_ids = item.get("recovered_source_candidate_ids")
        lineage_ids_valid = (
            isinstance(synthesis_source_ids, list)
            and bool(synthesis_source_ids)
            and all(
                isinstance(candidate_id, str)
                and bool(re.fullmatch(r"TRC-[A-F0-9]{12}", candidate_id))
                for candidate_id in synthesis_source_ids
            )
            and len(set(synthesis_source_ids)) == len(synthesis_source_ids)
            and isinstance(recovered_source_ids, list)
            and all(
                isinstance(candidate_id, str)
                and candidate_id in synthesis_source_ids
                for candidate_id in recovered_source_ids
            )
            and len(set(recovered_source_ids)) == len(recovered_source_ids)
        )
        expected_prediction_origin = None
        if lineage_ids_valid:
            if recovered_source_ids and set(recovered_source_ids) == set(
                synthesis_source_ids
            ):
                expected_prediction_origin = "recovered_draft_only"
            elif recovered_source_ids:
                expected_prediction_origin = "mixed_initial_and_recovered"
            else:
                expected_prediction_origin = "initial_audit_pass_only"
        item_lineage_provenance_available = (
            lineage_run_provenance_available
            and prediction_origin == expected_prediction_origin
        )
        if not item_lineage_provenance_available or prediction_origin not in {
            "initial_audit_pass_only", "recovered_draft_only",
            "mixed_initial_and_recovered",
        }:
            prediction_origin = "legacy_unknown"
            synthesis_source_ids = []
            recovered_source_ids = []
        else:
            synthesis_lineage_provenance_count += 1
        if prediction_origin in {
            "recovered_draft_only", "mixed_initial_and_recovered",
        }:
            recovered_origin_hypothesis_count += 1
        target_alias_pairs = []
        if target_alias_provenance_available:
            target_alias_provenance_hypothesis_count += 1
            adjacency_basis = item.get("adjacency_support_basis") or {}
            if (
                isinstance(adjacency_basis, dict)
                and adjacency_basis.get("reviewed_aliases_influence_support") is True
                and isinstance(adjacency_basis.get("reviewed_alias_matches"), list)
            ):
                for alias_match in adjacency_basis["reviewed_alias_matches"]:
                    if not isinstance(alias_match, dict):
                        continue
                    target_term = str(
                        alias_match.get("target_term") or ""
                    ).strip().casefold()
                    evidence_term = str(
                        alias_match.get("evidence_alias_term") or ""
                    ).strip().casefold()
                    pair = {
                        "target_term": target_term,
                        "evidence_term": evidence_term,
                    }
                    if (
                        re.fullmatch(r"[a-z0-9]{2,40}", target_term)
                        and re.fullmatch(r"[a-z0-9]{2,40}", evidence_term)
                        and evidence_term in (
                            active_target_aliases.get(target_term) or []
                        )
                        and pair not in target_alias_pairs
                    ):
                        target_alias_pairs.append(pair)
            target_alias_pairs = target_alias_pairs[:4]
            if target_alias_pairs:
                alias_supported_hypothesis_count += 1
        threshold_info = (
            (category_thresholds or {}).get(target)
            if isinstance(category_thresholds, dict) else None
        )
        if not isinstance(threshold_info, dict):
            threshold_info = {
                "threshold": feature_match_threshold,
                "source": "global_or_default",
                "status": str(
                    (threshold_calibration or {}).get("status")
                    or "legacy_default"
                ),
            }
        try:
            hypothesis_threshold = float(
                threshold_info.get("threshold", feature_match_threshold)
            )
        except (TypeError, ValueError):
            hypothesis_threshold = float(feature_match_threshold)
        try:
            threshold_decisive_review_count = int(
                threshold_info.get("decisive_review_count") or 0
            )
        except (TypeError, ValueError):
            threshold_decisive_review_count = 0
        try:
            threshold_effective_decisive_review_count = float(
                threshold_info.get("effective_decisive_review_count") or 0
            )
        except (TypeError, ValueError):
            threshold_effective_decisive_review_count = 0.0
        try:
            threshold_reviewed_cohort_count = int(
                threshold_info.get("reviewed_cohort_count") or 0
            )
            threshold_reviewed_hypothesis_count = int(
                threshold_info.get("reviewed_hypothesis_count") or 0
            )
        except (TypeError, ValueError):
            threshold_reviewed_cohort_count = 0
            threshold_reviewed_hypothesis_count = 0
        scenario_reasons = scenario_evaluation_reasons(item)
        scenario_evaluable = not scenario_reasons
        if not target or target == "Cross-cutting / unknown":
            result = "unscorable_cross_cutting"
            matches = []
        else:
            category_evaluable_count += 1
            if scenario_evaluable:
                scenario_evaluable_count += 1
            else:
                insufficient_prediction_detail_count += 1
            matches = observed.get(target, [])
            if matches:
                overlap_count += 1
                if scenario_evaluable:
                    scenario_evaluable_overlap_count += 1
                    feature_candidates = feature_match_candidates(
                        item, observed_bugs.get(target, []),
                        threshold=hypothesis_threshold,
                    )
                    audit_candidates = threshold_audit_candidates(
                        item, observed_bugs.get(target, []),
                        threshold=hypothesis_threshold,
                    )
                    if audit_candidates:
                        threshold_audit_hypothesis_count += 1
                        threshold_audit_candidate_count += len(audit_candidates)
                    if feature_candidates:
                        result = "feature_match_candidate"
                        feature_candidate_count += 1
                    else:
                        result = "category_overlap"
                else:
                    result = "category_overlap_insufficient_prediction_detail"
                    feature_candidates = []
                    audit_candidates = []
            else:
                result = "not_observed_in_current_cohort"
                feature_candidates = []
                audit_candidates = []
        if result == "unscorable_cross_cutting":
            feature_candidates = []
            audit_candidates = []
        hypotheses.append({
            "hypothesis_id": str(item.get("hypothesis_id") or ""),
            "target_category": target,
            "area": str(item.get("area") or ""),
            "trigger": str(item.get("trigger") or ""),
            "expected_failure_mode": str(
                item.get("expected_failure_mode") or ""
            ),
            "scope_boundary": str(item.get("scope_boundary") or ""),
            "disconfirming_observation": str(
                item.get("disconfirming_observation") or ""
            ),
            "prediction_boundary_provenance": (
                "schema7_validated"
                if boundary_provenance_available else "unavailable_legacy"
            ),
            "prediction_origin": prediction_origin,
            "prediction_origin_provenance": (
                "schema11_exact_claim_lineage"
                if item_lineage_provenance_available
                and int(prior_analysis.get("schema_version") or 0) >= 11
                else "schema10_exact_synthesis_lineage"
                if item_lineage_provenance_available
                else "unavailable_legacy"
            ),
            "synthesis_source_candidate_ids": list(synthesis_source_ids),
            "recovered_source_candidate_ids": list(recovered_source_ids),
            "target_alias_provenance": (
                "schema27_exact_basis"
                if target_alias_provenance_available else "unavailable_legacy"
            ),
            "target_alias_pairs": target_alias_pairs,
            "analysis_guardrail_provenance": (
                "exact_policy"
                if guardrail_provenance_available else "legacy_unknown"
            ),
            "analysis_guardrail": (
                str(applied_guardrail.get("guardrail") or "") or None
            ),
            "analysis_guardrail_source_rejection_reason": (
                str(applied_guardrail.get("source_rejection_reason") or "")
                or None
            ),
            "analysis_model": (
                prior_analysis_model
                if guardrail_provenance_available
                or item_lineage_provenance_available else None
            ),
            "analysis_engine_sha256": (
                prior_analysis_engine
                if guardrail_provenance_available
                or item_lineage_provenance_available else None
            ),
            "historical_support": (
                bool(historical_codes)
                if historical_provenance_available else None
            ),
            "historical_support_provenance": (
                "schema6_exact_refs"
                if historical_provenance_available else "unavailable_legacy"
            ),
            "historical_analog_count": len(historical_codes),
            "historical_analog_codes": historical_codes,
            "historical_analog_max_similarity": (
                round(max(historical_scores), 6)
                if historical_scores else None
            ),
            "feature_match_threshold": round(hypothesis_threshold, 3),
            "threshold_source": str(
                threshold_info.get("source") or "global_or_default"
            ),
            "threshold_status": str(
                threshold_info.get("status") or "legacy_default"
            ),
            "threshold_decisive_review_count": threshold_decisive_review_count,
            "threshold_effective_decisive_review_count": round(
                threshold_effective_decisive_review_count, 3
            ),
            "threshold_reviewed_cohort_count": threshold_reviewed_cohort_count,
            "threshold_reviewed_hypothesis_count": (
                threshold_reviewed_hypothesis_count
            ),
            "scenario_evaluation_status": (
                "unscorable_target"
                if not target or target == "Cross-cutting / unknown"
                else "evaluable" if scenario_evaluable
                else "insufficient_prediction_detail"
            ),
            "scenario_evaluation_reasons": scenario_reasons,
            "result": result,
            "matching_bug_codes": matches,
            "feature_match_candidates": feature_candidates,
            "threshold_audit_candidates": audit_candidates,
        })

    return {
        "schema_version": 10,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "prior_close_build": prior_close_build,
        "current_close_build": current_close_build,
        "prior_generated_at": str(prior_analysis.get("generated_at") or ""),
        "scope": "module-category overlap in the current fixed-bug TR cohort",
        "matcher_policy": FEATURE_MATCHER_POLICY,
        "limitation": (
            "Category overlap is a coarse retrospective signal. It does not prove "
            "the predicted scenario occurred, and absence does not prove a false positive. "
            "Feature matches require lexical coverage of the predicted area plus "
            "at least one trigger or failure-mode facet, and still require human "
            "confirmation."
            " Below-threshold audit samples are not predictions; they measure possible "
            "matcher false negatives."
        ),
        # scoreable_hypothesis_count is retained as a compatibility alias for
        # category-level evaluation; it must not be used as the scenario-rate
        # denominator.
        "scoreable_hypothesis_count": category_evaluable_count,
        "category_evaluable_hypothesis_count": category_evaluable_count,
        "scenario_evaluable_hypothesis_count": scenario_evaluable_count,
        "insufficient_prediction_detail_count": (
            insufficient_prediction_detail_count
        ),
        "scenario_evaluable_category_overlap_count": (
            scenario_evaluable_overlap_count
        ),
        "category_overlap_count": overlap_count,
        "category_overlap_rate": (
            overlap_count / category_evaluable_count
            if category_evaluable_count else None
        ),
        "feature_candidate_hypothesis_count": feature_candidate_count,
        "feature_candidate_rate": (
            feature_candidate_count / scenario_evaluable_count
            if scenario_evaluable_count else None
        ),
        "feature_candidate_rate_given_category_observed": (
            feature_candidate_count / scenario_evaluable_overlap_count
            if scenario_evaluable_overlap_count else None
        ),
        "threshold_audit_hypothesis_count": threshold_audit_hypothesis_count,
        "threshold_audit_candidate_count": threshold_audit_candidate_count,
        "historical_support_provenance_hypothesis_count": (
            historical_support_provenance_count
        ),
        "historical_support_hypothesis_count": (
            historical_support_hypothesis_count
        ),
        "analysis_guardrail_provenance_hypothesis_count": (
            len(hypotheses) if guardrail_provenance_available else 0
        ),
        "applied_analysis_guardrail_hypothesis_count": (
            applied_guardrail_hypothesis_count
        ),
        "synthesis_lineage_provenance_hypothesis_count": (
            synthesis_lineage_provenance_count
        ),
        "recovered_origin_hypothesis_count": recovered_origin_hypothesis_count,
        "target_alias_provenance_hypothesis_count": (
            target_alias_provenance_hypothesis_count
        ),
        "alias_supported_hypothesis_count": alias_supported_hypothesis_count,
        "feature_match_threshold": round(float(feature_match_threshold), 3),
        "threshold_calibration": threshold_calibration,
        "category_thresholds": category_thresholds or {},
        "observed_categories": sorted(observed, key=casefold_order),
        "hypotheses": hypotheses,
    }


def audit_report_manifest(product_dir: Path, manifest: dict) -> dict:
    """Read-only integrity audit used by the dashboard and tests."""
    warnings = []
    paths = {
        key: resolve_product_artifact(product_dir, manifest.get(key))
        for key in ("details_path", "analysis_path", "report_path")
    }
    unsafe = [key for key, path in paths.items() if manifest.get(key) and path is None]
    if unsafe:
        warnings.append("Unsafe artifact path(s): " + ", ".join(unsafe))
    schema_v2 = manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION
    if not schema_v2:
        warnings.append(
            "This is a legacy report without immutable source and artifact hashes; "
            "refresh it before treating the evidence as current."
        )
    else:
        missing_paths = [
            key for key in ("details_path", "analysis_path", "report_path")
            if not manifest.get(key)
        ]
        if missing_paths:
            warnings.append("Manifest is missing artifact path(s): " + ", ".join(missing_paths))
    for path_key, hash_key in (
        ("details_path", "details_sha256"),
        ("analysis_path", "analysis_sha256"),
        ("report_path", "report_sha256"),
    ):
        path = paths[path_key]
        if not path:
            continue
        if not path.is_file():
            warnings.append(f"Referenced {path_key} is missing.")
        elif schema_v2 and sha256_file(path) != manifest.get(hash_key):
            warnings.append(f"Referenced {path_key} failed its integrity hash check.")
    details_path = paths.get("details_path")
    if details_path and details_path.is_file():
        try:
            details = json.loads(details_path.read_text(encoding="utf-8"))
            detail_codes = [
                str(bug.get("BugCode"))
                for bugs in details.get("products", {}).values() for bug in bugs
                if bug.get("BugCode")
            ]
            manifest_codes = [str(code) for code in manifest.get("bug_codes", [])]
            if detail_codes != manifest_codes:
                warnings.append(
                    "Stored evidence BugCodes/count do not match the report manifest."
                )
        except (OSError, json.JSONDecodeError, AttributeError):
            warnings.append("Stored bug evidence is unreadable.")
    if schema_v2:
        analysis_path = paths.get("analysis_path")
        if analysis_path and analysis_path.is_file():
            try:
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
                if [str(code) for code in analysis.get("bug_codes", [])] != [
                    str(code) for code in manifest.get("bug_codes", [])
                ]:
                    warnings.append("Structured analysis BugCodes do not match the manifest.")
                if analysis.get("source_sha256") != manifest.get("source_sha256"):
                    warnings.append("Structured analysis source hash does not match the manifest.")
                if analysis.get("engine_sha256") != manifest.get("engine_sha256"):
                    warnings.append("Structured analysis engine hash does not match the manifest.")
                warnings.extend(structured_prediction_errors(analysis))
            except (OSError, json.JSONDecodeError, AttributeError):
                warnings.append("Structured analysis is unreadable.")
        report_path = paths.get("report_path")
        if report_path and report_path.is_file():
            try:
                report_text = report_path.read_text(encoding="utf-8", errors="replace")
                report_codes = re.findall(
                    r"(?m)^##[ \t]+([A-Za-z0-9._-]+)[ \t]*$", report_text
                )
                if report_codes != [str(code) for code in manifest.get("bug_codes", [])]:
                    warnings.append("Rendered report checklists do not match the manifest BugCodes.")
            except OSError:
                warnings.append("Rendered report is unreadable.")
    return {
        "warnings": list(dict.fromkeys(warnings)),
        "paths": paths,
        "report_safe": paths.get("report_path") is not None,
    }


def extract_manifest_checklists(verification_markdown: str,
                                bug_codes: list[str]) -> list[tuple[str, str]]:
    """Extract only exact manifest-backed level-2 checklist blocks."""
    allowed = set(bug_codes)
    if not allowed:
        return []
    headings = list(re.finditer(
        r"^##[ \t]+(.+?)[ \t]*$", verification_markdown, flags=re.M,
    ))
    blocks = []
    for index, heading in enumerate(headings):
        code = heading.group(1).strip()
        if code not in allowed:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(verification_markdown)
        blocks.append((
            code,
            verification_markdown[heading.start():end].strip(),
        ))
    return blocks


def response_detail_codes(data: dict, expected_product: str) -> tuple[set[str], set[str]]:
    """Return (usable, degraded) BugCodes for the requested product only."""
    usable: set[str] = set()
    degraded: set[str] = set()
    for product, bugs in data.get("products", {}).items():
        if product != expected_product:
            continue
        for bug in bugs:
            code = str(bug.get("BugCode") or "").strip()
            if not code:
                continue
            if has_usable_full_detail(bug):
                usable.add(code)
            else:
                degraded.add(code)
    return usable, degraded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process newly closed bugs in the newest TR into a QA test-focus report."
    )
    parser.add_argument("--product", required=True, choices=list(PRODUCT_MAP))
    parser.add_argument("--product-dir", default=None,
                        help="Default: data/products/<product>")
    parser.add_argument("--parsed-csv", default=None,
                        help="Default: <product-dir>/ecl_parsed.csv")
    parser.add_argument("--webhook-url", default=DEFAULT_WEBHOOK_URL)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--summary-timeout", type=int, default=600)
    parser.add_argument("--safety-limit", type=int, default=500,
                        help="Fail safely when a cohort exceeds this size (default: 500)")
    args = parser.parse_args()

    product_dir = Path(args.product_dir or f"data/products/{args.product}")
    report_root = product_dir / "tr_reports"
    latest_path = report_root / "latest.json"
    prior_manifest = {}
    if latest_path.exists():
        try:
            prior_manifest = json.loads(latest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            prior_manifest = {}
    latest_success = str(prior_manifest.get("generated_at") or "")
    parsed_csv = Path(args.parsed_csv or product_dir / "ecl_parsed.csv")
    if not parsed_csv.exists():
        print(f"ERROR: parsed bug file not found: {parsed_csv}", file=sys.stderr)
        write_attempt_status(
            report_root, args.product, "fetch_failed",
            message=f"parsed bug file not found: {parsed_csv}",
            latest_success=latest_success,
        )
        return 1

    close_build, bug_codes = select_newest_tr_cohort(parsed_csv)
    if not bug_codes:
        print(f"[{args.product}] No TRCreated cohort found; summary unchanged.")
        write_attempt_status(
            report_root, args.product, "no_cohort",
            message="No TRCreated cohort exists in the newest parsed product data.",
            latest_success=latest_success,
        )
        return 0

    label = sanitize_label(close_build)
    if not label:
        label = "unknown_build"
    print(f"[{args.product}] New TR cohort: build={close_build!r}, bugs={len(bug_codes)}")

    payload = {
        "product_name": PRODUCT_MAP[args.product],
        "close_build": close_build,
        "bug_codes": bug_codes,
    }
    try:
        incoming = normalize_response(
            post_json(args.webhook_url, payload, timeout=args.timeout)
        )
    except FetchError as exc:
        print(f"ERROR: TR detail fetch failed: {exc}", file=sys.stderr)
        write_attempt_status(
            report_root, args.product, "fetch_failed",
            close_build=close_build, bug_codes=bug_codes, message=exc,
            latest_success=latest_success,
        )
        return 1

    # The n8n workflow fetches one detail record per BugCode. Its eCL node can
    # return HTTP 200 with a per-item transient error, so compare the response
    # to the requested cohort and retry only missing codes (up to two rounds).
    expected_product = PRODUCT_MAP[args.product]

    for retry_round in range(1, 3):
        usable, degraded = response_detail_codes(incoming, expected_product)
        retry_codes = [code for code in bug_codes if code not in usable]
        if not retry_codes:
            break
        print(f"[{args.product}] Retrying {len(retry_codes)} missing/degraded bug detail(s) "
              f"(round {retry_round}/2)")
        retry_payload = {**payload, "bug_codes": retry_codes}
        try:
            retry_data = normalize_response(
                post_json(args.webhook_url, retry_payload, timeout=args.timeout)
            )
        except FetchError as exc:
            print(f"WARNING: missing-detail retry failed: {exc}")
            continue
        incoming = merge_detail_data(incoming, retry_data)

    usable, degraded = response_detail_codes(incoming, expected_product)
    missing = [code for code in bug_codes if code not in usable]
    if missing:
        print(f"WARNING: n8n did not return {len(missing)} requested bug detail(s): "
              f"{', '.join(missing)}. Using parsed-list fallback.")
        fallback_records = local_fallback_bugs(parsed_csv, missing)
        found_fallbacks = {bug["BugCode"] for bug in fallback_records}
        still_missing = [code for code in missing if code not in found_fallbacks]
        if still_missing:
            print(f"ERROR: no local fallback found for: {', '.join(still_missing)}",
                  file=sys.stderr)
            write_attempt_status(
                report_root, args.product, "fetch_failed",
                close_build=close_build, bug_codes=bug_codes,
                message="No local fallback found for: " + ", ".join(still_missing),
                latest_success=latest_success,
            )
            return 1
        incoming.setdefault("products", {}).setdefault(expected_product, []).extend(fallback_records)
    transient_errors = list(incoming.get("meta", {}).get("errors", []))
    if transient_errors:
        incoming["meta"]["recovered_errors"] = transient_errors
        incoming["meta"]["errors"] = []

    existing = None
    prior_details_path = resolve_product_artifact(product_dir, prior_manifest.get("details_path"))
    legacy_details_path = report_root / label / "bug_details.json"
    for candidate in (prior_details_path, legacy_details_path):
        if not candidate or not candidate.exists():
            continue
        try:
            existing = json.loads(candidate.read_text(encoding="utf-8"))
            break
        except (json.JSONDecodeError, OSError):
            print(f"WARNING: ignoring unreadable prior detail file: {candidate}")

    details = filter_detail_data(
        merge_detail_data(existing, incoming), bug_codes,
        expected_product=expected_product,
    )
    review_ledger = None
    historical_analog_threshold_policy = None
    analysis_guardrail_policy = None
    target_alias_policy = None
    try:
        review_ledger = load_review_ledger(
            report_root / "prediction_reviews.json", args.product
        )
        historical_analog_threshold_policy = (
            calibrate_historical_analog_policy(review_ledger)
        )
        analysis_guardrail_policy = compact_analysis_guardrail_policy(
            build_analysis_guardrail_policy(review_ledger)
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(
            "WARNING: reviewed calibration and analysis guardrails unavailable; "
            f"using default policies: {exc}"
        )
    try:
        grounding_review_ledger = load_grounding_review_ledger(
            report_root / "grounding_reviews.json", args.product
        )
        target_alias_policy = compact_target_alias_policy(
            build_target_alias_policy(grounding_review_ledger, review_ledger)
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(
            "WARNING: reviewed target aliases unavailable; using no aliases: "
            f"{exc}"
        )
    details = attach_module_category_context(
        details, parsed_csv,
        historical_analog_threshold_policy=historical_analog_threshold_policy,
    )
    details.setdefault("meta", {})["analysis_guardrail_policy"] = (
        analysis_guardrail_policy or {
            "schema_version": 1,
            "policy": (
                "Controlled category guardrail codes derived from diverse "
                "structured reviews; reviewer notes are excluded."
            ),
            "active_guardrails": {},
        }
    )
    details.setdefault("meta", {})["target_alias_policy"] = (
        target_alias_policy or compact_target_alias_policy(None)
    )
    if transient_errors:
        details.setdefault("meta", {})["recovered_errors"] = transient_errors
    detail_codes = [
        str(bug.get("BugCode"))
        for bugs in details.get("products", {}).values() for bug in bugs
        if bug.get("BugCode")
    ]
    absent = [code for code in bug_codes if code not in set(detail_codes)]
    if absent:
        message = "Final evidence is missing BugCodes: " + ", ".join(absent)
        print(f"ERROR: {message}", file=sys.stderr)
        write_attempt_status(
            report_root, args.product, "fetch_failed",
            close_build=close_build, bug_codes=bug_codes, message=message,
            latest_success=latest_success,
        )
        return 1

    source_hash, analysis_schema = analysis_source_hash(details)
    engine_hash = analysis_engine_sha256()
    if manifest_cache_hit(
        product_dir, prior_manifest, source_hash=source_hash, engine_sha256=engine_hash,
        model=args.model, analysis_schema=analysis_schema,
        close_build=close_build, bug_codes=bug_codes,
    ):
        print(f"[{args.product}] Evidence/model unchanged; keeping validated report.")
        write_attempt_status(
            report_root, args.product, "unchanged",
            close_build=close_build, bug_codes=bug_codes,
            message="Evidence, model, and analysis schema are unchanged.",
            latest_success=latest_success,
        )
        return 0

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = f"{run_stamp}_{source_hash[:8]}"
    stage_dir = report_root / ".staging" / run_name
    run_dir = report_root / label / "runs" / run_name
    stage_dir.mkdir(parents=True, exist_ok=False)
    stage_details = stage_dir / "bug_details.json"
    stage_analysis = stage_dir / "analysis.json"
    stage_report = stage_dir / "test_focus_summary.md"
    write_json(stage_details, details)

    command = [
        sys.executable, str(Path(__file__).with_name("summarize_tr_bugs.py")),
        str(stage_details), "--output", str(stage_report),
        "--analysis-output", str(stage_analysis), "--model", args.model,
        "--max-bugs", "0", "--safety-limit", str(args.safety_limit),
        "--timeout", str(args.summary_timeout),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        shutil.rmtree(stage_dir, ignore_errors=True)
        write_attempt_status(
            report_root, args.product, "analysis_failed",
            close_build=close_build, bug_codes=bug_codes,
            message=f"Structured analysis exited with code {result.returncode}.",
            latest_success=latest_success,
        )
        return result.returncode

    try:
        analysis = json.loads(stage_analysis.read_text(encoding="utf-8"))
        if analysis.get("schema_version") != analysis_schema:
            raise ValueError("analysis schema mismatch")
        if analysis.get("source_sha256") != source_hash:
            raise ValueError("analysis source hash mismatch")
        if analysis.get("engine_sha256") != engine_hash:
            raise ValueError("analysis engine hash mismatch")
        if analysis.get("bug_codes") != bug_codes:
            raise ValueError("analysis BugCodes do not match cohort order")
        prediction_errors = structured_prediction_errors(analysis)
        if prediction_errors:
            raise ValueError("; ".join(prediction_errors))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        shutil.rmtree(stage_dir, ignore_errors=True)
        write_attempt_status(
            report_root, args.product, "validation_failed",
            close_build=close_build, bug_codes=bug_codes, message=exc,
            latest_success=latest_success,
        )
        return 1

    prior_analysis_path, prior_scored_build = resolve_prior_scoreable_analysis(
        report_root, product_dir, prior_manifest, close_build
    )
    if not prior_analysis_path:
        analysis["prior_hypothesis_evaluation_skipped"] = {
            "reason": (
                "same_close_build"
                if str(prior_manifest.get("close_build") or "") == close_build
                else "no_prior_report"
            ),
            "current_close_build": close_build,
            "prior_close_build": str(prior_manifest.get("close_build") or ""),
        }
        write_json(stage_analysis, analysis)
    if prior_analysis_path and prior_analysis_path.is_file():
        try:
            prior_analysis = json.loads(prior_analysis_path.read_text(encoding="utf-8"))
            if (
                int(prior_analysis.get("schema_version") or 0) >= 4
                and not structured_prediction_errors(prior_analysis)
            ):
                feature_threshold = DEFAULT_FEATURE_MATCH_THRESHOLD
                threshold_calibration = None
                category_thresholds = {}
                try:
                    if review_ledger is None:
                        review_ledger = load_review_ledger(
                            report_root / "prediction_reviews.json", args.product
                        )
                    threshold_calibration = calibrate_threshold_policy(
                        review_ledger, matcher_policy=FEATURE_MATCHER_POLICY
                    )
                    global_threshold = resolve_threshold_for_category(
                        threshold_calibration, ""
                    )
                    feature_threshold = float(global_threshold["threshold"])
                    prior_categories = {
                        str(item.get("target_category") or "").strip()
                        for item in prior_analysis.get("recommendations") or []
                        if isinstance(item, dict)
                    }
                    category_thresholds = {
                        category: resolve_threshold_for_category(
                            threshold_calibration, category
                        )
                        for category in prior_categories if category
                    }
                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                    print(
                        "WARNING: prediction-review calibration unavailable; "
                        f"using default threshold: {exc}"
                    )
                analysis["prior_hypothesis_evaluation"] = evaluate_prior_hypotheses(
                    prior_analysis,
                    details,
                    prior_close_build=prior_scored_build,
                    current_close_build=close_build,
                    feature_match_threshold=feature_threshold,
                    threshold_calibration=threshold_calibration,
                    category_thresholds=category_thresholds,
                )
                write_json(stage_analysis, analysis)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"WARNING: could not evaluate prior TR hypotheses: {exc}")

    hashes = {
        "details_sha256": sha256_file(stage_details),
        "analysis_sha256": sha256_file(stage_analysis),
        "report_sha256": sha256_file(stage_report),
    }
    try:
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        stage_dir.replace(run_dir)
    except OSError as exc:
        shutil.rmtree(stage_dir, ignore_errors=True)
        write_attempt_status(
            report_root, args.product, "validation_failed",
            close_build=close_build, bug_codes=bug_codes,
            message=f"Could not promote immutable run: {exc}",
            latest_success=latest_success,
        )
        return 1
    final_details = run_dir / stage_details.name
    final_analysis = run_dir / stage_analysis.name
    final_report = run_dir / stage_report.name
    meta = details.get("meta", {})
    generated_at = str(analysis.get("generated_at") or datetime.now(timezone.utc).isoformat())
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "product": args.product,
        "close_build": close_build,
        "tr_codes": meta.get("tr_codes", []),
        "bug_creation_trs": meta.get("tr_codes", []),
        "bug_count": len(bug_codes),
        "bug_count_total": len(bug_codes),
        "bug_count_analyzed": int(analysis.get("bug_count") or 0),
        "bug_codes": bug_codes,
        "missing_detail_bug_codes": meta.get("missing_detail_bug_codes", []),
        "generated_at": generated_at,
        "model": args.model,
        "analysis_schema_version": analysis_schema,
        "source_sha256": source_hash,
        "engine_sha256": engine_hash,
        **hashes,
        "details_path": final_details.relative_to(product_dir).as_posix(),
        "analysis_path": final_analysis.relative_to(product_dir).as_posix(),
        "report_path": final_report.relative_to(product_dir).as_posix(),
    }
    try:
        write_json(latest_path, manifest)
    except OSError as exc:
        write_attempt_status(
            report_root, args.product, "validation_failed",
            close_build=close_build, bug_codes=bug_codes,
            message=f"Could not switch latest manifest: {exc}",
            latest_success=latest_success,
        )
        return 1
    write_attempt_status(
        report_root, args.product, "published",
        close_build=close_build, bug_codes=bug_codes,
        message="Published a validated immutable TR Test Focus run.",
        latest_success=generated_at,
    )
    print(f"[{args.product}] Latest TR test focus -> {final_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
