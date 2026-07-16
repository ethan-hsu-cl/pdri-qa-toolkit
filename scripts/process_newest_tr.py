#!/usr/bin/env python3
"""Fetch and summarize the fixed bugs in the newest TR detected by a refresh.

The regular eBug refresh marks status transitions with ``_status_changed``.
This script selects the newest product version's ``TRCreated`` cohort (the bugs
from which the Testing Request is created), derives its dominant Close Build#
for display, fetches their full status logs through n8n, and
generates the QA test-focus report. Repeated runs for the same build merge bugs
instead of replacing the earlier report cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
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
except ModuleNotFoundError:  # Imported as scripts.process_newest_tr in tests/tools
    from scripts.fetch_from_n8n import PRODUCT_MAP
    from scripts.fetch_tr_bug_details import (
        DEFAULT_WEBHOOK_URL,
        FetchError,
        normalize_response,
        post_json,
        sanitize_label,
    )


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
        mm, dd = int(token[:2]), int(token[2:])
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            month_days.append((datetime.now().year * 10_000 + mm * 100 + dd, token))
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
    """Return (dominant_close_build, bug_codes) for the newest Testing Request.

    eCL marks the authoritative "RD Resolved Bug" list as TRCreated. Select all
    TRCreated bugs in the newest parsed version, including component-build
    outliers, then use the cohort's most common Close Build# as its display build.
    """
    with parsed_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    tr_rows = [
        row for row in rows
        if str(row.get("Status") or "").strip().lower() == "trcreated"
        and str(row.get("BugCode") or "").strip()
        and str(row.get("parsed_version") or "").strip()
    ]
    if not tr_rows:
        return "", []

    def version_key(value: str) -> tuple[int, ...]:
        return tuple(int(part) for part in re.findall(r"\d+", value))

    newest_version = max(
        {str(row.get("parsed_version") or "").strip() for row in tr_rows},
        key=version_key,
    )
    cohort = [
        row for row in tr_rows
        if str(row.get("parsed_version") or "").strip() == newest_version
    ]
    cohort_codes = [str(row.get("BugCode") or "").strip() for row in cohort]

    build_counts: dict[str, int] = defaultdict(int)
    for row in cohort:
        build = str(row.get("Close Build#") or "").strip()
        if build and re.search(r"\d", build):
            build_counts[build] += 1
    if not build_counts:
        return newest_version, list(dict.fromkeys(cohort_codes))
    close_build = max(
        build_counts,
        key=lambda build: (build_counts[build], close_build_sort_key(build)),
    )
    return close_build, list(dict.fromkeys(cohort_codes))


def merge_detail_data(existing: dict | None, incoming: dict) -> dict:
    """Merge organized n8n responses by product and BugCode."""
    if not existing:
        return incoming

    merged_products: dict[str, list[dict]] = {}
    product_names = set(existing.get("products", {})) | set(incoming.get("products", {}))
    for product in product_names:
        by_code: dict[str, dict] = {}
        for source in (existing, incoming):
            for bug in source.get("products", {}).get(product, []):
                code = str(bug.get("BugCode", "")).strip()
                if code:
                    by_code[code] = bug
        merged_products[product] = list(by_code.values())

    meta = {**existing.get("meta", {}), **incoming.get("meta", {})}
    all_bugs = [bug for bugs in merged_products.values() for bug in bugs]
    meta["bug_count"] = len(all_bugs)
    meta["products"] = sorted(merged_products)
    meta["tr_codes"] = sorted({
        str(bug.get("ParentTRCode")).strip()
        for bug in all_bugs if bug.get("ParentTRCode")
    })
    meta["errors"] = list(dict.fromkeys(
        list(existing.get("meta", {}).get("errors", []))
        + list(incoming.get("meta", {}).get("errors", []))
    ))
    return {"meta": meta, "products": merged_products}


def filter_detail_data(data: dict, bug_codes: list[str]) -> dict:
    """Prune cached details to the authoritative current TR cohort."""
    wanted = set(bug_codes)
    products = {}
    for product, bugs in data.get("products", {}).items():
        kept = [bug for bug in bugs if str(bug.get("BugCode") or "").strip() in wanted]
        if kept:
            products[product] = kept
    all_bugs = [bug for bugs in products.values() for bug in bugs]
    meta = dict(data.get("meta", {}))
    meta["bug_count"] = len(all_bugs)
    meta["products"] = sorted(products)
    meta["tr_codes"] = sorted({
        str(bug.get("ParentTRCode")).strip()
        for bug in all_bugs if bug.get("ParentTRCode")
    })
    meta["missing_detail_bug_codes"] = [
        code for code in meta.get("missing_detail_bug_codes", []) if code in wanted
    ]
    return {"meta": meta, "products": products}


def local_fallback_bugs(parsed_csv: Path, missing_codes: list[str]) -> list[dict]:
    """Build transparent summary-only records when getOldeBug is unavailable."""
    wanted = set(missing_codes)
    fallbacks = []
    with parsed_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            code = str(row.get("BugCode") or "").strip()
            if code not in wanted:
                continue
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


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


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
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--summary-timeout", type=int, default=600)
    args = parser.parse_args()

    product_dir = Path(args.product_dir or f"data/products/{args.product}")
    parsed_csv = Path(args.parsed_csv or product_dir / "ecl_parsed.csv")
    if not parsed_csv.exists():
        print(f"ERROR: parsed bug file not found: {parsed_csv}", file=sys.stderr)
        return 1

    close_build, bug_codes = select_newest_tr_cohort(parsed_csv)
    if not bug_codes:
        print(f"[{args.product}] No TRCreated cohort found; summary unchanged.")
        return 0

    label = sanitize_label(close_build)
    report_dir = product_dir / "tr_reports" / label
    details_path = report_dir / "bug_details.json"
    report_path = report_dir / "test_focus_summary.md"
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
        return 1

    # The n8n workflow fetches one detail record per BugCode. Its eCL node can
    # return HTTP 200 with a per-item transient error, so compare the response
    # to the requested cohort and retry only missing codes (up to two rounds).
    def fetched_codes(response: dict) -> set[str]:
        return {
            str(bug.get("BugCode") or "").strip()
            for bugs in response.get("products", {}).values() for bug in bugs
            if bug.get("BugCode")
        }

    for retry_round in range(1, 3):
        missing = [code for code in bug_codes if code not in fetched_codes(incoming)]
        if not missing:
            break
        print(f"[{args.product}] Retrying {len(missing)} missing bug detail(s) "
              f"(round {retry_round}/2)")
        retry_payload = {**payload, "bug_codes": missing}
        try:
            retry_data = normalize_response(
                post_json(args.webhook_url, retry_payload, timeout=args.timeout)
            )
        except FetchError as exc:
            print(f"WARNING: missing-detail retry failed: {exc}")
            continue
        incoming = merge_detail_data(incoming, retry_data)

    missing = [code for code in bug_codes if code not in fetched_codes(incoming)]
    if missing:
        print(f"WARNING: n8n did not return {len(missing)} requested bug detail(s): "
              f"{', '.join(missing)}. Using parsed-list fallback.")
        fallback_records = local_fallback_bugs(parsed_csv, missing)
        found_fallbacks = {bug["BugCode"] for bug in fallback_records}
        still_missing = [code for code in missing if code not in found_fallbacks]
        if still_missing:
            print(f"ERROR: no local fallback found for: {', '.join(still_missing)}",
                  file=sys.stderr)
            return 1
        product_name = PRODUCT_MAP[args.product]
        incoming.setdefault("products", {}).setdefault(product_name, []).extend(fallback_records)
        incoming.setdefault("meta", {})["missing_detail_bug_codes"] = missing
    transient_errors = list(incoming.get("meta", {}).get("errors", []))
    if transient_errors:
        incoming["meta"]["recovered_errors"] = transient_errors
        incoming["meta"]["errors"] = []

    existing = None
    if details_path.exists():
        try:
            existing = json.loads(details_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print(f"WARNING: ignoring unreadable prior detail file: {details_path}")
    details = filter_detail_data(merge_detail_data(existing, incoming), bug_codes)
    write_json(details_path, details)

    command = [
        sys.executable, str(Path(__file__).with_name("summarize_tr_bugs.py")),
        str(details_path), "--output", str(report_path), "--model", args.model,
        "--timeout", str(args.summary_timeout),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        return result.returncode

    meta = details.get("meta", {})
    manifest = {
        "product": args.product,
        "close_build": close_build,
        "tr_codes": meta.get("tr_codes", []),
        "bug_count": meta.get("bug_count", 0),
        "bug_codes": [
            bug.get("BugCode")
            for bugs in details.get("products", {}).values() for bug in bugs
            if bug.get("BugCode")
        ],
        "missing_detail_bug_codes": meta.get("missing_detail_bug_codes", []),
        "generated_at": datetime.now().astimezone().isoformat(),
        "details_path": str(details_path),
        "report_path": str(report_path),
    }
    write_json(product_dir / "tr_reports" / "latest.json", manifest)
    print(f"[{args.product}] Latest TR test focus -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
