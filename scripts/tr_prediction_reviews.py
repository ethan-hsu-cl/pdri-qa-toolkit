#!/usr/bin/env python3
"""Durable human-review ledger for TR Test Focus prediction candidates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

try:  # Available on macOS/Linux; tests and production run on macOS.
    import fcntl
except ImportError:  # pragma: no cover - portability fallback
    fcntl = None


REVIEW_SCHEMA_VERSION = 1
REVIEW_DECISIONS = {"confirmed", "rejected", "unclear"}
GROUNDING_REVIEW_SCHEMA_VERSION = 1
GROUNDING_REVIEW_DECISIONS = {"correctly_withheld", "false_negative", "unclear"}
GROUNDING_AUDIT_STAGES = {
    "deterministic_adjacency_gate",
    "draft_before_synthesis",
    "final_after_synthesis",
}
GROUNDING_WITHHELD_REASONS = {
    "adjacent_target_absent_from_current_signal",
    "target_not_supported", "trigger_not_supported",
    "failure_mode_not_supported", "shared_path_not_concrete",
    "boundary_not_supported", "direct_fix_restatement",
}
GROUNDING_AUDIT_CHECKS = {
    "target_supported", "trigger_supported", "failure_mode_supported",
    "shared_path_concrete", "boundary_supported", "adjacent_not_direct_fix",
}
TARGET_ALIAS_MIN_CONFIRMED_REVIEWS = 4
TARGET_ALIAS_MIN_DISTINCT_CLOSE_BUILDS = 3
TARGET_ALIAS_MIN_DISTINCT_SOURCE_RUNS = 4
TARGET_ALIAS_RETIRE_MIN_DECISIVE_REVIEWS = 12
TARGET_ALIAS_RETIRE_MIN_EFFECTIVE_REVIEWS = 8.0
TARGET_ALIAS_RETIRE_MIN_MISLEADING_REVIEWS = 6
TARGET_ALIAS_RETIRE_MIN_DISTINCT_COHORTS = 3
TARGET_ALIAS_RETIRE_MIN_DISTINCT_HYPOTHESES = 3
TARGET_ALIAS_RETIRE_MIN_MISLEADING_RATE = 0.50
TARGET_ALIAS_RETIRE_MIN_WILSON_LOWER_BOUND = 0.30
TARGET_ALIAS_GENERIC_TERMS = {
    "another", "area", "backed", "based", "behavior", "component",
    "cache", "changed", "editor", "feature", "fixed", "flow", "manager",
    "operation", "other", "path", "preview", "related", "renderer", "result",
    "screen", "service", "shared", "state", "surface", "system", "updated",
    "workflow",
}
REVIEW_CANDIDATE_KINDS = {"feature_match", "threshold_audit"}
FEATURE_MATCHER_POLICY = "facet_coverage_v2"
REJECTION_REASON_LABELS = {
    "lexical_overlap_only": "Lexical overlap only",
    "wrong_target": "Wrong target area",
    "wrong_trigger": "Wrong trigger or precondition",
    "wrong_failure_mode": "Wrong failure mode",
    "outside_scope": "Outside the prediction boundary",
    "disconfirming_observation_seen": "Disconfirming observation was seen",
    "unsupported_shared_path": "Shared-path inference was unsupported",
    "historical_corroboration_misleading": (
        "Historical corroboration was misleading"
    ),
    "reviewed_target_alias_misleading": (
        "Reviewed target alias was misleading"
    ),
    "other": "Other reason",
}
REJECTION_REASONS = set(REJECTION_REASON_LABELS)
NON_MATCHER_REJECTION_REASONS = {
    "disconfirming_observation_seen",
    "unsupported_shared_path",
    "historical_corroboration_misleading",
    "reviewed_target_alias_misleading",
    "other",
}
ANALOG_NEGATIVE_REJECTION_REASONS = {
    "historical_corroboration_misleading",
}
AI_GUARDRAIL_BY_REJECTION_REASON = {
    "unsupported_shared_path": "require_stronger_shared_path_grounding",
    "disconfirming_observation_seen": "narrow_scope_and_disconfirmation",
}
ANALYSIS_GUARDRAIL_PROVENANCE_VALUES = {
    "exact_policy", "legacy_unknown",
}
EXACT_PREDICTION_ORIGIN_PROVENANCE_VALUES = {
    "schema10_exact_synthesis_lineage", "schema11_exact_claim_lineage",
}
PREDICTION_ORIGIN_PROVENANCE_VALUES = {
    *EXACT_PREDICTION_ORIGIN_PROVENANCE_VALUES, "unavailable_legacy",
}
PREDICTION_ORIGINS = {
    "initial_audit_pass_only", "recovered_draft_only",
    "mixed_initial_and_recovered", "legacy_unknown",
}
TARGET_ALIAS_PROVENANCE_VALUES = {
    "schema27_exact_basis", "unavailable_legacy",
}
ANALYSIS_GUARDRAILS = set(AI_GUARDRAIL_BY_REJECTION_REASON.values())
MAX_REVIEWER_CHARS = 100
MAX_NOTE_CHARS = 1000
DEFAULT_FEATURE_MATCH_THRESHOLD = 0.24
DEFAULT_HISTORICAL_ANALOG_THRESHOLD = 0.18
MIN_CALIBRATION_DECISIVE = 12
MIN_CALIBRATION_SELECTED = 8
TARGET_CONFIRMATION_RATE = 0.80
MIN_WILSON_LOWER_BOUND = 0.50
MIN_CONFIRMED_RETENTION_RATE = 0.80
MIN_CALIBRATION_COHORTS = 3
MIN_CALIBRATION_HYPOTHESES = 2
MAX_EFFECTIVE_REVIEWS_PER_COHORT = 4.0
MAX_EFFECTIVE_REVIEWS_PER_HYPOTHESIS_COHORT = 2.0


class ReviewLedgerError(RuntimeError):
    """Raised rather than replacing malformed or incompatible review history."""


def casefold_order(value: str) -> tuple[str, str]:
    """Total order for case-insensitive sorts of hashed analysis input.

    ``str.casefold`` alone leaves distinct case variants ("PIP" / "PiP") tied,
    so their relative order falls back to set iteration order, which Python
    randomizes per process. The parent hashes the analysis source in-process
    while the summarizer re-hashes it in a subprocess, so a tie that resolves
    differently between them breaks the source-hash check.
    """
    return (value.casefold(), value)


def _is_candidate_id_list(value: object, *, require_nonempty: bool) -> bool:
    return (
        isinstance(value, list)
        and (bool(value) or not require_nonempty)
        and all(
            isinstance(candidate_id, str)
            and bool(re.fullmatch(r"TRC-[A-F0-9]{12}", candidate_id))
            for candidate_id in value
        )
        and len(set(value)) == len(value)
    )


def review_key(hypothesis_id: str, prior_close_build: str,
               current_close_build: str, bug_code: str,
               matcher_policy: str | None = None) -> str:
    values = [hypothesis_id, prior_close_build, current_close_build, bug_code]
    if not re.fullmatch(r"TRH-[A-F0-9]{12}", str(hypothesis_id or "")):
        raise ReviewLedgerError("invalid hypothesis_id")
    if any(not str(value or "").strip() for value in values[1:]):
        raise ReviewLedgerError("review identity fields must be non-empty")
    if matcher_policy is not None:
        clean_policy = str(matcher_policy).strip()
        if not re.fullmatch(r"[a-z0-9_]{1,80}", clean_policy):
            raise ReviewLedgerError("invalid matcher_policy")
        values.append(clean_policy)
    encoded = "\0".join(str(value).strip() for value in values).encode("utf-8")
    return "TRR-" + hashlib.sha256(encoded).hexdigest()[:16].upper()


def _clean_text(value: object, max_chars: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:max_chars]


def _empty_ledger(product: str) -> dict:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "product": product,
        "updated_at": None,
        "reviews": {},
    }


def _validate_entry(key: str, entry: object) -> list[str]:
    if not isinstance(entry, dict):
        return [f"review {key} must be an object"]
    errors = []
    expected_key = None
    try:
        expected_key = review_key(
            str(entry.get("hypothesis_id") or ""),
            str(entry.get("prior_close_build") or ""),
            str(entry.get("current_close_build") or ""),
            str(entry.get("bug_code") or ""),
            entry.get("matcher_policy"),
        )
    except ReviewLedgerError as exc:
        errors.append(f"review {key}: {exc}")
    if expected_key and expected_key != key:
        errors.append(f"review {key} identity hash mismatch")
    if entry.get("decision") not in REVIEW_DECISIONS:
        errors.append(f"review {key} has invalid decision")
    rejection_reason = entry.get("rejection_reason")
    if rejection_reason is not None and rejection_reason not in REJECTION_REASONS:
        errors.append(f"review {key} has invalid rejection_reason")
    if rejection_reason is not None and entry.get("decision") != "rejected":
        errors.append(f"review {key} has rejection_reason for non-rejected decision")
    if rejection_reason == "other" and not str(entry.get("note") or "").strip():
        errors.append(f"review {key} other rejection requires a note")
    if len(str(entry.get("reviewer") or "")) > MAX_REVIEWER_CHARS:
        errors.append(f"review {key} reviewer is too long")
    if len(str(entry.get("note") or "")) > MAX_NOTE_CHARS:
        errors.append(f"review {key} note is too long")
    score = entry.get("match_score")
    if score is not None and (
        isinstance(score, bool) or not isinstance(score, (int, float))
        or not math.isfinite(float(score)) or not 0 <= float(score) <= 1
    ):
        errors.append(f"review {key} has invalid match_score")
    candidate_kind = entry.get("candidate_kind")
    if (
        candidate_kind is not None
        and candidate_kind not in REVIEW_CANDIDATE_KINDS
    ):
        errors.append(f"review {key} has invalid candidate_kind")
    applied_threshold = entry.get("applied_threshold")
    if applied_threshold is not None and (
        isinstance(applied_threshold, bool)
        or not isinstance(applied_threshold, (int, float))
        or not math.isfinite(float(applied_threshold))
        or not 0 <= float(applied_threshold) <= 1
    ):
        errors.append(f"review {key} has invalid applied_threshold")
    terms = entry.get("overlap_terms")
    if terms is not None and (
        not isinstance(terms, list) or len(terms) > 30
        or any(not isinstance(term, str) or len(term) > 80 for term in terms)
    ):
        errors.append(f"review {key} has invalid overlap_terms")
    matcher_policy = entry.get("matcher_policy")
    overlap_facets = entry.get("overlap_facets")
    if matcher_policy is not None and (
        not isinstance(matcher_policy, str)
        or not re.fullmatch(r"[a-z0-9_]{1,80}", matcher_policy)
    ):
        errors.append(f"review {key} has invalid matcher_policy")
    if overlap_facets is not None:
        if (
            not isinstance(overlap_facets, dict)
            or set(overlap_facets) != {"area", "trigger", "failure_mode"}
            or any(
                not isinstance(values, list) or len(values) > 30
                or any(
                    not isinstance(term, str) or not term or len(term) > 80
                    for term in values
                )
                for values in overlap_facets.values()
            )
        ):
            errors.append(f"review {key} has invalid overlap_facets")
    if (matcher_policy is None) != (overlap_facets is None):
        errors.append(f"review {key} has incomplete matcher provenance")
    historical_support = entry.get("historical_support")
    historical_codes = entry.get("historical_analog_codes")
    historical_count = entry.get("historical_analog_count")
    historical_score = entry.get("historical_analog_max_similarity")
    if historical_support is not None and not isinstance(
        historical_support, bool
    ):
        errors.append(f"review {key} has invalid historical_support")
    if historical_codes is not None and (
        not isinstance(historical_codes, list) or len(historical_codes) > 3
        or any(
            not isinstance(code, str)
            or not re.fullmatch(r"[A-Za-z0-9._-]+", code)
            for code in historical_codes
        )
        or len(set(historical_codes)) != len(historical_codes)
    ):
        errors.append(f"review {key} has invalid historical_analog_codes")
    if historical_count is not None and (
        isinstance(historical_count, bool)
        or not isinstance(historical_count, int)
        or not 0 <= historical_count <= 3
        or (
            isinstance(historical_codes, list)
            and historical_count != len(historical_codes)
        )
    ):
        errors.append(f"review {key} has invalid historical_analog_count")
    if historical_score is not None and (
        isinstance(historical_score, bool)
        or not isinstance(historical_score, (int, float))
        or not math.isfinite(float(historical_score))
        or not 0 <= float(historical_score) <= 1
    ):
        errors.append(
            f"review {key} has invalid historical_analog_max_similarity"
        )
    if historical_support is True and (
        not historical_codes or historical_score is None
    ):
        errors.append(f"review {key} historical support is incomplete")
    if historical_support is False and (
        historical_codes or historical_count not in {None, 0}
        or historical_score is not None
    ):
        errors.append(f"review {key} has contradictory historical support")
    scope_boundary = entry.get("scope_boundary")
    disconfirming_observation = entry.get("disconfirming_observation")
    if scope_boundary is not None and (
        not isinstance(scope_boundary, str) or not scope_boundary.strip()
        or len(scope_boundary) > 320
    ):
        errors.append(f"review {key} has invalid scope_boundary")
    if disconfirming_observation is not None and (
        not isinstance(disconfirming_observation, str)
        or not disconfirming_observation.strip()
        or len(disconfirming_observation) > 320
    ):
        errors.append(f"review {key} has invalid disconfirming_observation")
    if (scope_boundary is None) != (disconfirming_observation is None):
        errors.append(f"review {key} has incomplete prediction boundaries")
    guardrail_provenance = entry.get("analysis_guardrail_provenance")
    guardrail = entry.get("analysis_guardrail")
    guardrail_reason = entry.get("analysis_guardrail_source_rejection_reason")
    analysis_model = entry.get("analysis_model")
    analysis_engine = entry.get("analysis_engine_sha256")
    origin_provenance = entry.get("prediction_origin_provenance")
    prediction_origin = entry.get("prediction_origin")
    synthesis_ids = entry.get("synthesis_source_candidate_ids")
    recovered_ids = entry.get("recovered_source_candidate_ids")
    target_alias_provenance = entry.get("target_alias_provenance")
    target_alias_pairs = entry.get("target_alias_pairs")
    if (
        guardrail_provenance is not None
        and guardrail_provenance not in ANALYSIS_GUARDRAIL_PROVENANCE_VALUES
    ):
        errors.append(f"review {key} has invalid analysis_guardrail_provenance")
    if guardrail is not None and guardrail not in ANALYSIS_GUARDRAILS:
        errors.append(f"review {key} has invalid analysis_guardrail")
    if guardrail_reason is not None and (
        guardrail_reason not in AI_GUARDRAIL_BY_REJECTION_REASON
        or AI_GUARDRAIL_BY_REJECTION_REASON[guardrail_reason] != guardrail
    ):
        errors.append(f"review {key} has invalid analysis guardrail reason")
    if guardrail is not None and guardrail_reason is None:
        errors.append(f"review {key} analysis guardrail reason is missing")
    if guardrail_provenance != "exact_policy" and any(
        value is not None for value in (guardrail, guardrail_reason)
    ):
        errors.append(f"review {key} guardrail lacks exact provenance")
    if guardrail_provenance == "exact_policy" and (
        not isinstance(analysis_model, str) or not analysis_model.strip()
        or len(analysis_model) > 120
        or not isinstance(analysis_engine, str)
        or not re.fullmatch(r"[a-f0-9]{64}", analysis_engine)
    ):
        errors.append(f"review {key} exact analysis configuration is incomplete")
    if (
        guardrail_provenance != "exact_policy"
        and origin_provenance not in EXACT_PREDICTION_ORIGIN_PROVENANCE_VALUES
        and any(
            value is not None for value in (analysis_model, analysis_engine)
        )
    ):
        errors.append(
            f"review {key} has analysis configuration without exact provenance"
        )
    if (
        origin_provenance is not None
        and origin_provenance not in PREDICTION_ORIGIN_PROVENANCE_VALUES
    ):
        errors.append(f"review {key} has invalid prediction origin provenance")
    if prediction_origin is not None and prediction_origin not in PREDICTION_ORIGINS:
        errors.append(f"review {key} has invalid prediction origin")
    if origin_provenance in EXACT_PREDICTION_ORIGIN_PROVENANCE_VALUES:
        if prediction_origin not in PREDICTION_ORIGINS - {"legacy_unknown"}:
            errors.append(f"review {key} exact lineage has invalid prediction origin")
        if not _is_candidate_id_list(synthesis_ids, require_nonempty=True):
            errors.append(f"review {key} has invalid synthesis lineage IDs")
            synthesis_ids = []
        if not _is_candidate_id_list(
            recovered_ids, require_nonempty=False
        ) or any(
            candidate_id not in synthesis_ids for candidate_id in recovered_ids
        ):
            errors.append(f"review {key} has invalid recovered lineage IDs")
            recovered_ids = []
        expected_recovered = prediction_origin in {
            "recovered_draft_only", "mixed_initial_and_recovered",
        }
        if bool(recovered_ids) != expected_recovered:
            errors.append(f"review {key} prediction origin contradicts recovered IDs")
        if (
            prediction_origin == "recovered_draft_only"
            and set(recovered_ids) != set(synthesis_ids)
        ):
            errors.append(f"review {key} recovered-only lineage is incomplete")
        if (
            prediction_origin == "mixed_initial_and_recovered"
            and set(recovered_ids) == set(synthesis_ids)
        ):
            errors.append(f"review {key} mixed lineage lacks an initial-pass source")
        if (
            not isinstance(analysis_model, str) or not analysis_model.strip()
            or not isinstance(analysis_engine, str)
            or not re.fullmatch(r"[a-f0-9]{64}", analysis_engine)
        ):
            errors.append(f"review {key} exact lineage configuration is incomplete")
    elif any(value is not None for value in (
        prediction_origin, synthesis_ids, recovered_ids,
    )):
        if not (
            origin_provenance == "unavailable_legacy"
            and prediction_origin == "legacy_unknown"
            and synthesis_ids in (None, []) and recovered_ids in (None, [])
        ):
            errors.append(f"review {key} lineage fields lack exact provenance")
    if (
        target_alias_provenance is not None
        and target_alias_provenance not in TARGET_ALIAS_PROVENANCE_VALUES
    ):
        errors.append(f"review {key} has invalid target alias provenance")
    try:
        normalized_alias_pairs = _normalized_target_alias_pairs(target_alias_pairs)
    except ReviewLedgerError:
        normalized_alias_pairs = []
        errors.append(f"review {key} has invalid target alias pairs")
    if target_alias_provenance == "schema27_exact_basis":
        if target_alias_pairs != normalized_alias_pairs:
            errors.append(f"review {key} target alias pairs are not normalized")
    elif target_alias_pairs not in (None, []):
        errors.append(f"review {key} target alias pairs lack exact provenance")
    if (
        rejection_reason == "reviewed_target_alias_misleading"
        and (
            target_alias_provenance != "schema27_exact_basis"
            or len(normalized_alias_pairs) != 1
            or candidate_kind == "threshold_audit"
        )
    ):
        errors.append(
            f"review {key} alias rejection requires one exact alias on a feature match"
        )
    history = entry.get("history")
    if history is not None:
        if not isinstance(history, list):
            errors.append(f"review {key} history must be an array")
        else:
            for index, event in enumerate(history, start=1):
                if not isinstance(event, dict):
                    errors.append(f"review {key} history event {index} must be an object")
                    continue
                if event.get("decision") not in REVIEW_DECISIONS:
                    errors.append(f"review {key} history event {index} has invalid decision")
                event_reason = event.get("rejection_reason")
                if event_reason is not None and event_reason not in REJECTION_REASONS:
                    errors.append(
                        f"review {key} history event {index} has invalid rejection_reason"
                    )
                if event_reason is not None and event.get("decision") != "rejected":
                    errors.append(
                        f"review {key} history event {index} has rejection_reason "
                        "for non-rejected decision"
                    )
                if event_reason == "other" and not str(
                    event.get("note") or ""
                ).strip():
                    errors.append(
                        f"review {key} history event {index} other rejection "
                        "requires a note"
                    )
                if len(str(event.get("reviewer") or "")) > MAX_REVIEWER_CHARS:
                    errors.append(f"review {key} history event {index} reviewer is too long")
                if len(str(event.get("note") or "")) > MAX_NOTE_CHARS:
                    errors.append(f"review {key} history event {index} note is too long")
                if (
                    event.get("candidate_kind") is not None
                    and event.get("candidate_kind") not in REVIEW_CANDIDATE_KINDS
                ):
                    errors.append(
                        f"review {key} history event {index} has invalid candidate_kind"
                    )
                event_threshold = event.get("applied_threshold")
                if event_threshold is not None and (
                    isinstance(event_threshold, bool)
                    or not isinstance(event_threshold, (int, float))
                    or not math.isfinite(float(event_threshold))
                    or not 0 <= float(event_threshold) <= 1
                ):
                    errors.append(
                        f"review {key} history event {index} has invalid applied_threshold"
                    )
    return errors


def load_review_ledger(path: Path, product: str) -> dict:
    if not path.exists():
        return _empty_ledger(product)
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewLedgerError(f"review ledger is unreadable: {exc}") from exc
    if not isinstance(ledger, dict):
        raise ReviewLedgerError("review ledger must be a JSON object")
    if ledger.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ReviewLedgerError("unsupported review ledger schema")
    if str(ledger.get("product") or "") != product:
        raise ReviewLedgerError("review ledger product mismatch")
    reviews = ledger.get("reviews")
    if not isinstance(reviews, dict):
        raise ReviewLedgerError("review ledger reviews must be an object")
    errors = [
        error for key, entry in reviews.items()
        for error in _validate_entry(str(key), entry)
    ]
    if errors:
        raise ReviewLedgerError("; ".join(errors))
    return ledger


@contextmanager
def _ledger_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_ledger_atomic(path: Path, ledger: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(
            json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def upsert_review(path: Path, *, product: str, hypothesis_id: str,
                  prior_close_build: str, current_close_build: str,
                  bug_code: str, target_category: str, predicted_area: str,
                  decision: str, reviewer: str = "", note: str = "",
                  match_score: float | None = None,
                  overlap_terms: list[str] | None = None,
                  matcher_policy: str | None = None,
                  overlap_facets: dict[str, list[str]] | None = None,
                  trigger: str | None = None,
                  expected_failure_mode: str | None = None,
                  candidate_kind: str | None = None,
                  applied_threshold: float | None = None,
                  historical_analog_codes: list[str] | None = None,
                  historical_analog_max_similarity: float | None = None,
                  scope_boundary: str | None = None,
                  disconfirming_observation: str | None = None,
                  rejection_reason: str | None = None,
                  analysis_guardrail_provenance: str | None = None,
                  analysis_guardrail: str | None = None,
                  analysis_guardrail_source_rejection_reason: str | None = None,
                  analysis_model: str | None = None,
                  analysis_engine_sha256: str | None = None,
                  prediction_origin_provenance: str | None = None,
                  prediction_origin: str | None = None,
                  synthesis_source_candidate_ids: list[str] | None = None,
                  recovered_source_candidate_ids: list[str] | None = None,
                  target_alias_provenance: str | None = None,
                  target_alias_pairs: list[dict] | None = None) -> dict:
    if decision not in REVIEW_DECISIONS:
        raise ReviewLedgerError("decision must be confirmed, rejected, or unclear")
    reviewer = _clean_text(reviewer, MAX_REVIEWER_CHARS + 1)
    note = _clean_text(note, MAX_NOTE_CHARS + 1)
    if len(reviewer) > MAX_REVIEWER_CHARS:
        raise ReviewLedgerError("reviewer is too long")
    if len(note) > MAX_NOTE_CHARS:
        raise ReviewLedgerError("note is too long")
    if decision == "rejected" and rejection_reason not in REJECTION_REASONS:
        raise ReviewLedgerError(
            "rejected decisions require a supported rejection_reason"
        )
    if decision != "rejected" and rejection_reason is not None:
        raise ReviewLedgerError(
            "rejection_reason is only valid for rejected decisions"
        )
    if rejection_reason == "other" and not note:
        raise ReviewLedgerError("other rejection_reason requires a note")
    if (
        analysis_guardrail_provenance is not None
        and analysis_guardrail_provenance
        not in ANALYSIS_GUARDRAIL_PROVENANCE_VALUES
    ):
        raise ReviewLedgerError("invalid analysis guardrail provenance")
    if analysis_guardrail is not None and analysis_guardrail not in ANALYSIS_GUARDRAILS:
        raise ReviewLedgerError("invalid analysis guardrail")
    if analysis_guardrail_source_rejection_reason is not None and (
        analysis_guardrail_source_rejection_reason
        not in AI_GUARDRAIL_BY_REJECTION_REASON
        or AI_GUARDRAIL_BY_REJECTION_REASON[
            analysis_guardrail_source_rejection_reason
        ] != analysis_guardrail
    ):
        raise ReviewLedgerError("analysis guardrail reason does not match guardrail")
    if analysis_guardrail is not None and (
        analysis_guardrail_source_rejection_reason is None
    ):
        raise ReviewLedgerError("analysis guardrail reason is required")
    if analysis_guardrail_provenance != "exact_policy" and any(
        value is not None for value in (
            analysis_guardrail, analysis_guardrail_source_rejection_reason,
        )
    ):
        raise ReviewLedgerError("analysis guardrail requires exact provenance")
    clean_analysis_model = (
        _clean_text(analysis_model, 121) if analysis_model is not None else None
    )
    clean_analysis_engine = (
        str(analysis_engine_sha256).strip().lower()
        if analysis_engine_sha256 is not None else None
    )
    if analysis_guardrail_provenance == "exact_policy" and (
        not clean_analysis_model or len(clean_analysis_model) > 120
        or not clean_analysis_engine
        or not re.fullmatch(r"[a-f0-9]{64}", clean_analysis_engine)
    ):
        raise ReviewLedgerError(
            "exact guardrail provenance requires model and engine SHA-256"
        )
    if (
        analysis_guardrail_provenance != "exact_policy"
        and prediction_origin_provenance
        not in EXACT_PREDICTION_ORIGIN_PROVENANCE_VALUES
        and any(
            value is not None for value in (analysis_model, analysis_engine_sha256)
        )
    ):
        raise ReviewLedgerError(
            "analysis configuration requires exact guardrail or lineage provenance"
        )
    if (
        prediction_origin_provenance is not None
        and prediction_origin_provenance not in PREDICTION_ORIGIN_PROVENANCE_VALUES
    ):
        raise ReviewLedgerError("invalid prediction origin provenance")
    if prediction_origin is not None and prediction_origin not in PREDICTION_ORIGINS:
        raise ReviewLedgerError("invalid prediction origin")
    clean_synthesis_ids = None
    clean_recovered_ids = None
    if prediction_origin_provenance in EXACT_PREDICTION_ORIGIN_PROVENANCE_VALUES:
        clean_synthesis_ids = list(dict.fromkeys(
            str(candidate_id) for candidate_id in (
                synthesis_source_candidate_ids or []
            )
        ))
        clean_recovered_ids = list(dict.fromkeys(
            str(candidate_id) for candidate_id in (
                recovered_source_candidate_ids or []
            )
        ))
        if prediction_origin not in PREDICTION_ORIGINS - {"legacy_unknown"}:
            raise ReviewLedgerError("exact lineage requires a supported prediction origin")
        if not clean_synthesis_ids or any(
            not re.fullmatch(r"TRC-[A-F0-9]{12}", candidate_id)
            for candidate_id in clean_synthesis_ids
        ):
            raise ReviewLedgerError("invalid synthesis source candidate IDs")
        if any(
            candidate_id not in clean_synthesis_ids
            for candidate_id in clean_recovered_ids
        ):
            raise ReviewLedgerError("recovered candidate IDs must be synthesis sources")
        expected_recovered = prediction_origin in {
            "recovered_draft_only", "mixed_initial_and_recovered",
        }
        if bool(clean_recovered_ids) != expected_recovered:
            raise ReviewLedgerError("prediction origin contradicts recovered candidate IDs")
        if (
            prediction_origin == "recovered_draft_only"
            and set(clean_recovered_ids) != set(clean_synthesis_ids)
        ):
            raise ReviewLedgerError("recovered-only prediction has incomplete lineage")
        if (
            prediction_origin == "mixed_initial_and_recovered"
            and set(clean_recovered_ids) == set(clean_synthesis_ids)
        ):
            raise ReviewLedgerError("mixed prediction lacks an initial-pass source")
        if (
            not clean_analysis_model or len(clean_analysis_model) > 120
            or not clean_analysis_engine
            or not re.fullmatch(r"[a-f0-9]{64}", clean_analysis_engine)
        ):
            raise ReviewLedgerError(
                "exact lineage provenance requires model and engine SHA-256"
            )
    elif prediction_origin_provenance == "unavailable_legacy":
        if prediction_origin != "legacy_unknown" or any((
            synthesis_source_candidate_ids, recovered_source_candidate_ids,
        )):
            raise ReviewLedgerError("legacy lineage provenance must remain unknown")
        clean_synthesis_ids = []
        clean_recovered_ids = []
    elif any(value is not None for value in (
        prediction_origin, synthesis_source_candidate_ids,
        recovered_source_candidate_ids,
    )):
        raise ReviewLedgerError("prediction lineage fields require provenance")
    if (
        target_alias_provenance is not None
        and target_alias_provenance not in TARGET_ALIAS_PROVENANCE_VALUES
    ):
        raise ReviewLedgerError("invalid target alias provenance")
    clean_target_alias_pairs = _normalized_target_alias_pairs(target_alias_pairs)
    if target_alias_provenance == "schema27_exact_basis":
        if target_alias_pairs is None:
            raise ReviewLedgerError("exact target alias provenance requires pair array")
    elif clean_target_alias_pairs:
        raise ReviewLedgerError("target alias pairs require exact provenance")
    elif target_alias_provenance == "unavailable_legacy":
        clean_target_alias_pairs = []
    elif target_alias_pairs is not None:
        raise ReviewLedgerError("target alias pairs require provenance")
    if match_score is not None and (
        isinstance(match_score, bool) or not isinstance(match_score, (int, float))
        or not math.isfinite(float(match_score)) or not 0 <= float(match_score) <= 1
    ):
        raise ReviewLedgerError("match_score must be between 0 and 1")
    if candidate_kind is not None and candidate_kind not in REVIEW_CANDIDATE_KINDS:
        raise ReviewLedgerError(
            "candidate_kind must be feature_match or threshold_audit"
        )
    if rejection_reason == "reviewed_target_alias_misleading" and (
        target_alias_provenance != "schema27_exact_basis"
        or len(clean_target_alias_pairs) != 1
        or candidate_kind == "threshold_audit"
    ):
        raise ReviewLedgerError(
            "alias rejection requires one exact alias on a feature match"
        )
    if applied_threshold is not None and (
        isinstance(applied_threshold, bool)
        or not isinstance(applied_threshold, (int, float))
        or not math.isfinite(float(applied_threshold))
        or not 0 <= float(applied_threshold) <= 1
    ):
        raise ReviewLedgerError("applied_threshold must be between 0 and 1")
    clean_terms = None if overlap_terms is None else list(dict.fromkeys(
        _clean_text(term, 80) for term in overlap_terms if _clean_text(term, 80)
    ))[:30]
    clean_matcher_policy = (
        str(matcher_policy).strip() if matcher_policy is not None else None
    )
    if clean_matcher_policy is not None and not re.fullmatch(
        r"[a-z0-9_]{1,80}", clean_matcher_policy
    ):
        raise ReviewLedgerError("invalid matcher_policy")
    clean_overlap_facets = None
    if overlap_facets is not None:
        if not isinstance(overlap_facets, dict) or set(overlap_facets) != {
            "area", "trigger", "failure_mode",
        }:
            raise ReviewLedgerError("invalid overlap_facets")
        clean_overlap_facets = {
            facet: list(dict.fromkeys(
                _clean_text(term, 80)
                for term in overlap_facets[facet]
                if _clean_text(term, 80)
            ))[:30]
            if isinstance(overlap_facets[facet], list) else None
            for facet in ("area", "trigger", "failure_mode")
        }
        if any(values is None for values in clean_overlap_facets.values()):
            raise ReviewLedgerError("invalid overlap_facets")
    if (clean_matcher_policy is None) != (clean_overlap_facets is None):
        raise ReviewLedgerError(
            "matcher_policy and overlap_facets must be supplied together"
        )
    clean_historical_codes = None
    if historical_analog_codes is not None:
        clean_historical_codes = list(dict.fromkeys(
            _clean_text(code, 80) for code in historical_analog_codes
            if _clean_text(code, 80)
        ))[:3]
        if any(
            not re.fullmatch(r"[A-Za-z0-9._-]+", code)
            for code in clean_historical_codes
        ):
            raise ReviewLedgerError("historical analog BugCode is invalid")
        if clean_historical_codes and historical_analog_max_similarity is None:
            raise ReviewLedgerError(
                "historical analog similarity is required when support exists"
            )
    if historical_analog_max_similarity is not None and (
        isinstance(historical_analog_max_similarity, bool)
        or not isinstance(historical_analog_max_similarity, (int, float))
        or not math.isfinite(float(historical_analog_max_similarity))
        or not 0 <= float(historical_analog_max_similarity) <= 1
    ):
        raise ReviewLedgerError(
            "historical analog similarity must be between 0 and 1"
        )
    if (scope_boundary is None) != (disconfirming_observation is None):
        raise ReviewLedgerError(
            "scope boundary and disconfirming observation must be supplied together"
        )
    clean_scope_boundary = (
        _clean_text(scope_boundary, 321) if scope_boundary is not None else None
    )
    clean_disconfirming_observation = (
        _clean_text(disconfirming_observation, 321)
        if disconfirming_observation is not None else None
    )
    if clean_scope_boundary is not None and (
        not clean_scope_boundary or not clean_disconfirming_observation
        or len(clean_scope_boundary) > 320
        or len(clean_disconfirming_observation) > 320
    ):
        raise ReviewLedgerError("prediction boundaries are empty or too long")
    key = review_key(
        hypothesis_id, prior_close_build, current_close_build, bug_code,
        clean_matcher_policy,
    )
    now = datetime.now(timezone.utc).isoformat()
    with _ledger_lock(path):
        ledger = load_review_ledger(path, product)
        existing = ledger["reviews"].get(key) or {}
        if clean_matcher_policy is not None and "matcher_policy" in existing:
            if (
                existing.get("matcher_policy") != clean_matcher_policy
                or existing.get("overlap_facets") != clean_overlap_facets
            ):
                raise ReviewLedgerError(
                    "matcher provenance cannot change for an existing review"
                )
        effective_historical_support = (
            bool(clean_historical_codes)
            if clean_historical_codes is not None
            else existing.get("historical_support")
        )
        effective_boundary_available = (
            clean_scope_boundary is not None
            or bool(existing.get("scope_boundary"))
        )
        if (
            rejection_reason == "historical_corroboration_misleading"
            and effective_historical_support is not True
        ):
            raise ReviewLedgerError(
                "historical corroboration rejection requires cited analog support"
            )
        if rejection_reason in {
            "outside_scope", "disconfirming_observation_seen",
        } and not effective_boundary_available:
            raise ReviewLedgerError(
                "boundary rejection requires validated prediction boundaries"
            )
        if clean_historical_codes is not None and "historical_support" in existing:
            existing_codes = existing.get("historical_analog_codes") or []
            existing_score = existing.get("historical_analog_max_similarity")
            incoming_score = (
                round(float(historical_analog_max_similarity), 6)
                if clean_historical_codes else None
            )
            if (
                clean_historical_codes != existing_codes
                or incoming_score != existing_score
            ):
                raise ReviewLedgerError(
                    "historical support provenance cannot change for an existing review"
                )
        if clean_scope_boundary is not None and "scope_boundary" in existing:
            if (
                clean_scope_boundary != existing.get("scope_boundary")
                or clean_disconfirming_observation
                != existing.get("disconfirming_observation")
            ):
                raise ReviewLedgerError(
                    "prediction boundaries cannot change for an existing review"
                )
        incoming_guardrail_provenance = {
            "analysis_guardrail_provenance": analysis_guardrail_provenance,
            "analysis_guardrail": analysis_guardrail,
            "analysis_guardrail_source_rejection_reason": (
                analysis_guardrail_source_rejection_reason
            ),
            "analysis_model": clean_analysis_model,
            "analysis_engine_sha256": clean_analysis_engine,
        }
        incoming_lineage_provenance = {
            "prediction_origin_provenance": prediction_origin_provenance,
            "prediction_origin": prediction_origin,
            "synthesis_source_candidate_ids": clean_synthesis_ids,
            "recovered_source_candidate_ids": clean_recovered_ids,
            "analysis_model": clean_analysis_model,
            "analysis_engine_sha256": clean_analysis_engine,
        }
        incoming_target_alias_provenance = {
            "target_alias_provenance": target_alias_provenance,
            "target_alias_pairs": clean_target_alias_pairs,
        }
        if (
            analysis_guardrail_provenance is not None
            and "analysis_guardrail_provenance" in existing
            and any(
                existing.get(field) != value
                for field, value in incoming_guardrail_provenance.items()
            )
        ):
            raise ReviewLedgerError(
                "analysis guardrail provenance cannot change for an existing review"
            )
        if (
            prediction_origin_provenance is not None
            and "prediction_origin_provenance" in existing
            and any(
                existing.get(field) != value
                for field, value in incoming_lineage_provenance.items()
            )
        ):
            raise ReviewLedgerError(
                "prediction lineage provenance cannot change for an existing review"
            )
        if (
            target_alias_provenance is not None
            and "target_alias_provenance" in existing
            and any(
                existing.get(field) != value
                for field, value in incoming_target_alias_provenance.items()
            )
        ):
            raise ReviewLedgerError(
                "target alias provenance cannot change for an existing review"
            )
        history = list(existing.get("history") or [])
        history.append({
            "decision": decision,
            "rejection_reason": (
                rejection_reason if decision == "rejected" else None
            ),
            "reviewer": reviewer,
            "note": note,
            "reviewed_at": now,
            "match_score": (
                round(float(match_score), 6) if match_score is not None
                else existing.get("match_score")
            ),
            "candidate_kind": (
                candidate_kind or existing.get("candidate_kind")
                or "feature_match"
            ),
            "applied_threshold": (
                round(float(applied_threshold), 3)
                if applied_threshold is not None
                else existing.get("applied_threshold")
            ),
            "matcher_policy": (
                clean_matcher_policy or existing.get("matcher_policy")
            ),
        })
        entry = {
            "review_key": key,
            "hypothesis_id": hypothesis_id,
            "prior_close_build": str(prior_close_build),
            "current_close_build": str(current_close_build),
            "bug_code": str(bug_code),
            "target_category": _clean_text(target_category, 120),
            "predicted_area": _clean_text(predicted_area, 160),
            "trigger": (
                _clean_text(trigger, 320) if trigger is not None
                else existing.get("trigger") or ""
            ),
            "expected_failure_mode": (
                _clean_text(expected_failure_mode, 320)
                if expected_failure_mode is not None
                else existing.get("expected_failure_mode") or ""
            ),
            "match_score": (
                round(float(match_score), 6) if match_score is not None
                else existing.get("match_score")
            ),
            "overlap_terms": (
                clean_terms if clean_terms is not None
                else existing.get("overlap_terms") or []
            ),
            "candidate_kind": (
                candidate_kind or existing.get("candidate_kind")
                or "feature_match"
            ),
            "applied_threshold": (
                round(float(applied_threshold), 3)
                if applied_threshold is not None
                else existing.get("applied_threshold")
            ),
            "decision": decision,
            "rejection_reason": (
                rejection_reason if decision == "rejected" else None
            ),
            "reviewer": reviewer,
            "note": note,
            "created_at": existing.get("created_at") or now,
            "reviewed_at": now,
            "history": history,
        }
        if clean_matcher_policy is not None:
            entry.update({
                "matcher_policy": clean_matcher_policy,
                "overlap_facets": clean_overlap_facets,
            })
        elif "matcher_policy" in existing:
            entry.update({
                "matcher_policy": existing.get("matcher_policy"),
                "overlap_facets": existing.get("overlap_facets"),
            })
        if clean_historical_codes is not None:
            entry.update({
                "historical_support": bool(clean_historical_codes),
                "historical_analog_codes": clean_historical_codes,
                "historical_analog_count": len(clean_historical_codes),
                "historical_analog_max_similarity": (
                    round(float(historical_analog_max_similarity), 6)
                    if clean_historical_codes else None
                ),
            })
        elif "historical_support" in existing:
            entry.update({
                "historical_support": existing.get("historical_support"),
                "historical_analog_codes": (
                    existing.get("historical_analog_codes") or []
                ),
                "historical_analog_count": int(
                    existing.get("historical_analog_count") or 0
                ),
                "historical_analog_max_similarity": existing.get(
                    "historical_analog_max_similarity"
                ),
            })
        if clean_scope_boundary is not None:
            entry.update({
                "scope_boundary": clean_scope_boundary,
                "disconfirming_observation": clean_disconfirming_observation,
            })
        elif "scope_boundary" in existing:
            entry.update({
                "scope_boundary": existing.get("scope_boundary"),
                "disconfirming_observation": existing.get(
                    "disconfirming_observation"
                ),
            })
        if analysis_guardrail_provenance is not None:
            entry.update(incoming_guardrail_provenance)
        elif "analysis_guardrail_provenance" in existing:
            entry.update({
                field: existing.get(field)
                for field in incoming_guardrail_provenance
            })
        if prediction_origin_provenance is not None:
            entry.update(incoming_lineage_provenance)
        elif "prediction_origin_provenance" in existing:
            entry.update({
                field: existing.get(field)
                for field in incoming_lineage_provenance
            })
        if target_alias_provenance is not None:
            entry.update(incoming_target_alias_provenance)
        elif "target_alias_provenance" in existing:
            entry.update({
                field: existing.get(field)
                for field in incoming_target_alias_provenance
            })
        errors = _validate_entry(key, entry)
        if errors:
            raise ReviewLedgerError("; ".join(errors))
        ledger["reviews"][key] = entry
        ledger["updated_at"] = now
        _write_ledger_atomic(path, ledger)
    return entry


def review_summary(ledger: dict, review_keys: set[str] | None = None) -> dict:
    entries = [
        entry for key, entry in (ledger.get("reviews") or {}).items()
        if review_keys is None or key in review_keys
    ]
    counts = {
        decision: sum(entry.get("decision") == decision for entry in entries)
        for decision in sorted(REVIEW_DECISIONS)
    }
    decisive = counts["confirmed"] + counts["rejected"]
    audit_entries = [
        entry for entry in entries
        if entry.get("candidate_kind") == "threshold_audit"
    ]
    prediction_entries = [
        entry for entry in entries
        if entry.get("candidate_kind") != "threshold_audit"
    ]
    prediction_counts = {
        decision: sum(
            entry.get("decision") == decision for entry in prediction_entries
        )
        for decision in sorted(REVIEW_DECISIONS)
    }
    prediction_decisive = (
        prediction_counts["confirmed"] + prediction_counts["rejected"]
    )
    prediction_rejection_reasons = Counter(
        str(entry.get("rejection_reason"))
        for entry in prediction_entries
        if entry.get("decision") == "rejected"
        and entry.get("rejection_reason") in REJECTION_REASONS
    )
    audit_rejection_reasons = Counter(
        str(entry.get("rejection_reason"))
        for entry in audit_entries
        if entry.get("decision") == "rejected"
        and entry.get("rejection_reason") in REJECTION_REASONS
    )
    rejected_without_reason = sum(
        entry.get("decision") == "rejected"
        and entry.get("rejection_reason") not in REJECTION_REASONS
        for entry in entries
    )
    summary = {
        "reviewed_count": len(entries),
        **{f"{decision}_count": count for decision, count in counts.items()},
        "decisive_count": decisive,
        "reviewed_confirmation_rate": (
            prediction_counts["confirmed"] / prediction_decisive
            if prediction_decisive else None
        ),
        "prediction_reviewed_count": len(prediction_entries),
        "prediction_confirmed_count": prediction_counts["confirmed"],
        "prediction_rejected_count": prediction_counts["rejected"],
        "prediction_unclear_count": prediction_counts["unclear"],
        "prediction_decisive_count": prediction_decisive,
        "threshold_audit_reviewed_count": len(audit_entries),
        "threshold_audit_confirmed_count": sum(
            entry.get("decision") == "confirmed" for entry in audit_entries
        ),
        "prediction_rejection_reason_counts": dict(sorted(
            prediction_rejection_reasons.items()
        )),
        "threshold_audit_rejection_reason_counts": dict(sorted(
            audit_rejection_reasons.items()
        )),
        "rejected_without_reason_count": rejected_without_reason,
    }
    if any(isinstance(entry.get("historical_support"), bool) for entry in entries):
        summary["historical_support_comparison"] = (
            historical_support_comparison(ledger, review_keys)
        )
    return summary


def _wilson_lower_bound(confirmed: float, total: float,
                        z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    rate = confirmed / total
    denominator = 1 + (z * z / total)
    centre = rate + (z * z / (2 * total))
    margin = z * math.sqrt(
        (rate * (1 - rate) / total) + (z * z / (4 * total * total))
    )
    return max(0.0, (centre - margin) / denominator)


def _wilson_interval(confirmed: float, total: float,
                     z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    rate = confirmed / total
    denominator = 1 + (z * z / total)
    centre = rate + (z * z / (2 * total))
    margin = z * math.sqrt(
        (rate * (1 - rate) / total) + (z * z / (4 * total * total))
    )
    return (
        max(0.0, (centre - margin) / denominator),
        min(1.0, (centre + margin) / denominator),
    )


def _historical_support_group(entries: list[dict]) -> dict:
    confirmed_raw = sum(entry["decision"] == "confirmed" for entry in entries)
    rejected_raw = sum(entry["decision"] == "rejected" for entry in entries)
    confirmed = sum(
        float(entry.get("_calibration_weight", 1.0))
        for entry in entries if entry["decision"] == "confirmed"
    )
    rejected = sum(
        float(entry.get("_calibration_weight", 1.0))
        for entry in entries if entry["decision"] == "rejected"
    )
    effective = confirmed + rejected
    lower, upper = _wilson_interval(confirmed, effective)
    return {
        "decisive_review_count": len(entries),
        "effective_decisive_review_count": round(effective, 3),
        "confirmed_review_count": confirmed_raw,
        "rejected_review_count": rejected_raw,
        "confirmation_rate": confirmed / effective if effective else None,
        "wilson_lower_bound": lower,
        "wilson_upper_bound": upper,
        "reviewed_cohort_count": len({
            entry["_cohort_id"] for entry in entries
        }),
        "reviewed_hypothesis_count": len({
            entry["_hypothesis_id"] for entry in entries
        }),
        "has_unknown_identity": any(
            entry["_cohort_id"] == "unknown_cohort"
            or entry["_hypothesis_id"] == "unknown_hypothesis"
            for entry in entries
        ),
    }


def historical_support_comparison(
    ledger: dict, review_keys: set[str] | None = None, *,
    min_decisive_per_group: int = 8,
    min_effective_per_group: float = 8.0,
    min_cohorts_per_group: int = 3,
    min_hypotheses_per_group: int = 3,
    max_per_cohort: float = MAX_EFFECTIVE_REVIEWS_PER_COHORT,
    max_per_hypothesis_cohort: float = (
        MAX_EFFECTIVE_REVIEWS_PER_HYPOTHESIS_COHORT
    ),
) -> dict:
    """Compare reviewed predictions with/without analog support observationally."""
    reviews = ledger.get("reviews") or {}
    relevant = [
        (str(key), entry) for key, entry in reviews.items()
        if (review_keys is None or key in review_keys)
        and isinstance(entry, dict)
        and entry.get("candidate_kind") != "threshold_audit"
        and entry.get("decision") in {"confirmed", "rejected"}
    ]
    known = [
        pair for pair in relevant
        if isinstance(pair[1].get("historical_support"), bool)
    ]
    weighted = _correlation_weighted_entries(
        known,
        max_per_cohort=max_per_cohort,
        max_per_hypothesis_cohort=max_per_hypothesis_cohort,
    )
    supported_entries = [
        entry for entry in weighted if entry["historical_support"] is True
    ]
    unsupported_entries = [
        entry for entry in weighted if entry["historical_support"] is False
    ]
    supported = _historical_support_group(supported_entries)
    unsupported = _historical_support_group(unsupported_entries)
    result = {
        "schema_version": 1,
        "status": "insufficient_reviews",
        "observational_only": True,
        "eligible_for_policy_change": False,
        "unknown_provenance_decisive_review_count": len(relevant) - len(known),
        "minimum_decisive_per_group": min_decisive_per_group,
        "minimum_effective_per_group": min_effective_per_group,
        "minimum_cohorts_per_group": min_cohorts_per_group,
        "minimum_hypotheses_per_group": min_hypotheses_per_group,
        "supported": supported,
        "unsupported": unsupported,
        "confirmation_rate_difference": None,
        "difference_confidence_interval": None,
    }
    groups = (supported, unsupported)
    if any(
        group["decisive_review_count"] < min_decisive_per_group
        or group["effective_decisive_review_count"] < min_effective_per_group
        for group in groups
    ):
        return result
    if any(
        group["reviewed_cohort_count"] < min_cohorts_per_group
        or group["reviewed_hypothesis_count"] < min_hypotheses_per_group
        or group["has_unknown_identity"]
        for group in groups
    ):
        result["status"] = "insufficient_diversity"
        return result
    supported_rate = float(supported["confirmation_rate"])
    unsupported_rate = float(unsupported["confirmation_rate"])
    difference = supported_rate - unsupported_rate
    difference_interval = (
        supported["wilson_lower_bound"] - unsupported["wilson_upper_bound"],
        supported["wilson_upper_bound"] - unsupported["wilson_lower_bound"],
    )
    result["confirmation_rate_difference"] = difference
    result["difference_confidence_interval"] = list(difference_interval)
    if difference_interval[0] > 0:
        result["status"] = "supported_higher"
    elif difference_interval[1] < 0:
        result["status"] = "supported_lower"
    else:
        result["status"] = "no_clear_difference"
    return result


def _threshold_evidence(entries: list[dict], threshold: float) -> dict:
    selected = [entry for entry in entries if float(entry["match_score"]) >= threshold]
    confirmed = sum(
        float(entry.get("_calibration_weight", 1.0))
        for entry in selected if entry["decision"] == "confirmed"
    )
    rejected = sum(
        float(entry.get("_calibration_weight", 1.0))
        for entry in selected if entry["decision"] == "rejected"
    )
    total = confirmed + rejected
    return {
        "selected_review_count": len(selected),
        "selected_count": round(total, 3),
        "confirmed_count": round(confirmed, 3),
        "rejected_count": round(rejected, 3),
        "confirmation_rate": confirmed / total if total else None,
        "wilson_lower_bound": _wilson_lower_bound(confirmed, total),
    }


def _correlation_weighted_entries(
    entries: list[tuple[str, dict]], *,
    max_per_cohort: float,
    max_per_hypothesis_cohort: float,
) -> list[dict]:
    """Cap correlated evidence without choosing arbitrary reviews to discard."""
    weighted = []
    for key, source in entries:
        entry = dict(source)
        prior_build = str(entry.get("prior_close_build") or "").strip()
        current_build = str(entry.get("current_close_build") or "").strip()
        entry["_cohort_id"] = (
            f"{prior_build}\0{current_build}"
            if prior_build and current_build else "unknown_cohort"
        )
        entry["_hypothesis_id"] = (
            str(entry.get("hypothesis_id") or "").strip()
            or "unknown_hypothesis"
        )
        entry["_review_key"] = key
        entry["_calibration_weight"] = 1.0
        weighted.append(entry)

    hypothesis_cohorts: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in weighted:
        hypothesis_cohorts[
            (entry["_cohort_id"], entry["_hypothesis_id"])
        ].append(entry)
    for grouped in hypothesis_cohorts.values():
        factor = min(1.0, max_per_hypothesis_cohort / len(grouped))
        for entry in grouped:
            entry["_calibration_weight"] *= factor

    cohorts: defaultdict[str, list[dict]] = defaultdict(list)
    for entry in weighted:
        cohorts[entry["_cohort_id"]].append(entry)
    for grouped in cohorts.values():
        current_weight = sum(entry["_calibration_weight"] for entry in grouped)
        factor = min(1.0, max_per_cohort / current_weight)
        for entry in grouped:
            entry["_calibration_weight"] *= factor
    return weighted


def build_analysis_guardrail_policy(
    ledger: dict, *,
    min_decisive: int = 12,
    min_rejections: int = 6,
    min_effective_rejections: float = 6.0,
    min_rejection_rate: float = 0.30,
    min_reason_effective: float = 4.0,
    min_reason_share: float = 0.50,
    min_cohorts: int = 3,
    min_hypotheses: int = 3,
    max_per_cohort: float = MAX_EFFECTIVE_REVIEWS_PER_COHORT,
    max_per_hypothesis_cohort: float = (
        MAX_EFFECTIVE_REVIEWS_PER_HYPOTHESIS_COHORT
    ),
) -> dict:
    """Derive controlled category guardrails from diverse structured outcomes."""
    eligible_pairs = [
        (str(key), entry)
        for key, entry in (ledger.get("reviews") or {}).items()
        if isinstance(entry, dict)
        and entry.get("candidate_kind") != "threshold_audit"
        and str(entry.get("target_category") or "").strip()
        and (
            entry.get("decision") == "confirmed"
            or (
                entry.get("decision") == "rejected"
                and entry.get("rejection_reason") in REJECTION_REASONS
            )
        )
    ]
    weighted = _correlation_weighted_entries(
        eligible_pairs,
        max_per_cohort=max_per_cohort,
        max_per_hypothesis_cohort=max_per_hypothesis_cohort,
    )
    categories = sorted({
        str(entry.get("target_category") or "").strip()
        for entry in weighted
    }, key=casefold_order)
    category_results = {}
    active_guardrails = {}
    for category in categories:
        category_entries = [
            entry for entry in weighted
            if str(entry.get("target_category") or "").strip() == category
        ]
        rejected = [
            entry for entry in category_entries
            if entry.get("decision") == "rejected"
        ]
        effective_decisive = sum(
            float(entry["_calibration_weight"])
            for entry in category_entries
        )
        effective_rejected = sum(
            float(entry["_calibration_weight"])
            for entry in rejected
        )
        raw_reason_counts = Counter(
            str(entry.get("rejection_reason")) for entry in rejected
        )
        effective_reason_counts: defaultdict[str, float] = defaultdict(float)
        for entry in rejected:
            effective_reason_counts[str(entry.get("rejection_reason"))] += float(
                entry["_calibration_weight"]
            )
        rejected_cohorts = {
            entry["_cohort_id"] for entry in rejected
        }
        rejected_hypotheses = {
            entry["_hypothesis_id"] for entry in rejected
        }
        result = {
            "status": "insufficient_reviews",
            "decisive_review_count": len(category_entries),
            "effective_decisive_review_count": round(effective_decisive, 3),
            "structured_rejection_count": len(rejected),
            "effective_structured_rejection_count": round(
                effective_rejected, 3
            ),
            "rejection_rate": (
                effective_rejected / effective_decisive
                if effective_decisive else None
            ),
            "rejected_cohort_count": len(rejected_cohorts),
            "rejected_hypothesis_count": len(rejected_hypotheses),
            "reason_counts": dict(sorted(raw_reason_counts.items())),
            "effective_reason_counts": {
                reason: round(count, 3)
                for reason, count in sorted(effective_reason_counts.items())
            },
            "dominant_ai_reason": None,
            "dominant_ai_reason_share": None,
            "active_guardrail": None,
        }
        if (
            len(category_entries) < min_decisive
            or effective_decisive < min_decisive
        ):
            category_results[category] = result
            continue
        if (
            len(rejected) < min_rejections
            or effective_rejected < min_effective_rejections
        ):
            result["status"] = "insufficient_rejections"
            category_results[category] = result
            continue
        if (
            len(rejected_cohorts) < min_cohorts
            or len(rejected_hypotheses) < min_hypotheses
            or "unknown_cohort" in rejected_cohorts
            or "unknown_hypothesis" in rejected_hypotheses
        ):
            result["status"] = "insufficient_diversity"
            category_results[category] = result
            continue
        rejection_rate = float(result["rejection_rate"] or 0)
        if rejection_rate < min_rejection_rate:
            result["status"] = "low_rejection_rate"
            category_results[category] = result
            continue
        ai_reason_counts = {
            reason: effective_reason_counts.get(reason, 0.0)
            for reason in AI_GUARDRAIL_BY_REJECTION_REASON
        }
        dominant_reason, dominant_count = max(
            ai_reason_counts.items(), key=lambda item: (item[1], item[0])
        )
        dominant_share = (
            dominant_count / effective_rejected if effective_rejected else 0
        )
        result["dominant_ai_reason"] = dominant_reason
        result["dominant_ai_reason_share"] = dominant_share
        if (
            dominant_count < min_reason_effective
            or dominant_share < min_reason_share
        ):
            result["status"] = "no_dominant_ai_failure"
            category_results[category] = result
            continue
        guardrail = AI_GUARDRAIL_BY_REJECTION_REASON[dominant_reason]
        result["status"] = "active"
        result["active_guardrail"] = guardrail
        active_guardrails[category] = {
            "guardrail": guardrail,
            "source_rejection_reason": dominant_reason,
            "effective_reason_count": round(dominant_count, 3),
            "reason_share": dominant_share,
            "effective_decisive_review_count": round(effective_decisive, 3),
            "effective_structured_rejection_count": round(
                effective_rejected, 3
            ),
            "rejected_cohort_count": len(rejected_cohorts),
            "rejected_hypothesis_count": len(rejected_hypotheses),
        }
        category_results[category] = result
    return {
        "schema_version": 1,
        "policy": (
            "Only controlled rejection codes are aggregated. Free-form notes are "
            "never included. A category guardrail requires diverse reviewed "
            "cohorts and hypotheses plus a dominant AI-layer failure reason."
        ),
        "minimum_decisive_required": min_decisive,
        "minimum_rejections_required": min_rejections,
        "minimum_effective_rejections_required": min_effective_rejections,
        "minimum_rejection_rate": min_rejection_rate,
        "minimum_reason_effective_required": min_reason_effective,
        "minimum_reason_share": min_reason_share,
        "minimum_cohorts_required": min_cohorts,
        "minimum_hypotheses_required": min_hypotheses,
        "active_guardrails": active_guardrails,
        "categories": category_results,
    }


def compact_analysis_guardrail_policy(policy: dict | None) -> dict:
    """Keep only controlled prompt inputs; omit live metrics and review text."""
    active = {}
    for category, item in sorted(
        ((policy or {}).get("active_guardrails") or {}).items(),
        key=lambda pair: str(pair[0]).casefold(),
    ):
        if not isinstance(item, dict):
            continue
        category = _clean_text(category, 120)
        guardrail = str(item.get("guardrail") or "").strip()
        reason = str(item.get("source_rejection_reason") or "").strip()
        if (
            category
            and reason in AI_GUARDRAIL_BY_REJECTION_REASON
            and AI_GUARDRAIL_BY_REJECTION_REASON[reason] == guardrail
        ):
            active[category] = {
                "guardrail": guardrail,
                "source_rejection_reason": reason,
            }
    return {
        "schema_version": 1,
        "policy": (
            "Controlled category guardrail codes derived from diverse structured "
            "reviews; reviewer notes are excluded."
        ),
        "active_guardrails": active,
    }


def _guardrail_effectiveness_group(entries: list[dict],
                                   target_reason: str) -> dict:
    effective_decisive = sum(
        float(entry.get("_calibration_weight", 1.0)) for entry in entries
    )
    effective_rejected = sum(
        float(entry.get("_calibration_weight", 1.0))
        for entry in entries if entry.get("decision") == "rejected"
    )
    effective_target_failures = sum(
        float(entry.get("_calibration_weight", 1.0))
        for entry in entries
        if entry.get("decision") == "rejected"
        and entry.get("rejection_reason") == target_reason
    )
    lower, upper = _wilson_interval(
        effective_target_failures, effective_decisive
    )
    return {
        "decisive_review_count": len(entries),
        "effective_decisive_review_count": round(effective_decisive, 3),
        "rejected_review_count": sum(
            entry.get("decision") == "rejected" for entry in entries
        ),
        "effective_rejected_review_count": round(effective_rejected, 3),
        "target_failure_review_count": sum(
            entry.get("decision") == "rejected"
            and entry.get("rejection_reason") == target_reason
            for entry in entries
        ),
        "effective_target_failure_review_count": round(
            effective_target_failures, 3
        ),
        "rejection_rate": (
            effective_rejected / effective_decisive
            if effective_decisive else None
        ),
        "target_failure_rate": (
            effective_target_failures / effective_decisive
            if effective_decisive else None
        ),
        "target_failure_wilson_lower_bound": lower,
        "target_failure_wilson_upper_bound": upper,
        "reviewed_cohort_count": len({
            entry["_cohort_id"] for entry in entries
        }),
        "reviewed_hypothesis_count": len({
            entry["_hypothesis_id"] for entry in entries
        }),
        "has_unknown_identity": any(
            entry["_cohort_id"] == "unknown_cohort"
            or entry["_hypothesis_id"] == "unknown_hypothesis"
            for entry in entries
        ),
    }


def guardrail_effectiveness_comparison(
    ledger: dict, *,
    min_decisive_per_group: int = 8,
    min_effective_per_group: float = 8.0,
    min_cohorts_per_group: int = 3,
    min_hypotheses_per_group: int = 3,
    max_per_cohort: float = MAX_EFFECTIVE_REVIEWS_PER_COHORT,
    max_per_hypothesis_cohort: float = (
        MAX_EFFECTIVE_REVIEWS_PER_HYPOTHESIS_COHORT
    ),
) -> dict:
    """Compare a guardrail only within an identical model/engine configuration."""
    decisive_pairs = [
        (str(key), entry)
        for key, entry in (ledger.get("reviews") or {}).items()
        if isinstance(entry, dict)
        and entry.get("candidate_kind") != "threshold_audit"
        and (
            entry.get("decision") == "confirmed"
            or (
                entry.get("decision") == "rejected"
                and entry.get("rejection_reason") in REJECTION_REASONS
            )
        )
    ]
    exact_pairs = [
        pair for pair in decisive_pairs
        if pair[1].get("analysis_guardrail_provenance") == "exact_policy"
        and str(pair[1].get("target_category") or "").strip()
        and str(pair[1].get("analysis_model") or "").strip()
        and re.fullmatch(
            r"[a-f0-9]{64}",
            str(pair[1].get("analysis_engine_sha256") or ""),
        )
    ]
    configurations = sorted({
        (
            str(entry.get("target_category") or "").strip(),
            str(entry.get("analysis_guardrail") or "").strip(),
            str(entry.get("analysis_guardrail_source_rejection_reason") or "")
            .strip(),
            str(entry.get("analysis_model") or "").strip(),
            str(entry.get("analysis_engine_sha256") or "").strip(),
        )
        for _key, entry in exact_pairs
        if entry.get("analysis_guardrail") in ANALYSIS_GUARDRAILS
        and entry.get("analysis_guardrail_source_rejection_reason")
        in AI_GUARDRAIL_BY_REJECTION_REASON
        and AI_GUARDRAIL_BY_REJECTION_REASON[
            entry["analysis_guardrail_source_rejection_reason"]
        ] == entry["analysis_guardrail"]
    }, key=lambda item: tuple(value.casefold() for value in item))
    comparisons = []
    for category, guardrail, target_reason, model, engine in configurations:
        relevant_pairs = [
            pair for pair in exact_pairs
            if str(pair[1].get("target_category") or "").strip() == category
            and str(pair[1].get("analysis_model") or "").strip() == model
            and str(pair[1].get("analysis_engine_sha256") or "").strip()
            == engine
            and pair[1].get("analysis_guardrail") in {None, guardrail}
        ]
        weighted = _correlation_weighted_entries(
            relevant_pairs,
            max_per_cohort=max_per_cohort,
            max_per_hypothesis_cohort=max_per_hypothesis_cohort,
        )
        guarded_entries = [
            entry for entry in weighted
            if entry.get("analysis_guardrail") == guardrail
        ]
        unguarded_entries = [
            entry for entry in weighted
            if entry.get("analysis_guardrail") is None
        ]
        guarded = _guardrail_effectiveness_group(
            guarded_entries, target_reason
        )
        unguarded = _guardrail_effectiveness_group(
            unguarded_entries, target_reason
        )
        comparison = {
            "target_category": category,
            "guardrail": guardrail,
            "source_rejection_reason": target_reason,
            "model": model,
            "engine_sha256": engine,
            "status": "insufficient_reviews",
            "observational_only": True,
            "eligible_for_policy_change": False,
            "minimum_decisive_per_group": min_decisive_per_group,
            "minimum_effective_per_group": min_effective_per_group,
            "minimum_cohorts_per_group": min_cohorts_per_group,
            "minimum_hypotheses_per_group": min_hypotheses_per_group,
            "unguarded": unguarded,
            "guarded": guarded,
            "target_failure_rate_reduction": None,
            "difference_confidence_interval": None,
        }
        groups = (unguarded, guarded)
        if any(
            group["decisive_review_count"] < min_decisive_per_group
            or group["effective_decisive_review_count"]
            < min_effective_per_group
            for group in groups
        ):
            comparisons.append(comparison)
            continue
        if any(
            group["reviewed_cohort_count"] < min_cohorts_per_group
            or group["reviewed_hypothesis_count"] < min_hypotheses_per_group
            or group["has_unknown_identity"]
            for group in groups
        ):
            comparison["status"] = "insufficient_diversity"
            comparisons.append(comparison)
            continue
        unguarded_rate = float(unguarded["target_failure_rate"] or 0)
        guarded_rate = float(guarded["target_failure_rate"] or 0)
        reduction = unguarded_rate - guarded_rate
        interval = (
            float(unguarded["target_failure_wilson_lower_bound"])
            - float(guarded["target_failure_wilson_upper_bound"]),
            float(unguarded["target_failure_wilson_upper_bound"])
            - float(guarded["target_failure_wilson_lower_bound"]),
        )
        comparison["target_failure_rate_reduction"] = reduction
        comparison["difference_confidence_interval"] = list(interval)
        if interval[0] > 0:
            comparison["status"] = "improved"
        elif interval[1] < 0:
            comparison["status"] = "worse"
        else:
            comparison["status"] = "no_clear_difference"
        comparisons.append(comparison)
    return {
        "schema_version": 1,
        "policy": (
            "Observational guarded-versus-unguarded comparisons use only exact "
            "policy provenance under the same model and analysis-engine SHA-256."
        ),
        "unknown_provenance_decisive_review_count": (
            len(decisive_pairs) - len(exact_pairs)
        ),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }


def repair_origin_effectiveness_comparison(
    ledger: dict, review_keys: set[str] | None = None, *,
    min_decisive_per_group: int = 8,
    min_effective_per_group: float = 8.0,
    min_cohorts_per_group: int = 3,
    min_hypotheses_per_group: int = 3,
    max_per_cohort: float = MAX_EFFECTIVE_REVIEWS_PER_COHORT,
    max_per_hypothesis_cohort: float = (
        MAX_EFFECTIVE_REVIEWS_PER_HYPOTHESIS_COHORT
    ),
) -> dict:
    """Compare repaired-origin predictions within matched run configurations."""
    decisive_pairs = [
        (str(key), entry)
        for key, entry in (ledger.get("reviews") or {}).items()
        if (review_keys is None or key in review_keys)
        and isinstance(entry, dict)
        and entry.get("candidate_kind") != "threshold_audit"
        and entry.get("decision") in {"confirmed", "rejected"}
    ]
    exact_pairs = [
        pair for pair in decisive_pairs
        if pair[1].get("prediction_origin_provenance")
        == "schema11_exact_claim_lineage"
        and pair[1].get("prediction_origin") in PREDICTION_ORIGINS - {
            "legacy_unknown"
        }
        and pair[1].get("analysis_guardrail_provenance") == "exact_policy"
        and (
            (
                pair[1].get("analysis_guardrail") is None
                and pair[1].get(
                    "analysis_guardrail_source_rejection_reason"
                ) is None
            )
            or (
                pair[1].get("analysis_guardrail") in ANALYSIS_GUARDRAILS
                and pair[1].get(
                    "analysis_guardrail_source_rejection_reason"
                ) in AI_GUARDRAIL_BY_REJECTION_REASON
                and AI_GUARDRAIL_BY_REJECTION_REASON[
                    pair[1]["analysis_guardrail_source_rejection_reason"]
                ] == pair[1]["analysis_guardrail"]
            )
        )
        and str(pair[1].get("target_category") or "").strip()
        and str(pair[1].get("analysis_model") or "").strip()
        and re.fullmatch(
            r"[a-f0-9]{64}",
            str(pair[1].get("analysis_engine_sha256") or ""),
        )
    ]
    configurations = sorted({
        (
            str(entry.get("target_category") or "").strip(),
            str(entry.get("analysis_model") or "").strip(),
            str(entry.get("analysis_engine_sha256") or "").strip(),
            str(entry.get("analysis_guardrail") or "").strip(),
            str(entry.get(
                "analysis_guardrail_source_rejection_reason"
            ) or "").strip(),
        )
        for _key, entry in exact_pairs
    }, key=lambda item: tuple(value.casefold() for value in item))
    comparisons = []
    repaired_origins = {
        "recovered_draft_only", "mixed_initial_and_recovered",
    }
    for category, model, engine, guardrail, guardrail_reason in configurations:
        relevant_pairs = [
            pair for pair in exact_pairs
            if str(pair[1].get("target_category") or "").strip() == category
            and str(pair[1].get("analysis_model") or "").strip() == model
            and str(pair[1].get("analysis_engine_sha256") or "").strip()
            == engine
            and str(pair[1].get("analysis_guardrail") or "").strip()
            == guardrail
            and str(pair[1].get(
                "analysis_guardrail_source_rejection_reason"
            ) or "").strip() == guardrail_reason
        ]
        weighted = _correlation_weighted_entries(
            relevant_pairs,
            max_per_cohort=max_per_cohort,
            max_per_hypothesis_cohort=max_per_hypothesis_cohort,
        )
        initial = _historical_support_group([
            entry for entry in weighted
            if entry.get("prediction_origin") == "initial_audit_pass_only"
        ])
        repaired = _historical_support_group([
            entry for entry in weighted
            if entry.get("prediction_origin") in repaired_origins
        ])
        comparison = {
            "target_category": category,
            "model": model,
            "engine_sha256": engine,
            "analysis_guardrail": guardrail or None,
            "analysis_guardrail_source_rejection_reason": (
                guardrail_reason or None
            ),
            "status": "insufficient_reviews",
            "observational_only": True,
            "eligible_for_policy_change": False,
            "minimum_decisive_per_group": min_decisive_per_group,
            "minimum_effective_per_group": min_effective_per_group,
            "minimum_cohorts_per_group": min_cohorts_per_group,
            "minimum_hypotheses_per_group": min_hypotheses_per_group,
            "initial_pass_only": initial,
            "repaired_origin": repaired,
            "confirmation_rate_difference": None,
            "difference_confidence_interval": None,
        }
        groups = (initial, repaired)
        if any(
            group["decisive_review_count"] < min_decisive_per_group
            or group["effective_decisive_review_count"]
            < min_effective_per_group
            for group in groups
        ):
            comparisons.append(comparison)
            continue
        if any(
            group["reviewed_cohort_count"] < min_cohorts_per_group
            or group["reviewed_hypothesis_count"] < min_hypotheses_per_group
            or group["has_unknown_identity"]
            for group in groups
        ):
            comparison["status"] = "insufficient_diversity"
            comparisons.append(comparison)
            continue
        initial_rate = float(initial["confirmation_rate"])
        repaired_rate = float(repaired["confirmation_rate"])
        difference = repaired_rate - initial_rate
        interval = (
            float(repaired["wilson_lower_bound"])
            - float(initial["wilson_upper_bound"]),
            float(repaired["wilson_upper_bound"])
            - float(initial["wilson_lower_bound"]),
        )
        comparison["confirmation_rate_difference"] = difference
        comparison["difference_confidence_interval"] = list(interval)
        if interval[0] > 0:
            comparison["status"] = "repaired_higher"
        elif interval[1] < 0:
            comparison["status"] = "repaired_lower"
        else:
            comparison["status"] = "no_clear_difference"
        comparisons.append(comparison)
    return {
        "schema_version": 1,
        "policy": (
            "Observational repaired-origin versus initial-pass comparisons use "
            "only exact schema-11 coherent claim lineage within the same target "
            "category, model, analysis-engine SHA-256, and applied category "
            "guardrail state. Results never change repair policy automatically."
        ),
        "unknown_provenance_decisive_review_count": (
            len(decisive_pairs) - len(exact_pairs)
        ),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }


def calibrate_match_threshold(
    ledger: dict, *,
    default_threshold: float = DEFAULT_FEATURE_MATCH_THRESHOLD,
    matcher_policy: str | None = None,
    min_decisive: int = MIN_CALIBRATION_DECISIVE,
    min_selected: int = MIN_CALIBRATION_SELECTED,
    target_rate: float = TARGET_CONFIRMATION_RATE,
    min_wilson: float = MIN_WILSON_LOWER_BOUND,
    min_confirmed_retention: float = MIN_CONFIRMED_RETENTION_RATE,
    max_threshold: float = 0.85,
    min_cohorts: int = MIN_CALIBRATION_COHORTS,
    min_hypotheses: int = MIN_CALIBRATION_HYPOTHESES,
    max_per_cohort: float = MAX_EFFECTIVE_REVIEWS_PER_COHORT,
    max_per_hypothesis_cohort: float = (
        MAX_EFFECTIVE_REVIEWS_PER_HYPOTHESIS_COHORT
    ),
    exclude_non_matcher_rejections: bool = True,
) -> dict:
    """Raise, never lower, the matcher threshold from sufficient reviewed evidence."""
    all_scored_pairs = [
        (str(key), entry)
        for key, entry in (ledger.get("reviews") or {}).items()
        if entry.get("decision") in {"confirmed", "rejected"}
        and isinstance(entry.get("match_score"), (int, float))
        and not isinstance(entry.get("match_score"), bool)
        and math.isfinite(float(entry["match_score"]))
        and 0 <= float(entry["match_score"]) <= 1
    ]
    scored_pairs = [
        pair for pair in all_scored_pairs
        if matcher_policy is None
        or pair[1].get("matcher_policy") == matcher_policy
    ]
    excluded_matcher_policy = len(all_scored_pairs) - len(scored_pairs)
    excluded_non_matcher = [
        pair for pair in scored_pairs
        if exclude_non_matcher_rejections
        and pair[1].get("decision") == "rejected"
        and pair[1].get("rejection_reason") in NON_MATCHER_REJECTION_REASONS
    ]
    excluded_unroutable = [
        pair for pair in scored_pairs
        if pair[1].get("decision") == "rejected"
        and pair[1].get("rejection_reason") not in REJECTION_REASONS
    ]
    decisive_pairs = [
        pair for pair in scored_pairs
        if pair not in excluded_non_matcher and pair not in excluded_unroutable
    ]
    decisive = _correlation_weighted_entries(
        decisive_pairs,
        max_per_cohort=max_per_cohort,
        max_per_hypothesis_cohort=max_per_hypothesis_cohort,
    )
    confirmed_total = sum(entry["decision"] == "confirmed" for entry in decisive)
    rejected_total = sum(entry["decision"] == "rejected" for entry in decisive)
    effective_decisive = sum(
        float(entry["_calibration_weight"]) for entry in decisive
    )
    cohort_counts: defaultdict[str, int] = defaultdict(int)
    for entry in decisive:
        cohort_counts[entry["_cohort_id"]] += 1
    cohort_count = len(cohort_counts)
    hypothesis_count = len({entry["_hypothesis_id"] for entry in decisive})
    base = _threshold_evidence(decisive, default_threshold)
    confirmed_below_default = [
        entry for entry in decisive
        if entry.get("candidate_kind") == "threshold_audit"
        and entry.get("decision") == "confirmed"
        and (
            float(entry["match_score"]) < default_threshold
            or (
                float(entry.get("applied_threshold") or 2) <= default_threshold
                and float(entry["match_score"]) <= default_threshold
            )
        )
    ]
    result = {
        "schema_version": 1,
        "matcher_policy": matcher_policy,
        "status": "insufficient_reviews",
        "eligible": False,
        "threshold": round(default_threshold, 3),
        "default_threshold": round(default_threshold, 3),
        "decisive_review_count": len(decisive),
        "effective_decisive_review_count": round(effective_decisive, 3),
        "confirmed_review_count": confirmed_total,
        "rejected_review_count": rejected_total,
        "excluded_non_matcher_rejection_count": len(excluded_non_matcher),
        "excluded_unroutable_rejection_count": len(excluded_unroutable),
        "excluded_matcher_policy_review_count": excluded_matcher_policy,
        "reviewed_cohort_count": cohort_count,
        "reviewed_hypothesis_count": hypothesis_count,
        "largest_cohort_review_count": max(cohort_counts.values(), default=0),
        "minimum_decisive_required": min_decisive,
        "minimum_selected_required": min_selected,
        "minimum_cohorts_required": min_cohorts,
        "minimum_hypotheses_required": min_hypotheses,
        "maximum_effective_reviews_per_cohort": max_per_cohort,
        "maximum_effective_reviews_per_hypothesis_cohort": (
            max_per_hypothesis_cohort
        ),
        "target_confirmation_rate": target_rate,
        "minimum_wilson_lower_bound": min_wilson,
        "minimum_confirmed_retention_rate": min_confirmed_retention,
        "confirmed_below_default_audit_count": len(confirmed_below_default),
        "effective_confirmed_below_default_audit_count": round(sum(
            float(entry["_calibration_weight"])
            for entry in confirmed_below_default
        ), 3),
        "evidence": base,
    }
    if (
        len(decisive) < min_decisive
        or confirmed_total < 3
        or rejected_total < 3
    ):
        return result
    if (
        effective_decisive < min_decisive
        or cohort_count < min_cohorts
        or hypothesis_count < min_hypotheses
        or "unknown_cohort" in cohort_counts
        or any(
            entry["_hypothesis_id"] == "unknown_hypothesis"
            for entry in decisive
        )
    ):
        result["status"] = "insufficient_diversity"
        return result
    if confirmed_below_default:
        result["status"] = "default_recall_warning"
        result["eligible"] = True
        return result
    if (
        base["confirmation_rate"] is not None
        and base["confirmation_rate"] >= target_rate
        and base["wilson_lower_bound"] >= min_wilson
    ):
        result["status"] = "default_performing_well"
        result["eligible"] = True
        return result

    thresholds = sorted({
        round(float(entry["match_score"]), 3) for entry in decisive
        if default_threshold < float(entry["match_score"]) <= max_threshold
    })
    best_higher_attempt = None
    for threshold in thresholds:
        evidence = _threshold_evidence(decisive, threshold)
        evidence["confirmed_retention_rate"] = (
            evidence["confirmed_count"] / base["confirmed_count"]
            if base["confirmed_count"] else None
        )
        if evidence["selected_count"] < min_selected:
            continue
        attempt = {"threshold": threshold, **evidence}
        if (
            best_higher_attempt is None
            or (
                float(attempt.get("confirmation_rate") or 0),
                float(attempt.get("wilson_lower_bound") or 0),
                float(attempt.get("confirmed_retention_rate") or 0),
                -threshold,
            )
            > (
                float(best_higher_attempt.get("confirmation_rate") or 0),
                float(best_higher_attempt.get("wilson_lower_bound") or 0),
                float(best_higher_attempt.get("confirmed_retention_rate") or 0),
                -float(best_higher_attempt["threshold"]),
            )
        ):
            best_higher_attempt = attempt
        if (
            evidence["confirmation_rate"] is not None
            and evidence["confirmation_rate"] >= target_rate
            and evidence["wilson_lower_bound"] >= min_wilson
            and evidence["confirmed_retention_rate"] is not None
            and evidence["confirmed_retention_rate"]
            >= min_confirmed_retention
        ):
            result.update({
                "status": "calibrated_higher",
                "eligible": True,
                "threshold": threshold,
                "evidence": evidence,
            })
            return result
    result["status"] = "no_safe_higher_threshold"
    result["eligible"] = True
    result["best_higher_threshold_attempt"] = best_higher_attempt
    return result


def calibrate_historical_analog_threshold(
    ledger: dict, *,
    default_threshold: float = DEFAULT_HISTORICAL_ANALOG_THRESHOLD,
    **calibration_kwargs,
) -> dict:
    """Safely raise the analog floor from reviewed, analog-supported outcomes."""
    transformed_reviews = {}
    excluded_ambiguous_rejections = 0
    for key, entry in (ledger.get("reviews") or {}).items():
        if (
            not isinstance(entry, dict)
            or entry.get("candidate_kind") == "threshold_audit"
            or entry.get("historical_support") is not True
            or not isinstance(
                entry.get("historical_analog_max_similarity"), (int, float)
            )
            or isinstance(entry.get("historical_analog_max_similarity"), bool)
        ):
            continue
        if (
            entry.get("decision") == "rejected"
            and entry.get("rejection_reason")
            not in ANALOG_NEGATIVE_REJECTION_REASONS
        ):
            excluded_ambiguous_rejections += 1
            continue
        transformed_reviews[str(key)] = {
            **entry,
            "match_score": float(entry["historical_analog_max_similarity"]),
            "candidate_kind": "feature_match",
        }
    result = calibrate_match_threshold(
        {"reviews": transformed_reviews},
        default_threshold=default_threshold,
        exclude_non_matcher_rejections=False,
        **calibration_kwargs,
    )
    result.update({
        "signal": "strongest_cited_historical_analog_similarity",
        "excluded_ambiguous_rejection_count": excluded_ambiguous_rejections,
        "observational_limit": (
            "The review outcome measures the full recommendation, not the analog "
            "alone. Calibration may only raise the selector floor and retains all "
            "feature-match diversity safeguards."
        ),
    })
    return result


def calibrate_historical_analog_policy(ledger: dict,
                                       **calibration_kwargs) -> dict:
    """Build global and category-specific analog similarity calibrations."""
    reviews = ledger.get("reviews") or {}
    global_calibration = calibrate_historical_analog_threshold(
        ledger, **calibration_kwargs
    )
    categories = sorted({
        str(entry.get("target_category") or "").strip()
        for entry in reviews.values()
        if isinstance(entry, dict)
        and str(entry.get("target_category") or "").strip()
        and entry.get("historical_support") is True
    }, key=casefold_order)
    category_calibrations = {
        category: calibrate_historical_analog_threshold(
            {
                "reviews": {
                    key: entry for key, entry in reviews.items()
                    if isinstance(entry, dict)
                    and str(entry.get("target_category") or "").strip()
                    == category
                }
            },
            **calibration_kwargs,
        )
        for category in categories
    }
    return {
        "schema_version": 1,
        "selection_policy": (
            "Use independently sufficient category evidence; otherwise use "
            "sufficient global evidence; otherwise retain the default analog floor."
        ),
        "global": global_calibration,
        "categories": category_calibrations,
    }


def resolve_historical_analog_threshold_for_category(
    policy: dict | None, category: str,
) -> dict:
    """Resolve a reviewed analog floor without ever falling below the default."""
    policy = policy if isinstance(policy, dict) else {}
    global_calibration = (
        policy.get("global") if isinstance(policy.get("global"), dict) else {}
    )
    category_calibration = (policy.get("categories") or {}).get(category)

    def eligible(value: object) -> bool:
        return isinstance(value, dict) and value.get("eligible") is True

    if eligible(category_calibration):
        selected = category_calibration
        source = "category_reviewed"
    elif eligible(global_calibration):
        selected = global_calibration
        source = "global_reviewed"
    else:
        selected = global_calibration
        source = "default"
    try:
        threshold = float(
            selected.get("threshold", DEFAULT_HISTORICAL_ANALOG_THRESHOLD)
        )
    except (TypeError, ValueError):
        threshold = DEFAULT_HISTORICAL_ANALOG_THRESHOLD
        source = "default"
    if not math.isfinite(threshold):
        threshold = DEFAULT_HISTORICAL_ANALOG_THRESHOLD
        source = "default"
    threshold = min(1.0, max(DEFAULT_HISTORICAL_ANALOG_THRESHOLD, threshold))
    return {
        "threshold": round(threshold, 3),
        "source": source,
        "status": str(selected.get("status") or "insufficient_reviews"),
        "decisive_review_count": int(
            selected.get("decisive_review_count") or 0
        ),
        "effective_decisive_review_count": float(
            selected.get("effective_decisive_review_count") or 0
        ),
        "reviewed_cohort_count": int(
            selected.get("reviewed_cohort_count") or 0
        ),
        "reviewed_hypothesis_count": int(
            selected.get("reviewed_hypothesis_count") or 0
        ),
    }


def calibrate_threshold_policy(ledger: dict, **calibration_kwargs) -> dict:
    """Build reviewed global and per-category matcher calibrations."""
    reviews = ledger.get("reviews") or {}
    global_calibration = calibrate_match_threshold(
        ledger, **calibration_kwargs
    )
    categories = sorted({
        str(entry.get("target_category") or "").strip()
        for entry in reviews.values()
        if isinstance(entry, dict)
        and str(entry.get("target_category") or "").strip()
    }, key=casefold_order)
    category_calibrations = {}
    for category in categories:
        category_calibrations[category] = calibrate_match_threshold(
            {
                "reviews": {
                    key: entry for key, entry in reviews.items()
                    if isinstance(entry, dict)
                    and str(entry.get("target_category") or "").strip()
                    == category
                }
            },
            **calibration_kwargs,
        )
    return {
        "schema_version": 1,
        "matcher_policy": calibration_kwargs.get("matcher_policy"),
        "selection_policy": (
            "Use independently sufficient category evidence; otherwise use "
            "sufficient global evidence; otherwise use the default threshold. "
            "When a matcher policy is supplied, reviews from other scoring "
            "policies are excluded."
        ),
        "global": global_calibration,
        "categories": category_calibrations,
    }


def resolve_threshold_for_category(policy: dict | None, category: str) -> dict:
    """Resolve a category threshold while supporting legacy global calibration."""
    policy = policy if isinstance(policy, dict) else {}
    if isinstance(policy.get("global"), dict):
        global_calibration = policy["global"]
        category_calibration = (policy.get("categories") or {}).get(category)
    else:
        global_calibration = policy
        category_calibration = None

    def is_eligible(calibration: object) -> bool:
        if not isinstance(calibration, dict):
            return False
        if "eligible" in calibration:
            return calibration.get("eligible") is True
        return bool(calibration.get("status")) and calibration.get("status") not in {
            "insufficient_reviews", "insufficient_diversity",
        }

    if (
        isinstance(category_calibration, dict)
        and is_eligible(category_calibration)
    ):
        selected = category_calibration
        source = "category_reviewed"
    elif (
        isinstance(global_calibration, dict)
        and is_eligible(global_calibration)
    ):
        selected = global_calibration
        source = "global_reviewed"
    else:
        selected = global_calibration if isinstance(global_calibration, dict) else {}
        source = "default"

    try:
        threshold = float(
            selected.get("threshold", DEFAULT_FEATURE_MATCH_THRESHOLD)
        )
    except (TypeError, ValueError):
        threshold = DEFAULT_FEATURE_MATCH_THRESHOLD
        source = "default"
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        threshold = DEFAULT_FEATURE_MATCH_THRESHOLD
        source = "default"
    return {
        "threshold": round(threshold, 3),
        "source": source,
        "status": str(selected.get("status") or "insufficient_reviews"),
        "decisive_review_count": int(
            selected.get("decisive_review_count") or 0
        ),
        "effective_decisive_review_count": float(
            selected.get("effective_decisive_review_count") or 0
        ),
        "reviewed_cohort_count": int(
            selected.get("reviewed_cohort_count") or 0
        ),
        "reviewed_hypothesis_count": int(
            selected.get("reviewed_hypothesis_count") or 0
        ),
    }


def overlay_reviews(evaluation: dict, ledger: dict) -> tuple[dict, dict]:
    """Overlay mutable decisions onto a copy of immutable evaluation evidence."""
    rendered = copy.deepcopy(evaluation)
    relevant_keys = set()
    prior_build = str(rendered.get("prior_close_build") or "")
    current_build = str(rendered.get("current_close_build") or "")
    for hypothesis in rendered.get("hypotheses") or []:
        hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
        candidates = (
            list(hypothesis.get("feature_match_candidates") or [])
            + list(hypothesis.get("threshold_audit_candidates") or [])
        )
        for candidate in candidates:
            try:
                key = review_key(
                    hypothesis_id, prior_build, current_build,
                    str(candidate.get("bug_code") or ""),
                    candidate.get("matcher_policy"),
                )
            except ReviewLedgerError:
                continue
            relevant_keys.add(key)
            candidate["review_key"] = key
            review = (ledger.get("reviews") or {}).get(key)
            if review:
                candidate["review_status"] = review["decision"]
                candidate["reviewer"] = review.get("reviewer") or ""
                candidate["review_note"] = review.get("note") or ""
                candidate["review_rejection_reason"] = review.get(
                    "rejection_reason"
                )
                candidate["reviewed_at"] = review.get("reviewed_at")
    return rendered, review_summary(ledger, relevant_keys)


def grounding_review_key(source_sha256: str, engine_sha256: str,
                         analysis_model: str, close_build: str,
                         audit_stage: str, candidate_id: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{64}", str(source_sha256 or "")):
        raise ReviewLedgerError("invalid grounding source SHA-256")
    if not re.fullmatch(r"[a-f0-9]{64}", str(engine_sha256 or "")):
        raise ReviewLedgerError("invalid grounding engine SHA-256")
    if not str(analysis_model or "").strip() or not str(close_build or "").strip():
        raise ReviewLedgerError("grounding model and Close Build must be non-empty")
    if audit_stage not in GROUNDING_AUDIT_STAGES:
        raise ReviewLedgerError("invalid grounding audit stage")
    if not re.fullmatch(r"TRC-[A-F0-9]{12}", str(candidate_id or "")):
        raise ReviewLedgerError("invalid grounding candidate_id")
    encoded = "\0".join((
        source_sha256, engine_sha256, str(analysis_model).strip(),
        str(close_build).strip(), audit_stage, candidate_id,
    )).encode("utf-8")
    return "TGR-" + hashlib.sha256(encoded).hexdigest()[:16].upper()


def _normalized_target_alias_pair(value: object) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "target_term", "evidence_term",
    }:
        raise ReviewLedgerError("target alias pair must contain exact term fields")
    pair = {
        "target_term": str(value.get("target_term") or "").strip().casefold(),
        "evidence_term": str(value.get("evidence_term") or "").strip().casefold(),
    }
    if any(
        not re.fullmatch(r"[a-z0-9]{2,40}", term)
        or term in TARGET_ALIAS_GENERIC_TERMS
        for term in pair.values()
    ) or pair["target_term"] == pair["evidence_term"]:
        raise ReviewLedgerError("target alias pair has invalid or generic terms")
    return pair


def _normalized_target_alias_pairs(value: object) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 4:
        raise ReviewLedgerError("target alias pairs must be an array of at most four pairs")
    pairs = []
    for raw_pair in value:
        pair = _normalized_target_alias_pair(raw_pair)
        if pair is None:
            raise ReviewLedgerError("target alias pair cannot be empty")
        if pair not in pairs:
            pairs.append(pair)
    if len(pairs) != len(value):
        raise ReviewLedgerError("target alias pairs must be unique")
    return pairs


def _target_alias_pair_errors(
    pair: object, *, audit_stage: object, withheld_reason: object,
    decision: object, failed_checks: object, overturned_checks: object,
    area: object, evidence_refs: object,
) -> list[str]:
    if pair is None:
        return []
    try:
        normalized = _normalized_target_alias_pair(pair)
    except ReviewLedgerError as exc:
        return [str(exc)]
    errors = []
    if (
        audit_stage != "deterministic_adjacency_gate"
        or withheld_reason != "adjacent_target_absent_from_current_signal"
        or decision != "false_negative"
        or "target_supported" not in (failed_checks or [])
        or "target_supported" not in (overturned_checks or [])
    ):
        errors.append(
            "target alias pair requires a target-gate false negative"
        )
    area_terms = set(re.findall(r"[a-z0-9]+", str(area or "").casefold()))
    evidence_terms = {
        term
        for ref in (evidence_refs or []) if isinstance(ref, dict)
        for term in re.findall(
            r"[a-z0-9]+", str(ref.get("quote") or "").casefold()
        )
    }
    if normalized and normalized["target_term"] not in area_terms:
        errors.append("target alias target_term is absent from the target area")
    if normalized and normalized["evidence_term"] not in evidence_terms:
        errors.append("target alias evidence_term is absent from cited evidence")
    return errors


def _empty_grounding_review_ledger(product: str) -> dict:
    return {
        "schema_version": GROUNDING_REVIEW_SCHEMA_VERSION,
        "product": product,
        "updated_at": None,
        "reviews": {},
    }


def _validate_grounding_review_entry(key: str, entry: object) -> list[str]:
    if not isinstance(entry, dict):
        return [f"grounding review {key} must be an object"]
    errors = []
    try:
        expected_key = grounding_review_key(
            str(entry.get("analysis_source_sha256") or ""),
            str(entry.get("analysis_engine_sha256") or ""),
            str(entry.get("analysis_model") or ""),
            str(entry.get("close_build") or ""),
            str(entry.get("audit_stage") or ""),
            str(entry.get("candidate_id") or ""),
        )
        if expected_key != key:
            errors.append(f"grounding review {key} identity hash mismatch")
    except ReviewLedgerError as exc:
        errors.append(f"grounding review {key}: {exc}")
    if entry.get("decision") not in GROUNDING_REVIEW_DECISIONS:
        errors.append(f"grounding review {key} has invalid decision")
    if entry.get("withheld_reason") not in GROUNDING_WITHHELD_REASONS:
        errors.append(f"grounding review {key} has invalid withheld_reason")
    engine = entry.get("analysis_engine_sha256")
    if not isinstance(engine, str) or not re.fullmatch(r"[a-f0-9]{64}", engine):
        errors.append(f"grounding review {key} has invalid engine SHA-256")
    if not str(entry.get("analysis_model") or "").strip():
        errors.append(f"grounding review {key} has missing model")
    if not str(entry.get("close_build") or "").strip():
        errors.append(f"grounding review {key} has missing Close Build")
    failed = entry.get("failed_checks")
    if (
        not isinstance(failed, list) or not failed
        or len(set(failed)) != len(failed)
        or any(check not in GROUNDING_AUDIT_CHECKS for check in failed)
    ):
        errors.append(f"grounding review {key} has invalid failed_checks")
        failed = []
    overturned = entry.get("overturned_checks")
    if (
        not isinstance(overturned, list)
        or len(set(overturned)) != len(overturned)
        or any(check not in failed for check in overturned)
    ):
        errors.append(f"grounding review {key} has invalid overturned_checks")
        overturned = []
    if entry.get("decision") == "false_negative" and not overturned:
        errors.append(f"grounding review {key} false negative needs overturned checks")
    if entry.get("decision") != "false_negative" and overturned:
        errors.append(f"grounding review {key} non-false-negative has overturned checks")
    codes = entry.get("source_bug_codes")
    if (
        not isinstance(codes, list) or not codes
        or len(set(codes)) != len(codes)
        or any(not re.fullmatch(r"[A-Za-z0-9._-]+", str(code)) for code in codes)
    ):
        errors.append(f"grounding review {key} has invalid source_bug_codes")
    refs = entry.get("evidence_refs")
    if not isinstance(refs, list) or not refs or any(
        not isinstance(ref, dict)
        or str(ref.get("bug_code") or "") not in (codes or [])
        or isinstance(ref.get("signal_index"), bool)
        or not isinstance(ref.get("signal_index"), int)
        or ref.get("signal_index") < 0
        for ref in (refs or [])
    ):
        errors.append(f"grounding review {key} has invalid evidence_refs")
    alias_errors = _target_alias_pair_errors(
        entry.get("target_alias_pair"),
        audit_stage=entry.get("audit_stage"),
        withheld_reason=entry.get("withheld_reason"),
        decision=entry.get("decision"),
        failed_checks=failed,
        overturned_checks=overturned,
        area=entry.get("area"),
        evidence_refs=refs,
    )
    errors.extend(
        f"grounding review {key} {error}" for error in alias_errors
    )
    if len(str(entry.get("reviewer") or "")) > MAX_REVIEWER_CHARS:
        errors.append(f"grounding review {key} reviewer is too long")
    if len(str(entry.get("note") or "")) > MAX_NOTE_CHARS:
        errors.append(f"grounding review {key} note is too long")
    history = entry.get("history")
    if not isinstance(history, list) or not history:
        errors.append(f"grounding review {key} history must be non-empty")
    else:
        for index, event in enumerate(history, start=1):
            if not isinstance(event, dict):
                errors.append(
                    f"grounding review {key} history event {index} must be an object"
                )
                continue
            event_decision = event.get("decision")
            event_overturned = event.get("overturned_checks")
            if event_decision not in GROUNDING_REVIEW_DECISIONS:
                errors.append(
                    f"grounding review {key} history event {index} has invalid decision"
                )
            if (
                not isinstance(event_overturned, list)
                or any(check not in failed for check in event_overturned)
            ):
                errors.append(
                    f"grounding review {key} history event {index} has invalid "
                    "overturned checks"
                )
            elif event_decision == "false_negative" and not event_overturned:
                errors.append(
                    f"grounding review {key} history event {index} false negative "
                    "needs overturned checks"
                )
            elif event_decision != "false_negative" and event_overturned:
                errors.append(
                    f"grounding review {key} history event {index} has unexpected "
                    "overturned checks"
                )
            history_alias_errors = _target_alias_pair_errors(
                event.get("target_alias_pair"),
                audit_stage=entry.get("audit_stage"),
                withheld_reason=entry.get("withheld_reason"),
                decision=event_decision,
                failed_checks=failed,
                overturned_checks=event_overturned,
                area=entry.get("area"),
                evidence_refs=refs,
            )
            errors.extend(
                f"grounding review {key} history event {index} {error}"
                for error in history_alias_errors
            )
            if len(str(event.get("reviewer") or "")) > MAX_REVIEWER_CHARS:
                errors.append(
                    f"grounding review {key} history event {index} reviewer is too long"
                )
            if len(str(event.get("note") or "")) > MAX_NOTE_CHARS:
                errors.append(
                    f"grounding review {key} history event {index} note is too long"
                )
    return errors


def load_grounding_review_ledger(path: Path, product: str) -> dict:
    if not path.exists():
        return _empty_grounding_review_ledger(product)
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewLedgerError(f"grounding review ledger is unreadable: {exc}") from exc
    if not isinstance(ledger, dict):
        raise ReviewLedgerError("grounding review ledger must be a JSON object")
    if ledger.get("schema_version") != GROUNDING_REVIEW_SCHEMA_VERSION:
        raise ReviewLedgerError("unsupported grounding review ledger schema")
    if str(ledger.get("product") or "") != product:
        raise ReviewLedgerError("grounding review ledger product mismatch")
    reviews = ledger.get("reviews")
    if not isinstance(reviews, dict):
        raise ReviewLedgerError("grounding review ledger reviews must be an object")
    errors = [
        error for key, entry in reviews.items()
        for error in _validate_grounding_review_entry(str(key), entry)
    ]
    if errors:
        raise ReviewLedgerError("; ".join(errors))
    return ledger


def upsert_grounding_review(path: Path, *, product: str,
                            analysis_source_sha256: str,
                            analysis_engine_sha256: str,
                            analysis_model: str, close_build: str,
                            candidate_id: str, audit_stage: str,
                            withheld_reason: str, failed_checks: list[str],
                            target_category: str, area: str,
                            source_bug_codes: list[str], evidence_refs: list[dict],
                            decision: str, overturned_checks: list[str] | None = None,
                            target_alias_pair: dict | None = None,
                            reviewer: str = "", note: str = "") -> dict:
    if decision not in GROUNDING_REVIEW_DECISIONS:
        raise ReviewLedgerError("invalid grounding review decision")
    clean_reviewer = _clean_text(reviewer, MAX_REVIEWER_CHARS + 1)
    clean_note = _clean_text(note, MAX_NOTE_CHARS + 1)
    if len(clean_reviewer) > MAX_REVIEWER_CHARS or len(clean_note) > MAX_NOTE_CHARS:
        raise ReviewLedgerError("grounding reviewer or note is too long")
    failed = list(dict.fromkeys(str(check) for check in failed_checks))
    overturned = list(dict.fromkeys(
        str(check) for check in (overturned_checks or [])
    ))
    clean_alias_pair = _normalized_target_alias_pair(target_alias_pair)
    if clean_alias_pair is not None and decision != "false_negative":
        raise ReviewLedgerError(
            "target alias pair requires a target-gate false negative"
        )
    key = grounding_review_key(
        analysis_source_sha256, str(analysis_engine_sha256).lower(),
        _clean_text(analysis_model, 120), _clean_text(close_build, 120),
        audit_stage, candidate_id,
    )
    immutable = {
        "analysis_source_sha256": str(analysis_source_sha256),
        "analysis_engine_sha256": str(analysis_engine_sha256).lower(),
        "analysis_model": _clean_text(analysis_model, 120),
        "close_build": _clean_text(close_build, 120),
        "candidate_id": str(candidate_id),
        "audit_stage": str(audit_stage),
        "withheld_reason": str(withheld_reason),
        "failed_checks": failed,
        "target_category": _clean_text(target_category, 120),
        "area": _clean_text(area, 240),
        "source_bug_codes": list(dict.fromkeys(str(code) for code in source_bug_codes)),
        "evidence_refs": copy.deepcopy(evidence_refs),
    }
    now = datetime.now(timezone.utc).isoformat()
    with _ledger_lock(path):
        ledger = load_grounding_review_ledger(path, product)
        existing = ledger["reviews"].get(key) or {}
        if existing and any(existing.get(field) != value for field, value in immutable.items()):
            raise ReviewLedgerError(
                "grounding evidence provenance cannot change for an existing review"
            )
        history = list(existing.get("history") or [])
        history.append({
            "decision": decision,
            "overturned_checks": overturned if decision == "false_negative" else [],
            "target_alias_pair": (
                clean_alias_pair if decision == "false_negative" else None
            ),
            "reviewer": clean_reviewer,
            "note": clean_note,
            "reviewed_at": now,
        })
        entry = {
            "review_key": key,
            **immutable,
            "decision": decision,
            "overturned_checks": overturned if decision == "false_negative" else [],
            "target_alias_pair": (
                clean_alias_pair if decision == "false_negative" else None
            ),
            "reviewer": clean_reviewer,
            "note": clean_note,
            "created_at": existing.get("created_at") or now,
            "reviewed_at": now,
            "history": history,
        }
        errors = _validate_grounding_review_entry(key, entry)
        if errors:
            raise ReviewLedgerError("; ".join(errors))
        ledger["reviews"][key] = entry
        ledger["updated_at"] = now
        _write_ledger_atomic(path, ledger)
    return entry


def grounding_review_summary(ledger: dict,
                             review_keys: set[str] | None = None) -> dict:
    entries = [
        entry for key, entry in (ledger.get("reviews") or {}).items()
        if review_keys is None or key in review_keys
    ]

    def group_summary(group: list[dict]) -> dict:
        correct = sum(item.get("decision") == "correctly_withheld" for item in group)
        false_negative = sum(item.get("decision") == "false_negative" for item in group)
        unclear = sum(item.get("decision") == "unclear" for item in group)
        decisive = correct + false_negative
        lower, upper = _wilson_interval(false_negative, decisive)
        return {
            "reviewed_count": len(group),
            "correctly_withheld_count": correct,
            "false_negative_count": false_negative,
            "unclear_count": unclear,
            "decisive_count": decisive,
            "false_negative_rate": (
                round(false_negative / decisive, 6) if decisive else None
            ),
            "false_negative_wilson_lower_bound": lower,
            "false_negative_wilson_upper_bound": upper,
        }

    summary = group_summary(entries)
    summary["by_stage"] = {
        stage: group_summary([
            item for item in entries if item.get("audit_stage") == stage
        ])
        for stage in sorted(GROUNDING_AUDIT_STAGES)
    }
    summary["by_reason"] = {
        reason: group_summary([
            item for item in entries if item.get("withheld_reason") == reason
        ])
        for reason in sorted(GROUNDING_WITHHELD_REASONS)
        if any(item.get("withheld_reason") == reason for item in entries)
    }
    return summary


def _target_alias_effectiveness(
    prediction_ledger: dict | None,
) -> dict[tuple[str, str], dict]:
    """Summarize only outcomes that directly adjudicate an exact alias pair."""
    grouped: defaultdict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    excluded_rejections: Counter[tuple[str, str]] = Counter()
    reviews = (
        prediction_ledger.get("reviews")
        if isinstance(prediction_ledger, dict) else {}
    ) or {}
    for key, entry in reviews.items():
        if (
            not isinstance(entry, dict)
            or entry.get("candidate_kind") == "threshold_audit"
            or entry.get("target_alias_provenance") != "schema27_exact_basis"
        ):
            continue
        try:
            pairs = _normalized_target_alias_pairs(entry.get("target_alias_pairs"))
        except ReviewLedgerError:
            continue
        if entry.get("decision") == "confirmed":
            for pair in pairs:
                grouped[(pair["target_term"], pair["evidence_term"])].append((
                    str(key), entry,
                ))
        elif (
            entry.get("decision") == "rejected"
            and entry.get("rejection_reason") == "reviewed_target_alias_misleading"
            and len(pairs) == 1
        ):
            pair = pairs[0]
            grouped[(pair["target_term"], pair["evidence_term"])].append((
                str(key), entry,
            ))
        elif entry.get("decision") == "rejected":
            for pair in pairs:
                excluded_rejections[(
                    pair["target_term"], pair["evidence_term"],
                )] += 1

    summaries = {}
    for pair, entries in grouped.items():
        weighted = _correlation_weighted_entries(
            entries,
            max_per_cohort=MAX_EFFECTIVE_REVIEWS_PER_COHORT,
            max_per_hypothesis_cohort=(
                MAX_EFFECTIVE_REVIEWS_PER_HYPOTHESIS_COHORT
            ),
        )
        confirmed = sum(
            entry.get("decision") == "confirmed" for entry in weighted
        )
        misleading = sum(
            entry.get("rejection_reason") == "reviewed_target_alias_misleading"
            for entry in weighted
        )
        effective_confirmed = sum(
            float(entry["_calibration_weight"])
            for entry in weighted if entry.get("decision") == "confirmed"
        )
        effective_misleading = sum(
            float(entry["_calibration_weight"])
            for entry in weighted
            if entry.get("rejection_reason") == "reviewed_target_alias_misleading"
        )
        effective_decisive = effective_confirmed + effective_misleading
        cohort_count = len({entry["_cohort_id"] for entry in weighted})
        hypothesis_count = len({entry["_hypothesis_id"] for entry in weighted})
        misleading_rate = (
            effective_misleading / effective_decisive
            if effective_decisive else None
        )
        misleading_wilson = _wilson_lower_bound(
            effective_misleading, effective_decisive,
        )
        retired = (
            len(weighted) >= TARGET_ALIAS_RETIRE_MIN_DECISIVE_REVIEWS
            and effective_decisive >= TARGET_ALIAS_RETIRE_MIN_EFFECTIVE_REVIEWS
            and misleading >= TARGET_ALIAS_RETIRE_MIN_MISLEADING_REVIEWS
            and cohort_count >= TARGET_ALIAS_RETIRE_MIN_DISTINCT_COHORTS
            and hypothesis_count >= TARGET_ALIAS_RETIRE_MIN_DISTINCT_HYPOTHESES
            and misleading_rate is not None
            and misleading_rate >= TARGET_ALIAS_RETIRE_MIN_MISLEADING_RATE
            and misleading_wilson >= TARGET_ALIAS_RETIRE_MIN_WILSON_LOWER_BOUND
        )
        summaries[pair] = {
            "decisive_review_count": len(weighted),
            "effective_decisive_review_count": round(effective_decisive, 3),
            "confirmed_review_count": confirmed,
            "alias_misleading_review_count": misleading,
            "effective_confirmed_review_count": round(effective_confirmed, 3),
            "effective_alias_misleading_review_count": round(
                effective_misleading, 3
            ),
            "excluded_other_rejection_count": excluded_rejections[pair],
            "distinct_cohort_count": cohort_count,
            "distinct_hypothesis_count": hypothesis_count,
            "observed_alias_misleading_rate": (
                round(misleading_rate, 6)
                if misleading_rate is not None else None
            ),
            "alias_misleading_wilson_lower_bound": round(
                misleading_wilson, 6
            ),
            "retired": retired,
        }
    for pair, count in excluded_rejections.items():
        summaries.setdefault(pair, {
            "decisive_review_count": 0,
            "effective_decisive_review_count": 0.0,
            "confirmed_review_count": 0,
            "alias_misleading_review_count": 0,
            "effective_confirmed_review_count": 0.0,
            "effective_alias_misleading_review_count": 0.0,
            "excluded_other_rejection_count": count,
            "distinct_cohort_count": 0,
            "distinct_hypothesis_count": 0,
            "observed_alias_misleading_rate": None,
            "alias_misleading_wilson_lower_bound": 0.0,
            "retired": False,
        })
    return summaries


def build_target_alias_policy(
    ledger: dict, prediction_ledger: dict | None = None,
) -> dict:
    """Activate reviewed aliases, then retire only from explicit diverse outcomes."""
    grouped: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in (ledger.get("reviews") or {}).values():
        if not isinstance(entry, dict) or entry.get("decision") != "false_negative":
            continue
        pair = entry.get("target_alias_pair")
        if pair is None:
            continue
        try:
            normalized = _normalized_target_alias_pair(pair)
        except ReviewLedgerError:
            continue
        if _target_alias_pair_errors(
            normalized,
            audit_stage=entry.get("audit_stage"),
            withheld_reason=entry.get("withheld_reason"),
            decision=entry.get("decision"),
            failed_checks=entry.get("failed_checks"),
            overturned_checks=entry.get("overturned_checks"),
            area=entry.get("area"),
            evidence_refs=entry.get("evidence_refs"),
        ):
            continue
        grouped[(
            normalized["target_term"], normalized["evidence_term"]
        )].append(entry)
    effectiveness = _target_alias_effectiveness(prediction_ledger)
    active_aliases: defaultdict[str, list[str]] = defaultdict(list)
    candidate_aliases = []
    for (target_term, evidence_term), entries in sorted(grouped.items()):
        source_runs = {
            str(entry.get("analysis_source_sha256") or "") for entry in entries
        }
        close_builds = {
            str(entry.get("close_build") or "") for entry in entries
        }
        candidate_ids = {
            str(entry.get("candidate_id") or "") for entry in entries
        }
        eligible = (
            len(entries) >= TARGET_ALIAS_MIN_CONFIRMED_REVIEWS
            and len(close_builds) >= TARGET_ALIAS_MIN_DISTINCT_CLOSE_BUILDS
            and len(source_runs) >= TARGET_ALIAS_MIN_DISTINCT_SOURCE_RUNS
            and len(candidate_ids) >= TARGET_ALIAS_MIN_CONFIRMED_REVIEWS
        )
        outcome = effectiveness.get((target_term, evidence_term)) or {
            "decisive_review_count": 0,
            "effective_decisive_review_count": 0.0,
            "confirmed_review_count": 0,
            "alias_misleading_review_count": 0,
            "effective_confirmed_review_count": 0.0,
            "effective_alias_misleading_review_count": 0.0,
            "excluded_other_rejection_count": 0,
            "distinct_cohort_count": 0,
            "distinct_hypothesis_count": 0,
            "observed_alias_misleading_rate": None,
            "alias_misleading_wilson_lower_bound": 0.0,
            "retired": False,
        }
        status = (
            "retired_low_precision" if eligible and outcome["retired"]
            else "active" if eligible else "insufficient_diverse_reviews"
        )
        candidate_aliases.append({
            "target_term": target_term,
            "evidence_term": evidence_term,
            "status": status,
            "confirmed_review_count": len(entries),
            "distinct_close_build_count": len(close_builds),
            "distinct_source_run_count": len(source_runs),
            "distinct_candidate_count": len(candidate_ids),
            "outcome_evidence": outcome,
        })
        if eligible and not outcome["retired"]:
            active_aliases[target_term].append(evidence_term)
    return {
        "schema_version": 1,
        "policy": "reviewed_directional_target_aliases_v1",
        "activation_thresholds": {
            "minimum_confirmed_reviews": TARGET_ALIAS_MIN_CONFIRMED_REVIEWS,
            "minimum_distinct_close_builds": (
                TARGET_ALIAS_MIN_DISTINCT_CLOSE_BUILDS
            ),
            "minimum_distinct_source_runs": (
                TARGET_ALIAS_MIN_DISTINCT_SOURCE_RUNS
            ),
            "minimum_distinct_candidates": TARGET_ALIAS_MIN_CONFIRMED_REVIEWS,
        },
        "retirement_thresholds": {
            "minimum_decisive_reviews": TARGET_ALIAS_RETIRE_MIN_DECISIVE_REVIEWS,
            "minimum_effective_decisive_reviews": (
                TARGET_ALIAS_RETIRE_MIN_EFFECTIVE_REVIEWS
            ),
            "minimum_alias_misleading_reviews": (
                TARGET_ALIAS_RETIRE_MIN_MISLEADING_REVIEWS
            ),
            "minimum_distinct_cohorts": TARGET_ALIAS_RETIRE_MIN_DISTINCT_COHORTS,
            "minimum_distinct_hypotheses": (
                TARGET_ALIAS_RETIRE_MIN_DISTINCT_HYPOTHESES
            ),
            "minimum_alias_misleading_rate": (
                TARGET_ALIAS_RETIRE_MIN_MISLEADING_RATE
            ),
            "minimum_alias_misleading_wilson_lower_bound": (
                TARGET_ALIAS_RETIRE_MIN_WILSON_LOWER_BOUND
            ),
            "retirement_rejection_reason": "reviewed_target_alias_misleading",
        },
        "active_aliases": {
            target: sorted(set(terms))
            for target, terms in sorted(active_aliases.items())
        },
        "candidate_aliases": candidate_aliases,
        "eligible_review_count": sum(len(rows) for rows in grouped.values()),
        "human_review_notes_included": False,
        "aliases_influence_confidence": False,
        "aliases_influence_risk": False,
        "aliases_influence_priority": False,
    }


def compact_target_alias_policy(policy: dict | None) -> dict:
    """Keep only active reviewed alias codes in immutable analysis input."""
    raw_aliases = (
        policy.get("active_aliases") if isinstance(policy, dict) else {}
    )
    active_aliases = {}
    if isinstance(raw_aliases, dict):
        for target, evidence_terms in sorted(raw_aliases.items()):
            try:
                target_pair_terms = [
                    _normalized_target_alias_pair({
                        "target_term": target,
                        "evidence_term": evidence_term,
                    })
                    for evidence_term in (
                        evidence_terms if isinstance(evidence_terms, list) else []
                    )
                ]
            except ReviewLedgerError:
                continue
            normalized_terms = sorted({
                pair["evidence_term"] for pair in target_pair_terms if pair
            })
            if normalized_terms:
                active_aliases[str(target).casefold()] = normalized_terms
    return {
        "schema_version": 1,
        "policy": "reviewed_directional_target_aliases_v1",
        "activation_thresholds": {
            "minimum_confirmed_reviews": TARGET_ALIAS_MIN_CONFIRMED_REVIEWS,
            "minimum_distinct_close_builds": (
                TARGET_ALIAS_MIN_DISTINCT_CLOSE_BUILDS
            ),
            "minimum_distinct_source_runs": (
                TARGET_ALIAS_MIN_DISTINCT_SOURCE_RUNS
            ),
            "minimum_distinct_candidates": TARGET_ALIAS_MIN_CONFIRMED_REVIEWS,
        },
        "active_aliases": active_aliases,
        "human_review_notes_included": False,
        "aliases_influence_confidence": False,
        "aliases_influence_risk": False,
        "aliases_influence_priority": False,
    }


def overlay_grounding_reviews(grounding_audit: dict, ledger: dict,
                              source_sha256: str, engine_sha256: str,
                              analysis_model: str,
                              close_build: str) -> tuple[dict, dict]:
    rendered = copy.deepcopy(grounding_audit)
    relevant_keys = set()
    collections = [
        rendered.get("withheld_candidates") or [],
        rendered.get("final_withheld_recommendations") or [],
        (
            (rendered.get("candidate_discovery_audit") or {}).get(
                "adjacency_gate_withheld_candidates"
            ) or []
        ),
    ]
    for collection in collections:
        for item in collection:
            if not isinstance(item, dict) or not item.get("human_review_sample"):
                continue
            try:
                key = grounding_review_key(
                    source_sha256, engine_sha256, analysis_model, close_build,
                    str(item.get("audit_stage") or ""),
                    str(item.get("candidate_id") or ""),
                )
            except ReviewLedgerError:
                continue
            relevant_keys.add(key)
            item["grounding_review_key"] = key
            review = (ledger.get("reviews") or {}).get(key)
            if review:
                item["grounding_review_status"] = review["decision"]
                item["grounding_review_overturned_checks"] = (
                    review.get("overturned_checks") or []
                )
                item["grounding_review_target_alias_pair"] = review.get(
                    "target_alias_pair"
                )
                item["grounding_reviewer"] = review.get("reviewer") or ""
                item["grounding_review_note"] = review.get("note") or ""
    return rendered, grounding_review_summary(ledger, relevant_keys)
