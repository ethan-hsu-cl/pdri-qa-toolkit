import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.fetch_tr_bug_details as fetch_details
import scripts.process_newest_tr as process_tr
import scripts.summarize_tr_bugs as summarize_tr
import scripts.tr_prediction_reviews as prediction_reviews

from scripts.process_newest_tr import (
    audit_report_manifest,
    close_build_sort_key,
    detail_quality_score,
    extract_manifest_checklists,
    filter_detail_data,
    local_fallback_bugs,
    merge_detail_data,
    mmdd_date_value,
    select_newest_tr_cohort,
)
from scripts.fetch_from_n8n import save_json
from scripts.summarize_tr_bugs import (
    build_regression_prompt,
    build_prompt,
    enrich_from_raw,
    generate_chunked_report,
    generate_structured_analysis,
    normalize_regression_bullets,
    render_analysis_markdown,
    regression_first,
    strip_chunk_wrappers,
    validate_report,
)


def _prediction_boundaries(area: str) -> dict:
    return {
        "scope_boundary": (
            f"Limit {area} to the cited shared behavior; independent paths are "
            "outside scope"
        ),
        "disconfirming_observation": (
            f"Lower priority if {area} remains correct under the stated trigger"
        ),
    }


def _candidate_discovery_audit(candidates=None) -> dict:
    candidates = list(candidates or [])
    for candidate in candidates:
        evidence_map = {}
        for code in candidate.get("source_bug_codes") or []:
            code_refs = [
                ref for ref in (candidate.get("evidence_refs") or [])
                if isinstance(ref, dict) and ref.get("bug_code") == code
            ]
            max_index = max(
                (int(ref.get("signal_index") or 0) for ref in code_refs),
                default=-1,
            )
            signals = [""] * (max_index + 1)
            for ref in code_refs:
                signals[int(ref.get("signal_index") or 0)] = str(
                    ref.get("quote") or ""
                )
            evidence_map[code] = {"adjacency_signal_quotes": signals}
        if "evidence_ref_completion_basis" not in candidate:
            final_refs, basis = summarize_tr.deterministic_evidence_ref_completion(
                str(candidate.get("area") or ""),
                list(candidate.get("source_bug_codes") or []),
                list(candidate.get("evidence_refs") or []),
                evidence_map,
            )
            candidate["evidence_refs"] = final_refs
            candidate["evidence_ref_completion_basis"] = basis
        else:
            final_refs = list(candidate.get("evidence_refs") or [])
        candidate["shared_path_support_basis"] = (
            summarize_tr.deterministic_shared_path_support_basis(
                str(candidate.get("area") or ""),
                str(candidate.get("rationale") or ""),
                final_refs,
                candidate.get("adjacency_support_basis"),
            )
        )
        candidate["scenario_specificity_basis"] = (
            summarize_tr.deterministic_scenario_specificity_basis(
                str(candidate.get("trigger") or ""),
                str(candidate.get("expected_failure_mode") or ""),
                list(candidate.get("source_bug_codes") or []),
                final_refs,
                evidence_map, action=str(candidate.get("action") or ""),
                disconfirming_observation=str(candidate.get("disconfirming_observation") or ""),
            )
        )
    candidate_ids = [
        summarize_tr.stable_candidate_audit_id(candidate)
        for candidate in candidates
    ]
    return {
        "schema_version": 1,
        "policy": "primary_plus_distinct_coverage_challenge_v1",
        "malformed_candidate_dropped_count": 0,
        "malformed_candidate_drop_records": [],
        "malformed_candidate_drop_reason_counts": {},
        "primary_candidate_count": len(candidate_ids),
        "primary_candidate_ids": candidate_ids,
        "coverage_challenge_candidate_count": 0,
        "coverage_challenge_candidate_ids": [],
        "coverage_challenge_distinct_count": 0,
        "coverage_challenge_distinct_candidate_ids": [],
        "coverage_overlap_filtered_count": 0,
        "coverage_overlap_filtered_candidates": [],
        "adjacency_gate_evaluated_count": len(candidate_ids),
        "adjacency_gate_supported_count": len(candidate_ids),
        "adjacency_gate_supported_candidate_ids": candidate_ids,
        "adjacency_gate_supported_candidates": [{
            "candidate_id": candidate_id,
            "discovery_origin": "primary_discovery",
            "candidate": candidate,
            "adjacency_support_basis": candidate[
                "adjacency_support_basis"
            ],
        } for candidate_id, candidate in zip(candidate_ids, candidates)],
        "adjacency_gate_withheld_count": 0,
        "adjacency_gate_withheld_candidates": [],
        "shared_path_gate_supported_count": len(candidate_ids),
        "shared_path_gate_withheld_count": 0,
        "shared_path_gate_withheld_candidate_ids": [],
        "evidence_ref_completion_applied_count": 0,
        "evidence_ref_completion_candidate_ids": [],
        "evidence_ref_completion_candidates": [],
        "reviewed_alias_supported_count": 0,
        "reviewed_alias_supported_candidate_ids": [],
        "reviewed_alias_supported_candidates": [],
        "grounding_input_candidate_count": len(candidate_ids),
        "unique_grounding_input_count": len(set(candidate_ids)),
        "exact_duplicate_candidate_count": (
            len(candidate_ids) - len(set(candidate_ids))
        ),
        "unique_candidate_origins": [{
            "candidate_id": candidate_id,
            "origins": ["primary_discovery"],
        } for candidate_id in dict.fromkeys(candidate_ids)],
        "challenge_coverage_overlap_can_reach_grounding": False,
        "adjacency_gate_withheld_can_reach_grounding": False,
        "model_critic_can_override_adjacency_gate": False,
        "model_critic_can_override_shared_path_gate": False,
        "evidence_ref_completion_scans_uncited_bugcodes": False,
        "evidence_ref_completion_changes_source_bugcodes": False,
        "model_critic_can_override_evidence_ref_completion": False,
        "reviewed_aliases_influence_confidence": False,
        "reviewed_aliases_influence_risk": False,
        "reviewed_aliases_influence_priority": False,
        "generation_origin_influences_confidence": False,
        "generation_origin_influences_risk": False,
        "generation_origin_influences_priority": False,
    }


def test_select_newest_tr_cohort_uses_trcreated_and_dominant_build(tmp_path):
    csv_path = tmp_path / "ecl_parsed.csv"
    csv_path.write_text(
        "BugCode,Status,parsed_version,Close Build#,_status_changed\n"
        "A,TRCreated,16.6.5,16.6.5.187153,true\n"
        "B,TRCreated,16.6.5,16.6.5.187153,True\n"
        "C,TRCreated,16.6.5,v0716 [abc1234],true\n"
        "D,BuildResolved,16.6.5,16.6.5.187153,true\n"
        "E,Close,16.6.5,16.6.5.187153,false\n",
        encoding="utf-8",
    )

    build, codes = select_newest_tr_cohort(csv_path)

    assert build == "16.6.5.187153"
    assert codes == ["A", "B", "C"]


def test_select_newest_tr_cohort_follows_the_build_not_the_reported_version(tmp_path):
    """A bug filed against a newer version can be fixed in an older build."""
    csv_path = tmp_path / "ecl_parsed.csv"
    csv_path.write_text(
        "BugCode,Status,parsed_version,Close Build#,_status_changed\n"
        # Filed against the newest version, but shipped in an older build.
        "STALE,TRCreated,16.8.0,16.7.5.188405,true\n"
        # The build QA is about to certify, carrying most of the fixes.
        "A,TRCreated,16.7.0,16.8.0.188591,true\n"
        "B,TRCreated,16.7.5,16.8.0.188591,true\n"
        "C,TRCreated,16.7.5,16.8.0.188591,true\n"
        "D,TRCreated,16.7.0,16.8.0.188318,true\n"
        "E,TRCreated,16.7.0,16.8.0.188318,true\n",
        encoding="utf-8",
    )

    build, codes = select_newest_tr_cohort(csv_path)

    assert build == "16.8.0.188591"
    assert codes == ["A", "B", "C"]


def test_select_newest_tr_cohort_skips_when_no_trcreated_bugs(tmp_path):
    csv_path = tmp_path / "ecl_parsed.csv"
    csv_path.write_text(
        "BugCode,Status,parsed_version,Close Build#,_status_changed\n"
        "A,Close,16.6.5,123,false\n",
        encoding="utf-8",
    )
    assert select_newest_tr_cohort(csv_path) == ("", [])


def test_select_newest_tr_cohort_prefers_newest_version_not_largest_batch(tmp_path):
    csv_path = tmp_path / "ecl_parsed.csv"
    rows = ["BugCode,Status,parsed_version,Close Build#,_status_changed"]
    rows += [f"OLD{i},TRCreated,25.0.6,0629_25.0.6_365,true" for i in range(8)]
    rows += ["NEW0,TRCreated,25.0.7,0702_25.0.7_365,true"]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    build, codes = select_newest_tr_cohort(csv_path)

    assert build == "0702_25.0.7_365"
    assert codes == ["NEW0"]


def test_select_newest_tr_excludes_buildresolved_but_keeps_component_build(tmp_path):
    csv_path = tmp_path / "ecl_parsed.csv"
    csv_path.write_text(
        "BugCode,Status,parsed_version,Close Build#,_status_changed\n"
        "OLD,TRCreated,16.5.5,16.5.5.186980,true\n"
        "A,TRCreated,16.6.5,16.6.5.187153,true\n"
        "A2,TRCreated,16.6.5,16.6.5.187153,true\n"
        "B,BuildResolved,16.6.5,16.6.5.187153,true\n"
        "C,TRCreated,16.6.5,v0716 [abc1234],false\n",
        encoding="utf-8",
    )
    build, codes = select_newest_tr_cohort(csv_path)
    assert build == "16.6.5.187153"
    assert codes == ["A", "A2", "C"]


def test_close_build_sort_key_handles_product_build_formats():
    assert close_build_sort_key("16.5.0.2607013") > close_build_sort_key("16.5.0.2606303")
    assert close_build_sort_key("18.0.7.0703.1204_hash") > close_build_sort_key("RC: 20260701_18.0.7")
    assert close_build_sort_key("18.0.7.0703.1204_hash") > close_build_sort_key("18.0.7.0702.1226_hash")
    assert close_build_sort_key("16.6.5.187073") > close_build_sort_key("16.6.5.186980")


def test_merge_detail_data_preserves_previous_bugs_and_updates_duplicates():
    old = {
        "meta": {"bug_count": 2, "tr_codes": ["TR-1"]},
        "products": {"PDR-I": [{"BugCode": "A", "Status": "Close"},
                                  {"BugCode": "B", "Status": "Close", "v": 1}]},
    }
    new = {
        "meta": {"bug_count": 2, "close_build": "123"},
        "products": {"PDR-I": [{"BugCode": "B", "Status": "Close", "v": 2,
                                  "ParentTRCode": "TR-1"},
                                  {"BugCode": "C", "Status": "Close",
                                   "ParentTRCode": "TR-1"}]},
    }

    merged = merge_detail_data(old, new)

    bugs = {b["BugCode"]: b for b in merged["products"]["PDR-I"]}
    assert set(bugs) == {"A", "B", "C"}
    assert bugs["B"]["v"] == 2
    assert merged["meta"]["bug_count"] == 3
    assert merged["meta"]["tr_codes"] == ["TR-1"]


def test_filter_detail_data_removes_bugs_no_longer_in_tr_cohort():
    data = {
        "meta": {"bug_count": 3, "missing_detail_bug_codes": ["EXTRA"]},
        "products": {"PDR-I": [
            {"BugCode": "A", "ParentTRCode": "TR-1"},
            {"BugCode": "B", "ParentTRCode": "TR-1"},
            {"BugCode": "EXTRA", "ParentTRCode": "TR-OLD"},
        ]},
    }
    filtered = filter_detail_data(data, ["A", "B"])
    assert [b["BugCode"] for b in filtered["products"]["PDR-I"]] == ["A", "B"]
    assert filtered["meta"]["bug_count"] == 2
    assert filtered["meta"]["tr_codes"] == ["TR-1"]
    assert filtered["meta"]["missing_detail_bug_codes"] == []


def test_summary_prompt_uses_root_cause_and_requests_actionable_sections():
    data = {
        "meta": {"close_build": "123"},
        "products": {"PDR-I": [{
            "BugCode": "A",
            "Short Description": "Export: wrong color",
            "Resolutions": [{"root_cause": "shared renderer", "solution": "fixed shader"}],
        }]},
    }

    prompt, count = build_prompt(data, 80, 1500)

    assert count == 1
    assert "root cause: shared renderer" in prompt
    assert "# Per-Bug Verification" in prompt
    assert "# Recommended Regression Areas" in prompt
    assert prompt.index("# Recommended Regression Areas") < prompt.index("# Per-Bug Verification")


def test_regression_first_reorders_an_existing_report():
    report = (
        "# Per-Bug Verification\n\n## A\n1. Verify the fix\n\n"
        "# Recommended Regression Areas\n\n## High\n- Test shared export paths"
    )

    reordered = regression_first(report)

    assert reordered.startswith("# Recommended Regression Areas")
    assert reordered.index("# Recommended Regression Areas") < reordered.index(
        "# Per-Bug Verification"
    )
    assert "## A\n1. Verify the fix" in reordered


def test_normalize_regression_bullets_separates_bold_module_entries():
    report = (
        "# Recommended Regression Areas\n"
        "**Export Module** - Critical: Test shared rendering.\n"
        "**Cloud Storage** - Medium: Test delete dialogs.\n\n"
        "# Per-Bug Verification\n## A\n1. Verify"
    )

    normalized = normalize_regression_bullets(report)

    assert "- **Export Module** - Critical:" in normalized
    assert "- **Cloud Storage** - Medium:" in normalized


def test_fetch_merge_clears_stale_change_flags_for_omitted_records(tmp_path):
    output = tmp_path / "ecl_raw.json"
    output.write_text(json.dumps([
        {"BugCode": "OLD", "Status": "Close", "_status_changed": True},
        {"BugCode": "CURRENT", "Status": "Assigned", "_status_changed": True},
    ]), encoding="utf-8")

    save_json([{"BugCode": "CURRENT", "Status": "Assigned"}], str(output))

    records = {r["BugCode"]: r for r in json.loads(output.read_text(encoding="utf-8"))}
    assert records["OLD"]["_status_changed"] is False
    assert records["CURRENT"]["_status_changed"] is False


def test_report_validation_rejects_truncated_output():
    data = {
        "products": {"PDR-I": [{"BugCode": "DRI000001-0001"},
                                  {"BugCode": "DRI000001-0002"}]},
    }
    truncated = "# Per-Bug Verification\n## DRI000001-0001\n- Verify it"
    assert validate_report(truncated, data, 80) == [
        "missing or duplicated '# Recommended Regression Areas' heading",
        "missing 1 BugCode(s): DRI000001-0002",
        "checklists must contain 2-4 numbered steps: DRI000001-0001 (0 steps)",
    ]


def test_report_validation_accepts_complete_output():
    data = {"products": {"PDR-I": [{"BugCode": "A"}, {"BugCode": "B"}]}}
    report = (
        "# Per-Bug Verification\n## A\n1. Test setup\n2. Verify result\n"
        "## B\n1. Test setup\n2. Verify result\n"
        "# Recommended Regression Areas\n"
        "- **Shared Export Queue** — **High** — Action: Export in a second "
        "format. Rationale: It shares renderer state. Evidence: A."
    )
    assert validate_report(report, data, 80) == []


def test_report_validation_requires_bugcode_as_checklist_heading():
    data = {"products": {"PDR-I": [{"BugCode": "A"}, {"BugCode": "B"}]}}
    report = (
        "# Recommended Regression Areas\n"
        "- **Export Queue** — **High** — Action: Test queue. "
        "Rationale: Shared state. Evidence: A, B.\n"
        "# Per-Bug Verification\n## A\n1. Set up\n2. Verify B is fixed\n"
    )
    assert "missing 1 BugCode(s): B" in validate_report(report, data, 80)


def test_report_validation_scopes_risk_check_to_regression_section():
    data = {"products": {"PDR-I": [{"BugCode": "A"}]}}
    report = (
        "# Recommended Regression Areas\n"
        "- **Export Queue** — Action: Test queue. Rationale: Shared state. Evidence: A.\n"
        "# Per-Bug Verification\n## A\n1. Use High quality\n2. Verify output\n"
    )
    assert "regression section has no risk-ranked recommendations" in validate_report(
        report, data, 80
    )


def test_regression_evidence_must_cite_bugcode_in_evidence_field():
    data = {"products": {"PDR-I": [{"BugCode": "BUG01"}]}}
    report = (
        "# Recommended Regression Areas\n"
        "- **Export Queue** — **High** — Action: Test the queue. "
        "Rationale: BUG01 changed shared state. Evidence: not supplied.\n"
        "# Per-Bug Verification\n## BUG01\n1. Set up\n2. Verify output\n"
    )
    assert "regression recommendation(s) lack BugCode evidence: 1" in validate_report(
        report, data, 80
    )


def test_report_validation_accepts_atypical_bugcode_heading():
    data = {"products": {"PHD-I": [{"BugCode": "PHI2626060087"}]}}
    report = (
        "# Recommended Regression Areas\n"
        "- **Photo Picker State** — **Medium** — Action: Reopen another picker. "
        "Rationale: It shares selection state. Evidence: PHI2626060087.\n"
        "# Per-Bug Verification\n## PHI2626060087\n1. Select a photo\n2. Verify state\n"
    )
    assert validate_report(report, data, 80) == []


def test_report_validation_accepts_explicit_no_grounded_adjacent_area():
    data = {"products": {"PDR-I": [{"BugCode": "A"}]}}
    report = (
        "# Recommended Regression Areas\n"
        "No evidence-grounded adjacent regression areas were identified from the "
        "supplied fix details.\n"
        "# Per-Bug Verification\n## A\n1. Set up\n2. Verify result\n"
    )
    assert validate_report(report, data, 80) == []


def test_enrich_from_raw_fills_resolution_when_repro_is_already_structured():
    bug = {
        "BugCode": "A",
        "ReproSteps": "Existing organized repro",
        "raw": {
            "StatusLog": json.dumps([
                {"RootCause": "shared renderer", "Solution": "reset shader cache"}
            ])
        },
    }
    enriched = enrich_from_raw(bug)
    assert enriched["ReproSteps"] == "Existing organized repro"
    assert enriched["Resolutions"][0]["root_cause"] == "shared renderer"


def test_regression_prompt_requires_grounded_adjacent_actions_and_risk_rubric():
    data = {
        "products": {"PDR-I": [{
            "BugCode": "A",
            "ModuleOwner": "Export",
            "Short Description": "Wrong color",
            "Resolutions": [{"root_cause": "shared renderer", "solution": "fixed shader"}],
        }]}
    }
    prompt = build_regression_prompt(data, 80, 1500)
    assert "ADJACENT" in prompt
    assert "Evidence: <BugCode(s)>" in prompt
    assert "Critical: credible adjacent risk of crash, data loss" in prompt
    assert "owner=Export" in prompt


def test_large_report_uses_evidence_batches_before_final_synthesis(monkeypatch):
    data = {
        "products": {"PDR-I": [
            {"BugCode": f"BUG{i:02d}", "Short Description": f"Issue {i}"}
            for i in range(13)
        ]}
    }
    prompts = []

    def fake_call(prompt, model, ollama_url, timeout):
        prompts.append(prompt)
        if "For EVERY bug" in prompt:
            codes = re.findall(r"^###\s+(\S+)", prompt, flags=re.M)
            return "\n\n".join(
                f"## {code}\n1. Set up {code}\n2. Verify {code}" for code in codes
            )
        if "Candidate areas:" in prompt:
            return (
                "# Recommended Regression Areas\n"
                "- **Shared Queue** — **High** — Action: Test another queued job. "
                "Rationale: It shares queue state. Evidence: BUG00."
            )
        return "- Candidate adjacent queue check; evidence BUG00"

    monkeypatch.setattr("scripts.summarize_tr_bugs.call_ollama", fake_call)
    report = generate_chunked_report(data, "model", "url", 10, 80, 1500)

    assert len([p for p in prompts if "Evidence batch:" in p]) == 2
    assert len([p for p in prompts if "Candidate areas:" in p]) == 1
    assert validate_report(report, data, 80) == []


def test_report_validation_rejects_non_actionable_regression_summary():
    data = {"products": {"PDR-I": [{"BugCode": "DRI000001-0001"}]}}
    report = (
        "# Per-Bug Verification\n## DRI000001-0001\n1. Test\n"
        "# Recommended Regression Areas\n## Summary of Design Guidelines\n- Tickets fixed"
    )
    errors = validate_report(report, data, 80)
    assert "regression section has no risk-ranked recommendations" in errors
    assert "regression section summarizes inputs instead of recommending tests" in errors


def test_local_fallback_bug_is_explicitly_marked(tmp_path):
    parsed = tmp_path / "ecl_parsed.csv"
    parsed.write_text(
        "BugCode,Status,Close Build#,Short Description,severity_label\n"
        "A,Close,123,Export crashes,2-Major\n",
        encoding="utf-8",
    )
    records = local_fallback_bugs(parsed, ["A"])
    assert records[0]["BugCode"] == "A"
    assert records[0]["DetailFetchError"].startswith("Full eBug status log unavailable")


def test_strip_chunk_wrappers_removes_unsolicited_sections():
    text = (
        "# Per-Bug Verification\n## A\n1. Verify\n\n"
        "# Recommended Regression Areas\n- Ignore this"
    )
    assert strip_chunk_wrappers(text) == "## A\n1. Verify"


def _evidence_from_prompt(prompt):
    match = re.search(r"<BUG_EVIDENCE>\n(.*?)\n</BUG_EVIDENCE>", prompt, re.S)
    return json.loads(match.group(1)) if match else []


def _supported_grounding_audits_from_prompt(prompt):
    match = re.search(
        r"<CANDIDATES_TO_AUDIT>\n(.*?)\n</CANDIDATES_TO_AUDIT>",
        prompt,
        re.S,
    )
    candidates = json.loads(match.group(1)) if match else []
    return {"audits": [{
        "candidate_id": item["candidate_id"],
        **{check: True for check in summarize_tr.GROUNDING_AUDIT_CHECKS},
        "verdict": "supported",
        "reason": "supported",
    } for item in candidates]}


def test_structured_analysis_batches_every_bug_and_applies_risk_rules(monkeypatch):
    data = {
        "products": {"PDR-I": [
            {
                "BugCode": f"BUG{i:02d}", "Short Description": f"Issue {i}",
                "ReproSteps": "Steps",
                "Resolutions": [{
                    "solution": "Updated shared queue scheduler service",
                }],
            }
            for i in range(13)
        ]}
    }
    calls = {"verification": 0, "candidate": 0, "grounding": 0, "synthesis": 0}

    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
            calls["verification"] += 1
            evidence = _evidence_from_prompt(prompt)
            return {"checklists": [
                {"bug_code": item["bug_code"], "steps": ["Set up", "Verify result"]}
                for item in evidence
            ]}
        if "Infer only defensible ADJACENT" in prompt:
            calls["candidate"] += 1
            evidence = _evidence_from_prompt(prompt)
            return {"candidates": [{
                "area": "Shared queue",
                "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
                "trigger": "Run a second queued operation",
                "expected_failure_mode": "The queued operation stalls",
                **_prediction_boundaries("Shared queue"),
                "action": "Run another queued operation",
                "rationale": "The workflow may share the queue scheduler state",
                "source_bug_codes": [evidence[0]["bug_code"]],
                "evidence_refs": [{
                    "bug_code": evidence[0]["bug_code"], "signal_index": 0,
                }],
                "confidence": "high",
            }]}
        if "Adversarially audit every proposed" in prompt:
            calls["grounding"] += 1
            return _supported_grounding_audits_from_prompt(prompt)
        calls["synthesis"] += 1
        candidates = json.loads(re.search(
            r"<CANDIDATES>\n(.*?)\n</CANDIDATES>", prompt, re.S
        ).group(1))
        return {"recommendations": [{
            **candidates[0],
            "risk": "Critical",
            "impact_basis": "functional",
        }]}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    report = render_analysis_markdown(analysis)

    assert calls == {
        "verification": 4, "candidate": 4, "grounding": 2, "synthesis": 1,
    }
    assert analysis["bug_count"] == 13
    assert analysis["recommendations"][0]["risk"] == "Medium"
    assert analysis["recommendations"][0]["confidence"] == "medium"
    assert analysis["recommendations"][0]["target_category"] == (
        summarize_tr.UNKNOWN_TARGET_CATEGORY
    )
    assert re.fullmatch(
        r"TRH-[A-F0-9]{12}", analysis["recommendations"][0]["hypothesis_id"]
    )
    assert analysis["target_category_taxonomy"] == [
        summarize_tr.UNKNOWN_TARGET_CATEGORY
    ]
    assert analysis["checklists"][0]["evidence_quality"] == "high"
    assert analysis["checklists"][0]["evidence_gaps"] == ["missing_root_cause"]
    discovery = analysis["candidate_grounding_audit"][
        "candidate_discovery_audit"
    ]
    assert discovery["primary_candidate_count"] == 2
    assert discovery["coverage_challenge_candidate_count"] == 2
    assert discovery["coverage_challenge_distinct_count"] == 0
    assert discovery["coverage_overlap_filtered_count"] == 2
    assert discovery["grounding_input_candidate_count"] == 2
    assert discovery["challenge_coverage_overlap_can_reach_grounding"] is False
    assert discovery["generation_origin_influences_confidence"] is False
    assert {
        tuple(row["origins"])
        for row in discovery["unique_candidate_origins"]
    } == {("primary_discovery",)}
    assert "**Confidence: Medium**" in report
    assert "Trigger: Run a second queued operation" in report
    assert "Watch for: The queued operation stalls" in report
    assert "Boundary: Limit Shared queue" in report
    assert "remains correct under the stated trigger" in report
    assert "Hypothesis: `TRH-" in report
    assert "Source evidence quality: **High**" in report
    assert validate_report(report, data, 0) == []


def test_schema24_coverage_challenge_recovers_distinct_missed_prediction(
    monkeypatch,
):
    data = {"products": {"P": [{
        "BugCode": "A",
        "Short Description": "Source generation issue",
        "ReproSteps": "Generate and open the source result",
        "Resolutions": [{
            "solution": (
                "Updated Timeline preview and Audio preview shared renderer state"
            ),
        }],
    }]}}
    common = {
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "rationale": "The adjacent preview may use the shared renderer",
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [],
        "confidence": "high",
    }
    primary = {
        **common,
        "area": "Audio preview",
        "trigger": "Open Audio preview after generation",
        "expected_failure_mode": "Audio preview crashes",
        **_prediction_boundaries("Audio preview"),
        "action": "Verify Audio preview after generation",
    }
    missed = {
        **common,
        "area": "Timeline preview",
        "trigger": "Open Timeline preview after generation",
        "expected_failure_mode": "Timeline preview crashes",
        **_prediction_boundaries("Timeline preview"),
        "action": "Verify Timeline preview after generation",
    }
    missed_wording_variant = {
        **missed,
        "action": "Verify after generation the Timeline preview",
    }

    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
            return {"checklists": [{
                "bug_code": "A", "steps": ["Set up", "Verify result"],
            }]}
        if "SECOND independent coverage-challenge" in prompt:
            return {"candidates": [missed, missed_wording_variant]}
        if "Infer only defensible ADJACENT" in prompt:
            return {"candidates": [primary]}
        if "Adversarially audit every proposed" in prompt:
            return _supported_grounding_audits_from_prompt(prompt)
        synthesis_candidates = json.loads(re.search(
            r"<CANDIDATES>\n(.*?)\n</CANDIDATES>", prompt, re.S
        ).group(1))
        return {"recommendations": [{
            **candidate, "risk": "High", "impact_basis": "crash",
        } for candidate in synthesis_candidates]}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    discovery = analysis["candidate_grounding_audit"][
        "candidate_discovery_audit"
    ]

    assert discovery["primary_candidate_count"] == 1
    assert discovery["coverage_challenge_candidate_count"] == 2
    assert discovery["coverage_challenge_distinct_count"] == 1
    assert discovery["coverage_overlap_filtered_count"] == 1
    assert discovery["grounding_input_candidate_count"] == 2
    assert discovery["unique_grounding_input_count"] == 2
    assert {
        item["area"]: item["candidate_generation_origins"]
        for item in analysis["recommendations"]
    } == {
        "Audio preview": ["primary_discovery"],
        "Timeline preview": ["coverage_challenge"],
    }
    rendered = render_analysis_markdown(analysis)
    assert "Target evidence: timeline" in rendered
    assert "Target evidence: audio" in rendered
    assert process_tr.structured_prediction_errors(analysis) == []

    tampered_adjacency = json.loads(json.dumps(analysis))
    tampered_adjacency["recommendations"][0][
        "adjacency_support_basis"
    ]["target_anchor_terms"] = []
    errors = process_tr.structured_prediction_errors(tampered_adjacency)
    assert any("adjacency support provenance mismatch" in error for error in errors)

    tampered_shared_path = json.loads(json.dumps(analysis))
    tampered_shared_path["recommendations"][0][
        "shared_path_support_basis"
    ]["strong_path_matches"] = []
    errors = process_tr.structured_prediction_errors(tampered_shared_path)
    assert any("shared-path support provenance mismatch" in error for error in errors)

    tampered_partition = json.loads(json.dumps(analysis))
    tampered_partition["candidate_grounding_audit"][
        "candidate_discovery_audit"
    ]["coverage_challenge_distinct_candidate_ids"] = []
    errors = process_tr.structured_prediction_errors(tampered_partition)
    assert any("challenge partition mismatch" in error for error in errors)

    tampered_overlap_claim = json.loads(json.dumps(analysis))
    tampered_overlap_claim["candidate_grounding_audit"][
        "candidate_discovery_audit"
    ]["coverage_overlap_filtered_candidates"][0]["candidate"]["action"] = (
        "Verify Timeline preview after import"
    )
    errors = process_tr.structured_prediction_errors(tampered_overlap_claim)
    assert any("coverage-overlap audit is invalid" in error for error in errors)

    tampered_origin = json.loads(json.dumps(analysis))
    timeline = next(
        item for item in tampered_origin["recommendations"]
        if item["area"] == "Timeline preview"
    )
    timeline["candidate_generation_origins"] = ["primary_discovery"]
    errors = process_tr.structured_prediction_errors(tampered_origin)
    assert any("candidate generation origin mismatch" in error for error in errors)

    tampered_influence = json.loads(json.dumps(analysis))
    tampered_influence["candidate_grounding_audit"][
        "candidate_discovery_audit"
    ]["generation_origin_influences_confidence"] = True
    errors = process_tr.structured_prediction_errors(tampered_influence)
    assert any(
        "generation_origin_influences_confidence" in error for error in errors
    )


def test_schema26_withholds_unanchored_target_before_model_critic(monkeypatch):
    data = {"products": {"P": [{
        "BugCode": "A",
        "Short Description": "Source preview issue",
        "ReproSteps": "Open source preview",
        "Resolutions": [{"solution": "Updated shared renderer state"}],
    }]}}
    candidate = {
        "area": "Timeline preview",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Open Timeline preview after generation",
        "expected_failure_mode": "Timeline preview crashes",
        **_prediction_boundaries("Timeline preview"),
        "action": "Verify Timeline preview after generation",
        "rationale": "The preview may use the shared renderer",
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [],
        "confidence": "high",
    }

    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
            return {"checklists": [{
                "bug_code": "A", "steps": ["Set up", "Verify result"],
            }]}
        if "SECOND independent coverage-challenge" in prompt:
            return {"candidates": []}
        if "Infer only defensible ADJACENT" in prompt:
            return {"candidates": [candidate]}
        if "Adversarially audit every proposed" in prompt:
            pytest.fail("the model critic cannot override the deterministic gate")
        pytest.fail("synthesis must not run for a gate-withheld target")

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    discovery = analysis["candidate_grounding_audit"][
        "candidate_discovery_audit"
    ]

    assert analysis["schema_version"] == 38
    assert analysis["recommendations"] == []
    assert discovery["adjacency_gate_evaluated_count"] == 1
    assert discovery["adjacency_gate_supported_count"] == 0
    assert discovery["adjacency_gate_withheld_count"] == 1
    assert discovery["grounding_input_candidate_count"] == 0
    assert discovery["model_critic_can_override_adjacency_gate"] is False
    withheld = discovery["adjacency_gate_withheld_candidates"][0]
    assert withheld["discovery_origin"] == "primary_discovery"
    assert withheld["reason"] == "adjacent_target_absent_from_current_signal"
    assert withheld["adjacency_support_basis"]["target_terms"] == ["timeline"]
    assert withheld["adjacency_support_basis"]["target_anchor_terms"] == []
    assert withheld["adjacency_support_basis"]["supported"] is False
    assert withheld["audit_stage"] == "deterministic_adjacency_gate"
    assert withheld["failed_checks"] == ["target_supported"]
    assert withheld["human_review_sample"] is True
    sampling = analysis["candidate_grounding_audit"]["human_review_sampling"]
    assert sampling["sampled_count"] == 1
    assert sampling["sampled_refs"] == [{
        "audit_stage": "deterministic_adjacency_gate",
        "candidate_id": withheld["candidate_id"],
    }]
    assert process_tr.structured_prediction_errors(analysis) == []

    tampered = json.loads(json.dumps(analysis))
    record = tampered["candidate_grounding_audit"][
        "candidate_discovery_audit"
    ]["adjacency_gate_withheld_candidates"][0]
    record["adjacency_support_basis"]["supported"] = True
    errors = process_tr.structured_prediction_errors(tampered)
    assert any("adjacency support" in error for error in errors)

    acronym_basis = summarize_tr.deterministic_adjacency_support_basis(
        "AI preview", [{
            "bug_code": "A", "signal_index": 0,
            "quote": "Updated AI preview shared renderer state",
        }],
    )
    assert acronym_basis["supported"] is True
    assert acronym_basis["target_anchor_terms"] == ["ai"]
    queue_basis = summarize_tr.deterministic_adjacency_support_basis(
        "Queue operation", [{
            "bug_code": "A", "signal_index": 0,
            "quote": "Updated queued operations in shared service",
        }],
    )
    assert queue_basis["supported"] is True
    collision_basis = summarize_tr.deterministic_adjacency_support_basis(
        "Timeline preview", [{
            "bug_code": "A", "signal_index": 0,
            "quote": "Updated timely preview refresh in shared service",
        }],
    )
    assert collision_basis["supported"] is False


def test_schema29_requires_distinctive_non_target_shared_path_term():
    refs = [{
        "bug_code": "A", "signal_index": 0,
        "quote": "Updated Timeline preview scheduler service state",
    }]
    adjacency = summarize_tr.deterministic_adjacency_support_basis(
        "Timeline preview", refs,
    )

    generic = summarize_tr.deterministic_shared_path_support_basis(
        "Timeline preview",
        "Timeline preview may use the shared service state",
        refs,
        adjacency,
    )
    concrete = summarize_tr.deterministic_shared_path_support_basis(
        "Timeline preview",
        "Timeline preview may use the shared scheduler service",
        refs,
        adjacency,
    )
    plural_generic = summarize_tr.deterministic_shared_path_support_basis(
        "Timeline preview",
        "Timeline preview may use shared services and paths",
        [{
            "bug_code": "A", "signal_index": 0,
            "quote": "Updated Timeline preview services and paths",
        }],
        adjacency,
    )

    assert adjacency["supported"] is True
    assert generic["supported"] is False
    assert generic["strong_path_matches"] == []
    assert {row["rationale_term"] for row in generic["weak_path_matches"]} == {
        "service",
    }
    assert concrete["supported"] is True
    assert concrete["matched_path_terms"] == ["scheduler", "service"]
    assert concrete["strong_path_matches"][0]["rationale_term"] == "scheduler"
    assert concrete["model_critic_can_override"] is False
    assert concrete["historical_analogs_influence_support"] is False
    assert concrete["shared_path_support_influences_confidence"] is False
    assert plural_generic["supported"] is False
    assert plural_generic["strong_path_matches"] == []

    first = {
        "area": "Timeline preview", "rationale": "Uses scheduler path",
        "shared_path_support_basis": concrete,
    }
    second = {**first, "rationale": "Uses scheduler buffer path"}
    assert summarize_tr.stable_hypothesis_id(first) == (
        summarize_tr.stable_hypothesis_id(second)
    )
    assert summarize_tr.stable_candidate_audit_id(first) != (
        summarize_tr.stable_candidate_audit_id(second)
    )


def test_schema29_withholds_generic_shared_path_before_model_critic(monkeypatch):
    data = {"products": {"P": [{
        "BugCode": "A",
        "Short Description": "Timeline preview issue",
        "ReproSteps": "Open Timeline preview",
        "Resolutions": [{
            "solution": "Updated Timeline preview shared service state",
        }],
    }]}}
    candidate = {
        "area": "Timeline preview",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Open Timeline preview after generation",
        "expected_failure_mode": "Timeline preview crashes",
        **_prediction_boundaries("Timeline preview"),
        "action": "Verify Timeline preview after generation",
        "rationale": "Timeline preview may use the shared service state",
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [],
        "confidence": "high",
    }

    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
            return {"checklists": [{
                "bug_code": "A", "steps": ["Set up", "Verify result"],
            }]}
        if "SECOND independent coverage-challenge" in prompt:
            return {"candidates": []}
        if "Infer only defensible ADJACENT" in prompt:
            return {"candidates": [candidate]}
        if "Adversarially audit every proposed" in prompt:
            pytest.fail("the model critic cannot override the shared-path gate")
        pytest.fail("synthesis must not run for a path-withheld candidate")

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    discovery = analysis["candidate_grounding_audit"][
        "candidate_discovery_audit"
    ]

    assert analysis["recommendations"] == []
    assert discovery["adjacency_gate_supported_count"] == 0
    assert discovery["shared_path_gate_supported_count"] == 0
    assert discovery["shared_path_gate_withheld_count"] == 1
    assert discovery["model_critic_can_override_shared_path_gate"] is False
    withheld = discovery["adjacency_gate_withheld_candidates"][0]
    assert withheld["reason"] == "shared_path_not_concrete"
    assert withheld["failed_checks"] == ["shared_path_concrete"]
    assert withheld["candidate"]["adjacency_support_basis"]["supported"] is True
    path_basis = withheld["candidate"]["shared_path_support_basis"]
    assert path_basis["supported"] is False
    assert path_basis["strong_path_matches"] == []
    assert withheld["human_review_sample"] is True
    assert process_tr.structured_prediction_errors(analysis) == []

    tampered = json.loads(json.dumps(analysis))
    tampered_basis = tampered["candidate_grounding_audit"][
        "candidate_discovery_audit"
    ]["adjacency_gate_withheld_candidates"][0]["candidate"][
        "shared_path_support_basis"
    ]
    tampered_basis["supported"] = True
    errors = process_tr.structured_prediction_errors(tampered)
    assert any("shared-path support provenance mismatch" in error for error in errors)


def test_schema26_completes_missed_target_reference_before_gate(monkeypatch):
    data = {"products": {"P": [{
        "BugCode": "A",
        "Short Description": "Source preview issue",
        "ReproSteps": "Open source preview",
        "Resolutions": [
            {"solution": "Updated shared renderer state"},
            {"solution": "Updated Timeline manager refresh path"},
        ],
    }]}}
    candidate = {
        "area": "Timeline preview",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Open Timeline preview after generation",
        "expected_failure_mode": "Timeline preview crashes",
        **_prediction_boundaries("Timeline preview"),
        "action": "Verify Timeline preview after generation",
        "rationale": "Timeline preview may use the shared renderer",
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [],
        "confidence": "high",
    }

    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
            return {"checklists": [{
                "bug_code": "A", "steps": ["Set up", "Verify result"],
            }]}
        if "SECOND independent coverage-challenge" in prompt:
            return {"candidates": []}
        if "Infer only defensible ADJACENT" in prompt:
            return {"candidates": [candidate]}
        if "Adversarially audit every proposed" in prompt:
            return _supported_grounding_audits_from_prompt(prompt)
        synthesis_candidates = json.loads(re.search(
            r"<CANDIDATES>\n(.*?)\n</CANDIDATES>", prompt, re.S
        ).group(1))
        return {"recommendations": [{
            **synthesis_candidates[0],
            "risk": "High",
            "impact_basis": "functional",
        }]}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    discovery = analysis["candidate_grounding_audit"][
        "candidate_discovery_audit"
    ]

    assert analysis["schema_version"] == 38
    assert len(analysis["recommendations"]) == 1
    assert discovery["adjacency_gate_supported_count"] == 1
    assert discovery["adjacency_gate_withheld_count"] == 0
    assert discovery["evidence_ref_completion_applied_count"] == 1
    completed = discovery["evidence_ref_completion_candidates"][0]
    assert completed["original_evidence_refs"][0]["signal_index"] == 0
    assert completed["added_evidence_refs"][0]["signal_index"] == 1
    assert "Timeline manager" in completed["added_evidence_refs"][0]["quote"]
    supported_candidate = discovery["adjacency_gate_supported_candidates"][0][
        "candidate"
    ]
    completion = supported_candidate["evidence_ref_completion_basis"]
    assert completion["status"] == "completed_from_candidate_bug_signal"
    assert completion["initial_target_supported"] is False
    assert completion["final_target_supported"] is True
    assert supported_candidate["adjacency_support_basis"]["supported"] is True
    assert process_tr.structured_prediction_errors(analysis) == []

    evidence_map = {
        "A": {"adjacency_signal_quotes": [
            "Updated shared renderer state",
            "Updated Timeline manager refresh path",
        ]},
        "B": {"adjacency_signal_quotes": [
            "Updated Timeline manager in another bug",
        ]},
    }
    limited_refs, limited = summarize_tr.deterministic_evidence_ref_completion(
        "Timeline preview", ["A"], [{
            "bug_code": "A", "signal_index": 0,
            "quote": "Updated shared renderer state",
        }], evidence_map, max_refs=1,
    )
    assert len(limited_refs) == 1
    assert limited["status"] == "reference_limit_reached"
    assert limited["final_target_supported"] is False
    assert all(
        ref["bug_code"] == "A"
        for ref in limited["matching_uncited_evidence_refs"]
    )

    tampered = json.loads(json.dumps(analysis))
    tampered_completion = tampered["candidate_grounding_audit"][
        "candidate_discovery_audit"
    ]["adjacency_gate_supported_candidates"][0]["candidate"][
        "evidence_ref_completion_basis"
    ]
    tampered_completion["added_evidence_refs"][0]["quote"] = "invented"
    errors = process_tr.structured_prediction_errors(tampered)
    assert any("reference completion provenance" in error for error in errors)


def test_schema27_reviewed_directional_alias_supports_gate_but_not_confidence(
    monkeypatch,
):
    alias_policy = prediction_reviews.compact_target_alias_policy({
        "active_aliases": {"captions": ["subtitle"]},
    })
    data = {
        "meta": {"target_alias_policy": alias_policy},
        "products": {"P": [
            {
                "BugCode": "A", "Short Description": "Source issue A",
                "ReproSteps": "Open source A",
                "Resolutions": [{
                    "solution": "Updated subtitle renderer service",
                }],
            },
            {
                "BugCode": "B", "Short Description": "Source issue B",
                "ReproSteps": "Open source B",
                "Resolutions": [{
                    "solution": "Changed subtitle cache manager",
                }],
            },
        ]},
    }
    candidate = {
        "area": "Captions editor",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Open Captions editor after generation",
        "expected_failure_mode": "Captions editor shows missing output",
        **_prediction_boundaries("Captions editor"),
        "action": "Verify Captions editor after generation",
        "rationale": "Captions may use the changed subtitle renderer services",
        "source_bug_codes": ["A", "B"],
        "evidence_refs": [
            {"bug_code": "A", "signal_index": 0},
            {"bug_code": "B", "signal_index": 0},
        ],
        "historical_analog_refs": [],
        "confidence": "high",
    }

    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
            evidence = _evidence_from_prompt(prompt)
            return {"checklists": [{
                "bug_code": item["bug_code"],
                "steps": ["Set up source", "Verify the fixed result"],
            } for item in evidence]}
        if "SECOND independent coverage-challenge" in prompt:
            return {"candidates": []}
        if "Infer only defensible ADJACENT" in prompt:
            return {"candidates": [candidate]}
        if "Adversarially audit every proposed" in prompt:
            return _supported_grounding_audits_from_prompt(prompt)
        synthesis_candidates = json.loads(re.search(
            r"<CANDIDATES>\n(.*?)\n</CANDIDATES>", prompt, re.S
        ).group(1))
        return {"recommendations": [{
            **synthesis_candidates[0],
            "risk": "High", "impact_basis": "functional",
        }]}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    discovery = analysis["candidate_grounding_audit"][
        "candidate_discovery_audit"
    ]

    assert analysis["schema_version"] == 38
    assert analysis["target_alias_policy"] == alias_policy
    assert len(analysis["recommendations"]) == 1
    assert discovery["reviewed_alias_supported_count"] == 1
    assert discovery["reviewed_aliases_influence_confidence"] is False
    supported = discovery["adjacency_gate_supported_candidates"][0][
        "candidate"
    ]
    basis = supported["adjacency_support_basis"]
    assert basis["reviewed_aliases_influence_support"] is True
    assert basis["reviewed_alias_matches"][0]["target_term"] == "captions"
    assert basis["reviewed_alias_matches"][0][
        "evidence_alias_term"
    ] == "subtitle"
    assert supported["confidence"] == "medium"
    assert supported["confidence_basis"][
        "evidence_label_before_target_cap"
    ] == "high"
    assert supported["confidence_basis"]["target_support_cap"] == "medium"
    assert process_tr.structured_prediction_errors(analysis) == []

    reverse = summarize_tr.deterministic_adjacency_support_basis(
        "Subtitle editor", [{
            "bug_code": "A", "signal_index": 0,
            "quote": "Updated captions renderer service",
        }], alias_policy,
    )
    assert reverse["supported"] is False

    tampered = json.loads(json.dumps(analysis))
    tampered["target_alias_policy"]["active_aliases"] = {
        "captions": ["voice"]
    }
    errors = process_tr.structured_prediction_errors(tampered)
    assert any("adjacency support provenance" in error for error in errors)


def test_structured_analysis_retries_truncated_or_invalid_batch(monkeypatch):
    data = {"products": {"PDR-I": [{"BugCode": "A", "Short Description": "Issue"}]}}
    attempts = {"verification": 0}

    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
            attempts["verification"] += 1
            if attempts["verification"] == 1:
                raise summarize_tr.AnalysisError("Ollama output reached its token limit")
            return {"checklists": [{"bug_code": "A", "steps": ["Set up", "Verify"]}]}
        return {"candidates": []}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    assert attempts["verification"] == 2
    assert analysis["recommendations"] == []
    assert analysis["checklists"][0]["steps"][0].startswith(
        "Confirm the missing reproduction context, root cause, implemented fix "
        "and expected behavior with the bug owner"
    )
    assert summarize_tr.NO_GROUNDED_REGRESSION in render_analysis_markdown(analysis)


def test_summary_only_evidence_caps_confidence_and_risk():
    evidence_map = {"A": {"evidence_quality": "low", "adjacency_signal_quotes": ["shared"]}}
    recommendations, errors = summarize_tr._validate_recommendations({
        "recommendations": [{
            "area": "Area", "action": "Verify adjacent behavior", "rationale": "Why",
            "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
            "trigger": "Use the adjacent state",
            "expected_failure_mode": "The adjacent state crashes",
            **_prediction_boundaries("Area"),
            "source_bug_codes": ["A"], "confidence": "high",
            "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
            "risk": "Critical", "impact_basis": "crash",
        }]
    }, ["A"], evidence_map)
    assert errors == []
    assert recommendations[0]["confidence"] == "low"
    assert recommendations[0]["risk"] == "Medium"


def test_deterministic_confidence_requires_independent_current_support():
    refs = [{"bug_code": "A", "signal_index": 0}]
    one_high = summarize_tr.deterministic_confidence_basis(
        ["A"], refs, [], {"A": {"evidence_quality": "high"}}
    )
    assert one_high["label"] == "medium"
    assert one_high["rule"] == "single_or_mixed_high_quality_current_source"

    high_and_medium = summarize_tr.deterministic_confidence_basis(
        ["A", "B"],
        refs + [{"bug_code": "B", "signal_index": 0}],
        [],
        {
            "A": {"evidence_quality": "high"},
            "B": {"evidence_quality": "medium"},
        },
    )
    assert high_and_medium["label"] == "high"
    assert high_and_medium["source_bug_count"] == 2

    one_medium_with_analogs = summarize_tr.deterministic_confidence_basis(
        ["A"], refs,
        [{"source_bug_code": "A", "analog_bug_code": f"OLD-{index}"}
         for index in range(3)],
        {"A": {"evidence_quality": "medium"}},
    )
    assert one_medium_with_analogs["label"] == "low"
    assert one_medium_with_analogs["historical_analog_ref_count"] == 3
    assert one_medium_with_analogs[
        "historical_analogs_influence_label"
    ] is False

    correlated_map = {
        "A": {
            "evidence_quality": "high", "parsed_module": "Timeline",
            "adjacency_signal_quotes": ["Updated shared renderer state"],
        },
        "B": {
            "evidence_quality": "medium", "parsed_module": "Timeline",
            "adjacency_signal_quotes": ["Updated shared renderer state"],
        },
    }
    correlated = summarize_tr.deterministic_confidence_basis(
        ["A", "B"],
        [
            {"bug_code": "A", "signal_index": 0,
             "quote": "Updated shared renderer state"},
            {"bug_code": "B", "signal_index": 0,
             "quote": "Updated shared renderer state"},
        ],
        [], correlated_map, correlate_sources=True,
    )
    assert correlated["label"] == "medium"
    assert correlated["rule"] == "correlated_sources_do_not_raise_high"
    assert correlated["independent_source_group_count"] == 1
    assert correlated["independence_groups"] == [["A", "B"]]
    assert correlated["correlation_links"][0]["reason"] == (
        "identical_cited_signal_terms"
    )

    independent_map = json.loads(json.dumps(correlated_map))
    independent_map["B"]["adjacency_signal_quotes"] = [
        "Export queue blocks another operation"
    ]
    independent = summarize_tr.deterministic_confidence_basis(
        ["A", "B"],
        [
            {"bug_code": "A", "signal_index": 0,
             "quote": "Updated shared renderer state"},
            {"bug_code": "B", "signal_index": 0,
             "quote": "Export queue blocks another operation"},
        ],
        [], independent_map, correlate_sources=True,
    )
    assert independent["label"] == "high"
    assert independent["independent_source_group_count"] == 2
    assert independent["independence_groups"] == [["A"], ["B"]]

    near_duplicate_map = json.loads(json.dumps(correlated_map))
    near_duplicate_map["A"]["adjacency_signal_quotes"] = [
        "Updated renderer queue state"
    ]
    near_duplicate_map["B"]["adjacency_signal_quotes"] = [
        "Updated renderer queue lifecycle"
    ]
    near_duplicate_refs = [
        {"bug_code": "A", "signal_index": 0,
         "quote": "Updated renderer queue state"},
        {"bug_code": "B", "signal_index": 0,
         "quote": "Updated renderer queue lifecycle"},
    ]
    near_duplicate = summarize_tr.deterministic_confidence_basis(
        ["B", "A"], near_duplicate_refs, [], near_duplicate_map,
        correlate_sources=True,
    )
    assert near_duplicate["independent_source_group_count"] == 1
    assert near_duplicate["correlation_links"][0]["reason"] == (
        "same_module_near_duplicate_signal"
    )
    assert near_duplicate["correlation_links"][0]["signal_jaccard"] == 0.6

    near_duplicate_map["B"]["parsed_module"] = "Export"
    different_modules = summarize_tr.deterministic_confidence_basis(
        ["A", "B"], near_duplicate_refs, [], near_duplicate_map,
        correlate_sources=True,
    )
    assert different_modules["independent_source_group_count"] == 2
    assert different_modules["correlation_links"] == []

    target_map = {
        "A": {
            "evidence_quality": "high", "parsed_module": "Export",
            "adjacency_signal_quotes": [
                "Timeline export shares queue behavior"
            ],
        },
        "B": {
            "evidence_quality": "medium", "parsed_module": "Playback",
            "adjacency_signal_quotes": [
                "Timeline playback uses common state"
            ],
        },
    }
    target_refs = [
        {"bug_code": "A", "signal_index": 0,
         "quote": "Timeline export shares queue behavior"},
        {"bug_code": "B", "signal_index": 0,
         "quote": "Timeline playback uses common state"},
    ]
    anchored_target = summarize_tr.deterministic_confidence_basis(
        ["A", "B"], target_refs, [], target_map,
        correlate_sources=True, target_area="Timeline preview",
    )
    assert anchored_target["label"] == "high"
    assert anchored_target["target_support_cap"] == "none"
    assert anchored_target["target_anchor_source_count"] == 2
    assert anchored_target["target_anchor_terms"] == ["timeline"]

    unanchored_target = summarize_tr.deterministic_confidence_basis(
        ["B", "A"], target_refs, [], target_map,
        correlate_sources=True, target_area="Caption editor",
    )
    assert unanchored_target["evidence_label_before_target_cap"] == "high"
    assert unanchored_target["label"] == "medium"
    assert unanchored_target["rule"] == "unanchored_target_caps_high"
    assert unanchored_target["target_support_cap"] == "medium"
    assert unanchored_target["target_anchor_source_count"] == 0
    assert unanchored_target["target_anchor_terms_by_bug"] == {
        "A": [], "B": [],
    }
    capped_risk = summarize_tr.deterministic_risk_basis(
        "Saving corrupts project data", unanchored_target["label"]
    )
    assert capped_risk["label"] == "High"
    assert capped_risk["confidence_cap"] == "critical_requires_high_confidence"


def test_deterministic_risk_uses_observable_impact_and_confidence_caps():
    data_loss = summarize_tr.deterministic_risk_basis(
        "Saving corrupts project data", "high"
    )
    assert data_loss["label"] == "Critical"
    assert data_loss["impact_basis"] == "data_loss"
    assert data_loss["confidence_cap"] == "none"

    capped_data_loss = summarize_tr.deterministic_risk_basis(
        "Saving corrupts project data", "medium"
    )
    assert capped_data_loss["label"] == "High"
    assert capped_data_loss["confidence_cap"] == (
        "critical_requires_high_confidence"
    )

    low_confidence_crash = summarize_tr.deterministic_risk_basis(
        "Timeline preview crashes", "low"
    )
    assert low_confidence_crash["uncapped_label"] == "High"
    assert low_confidence_crash["label"] == "Medium"
    assert low_confidence_crash["confidence_cap"] == (
        "low_confidence_maximum_medium"
    )

    presentation = summarize_tr.deterministic_risk_basis(
        "Caption alignment flickers", "high"
    )
    assert presentation["label"] == "Low"
    assert presentation["impact_basis"] == "presentation"

    functional = summarize_tr.deterministic_risk_basis(
        "Queued operation stalls", "high"
    )
    assert functional["label"] == "Medium"
    assert functional["impact_basis"] == "functional"

    negated_crash = summarize_tr.deterministic_risk_basis(
        "The preview does not crash but the layout flickers", "high",
        negation_aware=True,
    )
    assert negated_crash["label"] == "Low"
    assert negated_crash["impact_basis"] == "presentation"
    assert negated_crash["ignored_negated_signals"] == [{
        "impact_basis": "crash", "signal": "crash", "negation": "does not",
    }]

    negated_data_loss = summarize_tr.deterministic_risk_basis(
        "No data loss occurs, but export produces the wrong file", "high",
        negation_aware=True,
    )
    assert negated_data_loss["label"] == "Medium"
    assert negated_data_loss["impact_basis"] == "functional"
    assert negated_data_loss["ignored_negated_signals"][0][
        "impact_basis"
    ] == "data_loss"

    separate_clause_harm = summarize_tr.deterministic_risk_basis(
        "The preview does not crash, but data loss occurs", "high",
        negation_aware=True,
    )
    assert separate_clause_harm["label"] == "Critical"
    assert separate_clause_harm["impact_basis"] == "data_loss"
    assert separate_clause_harm["ignored_negated_signals"][0][
        "impact_basis"
    ] == "crash"

    postposed_negation = summarize_tr.deterministic_risk_basis(
        "A security breach does not occur; the privacy screen flickers",
        "high", negation_aware=True,
    )
    assert postposed_negation["label"] == "Low"
    assert postposed_negation["impact_basis"] == "presentation"
    assert postposed_negation["ignored_negated_signals"] == [{
        "impact_basis": "security_privacy",
        "signal": "security breach",
        "negation": "does not occur",
    }]

    rendered = render_analysis_markdown({
        "recommendations": [{
            "area": "Privacy preview",
            "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
            "risk": postposed_negation["label"],
            "risk_basis": postposed_negation,
            "confidence": "high",
            "trigger": "Open the privacy preview",
            "expected_failure_mode": (
                "A security breach does not occur; the privacy screen flickers"
            ),
            "scope_boundary": "Only the privacy preview is in scope",
            "disconfirming_observation": "The preview remains stable",
            "action": "Verify the privacy preview",
            "rationale": "The preview state changed",
            "source_bug_codes": ["A"],
            "historical_analog_refs": [],
            "hypothesis_id": "TRH-TEST",
        }],
        "checklists": [],
    })
    assert "ignored negated signals: security breach (does not occur)" in rendered

    subscription_layout = summarize_tr.deterministic_risk_basis(
        "The subscription screen alignment is wrong", "high",
        negation_aware=True,
    )
    assert subscription_layout["label"] == "Low"
    assert subscription_layout["impact_basis"] == "presentation"

    missing_payment_button = summarize_tr.deterministic_risk_basis(
        "The payment button is missing", "high", negation_aware=True,
    )
    assert missing_payment_button["label"] == "Medium"
    assert missing_payment_button["impact_basis"] == "functional"

    duplicate_charge = summarize_tr.deterministic_risk_basis(
        "The customer is charged twice", "high", negation_aware=True,
    )
    assert duplicate_charge["label"] == "Critical"
    assert duplicate_charge["impact_basis"] == "billing"

    privacy_layout = summarize_tr.deterministic_risk_basis(
        "The privacy screen alignment is wrong", "high", negation_aware=True,
    )
    assert privacy_layout["label"] == "Low"
    assert privacy_layout["impact_basis"] == "presentation"

    unauthorized_access = summarize_tr.deterministic_risk_basis(
        "The account allows unauthorized access", "high",
        negation_aware=True,
    )
    assert unauthorized_access["label"] == "Critical"
    assert unauthorized_access["impact_basis"] == "security_privacy"


def test_schema30_requires_current_evidence_before_severe_harm_can_rank_up():
    recommendation = {
        "expected_failure_mode": "Saving corrupts project data",
        "confidence": "high",
        "source_bug_codes": ["A"],
        "evidence_refs": [{
            "bug_code": "A", "signal_index": 0,
            "quote": "Updated Timeline preview shared renderer state",
        }],
    }
    unsupported = summarize_tr.deterministic_recommendation_risk_basis(
        recommendation,
        {"A": {
            "summary": "Timeline preview shows the wrong frame",
            "repro_or_description": "Open Timeline preview",
        }},
    )

    assert unsupported["schema_version"] == 3
    assert unsupported["impact_basis"] == "data_loss"
    assert unsupported["uncapped_label"] == "Critical"
    assert unsupported["label"] == "Medium"
    assert unsupported["harm_evidence_required"] is True
    assert unsupported["harm_evidence_supported"] is False
    assert unsupported["harm_evidence_matches"] == []
    assert unsupported["harm_evidence_cap"] == (
        "unsupported_predicted_harm_maximum_medium"
    )
    assert unsupported["current_evidence_can_only_cap_risk"] is True
    assert unsupported["historical_analogs_influence_harm_support"] is False
    assert unsupported["model_critic_can_override_harm_support"] is False

    supported = summarize_tr.deterministic_recommendation_risk_basis(
        recommendation,
        {"A": {
            "summary": "Saving corrupts project data",
            "repro_or_description": "Open Timeline preview",
        }},
    )
    assert supported["label"] == "Critical"
    assert supported["harm_evidence_supported"] is True
    assert supported["harm_evidence_cap"] == "none"
    assert supported["harm_evidence_matches"] == [{
        "bug_code": "A",
        "evidence_field": "summary",
        "signal": "corrupts project",
    }]

    negated = summarize_tr.deterministic_recommendation_risk_basis(
        recommendation,
        {"A": {
            "summary": "Saving causes no data loss",
            "repro_or_description": "Open Timeline preview",
        }},
    )
    assert negated["label"] == "Medium"
    assert negated["harm_evidence_supported"] is False

    functional = summarize_tr.deterministic_recommendation_risk_basis(
        {
            **recommendation,
            "expected_failure_mode": "The queued operation stalls",
        },
        {"A": {"summary": "Timeline preview issue"}},
    )
    assert functional["label"] == "Medium"
    assert functional["harm_evidence_required"] is False
    assert functional["harm_evidence_supported"] is True


def test_schema32_sensitive_scenario_details_require_current_bug_evidence():
    evidence_map = {
        "A": {
            "summary": "Background HDR preview issue on iPhone 16 with iOS 18",
            "repro_or_description": 'Reconnect in en-US and tap "Done"',
            "locale_or_os": "iPhone 16, iOS 18, en-US",
            "changes": [{
                "solution": "Updated the HDR renderer after reconnect",
            }],
            "qa_verify_notes": [],
        },
        "B": {
            "summary": "Offline preview issue on iPhone 17 with iOS 19",
            "repro_or_description": "Open another source",
            "locale_or_os": "iPhone 17, iOS 19",
            "changes": [],
            "qa_verify_notes": [],
        },
    }
    supported = summarize_tr.deterministic_scenario_specificity_basis(
        "Reconnect the background preview on iPhone 16 with iOS 18 in en-US",
        'The HDR renderer fails after showing "Done"',
        ["A"],
        [{
            "bug_code": "A", "signal_index": 0,
            "quote": "Updated the HDR renderer after reconnect",
        }],
        evidence_map,
    )
    unsupported = summarize_tr.deterministic_scenario_specificity_basis(
        "Open the offline preview on iPhone 17 with iOS 19",
        "The renderer fails",
        ["A"],
        [{
            "bug_code": "A", "signal_index": 0,
            "quote": "Updated the HDR renderer after reconnect",
        }],
        evidence_map,
    )

    assert supported["supported"] is True
    assert supported["unsupported_specificity"] == []
    assert {
        (row["claim_field"], row["specificity_kind"], row["term"])
        for row in supported["claim_specificity"]
    } >= {
        ("trigger", "numeric_or_version", "16"),
        ("trigger", "numeric_or_version", "18"),
        ("trigger", "locale", "en-us"),
        ("trigger", "protected_term", "background"),
        ("trigger", "protected_term", "iphone"),
        ("expected_failure_mode", "quoted_literal", "Done"),
        ("expected_failure_mode", "protected_term", "hdr"),
        ("expected_failure_mode", "protected_term", "renderer"),
    }
    assert unsupported["supported"] is False
    assert {
        (row["specificity_kind"], row["term"])
        for row in unsupported["unsupported_specificity"]
    } >= {
        ("numeric_or_version", "17"),
        ("numeric_or_version", "19"),
        ("protected_term", "offline"),
    }
    assert unsupported["model_critic_can_override"] is False
    assert unsupported["historical_analogs_influence_support"] is False
    assert unsupported["reviewed_feedback_influences_support"] is False


def test_schema32_corrects_invented_exact_scenario_before_model_critic(
    monkeypatch,
):
    data = {"products": {"P": [{
        "BugCode": "A",
        "Short Description": "Timeline preview issue",
        "ReproSteps": "Open Timeline preview after generation",
        "Resolutions": [{
            "solution": "Updated Timeline preview shared renderer state",
        }],
    }]}}
    candidate_calls = 0
    correction_prompts = []

    def candidate(trigger):
        return {
            "area": "Timeline preview",
            "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
            "trigger": trigger,
            "expected_failure_mode": "Timeline preview shows wrong renderer state",
            **_prediction_boundaries("Timeline preview"),
            "action": "Verify Timeline preview after generation",
            "rationale": "Timeline preview may use the shared renderer state",
            "source_bug_codes": ["A"],
            "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
            "historical_analog_refs": [],
            "confidence": "high",
        }

    def fake_json(prompt, model, url, timeout, **kwargs):
        nonlocal candidate_calls
        if "direct fix-verification" in prompt:
            return {"checklists": [{
                "bug_code": "A", "steps": ["Set up", "Verify result"],
            }]}
        if "SECOND independent coverage-challenge" in prompt:
            return {"candidates": []}
        if "Infer only defensible ADJACENT" in prompt:
            candidate_calls += 1
            correction_prompts.append(prompt)
            return {"candidates": [candidate(
                "Open Timeline preview on iPhone 17"
                if candidate_calls == 1
                else "Open Timeline preview after generation"
            )]}
        if "Adversarially audit every proposed" in prompt:
            return _supported_grounding_audits_from_prompt(prompt)
        synthesis_candidates = json.loads(re.search(
            r"<CANDIDATES>\n(.*?)\n</CANDIDATES>", prompt, re.S
        ).group(1))
        return {"recommendations": [{
            **synthesis_candidates[0],
            "risk": "Medium", "impact_basis": "functional",
        }]}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)

    assert candidate_calls == 2
    assert "numeric_or_version detail '17'" in correction_prompts[1]
    assert "protected_term detail 'iphone'" in correction_prompts[1]
    assert len(analysis["recommendations"]) == 1
    basis = analysis["recommendations"][0]["scenario_specificity_basis"]
    assert basis["supported"] is True
    assert basis["unsupported_specificity"] == []
    assert process_tr.structured_prediction_errors(analysis) == []
    rendered = render_analysis_markdown(analysis)
    assert "Exact scenario evidence:" in rendered
    assert "renderer (expected failure mode ← A changes" in rendered
    assert "A adjacency\\_signal\\_quotes" in rendered

    tampered = json.loads(json.dumps(analysis))
    tampered["recommendations"][0]["scenario_specificity_basis"][
        "supported"
    ] = False
    errors = process_tr.structured_prediction_errors(tampered)
    assert any("scenario specificity provenance mismatch" in error for error in errors)


@pytest.mark.parametrize("validator,key", [
    (summarize_tr._validate_candidates, "candidates"),
    (summarize_tr._validate_recommendations, "recommendations"),
])
def test_instruction_gate_rejects_cross_scenario_action(validator, key):
    evidence = {"A": {
        "summary": "Compare table shows 500 instead of 100 monthly free credits",
        "evidence_quality": "high",
        "adjacency_signal_quotes": ["Refresh the API for all IAP pages with credits"],
    }}
    item = {
        "area": "IAP pages with credits",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Open the IAP pages with credits compare table",
        "expected_failure_mode": "Compare table shows 500 instead of 100 monthly free credits",
        **_prediction_boundaries("IAP pages with credits"),
        "disconfirming_observation": "Lower priority if IAP pages with credits display 100 monthly free credits",
        "action": "Verify IAP pages with credits display 1200 Credits/year",
        "rationale": "IAP pages with credits use the API refresh path",
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [],
        "confidence": "medium", "risk": "Medium", "impact_basis": "functional",
    }
    values, errors = validator({key: [item]}, ["A"], evidence)
    assert values == []
    assert any("action invents numeric_or_version detail '1200'" in error for error in errors)
    assert any("action conflicts" in error for error in errors)
    item["action"] = "Verify IAP pages with credits display 100 monthly free credits"
    values, errors = validator({key: [item]}, ["A"], evidence)
    assert errors == []
    assert len(values) == 1


@pytest.mark.parametrize("action,conflict", [
    ("Verify 1200 Credits/year", True),
    ("Verify 100 Credits/year", True),
    ("Verify 100 Credits/month", False),
    ("Verify 100 monthly free credits after selecting the yearly plan", False),
    ("Verify the display does not show 500 credits", False),
])
def test_credit_instruction_consistency(action, conflict):
    errors = summarize_tr.prediction_instruction_consistency_errors(
        "Compare table shows 500 instead of 100 monthly free credits", action,
        "Lower priority if the compare table displays 100 monthly free credits",
    )
    assert bool(errors) is conflict


def test_instruction_specificity_preserves_legacy_audit():
    args = ("Open IAP", "Wrong credits", ["A"], [], {"A": {"summary": "100 credits"}})
    legacy = summarize_tr.deterministic_scenario_specificity_basis(*args)
    current = summarize_tr.deterministic_scenario_specificity_basis(
        *args, action="Verify 100 credits",
        disconfirming_observation="Lower priority if 1200 credits display",
    )
    assert legacy["schema_version"] == 1
    assert "instruction_consistency_errors" not in legacy
    assert current["schema_version"] == 2
    assert current["supported"] is False
    assert any(row["claim_field"] == "disconfirming_observation"
               for row in current["unsupported_specificity"])


def test_adjacent_prediction_requires_exact_evidence_reference():
    evidence_map = {
        "A": {
            "evidence_quality": "high",
            "adjacency_signal_quotes": ["Updated the shared renderer path"],
        }
    }
    base = {
        "area": "Preview rendering",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Open another renderer-backed preview",
        "expected_failure_mode": "The adjacent preview renders incorrectly",
        **_prediction_boundaries("Preview rendering"),
        "action": "Exercise another renderer-backed preview",
        "rationale": "The fix changed a shared renderer path",
        "source_bug_codes": ["A"],
        "confidence": "high",
    }
    candidates, errors = summarize_tr._validate_candidates({
        "candidates": [{
            **base,
            "evidence_refs": [{"bug_code": "A", "signal_index": 1}],
        }]
    }, ["A"], evidence_map)
    assert candidates == []
    assert any("invalid signal_index" in error for error in errors)

    candidates, errors = summarize_tr._validate_candidates({
        "candidates": [{
            **base,
            "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        }]
    }, ["A"], evidence_map)
    assert errors == []
    assert candidates[0]["evidence_refs"] == [{
        "bug_code": "A",
        "signal_index": 0,
        "quote": "Updated the shared renderer path",
    }]
    assert candidates[0]["confidence"] == "medium"
    assert candidates[0]["confidence_basis"]["rule"] == (
        "single_or_mixed_high_quality_current_source"
    )

    low_requested, errors = summarize_tr._validate_candidates({
        "candidates": [{
            **base,
            "confidence": "low",
            "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        }]
    }, ["A"], evidence_map)
    assert errors == []
    assert low_requested[0]["confidence"] == "medium"
    assert low_requested[0]["confidence_basis"] == candidates[0][
        "confidence_basis"
    ]

    candidates, errors = summarize_tr._validate_candidates({
        "candidates": [{
            **base,
            "target_category": "Invented category",
            "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        }]
    }, ["A"], evidence_map, ["AI Features"])
    assert candidates == []
    assert any("invalid target_category" in error for error in errors)

    candidates, errors = summarize_tr._validate_candidates({
        "candidates": [{
            **base,
            "action": "Reset the internal renderer lifecycle",
            "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        }]
    }, ["A"], evidence_map)
    assert candidates == []
    assert any("not QA-executable" in error for error in errors)

    candidates, errors = summarize_tr._validate_candidates({
        "candidates": [{
                **base,
                "area": "Editor preview",
                **_prediction_boundaries("Editor preview"),
                "trigger": "Open its preview after generation",
            "action": "Generate an image and open its preview",
            "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        }]
    }, ["A"], evidence_map)
    assert errors == []
    assert candidates[0]["trigger"].startswith("In Editor preview,")
    assert candidates[0]["action"].startswith("Verify Editor preview:")


def test_historical_analog_citations_are_exact_and_only_corroborative():
    evidence_map = {
        "A": {
            "evidence_quality": "high",
            "adjacency_signal_quotes": ["Updated the shared renderer path"],
            "historical_analogs": [
                {
                    "bug_code": "H1",
                    "module": "Timeline",
                    "module_category": "Editor Core",
                    "summary": "Timeline preview crashed in the renderer path",
                    "relationship": "same_category_adjacent_module",
                    "similarity_score": 0.61,
                    "overlap_terms": ["preview", "renderer"],
                },
                {
                    "bug_code": "H2",
                    "module": "Produce",
                    "module_category": "Editor Core",
                    "summary": "Timeline preview crashed after generation",
                    "relationship": "same_category_adjacent_module",
                    "similarity_score": 0.52,
                    "overlap_terms": ["preview", "generation"],
                },
                {
                    "bug_code": "H3",
                    "module": "Audio Mixer",
                    "module_category": "Editor Core",
                    "summary": "Music volume slider used the wrong color",
                    "relationship": "same_category_adjacent_module",
                    "similarity_score": 0.41,
                    "overlap_terms": ["color", "slider"],
                },
            ],
        }
    }
    candidate = {
        "area": "Timeline preview",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Open Timeline preview after generation",
        "expected_failure_mode": "Timeline preview crashes",
        **_prediction_boundaries("Timeline preview"),
        "action": "Exercise Timeline preview after generation",
        "rationale": "The fix changed a shared renderer path",
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [{
            "source_bug_code": "A", "analog_bug_code": "H1",
        }],
        "confidence": "high",
    }
    candidates, errors = summarize_tr._validate_candidates(
        {"candidates": [candidate]}, ["A"], evidence_map
    )
    assert errors == []
    assert candidates[0]["historical_analog_refs"][0]["summary"].startswith(
        "Timeline preview"
    )

    invented = {**candidate, "historical_analog_refs": [{
        "source_bug_code": "A", "analog_bug_code": "INVENTED",
    }]}
    candidates, errors = summarize_tr._validate_candidates(
        {"candidates": [invented]}, ["A"], evidence_map
    )
    assert candidates == []
    assert any("unavailable analog" in error for error in errors)

    irrelevant = {**candidate, "historical_analog_refs": [{
        "source_bug_code": "A", "analog_bug_code": "H3",
    }]}
    candidates, errors = summarize_tr._validate_candidates(
        {"candidates": [irrelevant]}, ["A"], evidence_map
    )
    assert candidates == []
    assert any("does not support" in error for error in errors)

    recommendation = {
        **candidate,
        "risk": "High",
        "impact_basis": "functional",
        "historical_analog_refs": [{
            "source_bug_code": "A", "analog_bug_code": "H2",
        }],
    }
    recommendations, errors = summarize_tr._validate_recommendations(
        {"recommendations": [recommendation]},
        ["A"], evidence_map,
        allowed_evidence_refs={("A", 0)},
        allowed_historical_refs={("A", "H1")},
    )
    assert recommendations == []
    assert any("invented historical_analog_refs" in error for error in errors)


def test_prediction_boundaries_must_limit_scope_and_define_a_passing_result():
    evidence_map = {
        "A": {
            "evidence_quality": "high",
            "adjacency_signal_quotes": ["Updated the shared renderer path"],
        }
    }
    base = {
        "area": "Timeline preview",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Open Timeline preview after image generation",
        "expected_failure_mode": "Timeline preview crashes",
        **_prediction_boundaries("Timeline preview"),
        "action": "Verify Timeline preview after image generation",
        "rationale": "The fix changed a shared renderer path",
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [],
        "confidence": "high",
    }
    candidates, errors = summarize_tr._validate_candidates(
        {"candidates": [base]}, ["A"], evidence_map
    )
    assert errors == []
    assert candidates[0]["scope_boundary"].startswith("Limit Timeline preview")

    invalid = {
        **base,
        "scope_boundary": "Timeline preview might be relevant",
        "disconfirming_observation": "Investigate Timeline preview further",
    }
    candidates, errors = summarize_tr._validate_candidates(
        {"candidates": [invalid]}, ["A"], evidence_map
    )
    assert candidates == []
    assert any("lacks explicit limiting language" in error for error in errors)
    assert any("observable passing result" in error for error in errors)

    invented_component = {
        **base,
        "scope_boundary": (
            "Limit Timeline preview to the database cache; independent paths "
            "are outside scope"
        ),
    }
    candidates, errors = summarize_tr._validate_candidates(
        {"candidates": [invented_component]}, ["A"], evidence_map
    )
    assert candidates == []
    assert any("invents component term" in error for error in errors)

    generic = {
        **base,
        "trigger": "Check Timeline preview",
        "expected_failure_mode": "Timeline preview result is wrong",
    }
    candidates, errors = summarize_tr._validate_candidates(
        {"candidates": [generic]}, ["A"], evidence_map
    )
    assert candidates == []
    assert any("merely restates the area" in error for error in errors)

    recommendations, errors = summarize_tr._validate_recommendations(
        {"recommendations": [{
            **generic, "risk": "Medium", "impact_basis": "functional",
        }]},
        ["A"], evidence_map,
    )
    assert recommendations == []
    assert any("merely restates the area" in error for error in errors)


def test_synthesis_cannot_rewrite_valid_candidate_boundaries():
    evidence_map = {
        "A": {
            "evidence_quality": "high",
            "adjacency_signal_quotes": ["Updated the shared renderer path"],
        }
    }
    boundaries = _prediction_boundaries("Timeline preview")
    recommendation = {
        "area": "Timeline preview",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Open Timeline preview after image generation",
        "expected_failure_mode": "Timeline preview crashes",
        **boundaries,
        "action": "Verify Timeline preview after image generation",
        "rationale": "The fix changed a shared renderer path",
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [],
        "confidence": "high",
        "risk": "High",
        "impact_basis": "crash",
    }
    normalized, errors = summarize_tr._validate_recommendations(
        {"recommendations": [recommendation]},
        ["A"], evidence_map,
        allowed_boundary_pairs={(
            boundaries["scope_boundary"],
            boundaries["disconfirming_observation"],
        )},
    )
    assert errors == []
    assert normalized[0]["disconfirming_observation"] == boundaries[
        "disconfirming_observation"
    ]

    normalized, errors = summarize_tr._validate_recommendations(
        {"recommendations": [recommendation]},
        ["A"], evidence_map,
        allowed_boundary_pairs={(
            boundaries["scope_boundary"],
            boundaries["disconfirming_observation"],
        )},
        allowed_boundary_evidence={(
            boundaries["scope_boundary"],
            boundaries["disconfirming_observation"],
        ): {("OTHER", 0)}},
    )
    assert normalized == []
    assert any("dropped the evidence tied" in error for error in errors)

    rewritten = {
        **recommendation,
        **_prediction_boundaries("Timeline preview export"),
    }
    normalized, errors = summarize_tr._validate_recommendations(
        {"recommendations": [rewritten]},
        ["A"], evidence_map,
        allowed_boundary_pairs={(
            boundaries["scope_boundary"],
            boundaries["disconfirming_observation"],
        )},
    )
    assert normalized == []
    assert any("invented or rewrote" in error for error in errors)


def test_structured_evidence_normalizes_historical_analogs():
    evidence = summarize_tr.structured_bug_evidence({
        "BugCode": "A",
        "Short Description": "Preview crashes",
        "HistoricalAnalogs": [{
            "bug_code": "H1",
            "module": "Timeline",
            "module_category": "Editor Core",
            "summary": "Timeline preview crashes",
            "created_at": "2026-07-01",
            "closed_at": "2026-07-02",
            "relationship": "same_category_adjacent_module",
            "similarity_score": 0.51234567,
            "overlap_terms": ["preview", "crashes"],
            "distinctive_overlap_terms": ["crashes"],
        }],
    })
    assert evidence["historical_analogs"] == [{
        "bug_code": "H1",
        "module": "Timeline",
        "module_category": "Editor Core",
        "summary": "Timeline preview crashes",
        "created_at": "2026-07-01",
        "closed_at": "2026-07-02",
        "relationship": "same_category_adjacent_module",
        "similarity_score": 0.512346,
        "overlap_terms": ["preview", "crashes"],
        "distinctive_overlap_terms": ["crashes"],
    }]


def test_checklist_evidence_gaps_are_deterministic_for_degraded_detail():
    evidence = summarize_tr.structured_bug_evidence({
        "BugCode": "A",
        "Short Description": "List-level summary",
        "DetailFetchError": "full detail unavailable after retries",
    })
    assert evidence["evidence_quality"] == "low"
    assert evidence["evidence_gaps"] == [
        "full_detail_unavailable",
        "missing_reproduction_context",
        "missing_root_cause",
        "missing_fix_description",
    ]


def test_generic_verification_note_is_not_adjacent_regression_evidence():
    generic = summarize_tr.structured_bug_evidence({
        "BugCode": "A",
        "Short Description": "Prompt is not localized",
        "ReproSteps": "Open the prompt box",
        "Resolutions": [{
            "solution": "Updated the prompt string",
            "risk": "Verify in the new builds",
        }],
    })
    assert generic["adjacency_signal_quotes"] == []

    broader = summarize_tr.structured_bug_evidence({
        "BugCode": "A",
        "Short Description": "Preview crashes",
        "ReproSteps": "Open preview",
        "Resolutions": [{
            "solution": "Updated the shared renderer",
            "risk": "Other preview surfaces may crash",
        }],
    })
    assert broader["adjacency_signal_quotes"] == [
        "Updated the shared renderer",
        "Other preview surfaces may crash",
    ]


def _stated_generalization_data():
    return {"products": {"P": [{
        "BugCode": "A",
        "Short Description": "Expand icon stays after Clear",
        "ReproSteps": "Tap Clear in Text to Image",
        "Resolutions": [{
            "solution": (
                "Call updateExpandButtonState() in clearPrompt(), matching "
                "every other text-mutation path."
            ),
        }],
    }]}}


def _generalized_candidate():
    return {
        "area": "Text to Video prompt",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Tap Clear in the Text to Video prompt box",
        "expected_failure_mode": (
            "The expand icon stays visible after clearing the prompt"
        ),
        **_prediction_boundaries("Text to Video prompt"),
        "action": "Verify the Text to Video prompt box after tapping Clear",
        "rationale": (
            "The fix says clearPrompt() now matches every other text-mutation "
            "path, so other prompt text mutations share the same expand-icon "
            "refresh"
        ),
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [],
        "confidence": "high",
    }


def _generalization_fake_json(candidate, failed_checks):
    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
            return {"checklists": [{
                "bug_code": "A", "steps": ["Set up source", "Verify result"],
            }]}
        if "SECOND independent coverage-challenge" in prompt:
            return {"candidates": []}
        if "Infer only defensible ADJACENT" in prompt:
            return {"candidates": [candidate]}
        if "Adversarially audit every proposed" in prompt:
            audited = json.loads(re.search(
                r"<CANDIDATES_TO_AUDIT>\n(.*?)\n</CANDIDATES_TO_AUDIT>",
                prompt, re.S,
            ).group(1))
            return {"audits": [{
                "candidate_id": item["candidate_id"],
                **{
                    check: check not in failed_checks
                    for check in summarize_tr.GROUNDING_AUDIT_CHECKS
                },
                "verdict": "withhold",
                "reason": summarize_tr.GROUNDING_AUDIT_REASON_BY_CHECK[
                    failed_checks[0]
                ],
            } for item in audited]}
        synthesis_candidates = json.loads(re.search(
            r"<CANDIDATES>\n(.*?)\n</CANDIDATES>", prompt, re.S
        ).group(1))
        return {"recommendations": [{
            **synthesis_candidates[0],
            "risk": "Medium", "impact_basis": "functional",
        }]}
    return fake_json


def test_direct_fix_restatements_are_withheld_before_the_critic():
    """The page must point somewhere the fix was not already verified."""
    evidence_map = {"A": {
        "parsed_module": "Image to Video",
        "summary": 'Tapping "Create another one" always returns to the Custom tab',
        "repro_or_description": "Generate from a template then tap Create another one",
        "adjacency_signal_quotes": ["Return to the existing Image-to-Video editor"],
    }}
    restatement = summarize_tr.deterministic_direct_fix_basis(
        "Image to Video",
        "Generating from a template in Image to Video",
        'Tapping "Create another one" returns to the "Custom" tab instead of "Template"',
        ["A"], evidence_map,
    )
    assert restatement["supported"] is False
    assert restatement["rule"] == "restates_the_cited_fix"

    # Padding the area name with the module does not buy adjacency.
    assert summarize_tr.deterministic_direct_fix_basis(
        "Image to Video template selection",
        'Tapping "Create another one" after generating from Template',
        'The app returns to the "Custom" tab instead of the "Template" tab',
        ["A"], evidence_map,
    )["supported"] is False

    # A different module is adjacent by construction.
    assert summarize_tr.deterministic_direct_fix_basis(
        "Text to Video", "Tap Clear in Text to Video",
        "Placeholder stays visible", ["A"], evidence_map,
    )["supported"] is True

    # So is a different surface inside the same module.
    assert summarize_tr.deterministic_direct_fix_basis(
        "Image to Video", "Open the gender tab filter",
        "Favorites category is skipped", ["A"], evidence_map,
    )["supported"] is True


def test_disconfirming_observation_accepts_any_observable_outcome():
    """A curated adjective list rejected valid phrasings like "remains selected"."""
    for observation in (
        "If the [Hot] tab remains selected after switching gender in Image to Video, lower this",
        "If tapping Create another one returns to the Template tab page in Image to Video, lower this",
        "If the warning correctly states the prompt was blocked in Image to Video, lower this",
    ):
        assert summarize_tr.prediction_boundary_errors(
            "Image to Video",
            "Limited to Image to Video tab restoration; unrelated paths are not implicated",
            observation,
            "candidate 1",
            "The wrong tab is shown after switching",
        ) == []


def test_disconfirming_observation_cannot_restate_the_predicted_failure():
    errors = summarize_tr.prediction_boundary_errors(
        "Image to Video",
        "Limited to Image to Video tab restoration; unrelated paths are not implicated",
        "If the wrong tab is shown after switching in Image to Video, lower this",
        "candidate 1",
        "The wrong tab is shown after switching in Image to Video",
    )
    assert any("restates the predicted failure" in error for error in errors)


def test_stated_generalization_family_overrides_only_the_target_check(monkeypatch):
    """An RD writing "every other <family>" names the family, not the member."""
    candidate = _generalized_candidate()
    monkeypatch.setattr(
        summarize_tr, "call_ollama_json",
        _generalization_fake_json(candidate, ["target_supported"]),
    )
    analysis = generate_structured_analysis(
        _stated_generalization_data(), "model", "url", 10,
    )
    audit = analysis["candidate_grounding_audit"]
    assert audit["generalization_target_override_count"] == 1
    assert len(analysis["recommendations"]) == 1
    basis = analysis["recommendations"][0]["generalization_support_basis"]
    assert basis["supported"] is True
    assert basis["overrides_only_target_support"] is True
    assert basis["influences_confidence"] is False
    assert basis["influences_priority"] is False
    # The override decides publication only; the deterministic label is
    # unchanged by it.
    assert analysis["recommendations"][0]["confidence"] in {"low", "medium"}


def test_stated_generalization_does_not_rescue_other_failed_checks(monkeypatch):
    candidate = _generalized_candidate()
    monkeypatch.setattr(
        summarize_tr, "call_ollama_json",
        _generalization_fake_json(
            candidate, ["target_supported", "boundary_supported"],
        ),
    )
    analysis = generate_structured_analysis(
        _stated_generalization_data(), "model", "url", 10,
    )
    audit = analysis["candidate_grounding_audit"]
    assert audit["generalization_target_override_count"] == 0
    assert analysis["recommendations"] == []
    assert audit["withheld_candidate_count"] >= 1


def test_generalization_support_requires_the_claim_to_invoke_the_family():
    refs = [{
        "bug_code": "A", "signal_index": 0,
        "quote": (
            "Call updateExpandButtonState() in clearPrompt(), matching every "
            "other text-mutation path."
        ),
    }]
    supported = summarize_tr.deterministic_generalization_support_basis(
        "Text to Video prompt", "Tap Clear in the Text to Video prompt box",
        "clearPrompt() now matches every other text-mutation path", refs,
    )
    assert supported["supported"] is True

    # Same quote, unrelated claim: the family is not invoked and the trigger
    # touches nothing the quote names.
    unrelated = summarize_tr.deterministic_generalization_support_basis(
        "Audio denoise", "Export a project",
        "Denoise shares the export pipeline", refs,
    )
    assert unrelated["supported"] is False

    # No generalization language in the evidence at all.
    no_family = summarize_tr.deterministic_generalization_support_basis(
        "Text to Video prompt", "Tap Clear in the Text to Video prompt box",
        "text mutation path", [{
            "bug_code": "A", "signal_index": 0,
            "quote": "Swap ttiDescriptionLabel.textColor to .Text.primary_high",
        }],
    )
    assert no_family["supported"] is False


def test_code_shaped_fix_notes_carry_adjacency_evidence():
    """RD notes name symbols, not architecture nouns; both must be usable."""
    generalized = summarize_tr.structured_bug_evidence({
        "BugCode": "A",
        "Short Description": "Expand icon stays after Clear",
        "ReproSteps": "Tap Clear",
        "Resolutions": [{
            "solution": (
                "Call updateExpandButtonState() in clearPrompt(), matching "
                "every other text-mutation path."
            ),
            "risk": "N/A",
        }],
    })
    assert generalized["adjacency_signal_quotes"] == [
        "Call updateExpandButtonState() in clearPrompt(), matching every "
        "other text-mutation path."
    ]

    shared_resource = summarize_tr.structured_bug_evidence({
        "BugCode": "B",
        "Short Description": "VIP wording",
        "ReproSteps": "Open the profile tab",
        "Resolutions": [{
            "solution": (
                'Reuse the existing NSLocalizedString("Pro+") key for both '
                "the .vip title and credit banner"
            ),
            "risk": "N/A",
        }],
    })
    assert shared_resource["adjacency_signal_quotes"]

    # A device name is camel-cased but is not a mechanism to extrapolate from,
    # so it must not qualify a note that names nothing else.
    device_only = summarize_tr.structured_bug_evidence({
        "BugCode": "C",
        "Short Description": "Spacing wrong",
        "ReproSteps": "Open the app",
        "Resolutions": [{"solution": "Adjust the spacing on iPhone", "risk": "N/A"}],
    })
    assert device_only["adjacency_signal_quotes"] == []

    # A named shared surface does qualify: several features route through the
    # same media picker, so a fix there is adjacency evidence.
    shared_surface = summarize_tr.structured_bug_evidence({
        "BugCode": "E",
        "Short Description": "Picker shows numbers",
        "ReproSteps": "Open Photo Enhance",
        "Resolutions": [{
            "solution": "Restore the direct selection flow and remove its "
                        "candidate-tray presentation from the media picker",
            "risk": "N/A",
        }],
    })
    assert shared_surface["adjacency_signal_quotes"]

    # Notes with no mechanism, no generalization and no shared architecture
    # stay silent so discovery cannot extrapolate from nothing.
    for empty_note in ("Adjust button layout", "Not sync guideline", "QA request."):
        assert summarize_tr.structured_bug_evidence({
            "BugCode": "D",
            "Short Description": "Layout",
            "ReproSteps": "Open the page",
            "Resolutions": [{"solution": empty_note, "risk": "N/A"}],
        })["adjacency_signal_quotes"] == []


def test_module_category_context_uses_current_parsed_taxonomy(tmp_path):
    parsed = tmp_path / "ecl_parsed.csv"
    parsed.write_text(
        "BugCode,parsed_module,module_category\n"
        "A,Text to Image,AI Features\n"
        "B,Timeline,Editor Core\n"
        "C,Timeline,Editor Core\n",
        encoding="utf-8",
    )
    details = {
        "meta": {"close_build": "1"},
        "products": {"P": [{
            "BugCode": "A",
            "ParsedModule": "stale",
            "ModuleCategory": "stale",
        }]},
    }
    enriched = process_tr.attach_module_category_context(details, parsed)
    bug = enriched["products"]["P"][0]
    assert bug["ParsedModule"] == "Text to Image"
    assert bug["ModuleCategory"] == "AI Features"
    assert enriched["meta"]["module_category_taxonomy"] == [
        {
            "category": "AI Features", "historical_bug_count": 1,
            "module_count": 1, "modules": ["Text to Image"],
        },
        {
            "category": "Editor Core", "historical_bug_count": 2,
            "module_count": 1, "modules": ["Timeline"],
        },
    ]
    assert details["products"]["P"][0]["ParsedModule"] == "stale"


def test_deterministic_target_category_uses_module_then_source_or_abstains():
    support = {
        "schema_version": 1,
        "policy": "known_target_module_then_unique_current_source_category_v1",
        "taxonomy": [
            {"category": "AI Features", "modules": ["Text to Image"]},
            {"category": "Editor Core", "modules": ["Timeline", "Shared"]},
            {"category": "Media", "modules": ["Shared"]},
        ],
        "source_category_by_bug": {
            "A": "AI Features", "B": "AI Features", "C": "Media",
        },
    }

    module_basis = summarize_tr.deterministic_target_category_basis(
        "AI Features", "Timeline preview", "Open Timeline preview",
        "Verify Timeline preview", ["A"], support,
    )
    assert module_basis["label"] == "Editor Core"
    assert module_basis["rule"] == "unambiguous_known_target_module"
    assert module_basis["matched_modules"] == ["Timeline"]

    source_basis = summarize_tr.deterministic_target_category_basis(
        "AI Features", "Generated preview", "Open generated preview",
        "Verify generated preview", ["A", "B"], support,
    )
    assert source_basis["label"] == "AI Features"
    assert source_basis["rule"] == "unique_current_source_category"

    unsupported = summarize_tr.deterministic_target_category_basis(
        "Editor Core", "Generated preview", "Open generated preview",
        "Verify generated preview", ["A"], support,
    )
    assert unsupported["label"] == summarize_tr.UNKNOWN_TARGET_CATEGORY
    assert unsupported["rule"] == "unsupported_model_category"

    ambiguous = summarize_tr.deterministic_target_category_basis(
        "Editor Core", "Shared preview", "Open Shared preview",
        "Verify Shared preview", ["A"], support,
    )
    assert ambiguous["label"] == summarize_tr.UNKNOWN_TARGET_CATEGORY
    assert ambiguous["rule"] == "ambiguous_known_target_module"


def test_historical_analogs_are_similar_resolved_older_and_leakage_safe(tmp_path):
    parsed = tmp_path / "ecl_parsed.csv"
    parsed.write_text(
        "BugCode,Status,Create Date,Closed Date,Short Description,"
        "parsed_description,parsed_module,module_category\n"
        "A,TRCreated,2026-08-10 10:00:00,,Current issue,"
        "Preview crashes after image generation,Image Generator,AI Features\n"
        "B,TRCreated,2026-08-10 09:00:00,,Current cohort analog,"
        "Preview crashes in renderer pipeline,Timeline,AI Features\n"
        "H-ADJ,Close,2026-07-01 09:00:00,2026-07-03 09:00:00,Historical,"
        "Preview crashes after image export in renderer pipeline,Timeline,AI Features\n"
        "H-SAME,Close,2026-06-01 09:00:00,2026-06-02 09:00:00,Historical,"
        "Image generation preview shows black frame in renderer pipeline,Image Generator,AI Features\n"
        "FUTURE,Close,2026-08-11 09:00:00,2026-08-12 09:00:00,Future,"
        "Preview crashes after image export in renderer pipeline,Timeline,AI Features\n"
        "ACTIVE,NewCreated,2026-07-02 09:00:00,,Active,"
        "Preview crashes after image export in renderer pipeline,Timeline,AI Features\n"
        "OTHER,Close,2026-07-02 09:00:00,2026-07-03 09:00:00,Other,"
        "Preview crashes after image export in renderer pipeline,Timeline,Editor Core\n"
        "WEAK,Close,2026-07-02 09:00:00,2026-07-03 09:00:00,Weak,"
        "Image Generator opens normally,Image Generator,AI Features\n",
        encoding="utf-8",
    )
    details = {"products": {"P": [
        {
            "BugCode": "A",
            "Short Description": "Preview crashes after image generation",
            "ReproSteps": "Generate an image and open preview",
            "Resolutions": [{
                "root_cause": "Shared renderer pipeline retained stale state",
                "solution": "Reset renderer pipeline state after generation",
            }],
        },
        {"BugCode": "B", "Short Description": "Current cohort bug"},
    ]}}

    enriched = process_tr.attach_module_category_context(details, parsed)
    source = enriched["products"]["P"][0]
    analogs = {item["bug_code"]: item for item in source["HistoricalAnalogs"]}

    assert set(analogs) == {"H-ADJ", "H-SAME"}
    assert analogs["H-ADJ"]["relationship"] == "same_category_adjacent_module"
    assert analogs["H-SAME"]["relationship"] == "same_module"
    assert all(item["similarity_score"] >= 0.18 for item in analogs.values())
    assert all(item["distinctive_overlap_terms"] for item in analogs.values())
    assert enriched["meta"]["historical_analog_count"] == 2
    assert enriched["meta"]["historical_analog_policy"][
        "requires_older_create_date"
    ] is True
    assert "HistoricalAnalogs" not in details["products"]["P"][0]

    source_hash, _schema = process_tr.analysis_source_hash(enriched)
    source["HistoricalAnalogs"][0]["summary"] += " changed"
    changed_hash, _schema = process_tr.analysis_source_hash(enriched)
    assert changed_hash != source_hash

    raised = process_tr.attach_module_category_context(
        details,
        parsed,
        historical_analog_threshold_policy={
            "global": {
                "status": "calibrated_higher",
                "eligible": True,
                "threshold": 0.90,
            },
            "categories": {},
        },
    )
    assert raised["products"]["P"][0]["HistoricalAnalogs"] == []
    applied = raised["meta"]["historical_analog_policy"][
        "applied_thresholds_by_category"
    ]["AI Features"]
    assert applied["source"] == "global_reviewed"
    assert applied["threshold"] == 0.90


def test_structured_analysis_rejects_cohort_above_safety_limit(monkeypatch):
    data = {"products": {"P": [
        {"BugCode": f"B{i}", "Short Description": "Issue"} for i in range(3)
    ]}}
    monkeypatch.setattr(
        summarize_tr, "call_ollama_json",
        lambda *args, **kwargs: pytest.fail("model must not be called"),
    )
    with pytest.raises(summarize_tr.AnalysisError, match="exceeding safety limit"):
        generate_structured_analysis(data, "model", "url", 10, safety_limit=2)


def test_structured_validators_reject_bad_steps_enums_and_citations():
    checklists, errors = summarize_tr._validate_verification({
        "checklists": [{"bug_code": "A", "steps": ["1. Set up", "2) Verify result"]}]
    }, ["A"])
    assert errors == []
    assert checklists[0]["steps"] == ["Set up", "Verify result"]

    checklists, errors = summarize_tr._validate_verification({
        "checklists": [{"bug_code": "A", "steps": [str(i) for i in range(9)]}]
    }, ["A"])
    assert checklists == []
    assert any("2-4 steps" in error for error in errors)

    evidence_map = {"A": {"evidence_quality": "high"}}
    candidates, errors = summarize_tr._validate_candidates({
        "candidates": [{
            "area": "Area", "action": "Verify adjacent behavior", "rationale": "Why",
            "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
            "trigger": "Use the adjacent state",
            "expected_failure_mode": "The adjacent state fails",
            **_prediction_boundaries("Area"),
            "source_bug_codes": ["INVENTED"], "confidence": "high",
            "evidence_refs": [{"bug_code": "INVENTED", "signal_index": 0}],
        }]
    }, ["A"], evidence_map)
    assert candidates == []
    assert any("invalid BugCodes" in error for error in errors)

    recommendations, errors = summarize_tr._validate_recommendations({
        "recommendations": [{
            "area": "Area", "action": "Verify adjacent behavior", "rationale": "Why",
            "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
            "trigger": "Use the adjacent state",
            "expected_failure_mode": "The adjacent state fails",
            **_prediction_boundaries("Area"),
            "source_bug_codes": ["A"], "confidence": "certain",
            "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
            "risk": "Urgent", "impact_basis": "unknown",
        }]
    }, ["A"], evidence_map)
    assert recommendations == []
    assert any("invalid risk" in error for error in errors)
    assert any("invalid confidence" in error for error in errors)


def test_checklist_grounding_rejects_invented_specifics_and_adds_anchors():
    evidence = {
        "summary": "Crop preview shows a black frame",
        "repro_or_description": "Open Crop and preview the selected photo",
        "locale_or_os": "",
        "changes": [{
            "root_cause": "", "solution": "Updated preview handling",
            "stated_risk": "",
        }],
        "qa_verify_notes": [],
        "evidence_quality": "high",
        "evidence_gaps": [],
    }
    rejected, errors = summarize_tr._validate_verification({
        "checklists": [{
            "bug_code": "A",
            "steps": [
                "Open Crop on iPhone 17 with iOS 18",
                "Verify the service displays \"Done\"",
            ],
        }],
    }, ["A"], {"A": evidence})
    assert rejected == []
    assert any("unsupported specificity" in error for error in errors)
    assert all(
        term in " ".join(errors)
        for term in ("17", "18", "iphone", "ios", "service", "quoted:Done")
    )

    non_executable, errors = summarize_tr._validate_verification({
        "checklists": [{
            "bug_code": "A",
            "steps": ["The crop screen is open", "Verify the preview"],
        }],
    }, ["A"], {"A": evidence})
    assert non_executable == []
    assert any("non-executable" in error for error in errors)

    supported_evidence = {
        **evidence,
        "repro_or_description": (
            "On iPhone 17 with iOS 18, open Crop and preview the selected photo"
        ),
    }
    validated, errors = summarize_tr._validate_verification({
        "checklists": [{
            "bug_code": "A",
            "steps": [
                "Open Crop on iPhone 17 with iOS 18",
                "Verify the crop preview",
            ],
        }],
    }, ["A"], {"A": supported_evidence})
    assert errors == []

    grounded = summarize_tr.deterministic_checklist_grounding(
        {
            "bug_code": "A", "steps": ["Set up", "Verify result"],
            "evidence_quality": "high", "evidence_gaps": [],
        },
        evidence,
    )
    assert grounded["steps"][0].startswith(
        "Reproduce the supplied scenario: Open Crop and preview"
    )
    assert "Crop preview shows a black frame" in grounded["steps"][-1]
    assert grounded["grounding_basis"]["reproduction_anchor_present"] is True
    assert grounded["grounding_basis"]["symptom_anchor_present"] is True
    assert grounded["grounding_basis"]["unsupported_specificity_terms"] == []
    assert summarize_tr.deterministic_checklist_grounding(
        grounded, evidence
    ) == grounded


def test_synthesis_lineage_requires_every_claim_and_source_to_contribute():
    evidence_map = {
        code: {
            "adjacency_signal_quotes": ["shared renderer state"],
            "historical_analogs": [],
            "evidence_quality": "high",
        }
        for code in ("A", "B")
    }
    base = {
        "area": "Timeline preview",
        "target_category": "Editor Core",
        "trigger": "Open Timeline preview after renderer state changes",
        "expected_failure_mode": "Timeline preview displays stale renderer state",
        **_prediction_boundaries("Timeline preview"),
        "action": "Verify Timeline preview after renderer state changes",
        "rationale": "Timeline preview uses shared renderer state",
        "historical_analog_refs": [],
        "confidence": "medium",
    }
    candidates = []
    for code in ("A", "B"):
        candidate = {
            **base,
            "source_bug_codes": [code],
            "evidence_refs": [{
                "bug_code": code,
                "signal_index": 0,
                "quote": "shared renderer state",
            }],
        }
        candidate["synthesis_source_candidate_ids"] = [
            summarize_tr.stable_candidate_audit_id(candidate)
        ]
        candidates.append(candidate)
    first_id = candidates[0]["synthesis_source_candidate_ids"][0]
    second_id = candidates[1]["synthesis_source_candidate_ids"][0]
    support = summarize_tr._synthesis_lineage_support(candidates)
    recommendation = {
        **base,
        "risk": "Medium",
        "impact_basis": "functional",
        "source_bug_codes": ["A", "B"],
        "evidence_refs": [
            {"bug_code": "A", "signal_index": 0},
            {"bug_code": "B", "signal_index": 0},
        ],
        "synthesis_source_candidate_ids": [first_id],
    }

    normalized, errors = summarize_tr._validate_recommendations(
        {"recommendations": [recommendation]},
        ["A", "B"],
        evidence_map,
        allowed_categories=["Editor Core"],
        allowed_lineage_support=support,
    )
    assert normalized == []
    assert any("lack cited candidate lineage" in error for error in errors)

    recommendation["synthesis_source_candidate_ids"] = [first_id, second_id]
    normalized, errors = summarize_tr._validate_recommendations(
        {"recommendations": [recommendation]},
        ["A", "B"],
        evidence_map,
        allowed_categories=["Editor Core"],
        allowed_lineage_support=support,
    )
    assert errors == []
    assert normalized[0]["synthesis_source_candidate_ids"] == [
        first_id, second_id,
    ]
    rewritten_trigger = {
        **recommendation,
        "trigger": "Open Timeline preview after shared renderer state changes",
    }
    rewritten, errors = summarize_tr._validate_recommendations(
        {"recommendations": [rewritten_trigger]},
        ["A", "B"],
        evidence_map,
        allowed_categories=["Editor Core"],
        allowed_lineage_support=support,
    )
    assert rewritten == []
    assert any("trigger/failure pair" in error for error in errors)

    hierarchical_support = summarize_tr._synthesis_lineage_support(
        normalized, support
    )
    assert hierarchical_support[first_id]["evidence_refs"] == {("A", 0)}
    assert hierarchical_support[second_id]["evidence_refs"] == {("B", 0)}

    noncontributing = {
        **recommendation,
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
    }
    normalized, errors = summarize_tr._validate_recommendations(
        {"recommendations": [noncontributing]},
        ["A", "B"],
        evidence_map,
        allowed_categories=["Editor Core"],
        allowed_lineage_support=support,
    )
    assert normalized == []
    assert any("noncontributing candidate IDs" in error for error in errors)

    distinct = {
        **candidates[1],
        "area": "Title preview",
        "trigger": "Open Title preview after renderer state changes",
        "expected_failure_mode": "Title preview displays stale renderer state",
        **_prediction_boundaries("Title preview"),
        "action": "Verify Title preview after renderer state changes",
    }
    distinct_id = summarize_tr.stable_candidate_audit_id(distinct)
    distinct["synthesis_source_candidate_ids"] = [distinct_id]
    chimera_support = summarize_tr._synthesis_lineage_support([
        candidates[0], distinct,
    ])
    chimera = {
        **recommendation,
        "synthesis_source_candidate_ids": [first_id, distinct_id],
    }
    normalized, errors = summarize_tr._validate_recommendations(
        {"recommendations": [chimera]},
        ["A", "B"],
        evidence_map,
        allowed_categories=["Editor Core"],
        allowed_lineage_support=chimera_support,
    )
    assert normalized == []
    assert any("different claim surface" in error for error in errors)


def test_grounding_audit_requires_complete_consistent_controlled_decisions():
    candidate = {
        "area": "Timeline preview",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Open Timeline preview after generation",
        "expected_failure_mode": "Timeline preview crashes",
        **_prediction_boundaries("Timeline preview"),
        "action": "Verify Timeline preview after generation",
        "rationale": "The preview may use the changed shared renderer",
        "source_bug_codes": ["A"],
        "evidence_refs": [{
            "bug_code": "A", "signal_index": 0, "quote": "shared renderer",
        }],
        "historical_analog_refs": [],
        "confidence": "high",
    }
    candidate_id = summarize_tr.stable_candidate_audit_id(candidate)
    supported = {
        "candidate_id": candidate_id,
        **{check: True for check in summarize_tr.GROUNDING_AUDIT_CHECKS},
        "verdict": "supported",
        "reason": "supported",
    }
    audits, errors = summarize_tr._validate_candidate_grounding_audit(
        {"audits": [supported]}, [candidate]
    )
    assert errors == []
    assert audits[0]["candidate_id"] == candidate_id

    contradictory = {
        **supported,
        "shared_path_concrete": False,
        "verdict": "withhold",
        "reason": "target_not_supported",
    }
    audits, errors = summarize_tr._validate_candidate_grounding_audit(
        {"audits": [contradictory]}, [candidate]
    )
    assert audits == []
    assert any("contradicts its failed checks" in error for error in errors)


def test_grounding_critic_withholds_semantically_weak_candidate(monkeypatch):
    data = {"products": {"P": [{
        "BugCode": "A",
        "Short Description": "Source preview issue",
        "ReproSteps": "Open source preview",
        "Resolutions": [{
            "solution": (
                "Updated Timeline preview and Audio preview shared renderer state"
            ),
        }],
    }]}}

    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
            return {"checklists": [{
                "bug_code": "A", "steps": ["Set up", "Verify result"],
            }]}
        if "Infer only defensible ADJACENT" in prompt:
            return {"candidates": [{
                "area": "Timeline preview",
                "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
                "trigger": "Open Timeline preview after generation",
                "expected_failure_mode": "Timeline preview crashes",
                **_prediction_boundaries("Timeline preview"),
                "action": "Verify Timeline preview after generation",
                "rationale": "The preview may use the shared renderer",
                "source_bug_codes": ["A"],
                "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
                "historical_analog_refs": [],
                "confidence": "high",
            }]}
        if "Adversarially audit every proposed" in prompt:
            payload = _supported_grounding_audits_from_prompt(prompt)
            audit = payload["audits"][0]
            audit.update({
                "shared_path_concrete": False,
                "verdict": "withhold",
                "reason": "shared_path_not_concrete",
            })
            return payload
        if "Repair only the narrowly failed grounding checks" in prompt:
            failed = json.loads(re.search(
                r"<FAILED_CANDIDATES>\n(.*?)\n</FAILED_CANDIDATES>",
                prompt, re.S,
            ).group(1))
            return {"repairs": [{
                "original_candidate_id": item["original_candidate_id"],
                "candidate": None,
            } for item in failed]}
        pytest.fail("synthesis must not run when every candidate is withheld")

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    audit = analysis["candidate_grounding_audit"]

    assert analysis["recommendations"] == []
    assert audit["audited_candidate_count"] == 1
    assert audit["supported_candidate_count"] == 0
    assert audit["withheld_candidate_count"] == 1
    assert audit["withheld_reason_counts"] == {
        "shared_path_not_concrete": 1
    }
    assert audit["withheld_candidates"][0]["failed_checks"] == [
        "shared_path_concrete"
    ]
    assert audit["withheld_candidates"][0]["audit_stage"] == (
        "draft_before_synthesis"
    )
    assert audit["withheld_candidates"][0]["human_review_sample"] is True
    assert audit["human_review_sampling"]["sampled_count"] == 1
    assert audit["repair_attempted_candidate_count"] == 0
    assert audit["repair_ineligible_candidate_count"] == 1
    assert audit["repair_declined_candidate_count"] == 0
    assert audit["recovered_candidate_count"] == 0
    assert audit["withheld_candidates"][0]["repair_status"] == (
        "ineligible_evidence_gap"
    )
    assert process_tr.structured_prediction_errors(analysis) == []
    assert summarize_tr.NO_GROUNDED_REGRESSION in render_analysis_markdown(
        analysis
    )


@pytest.mark.parametrize(
    ("failure_phase", "expected_status"),
    (("repair", "repair_unavailable"), ("reaudit", "reaudit_unavailable")),
)
def test_optional_grounding_repair_failure_falls_closed(
    monkeypatch, failure_phase, expected_status,
):
    data = {"products": {"P": [{
        "BugCode": "A",
        "Short Description": "Source preview issue",
        "ReproSteps": "Open source preview",
        "Resolutions": [{
            "solution": "Updated Timeline preview shared renderer state",
        }],
    }]}}
    audit_calls = 0

    def fake_json(prompt, model, url, timeout, **kwargs):
        nonlocal audit_calls
        if "direct fix-verification" in prompt:
            return {"checklists": [{
                "bug_code": "A", "steps": ["Set up", "Verify result"],
            }]}
        if "Infer only defensible ADJACENT" in prompt:
            return {"candidates": [{
                "area": "Timeline preview",
                "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
                "trigger": "Open Timeline preview",
                "expected_failure_mode": "Timeline preview shows the wrong state",
                **_prediction_boundaries("Timeline preview"),
                "action": "Verify Timeline preview after generation",
                "rationale": "Timeline preview may use the shared renderer state",
                "source_bug_codes": ["A"],
                "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
                "historical_analog_refs": [],
                "confidence": "medium",
            }]}
        if "Repair only the narrowly failed grounding checks" in prompt:
            if failure_phase == "repair":
                raise summarize_tr.AnalysisError("repair unavailable")
            failed = json.loads(re.search(
                r"<FAILED_CANDIDATES>\n(.*?)\n</FAILED_CANDIDATES>",
                prompt, re.S,
            ).group(1))
            repaired = dict(failed[0]["candidate"])
            repaired["trigger"] = (
                "Open Timeline preview after shared renderer state changes"
            )
            return {"repairs": [{
                "original_candidate_id": failed[0]["original_candidate_id"],
                "candidate": repaired,
            }]}
        if "Adversarially audit every proposed" in prompt:
            audit_calls += 1
            if failure_phase == "reaudit" and audit_calls > 1:
                raise summarize_tr.AnalysisError("reaudit unavailable")
            payload = _supported_grounding_audits_from_prompt(prompt)
            payload["audits"][0].update({
                "trigger_supported": False,
                "verdict": "withhold",
                "reason": "trigger_not_supported",
            })
            return payload
        pytest.fail("synthesis must not run without a repaired audit pass")

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    audit = analysis["candidate_grounding_audit"]

    assert analysis["recommendations"] == []
    assert audit["withheld_candidates"][0]["repair_status"] == expected_status
    assert audit["repair_unavailable_candidate_count"] == (
        1 if failure_phase == "repair" else 0
    )
    assert audit["repair_reaudit_unavailable_candidate_count"] == (
        1 if failure_phase == "reaudit" else 0
    )
    assert process_tr.structured_prediction_errors(analysis) == []


def test_final_grounding_critic_blocks_semantic_synthesis_drift(monkeypatch):
    data = {"products": {"P": [{
        "BugCode": "A",
        "Short Description": "Source preview issue",
        "ReproSteps": "Open source preview",
        "Resolutions": [{
            "solution": "Updated Timeline preview shared renderer state",
        }],
    }]}}
    audit_calls = 0

    def fake_json(prompt, model, url, timeout, **kwargs):
        nonlocal audit_calls
        if "direct fix-verification" in prompt:
            return {"checklists": [{
                "bug_code": "A", "steps": ["Set up", "Verify result"],
            }]}
        if "Infer only defensible ADJACENT" in prompt:
            return {"candidates": [{
                "area": "Timeline preview",
                "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
                "trigger": "Open Timeline preview after generation",
                "expected_failure_mode": "Timeline preview crashes",
                **_prediction_boundaries("Timeline preview"),
                "action": "Verify Timeline preview after generation",
                "rationale": "The preview may use the shared renderer",
                "source_bug_codes": ["A"],
                "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
                "historical_analog_refs": [],
                "confidence": "high",
            }]}
        if "Adversarially audit every proposed" in prompt:
            audit_calls += 1
            payload = _supported_grounding_audits_from_prompt(prompt)
            if audit_calls == 2:
                payload["audits"][0].update({
                    "failure_mode_supported": False,
                    "verdict": "withhold",
                    "reason": "failure_mode_not_supported",
                })
            return payload
        candidates = json.loads(re.search(
            r"<CANDIDATES>\n(.*?)\n</CANDIDATES>", prompt, re.S
        ).group(1))
        recommendation = {
            **candidates[0], "risk": "High", "impact_basis": "functional",
        }
        return {"recommendations": [recommendation, dict(recommendation)]}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    audit = analysis["candidate_grounding_audit"]

    assert audit_calls == 2
    assert audit["supported_candidate_count"] == 1
    assert audit["synthesized_recommendation_count"] == 2
    assert audit["duplicate_final_recommendation_count"] == 1
    assert audit["final_recommendation_audited_count"] == 1
    assert audit["final_recommendation_supported_count"] == 0
    assert audit["final_recommendation_withheld_count"] == 1
    assert audit["final_withheld_reason_counts"] == {
        "failure_mode_not_supported": 1
    }
    assert audit["final_withheld_recommendations"][0]["audit_stage"] == (
        "final_after_synthesis"
    )
    assert audit["final_withheld_recommendations"][0][
        "human_review_sample"
    ] is True
    assert analysis["recommendations"] == []
    assert process_tr.structured_prediction_errors(analysis) == []


def test_schema23_prioritizes_and_suppresses_grounded_coverage_equivalents(
    monkeypatch,
):
    data = {"products": {"P": [{
        "BugCode": "A",
        "Short Description": "Source preview issue",
        "ReproSteps": "Open source preview",
        "Resolutions": [{
            "solution": (
                "Updated Timeline preview and Audio preview shared renderer state"
            ),
        }],
    }]}}
    base = {
        "area": "Timeline preview",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Open Timeline preview after generation",
        "expected_failure_mode": "Timeline preview crashes",
        **_prediction_boundaries("Timeline preview"),
        "rationale": "The preview may use the shared renderer",
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [],
        "confidence": "high",
    }
    candidates = [
        {**base, "action": "Verify Timeline preview after generation"},
        {**base, "action": "Verify after generation the Timeline preview"},
        {
            **base,
            "area": "Audio preview",
            "trigger": "Open Audio preview after generation",
            "expected_failure_mode": "Audio preview crashes",
            **_prediction_boundaries("Audio preview"),
            "action": "Verify Audio preview after generation",
        },
    ]
    different_coverage = {
        **candidates[1],
        "action": "Verify after import the Timeline preview",
    }
    opposite_coverage = {
        **candidates[1],
        "action": "Verify without generation the Timeline preview",
    }

    assert summarize_tr.recommendation_coverage_signature(
        candidates[0]
    ) == summarize_tr.recommendation_coverage_signature(candidates[1])
    assert summarize_tr.stable_candidate_audit_id(
        candidates[0]
    ) != summarize_tr.stable_candidate_audit_id(candidates[1])
    assert summarize_tr.recommendation_coverage_signature(
        candidates[0]
    ) != summarize_tr.recommendation_coverage_signature(different_coverage)
    assert summarize_tr.recommendation_coverage_signature(
        candidates[0]
    ) != summarize_tr.recommendation_coverage_signature(opposite_coverage)

    audit_calls = 0

    def fake_json(prompt, model, url, timeout, **kwargs):
        nonlocal audit_calls
        if "direct fix-verification" in prompt:
            return {"checklists": [{
                "bug_code": "A", "steps": ["Set up", "Verify result"],
            }]}
        if "Infer only defensible ADJACENT" in prompt:
            return {"candidates": candidates}
        if "Adversarially audit every proposed" in prompt:
            audit_calls += 1
            return _supported_grounding_audits_from_prompt(prompt)
        synthesis_candidates = json.loads(re.search(
            r"<CANDIDATES>\n(.*?)\n</CANDIDATES>", prompt, re.S
        ).group(1))
        return {"recommendations": [{
            **candidate, "risk": "High", "impact_basis": "functional",
        } for candidate in synthesis_candidates]}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    audit = analysis["candidate_grounding_audit"]

    assert analysis["schema_version"] == 38
    assert audit_calls == 2
    assert audit["synthesized_recommendation_count"] == 3
    assert audit["duplicate_final_recommendation_count"] == 0
    assert audit["coverage_equivalent_recommendation_count"] == 1
    assert audit["final_recommendation_audited_count"] == 3
    assert audit["final_recommendation_grounded_count"] == 3
    assert len(analysis["recommendations"]) == 2
    assert [item["area"] for item in analysis["recommendations"]] == [
        "Timeline preview", "Audio preview",
    ]
    assert [item["priority_rank"] for item in analysis["recommendations"]] == [
        1, 2,
    ]
    assert all(
        item["risk"] == "Medium" for item in analysis["recommendations"]
    )
    assert all(
        item["risk_basis"]["harm_evidence_supported"] is False
        for item in analysis["recommendations"]
    )
    assert all(
        item["risk_basis"]["harm_evidence_cap"]
        == "unsupported_predicted_harm_maximum_medium"
        for item in analysis["recommendations"]
    )
    assert [
        item["priority_basis"][
            "target_anchored_independent_group_count"
        ]
        for item in analysis["recommendations"]
    ] == [1, 1]
    assert analysis["recommendation_priority_policy"][
        "model_order_influences_rank"
    ] is False
    safeguard = audit["synthesis_selection_safeguard"]
    assert safeguard["synthesis_omitted_candidate_count"] == 0
    assert safeguard["backfill_candidate_count"] == 0
    assert audit["portfolio_excluded_recommendation_count"] == 0
    rendered = render_analysis_markdown(analysis)
    assert "**Priority #1 · Timeline preview**" in rendered
    assert "**Priority #2 · Audio preview**" in rendered
    assert (
        "current harm evidence: absent; severe prediction capped to Medium"
        in rendered
    )
    record = audit["coverage_equivalent_recommendations"][0]
    assert record["suppressed_recommendation"]["action"] == (
        candidates[1]["action"]
    )
    assert record["coverage_signature"] == (
        summarize_tr.recommendation_coverage_signature(
            analysis["recommendations"][0]
        )
    )
    assert record["suppression_stage"] == "after_final_grounding_audit"
    assert record["suppressed_grounding_status"] == "supported"
    assert process_tr.structured_prediction_errors(analysis) == []

    tampered_harm_support = json.loads(json.dumps(analysis))
    tampered_harm_support["recommendations"][0]["risk_basis"][
        "harm_evidence_supported"
    ] = True
    errors = process_tr.structured_prediction_errors(tampered_harm_support)
    assert any("risk basis mismatch" in error for error in errors)

    tampered_checklist_basis = json.loads(json.dumps(analysis))
    tampered_checklist_basis["checklists"][0]["grounding_basis"][
        "grounded_step_count"
    ] = 0
    errors = process_tr.structured_prediction_errors(tampered_checklist_basis)
    assert any("checklist 1 grounding basis mismatch" in error for error in errors)

    tampered_checklist_step = json.loads(json.dumps(analysis))
    tampered_checklist_step["checklists"][0]["steps"][0] = (
        "Open the service on iPhone 17"
    )
    errors = process_tr.structured_prediction_errors(tampered_checklist_step)
    assert any("checklist 1 contains unsupported specificity" in error for error in errors)

    tampered_checklist_action = json.loads(json.dumps(analysis))
    tampered_checklist_action["checklists"][0]["steps"][0] = (
        "The source preview scenario is open"
    )
    errors = process_tr.structured_prediction_errors(
        tampered_checklist_action
    )
    assert any("checklist 1 contains non-executable steps" in error for error in errors)

    tampered = json.loads(json.dumps(analysis))
    tampered["candidate_grounding_audit"][
        "coverage_equivalent_recommendations"
    ][0]["suppressed_recommendation"]["action"] = (
        "Verify Timeline preview after import"
    )
    errors = process_tr.structured_prediction_errors(tampered)
    assert any("suppressed signature mismatch" in error for error in errors)

    tampered_lineage = json.loads(json.dumps(analysis))
    tampered_lineage["candidate_grounding_audit"][
        "coverage_equivalent_recommendations"
    ][0]["suppressed_recommendation"][
        "synthesis_source_candidate_ids"
    ] = ["TRC-FFFFFFFFFFFF"]
    errors = process_tr.structured_prediction_errors(tampered_lineage)
    assert any("invalid synthesis lineage" in error for error in errors)

    tampered_counts = json.loads(json.dumps(analysis))
    tampered_counts["candidate_grounding_audit"][
        "final_recommendation_grounded_count"
    ] -= 1
    errors = process_tr.structured_prediction_errors(tampered_counts)
    assert any(
        "grounded recommendation coverage count mismatch" in error
        for error in errors
    )

    tampered_confidence = json.loads(json.dumps(analysis))
    tampered_confidence["recommendations"][0]["confidence"] = "high"
    errors = process_tr.structured_prediction_errors(tampered_confidence)
    assert any("confidence label mismatch" in error for error in errors)

    tampered_confidence_groups = json.loads(json.dumps(analysis))
    tampered_confidence_groups["recommendations"][0]["confidence_basis"][
        "independent_source_group_count"
    ] = 2
    errors = process_tr.structured_prediction_errors(
        tampered_confidence_groups
    )
    assert any("confidence basis mismatch" in error for error in errors)

    tampered_target_anchor = json.loads(json.dumps(analysis))
    tampered_target_anchor["recommendations"][0]["confidence_basis"][
        "target_anchor_terms"
    ] = ["audio"]
    errors = process_tr.structured_prediction_errors(tampered_target_anchor)
    assert any("confidence basis mismatch" in error for error in errors)

    tampered_confidence_quote = json.loads(json.dumps(analysis))
    tampered_confidence_quote["recommendations"][0]["evidence_refs"][0][
        "quote"
    ] = "different stored signal"
    errors = process_tr.structured_prediction_errors(
        tampered_confidence_quote
    )
    assert any(
        "confidence reference provenance mismatch" in error
        for error in errors
    )

    tampered_quality = json.loads(json.dumps(analysis))
    tampered_quality["evidence_quality_by_bug"]["A"] = "low"
    errors = process_tr.structured_prediction_errors(tampered_quality)
    assert any("confidence basis mismatch" in error for error in errors)

    tampered_risk = json.loads(json.dumps(analysis))
    tampered_risk["recommendations"][0]["risk"] = "Critical"
    errors = process_tr.structured_prediction_errors(tampered_risk)
    assert any("risk label mismatch" in error for error in errors)

    tampered_risk_provenance = json.loads(json.dumps(analysis))
    tampered_risk_provenance["recommendations"][0]["risk_basis"][
        "ignored_negated_signals"
    ].append({
        "impact_basis": "crash",
        "signal": "crash",
        "negation": "does not",
    })
    errors = process_tr.structured_prediction_errors(tampered_risk_provenance)
    assert any("risk basis mismatch" in error for error in errors)

    tampered_priority_basis = json.loads(json.dumps(analysis))
    tampered_priority_basis["recommendations"][0]["priority_basis"][
        "independent_source_group_count"
    ] = 9
    errors = process_tr.structured_prediction_errors(tampered_priority_basis)
    assert any("priority basis mismatch" in error for error in errors)

    tampered_priority_rank = json.loads(json.dumps(analysis))
    tampered_priority_rank["recommendations"][0]["priority_rank"] = True
    errors = process_tr.structured_prediction_errors(tampered_priority_rank)
    assert any("priority rank mismatch" in error for error in errors)

    tampered_priority_numeric_type = json.loads(json.dumps(analysis))
    tampered_priority_numeric_type["recommendations"][1]["priority_basis"][
        "target_anchored_independent_group_count"
    ] = False
    errors = process_tr.structured_prediction_errors(
        tampered_priority_numeric_type
    )
    assert any("priority basis mismatch" in error for error in errors)

    tampered_priority_order = json.loads(json.dumps(analysis))
    tampered_priority_order["recommendations"].reverse()
    errors = process_tr.structured_prediction_errors(tampered_priority_order)
    assert any("priority order mismatch" in error for error in errors)

    tampered_priority_policy = json.loads(json.dumps(analysis))
    tampered_priority_policy["recommendation_priority_policy"][
        "model_order_influences_rank"
    ] = True
    errors = process_tr.structured_prediction_errors(tampered_priority_policy)
    assert any("priority policy mismatch" in error for error in errors)

    tampered_suppressed_risk = json.loads(json.dumps(analysis))
    tampered_suppressed_risk["candidate_grounding_audit"][
        "coverage_equivalent_recommendations"
    ][0]["suppressed_recommendation"]["impact_basis"] = "functional"
    errors = process_tr.structured_prediction_errors(tampered_suppressed_risk)
    assert any("risk provenance mismatch" in error for error in errors)

    tampered_category = json.loads(json.dumps(analysis))
    tampered_category["recommendations"][0]["target_category_basis"][
        "rule"
    ] = "unsupported_model_category"
    errors = process_tr.structured_prediction_errors(tampered_category)
    assert any("target category basis mismatch" in error for error in errors)

    tampered_suppressed_category = json.loads(json.dumps(analysis))
    tampered_suppressed_category["candidate_grounding_audit"][
        "coverage_equivalent_recommendations"
    ][0]["suppressed_recommendation"]["target_category"] = "invalid"
    errors = process_tr.structured_prediction_errors(
        tampered_suppressed_category
    )
    assert any("target category label mismatch" in error for error in errors)

    tampered_lineage_category = json.loads(json.dumps(analysis))
    tampered_lineage_category["candidate_grounding_audit"][
        "accepted_candidate_lineage"
    ][0]["target_category_basis"]["requested_category"] = "invalid"
    errors = process_tr.structured_prediction_errors(tampered_lineage_category)
    assert any(
        "accepted candidate lineage 1 target category basis mismatch" in error
        for error in errors
    )


def test_coverage_equivalent_fallback_survives_when_preferred_wording_is_withheld(
    monkeypatch,
):
    data = {"products": {"P": [{
        "BugCode": "A",
        "Short Description": "Source preview issue",
        "ReproSteps": "Open source preview",
        "Resolutions": [{
            "solution": "Updated Timeline preview shared renderer state",
        }],
    }]}}
    base = {
        "area": "Timeline preview",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Open Timeline preview after generation",
        "expected_failure_mode": "Timeline preview crashes",
        **_prediction_boundaries("Timeline preview"),
        "rationale": "The preview may use the shared renderer",
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [],
        "confidence": "high",
    }
    candidates = [
        {**base, "action": "Verify Timeline preview after generation"},
        {**base, "action": "Verify after generation the Timeline preview"},
    ]
    audit_calls = 0

    def fake_json(prompt, model, url, timeout, **kwargs):
        nonlocal audit_calls
        if "direct fix-verification" in prompt:
            return {"checklists": [{
                "bug_code": "A", "steps": ["Set up", "Verify result"],
            }]}
        if "Infer only defensible ADJACENT" in prompt:
            return {"candidates": candidates}
        if "Adversarially audit every proposed" in prompt:
            audit_calls += 1
            payload = _supported_grounding_audits_from_prompt(prompt)
            if audit_calls == 2:
                payload["audits"][0].update({
                    "failure_mode_supported": False,
                    "verdict": "withhold",
                    "reason": "failure_mode_not_supported",
                })
            return payload
        synthesis_candidates = json.loads(re.search(
            r"<CANDIDATES>\n(.*?)\n</CANDIDATES>", prompt, re.S
        ).group(1))
        return {"recommendations": [{
            **candidate, "risk": "High", "impact_basis": "functional",
        } for candidate in synthesis_candidates]}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)

    assert [item["action"] for item in analysis["recommendations"]] == [
        candidates[1]["action"]
    ]
    assert analysis["candidate_grounding_audit"][
        "final_recommendation_withheld_count"
    ] == 1
    assert process_tr.structured_prediction_errors(analysis) == []


def test_schema23_backfills_strong_grounded_candidate_omitted_by_synthesis(
    monkeypatch,
):
    data = {"products": {"P": [{
        "BugCode": "A",
        "Short Description": "Source preview issue",
        "ReproSteps": "Open source preview",
        "Resolutions": [{
            "solution": (
                "Updated Timeline preview and Audio preview shared renderer state"
            ),
        }],
    }]}}
    candidates = [
        {
            "area": "Timeline preview",
            "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
            "trigger": "Open Timeline preview after generation",
            "expected_failure_mode": "Timeline preview crashes",
            **_prediction_boundaries("Timeline preview"),
            "action": "Verify Timeline preview after generation",
            "rationale": "The preview may use the shared renderer",
            "source_bug_codes": ["A"],
            "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
            "historical_analog_refs": [],
            "confidence": "high",
        },
        {
            "area": "Audio preview",
            "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
            "trigger": "Open Audio preview after generation",
            "expected_failure_mode": "Audio preview crashes",
            **_prediction_boundaries("Audio preview"),
            "action": "Verify Audio preview after generation",
            "rationale": "The preview may use the shared renderer",
            "source_bug_codes": ["A"],
            "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
            "historical_analog_refs": [],
            "confidence": "high",
        },
    ]

    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
            return {"checklists": [{
                "bug_code": "A", "steps": ["Set up", "Verify result"],
            }]}
        if "Infer only defensible ADJACENT" in prompt:
            return {"candidates": candidates}
        if "Adversarially audit every proposed" in prompt:
            return _supported_grounding_audits_from_prompt(prompt)
        synthesis_candidates = json.loads(re.search(
            r"<CANDIDATES>\n(.*?)\n</CANDIDATES>", prompt, re.S
        ).group(1))
        audio = next(
            item for item in synthesis_candidates
            if item["area"] == "Audio preview"
        )
        return {"recommendations": [{
            **audio, "risk": "High", "impact_basis": "crash",
        }]}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    audit = analysis["candidate_grounding_audit"]
    safeguard = audit["synthesis_selection_safeguard"]

    assert [item["area"] for item in analysis["recommendations"]] == [
        "Timeline preview", "Audio preview",
    ]
    assert [
        item["selection_origin"] for item in analysis["recommendations"]
    ] == [
        "deterministic_synthesis_omission_backfill", "model_synthesis",
    ]
    assert safeguard["accepted_candidate_count"] == 2
    assert safeguard["model_selected_candidate_count"] == 1
    assert safeguard["synthesis_omitted_candidate_count"] == 1
    assert safeguard["backfill_candidate_count"] == 1
    assert audit["synthesized_recommendation_count"] == 1
    assert audit["synthesis_backfill_candidate_count"] == 1
    assert audit["final_recommendation_audited_count"] == 2
    assert process_tr.structured_prediction_errors(analysis) == []
    assert "Selection: deterministic recovery after AI synthesis omission" in (
        render_analysis_markdown(analysis)
    )

    tampered = json.loads(json.dumps(analysis))
    tampered["candidate_grounding_audit"]["synthesis_selection_safeguard"][
        "synthesis_omitted_candidate_ids"
    ] = []
    errors = process_tr.structured_prediction_errors(tampered)
    assert any(
        "synthesis selection safeguard provenance mismatch" in error
        for error in errors
    )

    tampered_origin = json.loads(json.dumps(analysis))
    tampered_origin["recommendations"][0][
        "selection_origin"
    ] = "model_synthesis"
    errors = process_tr.structured_prediction_errors(tampered_origin)
    assert any(
        "synthesis backfill selection origin mismatch" in error
        for error in errors
    )


def test_schema23_backfill_preserves_distinct_coverage_before_variants():
    confidence_basis = {
        "independence_groups": [["A"]],
        "target_anchor_terms_by_bug": {"A": ["timeline"]},
    }
    base = {
        "area": "Timeline preview",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Open Timeline preview after generation",
        "expected_failure_mode": "Timeline preview crashes",
        **_prediction_boundaries("Timeline preview"),
        "action": "Verify Timeline preview after generation",
        "rationale": "The preview may use shared state",
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [],
        "confidence": "medium",
        "confidence_basis": confidence_basis,
    }
    variant = {
        **base,
        "action": "Verify after generation the Timeline preview",
    }
    distinct = {
        **base,
        "area": "Audio preview",
        "trigger": "Open Audio preview after generation",
        "expected_failure_mode": "Audio preview layout flickers",
        **_prediction_boundaries("Audio preview"),
        "action": "Verify Audio preview after generation",
        "confidence_basis": {
            **confidence_basis,
            "target_anchor_terms_by_bug": {"A": []},
        },
    }
    selected, audit = summarize_tr.deterministic_synthesis_backfill(
        [base, variant, distinct], [], {"A": {"evidence_quality": "high"}},
        limit=2,
    )

    assert len(selected) == 2
    assert len({
        summarize_tr.recommendation_coverage_signature(item)
        for item in selected
    }) == 2
    assert {item["area"] for item in selected} == {
        "Timeline preview", "Audio preview",
    }
    assert audit["backfill_candidate_count"] == 2
    assert audit["unbackfilled_omitted_candidate_count"] == 1


def test_schema23_publishes_top_eight_and_preserves_supported_overflow(
    monkeypatch,
):
    bugs = []
    for index in range(24):
        code = f"A{index:02d}"
        bugs.append({
            "BugCode": code,
            "Short Description": f"Preview issue {code}",
            "ReproSteps": f"Open source preview {code}",
            "Resolutions": [{
                "root_cause": "Shared preview state",
                "solution": f"Updated {code} preview shared renderer state",
            }],
        })
    data = {"products": {"P": bugs}}

    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
                evidence = _evidence_from_prompt(prompt)
                return {"checklists": [{
                    "bug_code": item["bug_code"],
                    "steps": [
                        f"Open source preview {item['bug_code']}",
                        f"Verify preview issue {item['bug_code']} is resolved",
                    ],
                } for item in evidence]}
        if "Infer only defensible ADJACENT" in prompt:
            evidence = _evidence_from_prompt(prompt)
            return {"candidates": [{
                "area": f"{item['bug_code']} preview",
                "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
                "trigger": f"Open {item['bug_code']} preview after generation",
                "expected_failure_mode": (
                    f"{item['bug_code']} preview crashes after generation"
                ),
                **_prediction_boundaries(f"{item['bug_code']} preview"),
                "action": f"Verify {item['bug_code']} preview after generation",
                "rationale": "The preview may use the shared renderer",
                "source_bug_codes": [item["bug_code"]],
                    "evidence_refs": [{
                        "bug_code": item["bug_code"], "signal_index": 1,
                    }],
                "historical_analog_refs": [],
                "confidence": "high",
            } for item in evidence[:6]]}
        if "Adversarially audit every proposed" in prompt:
            return _supported_grounding_audits_from_prompt(prompt)
        synthesis_candidates = json.loads(re.search(
            r"<CANDIDATES>\n(.*?)\n</CANDIDATES>", prompt, re.S
        ).group(1))
        return {"recommendations": [{
            **item, "risk": "High", "impact_basis": "crash",
        } for item in synthesis_candidates[:8]]}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    audit = analysis["candidate_grounding_audit"]

    assert audit["supported_candidate_count"] == 12
    assert audit["synthesized_recommendation_count"] == 8
    assert audit["synthesis_backfill_candidate_count"] == 4
    assert audit["final_recommendation_audited_count"] == 12
    assert audit["final_recommendation_grounded_count"] == 12
    assert audit["final_recommendation_supported_count"] == 12
    assert audit["published_recommendation_count"] == 8
    assert audit["portfolio_excluded_recommendation_count"] == 4
    assert len(analysis["recommendations"]) == 8
    portfolio_policy = analysis["recommendation_portfolio_policy"]
    assert portfolio_policy["policy"] == (
        "risk_confidence_cutoff_coverage_diversity_v1"
    )
    assert portfolio_policy["cutoff_tier"] == {
        "risk": "Medium", "confidence": "medium",
    }
    assert portfolio_policy["cutoff_tier_candidate_count"] == 12
    assert portfolio_policy["cutoff_tier_slot_count"] == 8
    assert portfolio_policy["risk_tier_precedence_preserved"] is True
    assert portfolio_policy[
        "confidence_precedence_within_risk_preserved"
    ] is True
    assert portfolio_policy["diversity_influences_only_cutoff_tier"] is True
    assert [
        record["portfolio_position"]
        for record in audit["portfolio_excluded_recommendations"]
    ] == [9, 10, 11, 12]
    assert process_tr.structured_prediction_errors(analysis) == []

    tampered_position = json.loads(json.dumps(analysis))
    tampered_position["candidate_grounding_audit"][
        "portfolio_excluded_recommendations"
    ][0]["portfolio_position"] = True
    errors = process_tr.structured_prediction_errors(tampered_position)
    assert any("position mismatch" in error for error in errors)

    tampered_overflow = json.loads(json.dumps(analysis))
    tampered_overflow["candidate_grounding_audit"][
        "portfolio_excluded_recommendations"
    ][0]["recommendation"]["priority_basis"][
        "independent_source_group_count"
    ] = 99
    errors = process_tr.structured_prediction_errors(tampered_overflow)
    assert any("priority mismatch" in error for error in errors)

    tampered_overflow_identity = json.loads(json.dumps(analysis))
    tampered_overflow_identity["candidate_grounding_audit"][
        "portfolio_excluded_recommendations"
    ][0]["recommendation"]["hypothesis_id"] = "TRH-FFFFFFFFFFFF"
    errors = process_tr.structured_prediction_errors(
        tampered_overflow_identity
    )
    assert any("hypothesis identity mismatch" in error for error in errors)

    tampered_portfolio_policy = json.loads(json.dumps(analysis))
    tampered_portfolio_policy["recommendation_portfolio_policy"][
        "model_order_influences_selection"
    ] = True
    errors = process_tr.structured_prediction_errors(tampered_portfolio_policy)
    assert any("portfolio policy mismatch" in error for error in errors)


def test_schema31_diversifies_only_the_risk_confidence_cutoff_tier():
    def recommendation(
        index, *, risk="Medium", confidence="medium", path="renderer",
        category=summarize_tr.UNKNOWN_TARGET_CATEGORY, source="A",
    ):
        area = f"Area {index}"
        item = {
            "area": area,
            "target_category": category,
            "trigger": f"Open {area} after generation",
            "expected_failure_mode": f"{area} shows the wrong result",
            **_prediction_boundaries(area),
            "action": f"Verify {area} after generation",
            "rationale": f"{area} uses the shared {path}",
            "source_bug_codes": [source],
            "evidence_refs": [{
                "bug_code": source, "signal_index": 0,
                "quote": f"Updated {area} shared {path}",
            }],
            "historical_analog_refs": [],
            "risk": risk,
            "confidence": confidence,
            "shared_path_support_basis": {
                "strong_path_matches": [{
                    "rationale_term": path,
                }],
            },
        }
        item["priority_basis"] = {
            "sort_key": [
                summarize_tr.RISK_ORDER[risk],
                -summarize_tr.CONFIDENCE_ORDER[confidence],
                0, 0, 0, 0, f"{index:03d}",
            ],
        }
        return item

    equal_tier = [
        recommendation(index) for index in range(8)
    ] + [
        recommendation(
            8, path="cache", category="AI Features", source="B",
        ),
        recommendation(
            9, path="pipeline", category="Media", source="C",
        ),
    ]
    selected, excluded, policy = (
        summarize_tr.deterministic_recommendation_portfolio(
            equal_tier, limit=8,
        )
    )

    assert [item["area"] for item in selected] == [
        "Area 0", "Area 1", "Area 2", "Area 3", "Area 4", "Area 5",
        "Area 8", "Area 9",
    ]
    assert [item["area"] for item in excluded] == ["Area 6", "Area 7"]
    assert policy["cutoff_tier"] == {
        "risk": "Medium", "confidence": "medium",
    }
    assert policy["cutoff_selection_records"][0]["selection_reason"] == (
        "highest_priority_cutoff_tier_anchor"
    )
    assert {
        tuple(record["novel_path_terms"])
        for record in policy["cutoff_selection_records"]
    } >= {("cache",), ("pipeline",)}
    assert {
        record["reason"] for record in policy["exclusion_records"]
    } == {"equal_risk_confidence_tier_diversity_cutoff"}

    confidence_guard = [
        recommendation(index, risk="High", confidence="high")
        for index in range(9)
    ] + [
        recommendation(
            20, risk="High", confidence="medium", path="cache",
            category="AI Features", source="B",
        )
    ]
    guarded_selected, guarded_excluded, guarded_policy = (
        summarize_tr.deterministic_recommendation_portfolio(
            confidence_guard, limit=8,
        )
    )
    assert all(item["confidence"] == "high" for item in guarded_selected)
    assert any(item["confidence"] == "medium" for item in guarded_excluded)
    medium_id = summarize_tr.stable_hypothesis_id(confidence_guard[-1])
    medium_exclusion = next(
        row for row in guarded_policy["exclusion_records"]
        if row["hypothesis_id"] == medium_id
    )
    assert medium_exclusion["reason"] == (
        "below_risk_confidence_portfolio_cutoff"
    )

    risk_guard = [
        recommendation(index, risk="High", confidence="medium")
        for index in range(9)
    ] + [
        recommendation(
            30, risk="Medium", confidence="high", path="pipeline",
            category="Media", source="C",
        )
    ]
    risk_selected, risk_excluded, risk_policy = (
        summarize_tr.deterministic_recommendation_portfolio(
            risk_guard, limit=8,
        )
    )
    assert all(item["risk"] == "High" for item in risk_selected)
    assert any(item["risk"] == "Medium" for item in risk_excluded)
    lower_risk_id = summarize_tr.stable_hypothesis_id(risk_guard[-1])
    lower_risk_exclusion = next(
        row for row in risk_policy["exclusion_records"]
        if row["hypothesis_id"] == lower_risk_id
    )
    assert lower_risk_exclusion["reason"] == (
        "below_risk_confidence_portfolio_cutoff"
    )


def test_narrow_grounding_failure_is_repaired_and_reaudited(monkeypatch):
    data = {"products": {"P": [{
        "BugCode": "A",
        "Short Description": "Source preview issue",
        "ReproSteps": "Open source preview",
        "Resolutions": [{
            "solution": "Updated Timeline preview shared renderer state",
        }],
    }]}}
    audit_calls = 0

    def fake_json(prompt, model, url, timeout, **kwargs):
        nonlocal audit_calls
        if "direct fix-verification" in prompt:
            return {"checklists": [{
                "bug_code": "A", "steps": ["Set up", "Verify result"],
            }]}
        if "Infer only defensible ADJACENT" in prompt:
            return {"candidates": [{
                "area": "Timeline preview",
                "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
                "trigger": "Open Timeline preview",
                "expected_failure_mode": "Timeline preview shows the wrong state",
                **_prediction_boundaries("Timeline preview"),
                "action": "Verify Timeline preview after generation",
                "rationale": "Timeline preview may use the shared renderer state",
                "source_bug_codes": ["A"],
                "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
                "historical_analog_refs": [],
                "confidence": "medium",
            }]}
        if "Repair only the narrowly failed grounding checks" in prompt:
            failed = json.loads(re.search(
                r"<FAILED_CANDIDATES>\n(.*?)\n</FAILED_CANDIDATES>",
                prompt, re.S,
            ).group(1))
            repaired = dict(failed[0]["candidate"])
            repaired["trigger"] = (
                "Open Timeline preview after the shared renderer state changes"
            )
            return {"repairs": [{
                "original_candidate_id": failed[0]["original_candidate_id"],
                "candidate": repaired,
            }]}
        if "Adversarially audit every proposed" in prompt:
            audit_calls += 1
            payload = _supported_grounding_audits_from_prompt(prompt)
            if audit_calls == 1:
                payload["audits"][0].update({
                    "trigger_supported": False,
                    "verdict": "withhold",
                    "reason": "trigger_not_supported",
                })
            return payload
        candidates = json.loads(re.search(
            r"<CANDIDATES>\n(.*?)\n</CANDIDATES>", prompt, re.S
        ).group(1))
        return {"recommendations": [{
            **candidates[0], "risk": "Medium", "impact_basis": "functional",
        }]}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    audit = analysis["candidate_grounding_audit"]

    assert audit_calls == 3
    assert audit["initial_supported_candidate_count"] == 0
    assert audit["initial_withheld_candidate_count"] == 1
    assert audit["repair_attempted_candidate_count"] == 1
    assert audit["repair_supported_candidate_count"] == 1
    assert audit["recovered_candidate_count"] == 1
    assert audit["withheld_candidate_count"] == 0
    assert audit["recovered_candidates"][0]["changed_fields"] == ["trigger"]
    assert len(analysis["recommendations"]) == 1
    recommendation = analysis["recommendations"][0]
    recovered_id = audit["recovered_candidates"][0]["repaired_candidate_id"]
    assert recommendation["prediction_origin"] == "recovered_draft_only"
    assert recommendation["synthesis_source_candidate_ids"] == [recovered_id]
    assert recommendation["recovered_source_candidate_ids"] == [recovered_id]
    assert recommendation["synthesis_lineage"] == [{
        "candidate_id": recovered_id,
        "origin": "recovered_after_reaudit",
        "original_candidate_id": audit["recovered_candidates"][0][
            "original_candidate_id"
        ],
    }]
    assert analysis["recommendations_with_recovered_source_count"] == 1
    assert process_tr.structured_prediction_errors(analysis) == []

    tampered = json.loads(json.dumps(analysis))
    tampered["recommendations"][0]["synthesis_source_candidate_ids"] = [
        "TRC-FFFFFFFFFFFF"
    ]
    errors = process_tr.structured_prediction_errors(tampered)
    assert any("invalid synthesis lineage IDs" in error for error in errors)

    tampered = json.loads(json.dumps(analysis))
    tampered["candidate_grounding_audit"]["accepted_candidate_lineage"][0][
        "action"
    ] = "Verify a different adjacent feature"
    errors = process_tr.structured_prediction_errors(tampered)
    assert any("identity hash mismatch" in error for error in errors)

    tampered = json.loads(json.dumps(analysis))
    tampered["recommendations"][0].update({
        "trigger": "Check Timeline preview",
        "expected_failure_mode": "Timeline preview result is wrong",
    })
    tampered["candidate_grounding_audit"]["accepted_candidate_lineage"][0].update({
        "trigger": "Check Timeline preview",
        "expected_failure_mode": "Timeline preview result is wrong",
    })
    errors = process_tr.structured_prediction_errors(tampered)
    assert any(
        "recommendation 1 trigger/failure merely restates" in error
        for error in errors
    )
    assert any(
        "accepted candidate lineage 1 trigger/failure merely restates" in error
        for error in errors
    )


def test_grounding_repair_validator_rejects_provenance_drift():
    evidence_map = {
        "A": {
            "adjacency_signal_quotes": ["shared renderer state"],
            "historical_analogs": [],
            "evidence_quality": "high",
        }
    }
    original = {
        "area": "Timeline preview",
        "target_category": "Editor Core",
        "trigger": "Open Timeline preview",
        "expected_failure_mode": "Timeline preview shows the wrong state",
        **_prediction_boundaries("Timeline preview"),
        "action": "Verify Timeline preview after generation",
        "rationale": "Timeline preview may use the shared renderer state",
        "source_bug_codes": ["A"],
        "evidence_refs": [{
            "bug_code": "A", "signal_index": 0, "quote": "shared renderer state",
        }],
        "historical_analog_refs": [],
        "confidence": "medium",
    }
    original_id = summarize_tr.stable_candidate_audit_id(original)
    changed = dict(original)
    changed["target_category"] = summarize_tr.UNKNOWN_TARGET_CATEGORY
    changed["trigger"] = "Open Timeline preview after shared renderer state changes"
    changed["expected_failure_mode"] = "Timeline preview displays stale output"
    changed["confidence"] = "high"
    repairs, errors = summarize_tr._validate_candidate_grounding_repairs(
        {"repairs": [{
            "original_candidate_id": original_id,
            "candidate": changed,
        }]},
        [{
            "original_candidate_id": original_id,
            "failed_checks": ["trigger_supported"],
            "candidate": original,
        }],
        evidence_map,
        ["Editor Core", summarize_tr.UNKNOWN_TARGET_CATEGORY],
        {original_id},
    )

    assert repairs == []
    assert any("changed target_category" in error for error in errors)
    assert not any("increased confidence" in error for error in errors)
    assert any("changed unrelated field(s)" in error for error in errors)


def test_hierarchical_synthesis_keeps_original_evidence(monkeypatch):
    data = {"products": {"P": [
        {
            "BugCode": f"B{i:03d}", "Short Description": f"Issue {i}",
                "ReproSteps": "Steps", "Resolutions": [{
                    "solution": "Fix shared renderer buffer",
                }],
        }
        for i in range(49)
    ]}}
    synthesis_prompts = []

    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
            evidence = _evidence_from_prompt(prompt)
            return {"checklists": [
                {"bug_code": item["bug_code"], "steps": ["Set up", "Verify"]}
                for item in evidence
            ]}
        if "Infer only defensible ADJACENT" in prompt:
            evidence = _evidence_from_prompt(prompt)
            return {"candidates": [{
                "area": f"Renderer area {chr(65 + index)}",
                "action": "Verify adjacent behavior",
                "rationale": "The adjacent renderer uses the shared buffer path",
                "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
                "trigger": "Open another renderer-backed preview",
                    "expected_failure_mode": "The adjacent preview crashes",
                    **_prediction_boundaries(f"Renderer area {chr(65 + index)}"),
                "source_bug_codes": [evidence[index % len(evidence)]["bug_code"]],
                "evidence_refs": [{
                    "bug_code": evidence[index % len(evidence)]["bug_code"],
                    "signal_index": 0,
                }],
                "confidence": "high",
            } for index in range(6)]}
        if "Adversarially audit every proposed" in prompt:
            return _supported_grounding_audits_from_prompt(prompt)
        synthesis_prompts.append(prompt)
        candidates = json.loads(re.search(
            r"<CANDIDATES>\n(.*?)\n</CANDIDATES>", prompt, re.S
        ).group(1))
        first = candidates[0]
        return {"recommendations": [{
            **first, "risk": "Medium", "impact_basis": "functional",
        }]}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    assert len(synthesis_prompts) == 3
    assert all("<ORIGINAL_EVIDENCE>" in prompt for prompt in synthesis_prompts)
    assert analysis["bug_count"] == 49


def test_mmdd_year_rollover_and_invalid_calendar_date():
    assert mmdd_date_value("1231", datetime(2027, 1, 10)) == 20261231
    assert mmdd_date_value("0105", datetime(2027, 1, 10)) == 20270105
    assert mmdd_date_value("0231", datetime(2027, 3, 1)) is None


def test_fallback_deduplicates_and_prefers_current_trcreated_row(tmp_path):
    parsed = tmp_path / "ecl_parsed.csv"
    parsed.write_text(
        "BugCode,Status,parsed_version,Close Build#,Short Description\n"
        "A,Close,1.0,100,old\n"
        "A,TRCreated,2.0,200,current\n",
        encoding="utf-8",
    )
    records = local_fallback_bugs(parsed, ["A"])
    assert len(records) == 1
    assert records[0]["Short Description"] == "current"


def test_quality_merge_preserves_rich_evidence_and_clears_recovered_warning():
    rich = {
        "meta": {"missing_detail_bug_codes": ["A"]},
        "products": {"P": [{
            "BugCode": "A", "Short Description": "Issue",
            "Resolutions": [{"solution": "rich fix"}],
            "DetailFetchError": "old warning",
        }]},
    }
    degraded = {"meta": {}, "products": {"P": [{"BugCode": "A", "Resolutions": []}]}}
    merged = merge_detail_data(rich, degraded)
    filtered = filter_detail_data(merged, ["A"])
    bug = filtered["products"]["P"][0]
    assert detail_quality_score(bug) >= 3
    assert bug["Resolutions"][0]["solution"] == "rich fix"
    assert filtered["meta"]["missing_detail_bug_codes"] == ["A"]

    recovered = merge_detail_data(rich, {
        "meta": {}, "products": {"P": [{
            "BugCode": "A", "Short Description": "Issue",
            "ReproSteps": "Steps", "raw": {"a": 1, "b": 2, "c": 3, "d": 4},
        }]},
    })
    recovered = filter_detail_data(recovered, ["A"])
    assert "DetailFetchError" not in recovered["products"]["P"][0]
    assert recovered["meta"]["missing_detail_bug_codes"] == []


def test_filter_orders_cohort_and_quarantines_other_products():
    data = {"meta": {}, "products": {
        "Expected": [{"BugCode": "B"}, {"BugCode": "A"}],
        "Wrong": [{"BugCode": "C"}],
    }}
    result = filter_detail_data(data, ["A", "B", "C"], expected_product="Expected")
    assert [bug["BugCode"] for bug in result["products"]["Expected"]] == ["A", "B"]
    assert result["meta"]["unexpected_product_records"] == [
        {"bug_code": "C", "product": "Wrong"}
    ]


def test_immutable_publish_cache_hit_and_failed_rerun_preserve_latest(tmp_path, monkeypatch):
    product_dir = tmp_path / "pdri"
    product_dir.mkdir()
    parsed = product_dir / "ecl_parsed.csv"
    parsed.write_text(
        "BugCode,Status,parsed_version,Close Build#\n"
        "A,TRCreated,16.7.0,16.7.0.123456\n",
        encoding="utf-8",
    )
    state = {"summary": "Issue one", "fail": False, "runs": 0}

    def fake_post(url, payload, timeout=300, **kwargs):
        return {
            "meta": {"close_build": payload["close_build"], "errors": []},
            "products": {"PowerDirector Mobile for iOS": [{
                "BugCode": "A",
                "Short Description": state["summary"],
                "ReproSteps": "Steps",
                "ParentTRCode": "TR-1",
                "Resolutions": [{
                    "root_cause": "cause",
                    "solution": "Updated AI preview shared renderer state",
                }],
                "raw": {"a": 1, "b": 2, "c": 3, "d": 4},
            }]},
        }

    def fake_subprocess(command, check=False):
        state["runs"] += 1
        if state["fail"]:
            return SimpleNamespace(returncode=1)
        details_path = Path(command[2])
        report_path = Path(command[command.index("--output") + 1])
        analysis_path = Path(command[command.index("--analysis-output") + 1])
        details = json.loads(details_path.read_text(encoding="utf-8"))
        source_evidence = summarize_tr.structured_bug_evidence(
            next(iter(details["products"].values()))[0]
        )
        checklist_evidence = summarize_tr.checklist_evidence_payload(
            source_evidence
        )
        checklist = summarize_tr.deterministic_checklist_grounding({
            "bug_code": "A", "steps": ["Set up", "Verify"],
            "evidence_quality": source_evidence["evidence_quality"],
            "evidence_gaps": source_evidence["evidence_gaps"],
        }, checklist_evidence)
        source_hash, schema = process_tr.analysis_source_hash(details)
        analysis_path.write_text(json.dumps({
            "schema_version": schema,
            "engine_sha256": process_tr.analysis_engine_sha256(),
            "model": "gemma4:e2b-it-q4_K_M",
            "generated_at": "2026-08-10T00:00:00+00:00",
            "source_sha256": source_hash,
                "bug_count": 1,
                "bug_codes": ["A"],
                "evidence_quality_by_bug": {"A": "high"},
                "evidence_independence_by_bug": {
                    "A": {
                        "evidence_quality": "high",
                        "parsed_module": "",
                        "adjacency_signal_quotes": [],
                    },
                },
                    "target_category_support": summarize_tr.target_category_support(
                    details,
                    [summarize_tr.structured_bug_evidence(
                        next(iter(details["products"].values()))[0]
                    )],
                    ),
                    "checklist_evidence_by_bug": {"A": checklist_evidence},
                    "target_alias_policy": (
                        prediction_reviews.compact_target_alias_policy(None)
                    ),
                "candidate_grounding_audit": {
                "schema_version": 1,
                "candidate_discovery_audit": _candidate_discovery_audit(),
                "generated_candidate_count": 0,
                "duplicate_candidate_count": 0,
                "audited_candidate_count": 0,
                "supported_candidate_count": 0,
                "withheld_candidate_count": 0,
                    "withheld_reason_counts": {},
                    "withheld_candidates": [],
                    "initial_supported_candidate_count": 0,
                    "initial_withheld_candidate_count": 0,
                    "initial_withheld_reason_counts": {},
                    "repair_eligible_candidate_count": 0,
                    "repair_ineligible_candidate_count": 0,
                    "repair_attempted_candidate_count": 0,
                    "repair_proposed_candidate_count": 0,
                    "repair_declined_candidate_count": 0,
                    "repair_unavailable_candidate_count": 0,
                    "repair_reaudited_candidate_count": 0,
                    "repair_reaudit_unavailable_candidate_count": 0,
                    "repair_supported_candidate_count": 0,
                    "repair_withheld_candidate_count": 0,
                        "recovered_candidate_count": 0,
                        "recovered_candidates": [],
                        "accepted_candidate_lineage_count": 0,
                        "accepted_candidate_lineage": [],
                    "synthesized_recommendation_count": 0,
                "duplicate_final_recommendation_count": 0,
                "synthesis_selection_safeguard": (
                    summarize_tr.deterministic_synthesis_backfill(
                        [], [], {},
                    )[1]
                ),
                "synthesis_backfill_candidate_count": 0,
                "coverage_equivalent_recommendation_count": 0,
                    "coverage_equivalent_recommendations": [],
                    "final_recommendation_audited_count": 0,
                    "final_recommendation_grounded_count": 0,
                    "final_recommendation_supported_count": 0,
                "published_recommendation_count": 0,
                "published_recommendation_limit": 8,
                "portfolio_excluded_recommendation_count": 0,
                "portfolio_excluded_recommendations": [],
                "final_recommendation_withheld_count": 0,
                    "final_withheld_reason_counts": {},
                    "final_withheld_recommendations": [],
                    "human_review_sampling": {
                        "policy": "deterministic",
                        "max_samples_per_run": 6,
                        "sampled_count": 0,
                        "sampled_refs": [],
                    },
                },
                "recommendations_with_recovered_source_count": 0,
                "recommendation_priority_policy": (
                    summarize_tr.deterministic_recommendation_priority_policy(
                        []
                    )
                ),
                "recommendation_portfolio_policy": (
                    summarize_tr.deterministic_recommendation_portfolio([])[2]
                ),
                "recommendations": [],
            "checklists": [checklist],
        }) + "\n", encoding="utf-8")
        report_path.write_text(
            "# Recommended Regression Areas\n\nNone.\n\n"
            "# Per-Bug Verification\n\n## A\n"
            + "\n".join(
                f"{index}. {step}" for index, step in enumerate(
                    checklist["steps"], start=1
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(process_tr, "post_json", fake_post)
    monkeypatch.setattr(process_tr.subprocess, "run", fake_subprocess)
    monkeypatch.setattr(sys, "argv", [
        "process_newest_tr.py", "--product", "pdri",
        "--product-dir", str(product_dir),
        "--parsed-csv", str(parsed),
        "--model", "gemma4:e2b-it-q4_K_M",
    ])

    assert process_tr.main() == 0
    latest_path = product_dir / "tr_reports" / "latest.json"
    first_latest = latest_path.read_text(encoding="utf-8")
    manifest = json.loads(first_latest)
    assert manifest["schema_version"] == 2
    assert not Path(manifest["report_path"]).is_absolute()
    assert len(list((product_dir / "tr_reports").glob("*/runs/*"))) == 1
    assert audit_report_manifest(product_dir, manifest)["warnings"] == []

    assert process_tr.main() == 0
    assert state["runs"] == 1
    status = json.loads((product_dir / "tr_reports" / "status.json").read_text())
    assert status["outcome"] == "unchanged"

    state["summary"] = "Promotion failure issue"
    original_replace = Path.replace

    def fail_run_promotion(path, target):
        if path.is_dir() and ".staging" in path.parts:
            raise OSError("injected run promotion failure")
        return original_replace(path, target)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "replace", fail_run_promotion)
        assert process_tr.main() == 1
    assert latest_path.read_text(encoding="utf-8") == first_latest
    status = json.loads((product_dir / "tr_reports" / "status.json").read_text())
    assert status["outcome"] == "validation_failed"

    state["summary"] = "Manifest switch failure issue"
    original_write_json = process_tr.write_json

    def fail_latest_switch(path, data):
        if Path(path) == latest_path:
            raise OSError("injected manifest switch failure")
        return original_write_json(path, data)

    with monkeypatch.context() as scoped:
        scoped.setattr(process_tr, "write_json", fail_latest_switch)
        assert process_tr.main() == 1
    assert latest_path.read_text(encoding="utf-8") == first_latest
    status = json.loads((product_dir / "tr_reports" / "status.json").read_text())
    assert status["outcome"] == "validation_failed"

    state["summary"] = "Analysis failure issue"
    state["fail"] = True
    assert process_tr.main() == 1
    assert latest_path.read_text(encoding="utf-8") == first_latest
    # The manifest-switch failure leaves one unreferenced immutable run, but
    # the dashboard remains pinned to the original complete snapshot.
    assert len(list((product_dir / "tr_reports").glob("*/runs/*"))) == 2
    status = json.loads((product_dir / "tr_reports" / "status.json").read_text())
    assert status["outcome"] == "analysis_failed"


def test_new_cohort_publishes_prior_hypothesis_evaluation(tmp_path, monkeypatch):
    product_dir = tmp_path / "pdri"
    product_dir.mkdir()
    parsed = product_dir / "ecl_parsed.csv"

    def write_cohort(code, version, build):
        parsed.write_text(
            "BugCode,Status,parsed_version,Close Build#,parsed_module,module_category\n"
            f"{code},TRCreated,{version},{build},Text to Image,AI Features\n",
            encoding="utf-8",
        )

    write_cohort("A", "16.7.0", "100")

    def fake_post(url, payload, timeout=300, **kwargs):
        code = payload["bug_codes"][0]
        return {
            "meta": {"close_build": payload["close_build"], "errors": []},
            "products": {"PowerDirector Mobile for iOS": [{
                "BugCode": code,
                "Short Description": f"Issue {code}",
                "ReproSteps": "Steps",
                "Resolutions": [{
                    "root_cause": "cause",
                    "solution": "Updated AI preview shared renderer state",
                }],
                "raw": {"a": 1, "b": 2, "c": 3, "d": 4},
            }]},
        }

    def fake_subprocess(command, check=False):
        details_path = Path(command[2])
        report_path = Path(command[command.index("--output") + 1])
        analysis_path = Path(command[command.index("--analysis-output") + 1])
        details = json.loads(details_path.read_text(encoding="utf-8"))
        code = next(iter(details["products"].values()))[0]["BugCode"]
        source_evidence = summarize_tr.structured_bug_evidence(
            next(iter(details["products"].values()))[0]
        )
        evidence_map = {code: source_evidence}
        checklist_evidence = summarize_tr.checklist_evidence_payload(
            source_evidence
        )
        checklist = summarize_tr.deterministic_checklist_grounding({
            "bug_code": code, "steps": ["Set up", "Verify"],
            "evidence_quality": source_evidence["evidence_quality"],
            "evidence_gaps": source_evidence["evidence_gaps"],
        }, checklist_evidence)
        source_hash, schema = process_tr.analysis_source_hash(details)
        taxonomy = [
            item["category"]
            for item in details["meta"]["module_category_taxonomy"]
        ] + [summarize_tr.UNKNOWN_TARGET_CATEGORY]
        category_support = summarize_tr.target_category_support(
            details, [source_evidence],
        )
        recommendation = {
            "target_category": "AI Features",
            "area": "AI preview",
            "action": "Open an AI preview",
            "trigger": "Open an AI preview after generation",
            "expected_failure_mode": "The AI preview crashes",
            **_prediction_boundaries("AI preview"),
            "rationale": "The AI preview uses the shared renderer path",
            "source_bug_codes": [code],
            "evidence_refs": [{
                "bug_code": code, "signal_index": 0,
                "quote": source_evidence["adjacency_signal_quotes"][0],
            }],
            "historical_analog_refs": [],
        }
        recommendation["confidence"] = "medium"
        recommendation["target_category_basis"] = (
            summarize_tr.deterministic_target_category_basis(
                recommendation["target_category"], recommendation["area"],
                recommendation["trigger"], recommendation["action"],
                recommendation["source_bug_codes"], category_support,
            )
        )
        recommendation["confidence_basis"] = (
            summarize_tr.deterministic_confidence_basis(
                recommendation["source_bug_codes"],
                    recommendation["evidence_refs"], [], evidence_map,
                    correlate_sources=True,
                    target_area=recommendation["area"],
                )
        )
        recommendation["adjacency_support_basis"] = (
            summarize_tr.deterministic_adjacency_support_basis(
                recommendation["area"], recommendation["evidence_refs"],
            )
        )
        (
            recommendation["evidence_refs"],
            recommendation["evidence_ref_completion_basis"],
        ) = summarize_tr.deterministic_evidence_ref_completion(
            recommendation["area"],
            recommendation["source_bug_codes"],
            recommendation["evidence_refs"],
            evidence_map,
        )
        recommendation["shared_path_support_basis"] = (
            summarize_tr.deterministic_shared_path_support_basis(
                recommendation["area"], recommendation["rationale"],
                recommendation["evidence_refs"],
                recommendation["adjacency_support_basis"],
            )
        )
        recommendation["risk_basis"] = (
            summarize_tr.deterministic_recommendation_risk_basis(
                recommendation, evidence_map,
            )
        )
        recommendation["risk"] = recommendation["risk_basis"]["label"]
        recommendation["impact_basis"] = recommendation["risk_basis"][
            "impact_basis"
        ]
        recommendation["priority_basis"] = (
            summarize_tr.deterministic_recommendation_priority_basis(
                recommendation, evidence_map,
            )
        )
        recommendation["priority_rank"] = 1
        source_candidate_id = summarize_tr.stable_candidate_audit_id(
            recommendation
        )
        recommendation["synthesis_source_candidate_ids"] = [
            source_candidate_id
        ]
        recommendation["candidate_generation_origins"] = [
            "primary_discovery"
        ]
        recommendation["prediction_origin"] = "initial_audit_pass_only"
        recommendation["recovered_source_candidate_ids"] = []
        recommendation["synthesis_lineage"] = [{
            "candidate_id": source_candidate_id,
            "origin": "initial_audit_pass",
            "original_candidate_id": source_candidate_id,
        }]
        recommendation["hypothesis_id"] = summarize_tr.stable_hypothesis_id(
            recommendation
        )
        recommendation["grounding_audit_provenance"] = (
            "draft_and_final_adversarial_passes"
        )
        recommendation["selection_origin"] = "model_synthesis"
        analysis_path.write_text(json.dumps({
            "schema_version": schema,
            "engine_sha256": process_tr.analysis_engine_sha256(),
            "model": "model",
            "generated_at": f"2026-08-10T00:00:0{1 if code == 'A' else 2}+00:00",
            "source_sha256": source_hash,
                "bug_count": 1,
                "bug_codes": [code],
                "evidence_quality_by_bug": {code: "high"},
                "evidence_independence_by_bug": {
                    code: {
                        "evidence_quality": "high",
                        "parsed_module": source_evidence[
                            "parsed_module"
                        ],
                        "adjacency_signal_quotes": source_evidence[
                            "adjacency_signal_quotes"
                        ],
                    },
                },
                    "target_category_support": category_support,
                    "checklist_evidence_by_bug": {code: checklist_evidence},
                "target_alias_policy": (
                    prediction_reviews.compact_target_alias_policy(None)
                ),
                "target_category_taxonomy": taxonomy,
            "candidate_grounding_audit": {
                "schema_version": 1,
                "candidate_discovery_audit": _candidate_discovery_audit([
                    recommendation
                ]),
                "generated_candidate_count": 1,
                "duplicate_candidate_count": 0,
                "audited_candidate_count": 1,
                "supported_candidate_count": 1,
                "withheld_candidate_count": 0,
                    "withheld_reason_counts": {},
                    "withheld_candidates": [],
                    "initial_supported_candidate_count": 1,
                    "initial_withheld_candidate_count": 0,
                    "initial_withheld_reason_counts": {},
                    "repair_eligible_candidate_count": 0,
                    "repair_ineligible_candidate_count": 0,
                    "repair_attempted_candidate_count": 0,
                    "repair_proposed_candidate_count": 0,
                    "repair_declined_candidate_count": 0,
                    "repair_unavailable_candidate_count": 0,
                    "repair_reaudited_candidate_count": 0,
                    "repair_reaudit_unavailable_candidate_count": 0,
                    "repair_supported_candidate_count": 0,
                    "repair_withheld_candidate_count": 0,
                        "recovered_candidate_count": 0,
                        "recovered_candidates": [],
                        "accepted_candidate_lineage_count": 1,
                        "accepted_candidate_lineage": [{
                            "candidate_id": source_candidate_id,
                            "origin": "initial_audit_pass",
                            "original_candidate_id": source_candidate_id,
                                "candidate_generation_origins": [
                                    "primary_discovery"
                                ],
                                "target_category": "AI Features",
                                "target_category_basis": recommendation[
                                    "target_category_basis"
                                ],
                            "area": "AI preview",
                            "source_bug_codes": [code],
                                "evidence_refs": recommendation[
                                    "evidence_refs"
                                ],
                                    "evidence_ref_completion_basis": (
                                        recommendation[
                                            "evidence_ref_completion_basis"
                                        ]
                                    ),
                                "historical_analog_refs": [],
                                "confidence": recommendation["confidence"],
                                "confidence_basis": recommendation[
                                    "confidence_basis"
                                ],
                                    "adjacency_support_basis": recommendation[
                                        "adjacency_support_basis"
                                    ],
                                        "shared_path_support_basis": recommendation[
                                            "shared_path_support_basis"
                                        ],
                                        "scenario_specificity_basis": recommendation[
                                            "scenario_specificity_basis"
                                        ],
                                "action": recommendation["action"],
                                "rationale": recommendation["rationale"],
                            "trigger": recommendation["trigger"],
                            "expected_failure_mode": recommendation[
                                "expected_failure_mode"
                            ],
                            "scope_boundary": recommendation["scope_boundary"],
                            "disconfirming_observation": recommendation[
                                "disconfirming_observation"
                            ],
                        }],
                        "synthesized_recommendation_count": 1,
                    "duplicate_final_recommendation_count": 0,
                    "synthesis_selection_safeguard": (
                        summarize_tr.deterministic_synthesis_backfill(
                            [recommendation], [recommendation], evidence_map,
                        )[1]
                    ),
                    "synthesis_backfill_candidate_count": 0,
                    "coverage_equivalent_recommendation_count": 0,
                        "coverage_equivalent_recommendations": [],
                        "final_recommendation_audited_count": 1,
                    "final_recommendation_grounded_count": 1,
                    "final_recommendation_supported_count": 1,
                "published_recommendation_count": 1,
                "published_recommendation_limit": 8,
                "portfolio_excluded_recommendation_count": 0,
                "portfolio_excluded_recommendations": [],
                "final_recommendation_withheld_count": 0,
                    "final_withheld_reason_counts": {},
                    "final_withheld_recommendations": [],
                    "human_review_sampling": {
                        "policy": "deterministic",
                        "max_samples_per_run": 6,
                        "sampled_count": 0,
                        "sampled_refs": [],
                    },
                },
                "recommendations_with_recovered_source_count": 0,
                "recommendation_priority_policy": (
                    summarize_tr.deterministic_recommendation_priority_policy(
                        [recommendation]
                    )
                ),
                "recommendation_portfolio_policy": (
                    summarize_tr.deterministic_recommendation_portfolio(
                        [recommendation]
                    )[2]
                ),
                "recommendations": [recommendation],
            "checklists": [checklist],
        }) + "\n", encoding="utf-8")
        report_path.write_text(
            "# Recommended Regression Areas\n\nNone.\n\n"
            f"# Per-Bug Verification\n\n## {code}\n"
            + "\n".join(
                f"{index}. {step}" for index, step in enumerate(
                    checklist["steps"], start=1
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(process_tr, "post_json", fake_post)
    monkeypatch.setattr(process_tr.subprocess, "run", fake_subprocess)
    monkeypatch.setattr(sys, "argv", [
        "process_newest_tr.py", "--product", "pdri",
        "--product-dir", str(product_dir),
        "--parsed-csv", str(parsed),
        "--model", "model",
    ])

    assert process_tr.main() == 0
    calibration_reviews = (
        [("confirmed", score) for score in (
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
        )]
        + [("rejected", score) for score in (0.25, 0.28, 0.30, 0.35)]
    )
    for index, (decision, score) in enumerate(calibration_reviews):
        prediction_reviews.upsert_review(
            product_dir / "tr_reports" / "prediction_reviews.json",
            product="pdri",
            hypothesis_id=(
                "TRH-AAAAAAAAAAAA" if index % 2 == 0
                else "TRH-BBBBBBBBBBBB"
            ),
            prior_close_build=f"9{index % 3}",
            current_close_build=f"9{index % 3 + 3}",
            bug_code=f"CAL-{index}",
            target_category="AI Features",
            predicted_area="AI preview",
            decision=decision,
            rejection_reason=(
                "lexical_overlap_only" if decision == "rejected" else None
            ),
            match_score=score,
            overlap_terms=["ai", "preview"],
            matcher_policy=prediction_reviews.FEATURE_MATCHER_POLICY,
            overlap_facets={
                "area": ["ai", "preview"],
                "trigger": ["generation"],
                "failure_mode": ["crash"],
            },
        )
    write_cohort("B", "16.8.0", "101")
    assert process_tr.main() == 0

    manifest = json.loads(
        (product_dir / "tr_reports" / "latest.json").read_text(encoding="utf-8")
    )
    analysis_path = process_tr.resolve_product_artifact(
        product_dir, manifest["analysis_path"]
    )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    evaluation = analysis["prior_hypothesis_evaluation"]
    assert evaluation["schema_version"] == 10
    assert evaluation["matcher_policy"] == (
        prediction_reviews.FEATURE_MATCHER_POLICY
    )
    assert evaluation["prior_close_build"] == "100"
    assert evaluation["current_close_build"] == "101"
    assert evaluation["category_overlap_count"] == 1
    assert evaluation["hypotheses"][0]["matching_bug_codes"] == ["B"]
    assert evaluation["feature_match_threshold"] > 0.24
    assert (
        evaluation["threshold_calibration"]["global"]["status"]
        == "calibrated_higher"
    )
    assert (
        evaluation["threshold_calibration"]["categories"]["AI Features"]["status"]
        == "calibrated_higher"
    )
    assert evaluation["hypotheses"][0]["threshold_source"] == "category_reviewed"
    assert evaluation["hypotheses"][0]["feature_match_threshold"] > 0.24
    assert evaluation["hypotheses"][0]["threshold_decisive_review_count"] == 12
    assert (
        evaluation["hypotheses"][0]["threshold_effective_decisive_review_count"]
        == 12
    )
    assert evaluation["hypotheses"][0]["threshold_reviewed_cohort_count"] == 3


def test_manifest_audit_flags_legacy_mismatch_and_rejects_path_escape(tmp_path):
    product_dir = tmp_path / "pdri"
    report_dir = product_dir / "tr_reports" / "old"
    report_dir.mkdir(parents=True)
    details = report_dir / "bug_details.json"
    report = report_dir / "test_focus_summary.md"
    details.write_text(json.dumps({"products": {"P": [{"BugCode": "OTHER"}]}}))
    report.write_text("report")
    legacy = {
        "bug_codes": ["A"],
        "details_path": str(details),
        "report_path": str(report),
    }
    audit = audit_report_manifest(product_dir, legacy)
    assert any("legacy report" in warning for warning in audit["warnings"])
    assert any("do not match" in warning for warning in audit["warnings"])

    escaped = audit_report_manifest(product_dir, {"report_path": "../outside.md"})
    assert escaped["report_safe"] is False
    assert any("Unsafe artifact" in warning for warning in escaped["warnings"])


def test_manifest_cache_invalidates_when_analysis_engine_changes(tmp_path):
    product_dir = tmp_path / "pdri"
    run_dir = product_dir / "tr_reports" / "build" / "runs" / "run"
    run_dir.mkdir(parents=True)
    paths = {}
    for name in ("bug_details.json", "analysis.json", "test_focus_summary.md"):
        path = run_dir / name
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    manifest = {
        "schema_version": process_tr.MANIFEST_SCHEMA_VERSION,
        "source_sha256": "source",
        "engine_sha256": "engine-v1",
        "model": "model",
        "analysis_schema_version": summarize_tr.ANALYSIS_SCHEMA_VERSION,
        "close_build": "build",
        "bug_codes": ["A"],
        "details_path": paths["bug_details.json"].relative_to(product_dir).as_posix(),
        "details_sha256": process_tr.sha256_file(paths["bug_details.json"]),
        "analysis_path": paths["analysis.json"].relative_to(product_dir).as_posix(),
        "analysis_sha256": process_tr.sha256_file(paths["analysis.json"]),
        "report_path": paths["test_focus_summary.md"].relative_to(product_dir).as_posix(),
        "report_sha256": process_tr.sha256_file(paths["test_focus_summary.md"]),
    }
    kwargs = {
        "source_hash": "source",
        "model": "model",
        "analysis_schema": summarize_tr.ANALYSIS_SCHEMA_VERSION,
        "close_build": "build",
        "bug_codes": ["A"],
    }
    assert process_tr.manifest_cache_hit(
        product_dir, manifest, engine_sha256="engine-v1", **kwargs
    )
    assert not process_tr.manifest_cache_hit(
        product_dir, manifest, engine_sha256="engine-v2", **kwargs
    )


def test_structured_prediction_identity_validation():
    analysis = {
        "schema_version": 4,
        "target_category_taxonomy": ["AI Features"],
        "recommendations": [{
            "hypothesis_id": "TRH-0123456789AB",
            "target_category": "AI Features",
        }],
    }
    assert process_tr.structured_prediction_errors(analysis) == []

    analysis["recommendations"].append({
        "hypothesis_id": "TRH-0123456789AB",
        "target_category": "Invented",
    })
    errors = process_tr.structured_prediction_errors(analysis)
    assert any("invalid target_category" in error for error in errors)
    assert any("duplicate hypothesis_id" in error for error in errors)


def test_schema5_prediction_identity_requires_feature_fields_and_exact_hash():
    recommendation = {
        "target_category": "Editor Core",
        "area": "Editor preview",
        "action": "Verify preview stability",
        "trigger": "Open the editor preview after image generation",
        "expected_failure_mode": "The editor preview crashes or shows a black frame",
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
    }
    recommendation["hypothesis_id"] = summarize_tr.stable_hypothesis_id(
        recommendation
    )
    analysis = {
        "schema_version": 5,
        "target_category_taxonomy": ["Editor Core"],
        "recommendations": [recommendation],
    }
    assert process_tr.structured_prediction_errors(analysis) == []

    recommendation["expected_failure_mode"] = ""
    errors = process_tr.structured_prediction_errors(analysis)
    assert any("invalid feature field" in error for error in errors)

    recommendation["expected_failure_mode"] = "The editor preview crashes"
    errors = process_tr.structured_prediction_errors(analysis)
    assert any("does not match structured feature identity" in error for error in errors)


def test_schema6_prediction_identity_validates_historical_analog_refs():
    recommendation = {
        "target_category": "Editor Core",
        "area": "Timeline preview",
        "action": "Verify Timeline preview stability",
        "trigger": "Open Timeline preview after image generation",
        "expected_failure_mode": "Timeline preview crashes",
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [{
            "source_bug_code": "A",
            "analog_bug_code": "H1",
            "module": "Timeline",
            "summary": "Timeline preview crashed",
            "similarity_score": 0.60,
        }],
    }
    recommendation["hypothesis_id"] = summarize_tr.stable_hypothesis_id(
        recommendation
    )
    analysis = {
        "schema_version": 6,
        "target_category_taxonomy": ["Editor Core"],
        "recommendations_with_historical_support_count": 1,
        "recommendations": [recommendation],
    }
    assert process_tr.structured_prediction_errors(analysis) == []

    recommendation["historical_analog_refs"][0]["similarity_score"] = 2
    errors = process_tr.structured_prediction_errors(analysis)
    assert any("invalid similarity score" in error for error in errors)

    recommendation["historical_analog_refs"][0]["similarity_score"] = 0.60
    recommendation["historical_analog_refs"][0]["source_bug_code"] = "OTHER"
    errors = process_tr.structured_prediction_errors(analysis)
    assert any("invalid source BugCode" in error for error in errors)


def test_schema7_prediction_identity_requires_valid_boundaries():
    recommendation = {
        "target_category": "Editor Core",
        "area": "Timeline preview",
        "action": "Verify Timeline preview stability",
        "trigger": "Open Timeline preview after image generation",
        "expected_failure_mode": "Timeline preview crashes",
        **_prediction_boundaries("Timeline preview"),
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [],
    }
    recommendation["hypothesis_id"] = summarize_tr.stable_hypothesis_id(
        recommendation
    )
    analysis = {
        "schema_version": 7,
        "target_category_taxonomy": ["Editor Core"],
        "recommendations_with_historical_support_count": 0,
        "recommendations": [recommendation],
    }
    assert process_tr.structured_prediction_errors(analysis) == []
    evaluation = process_tr.evaluate_prior_hypotheses(
        analysis,
        {"products": {"P": [{
            "BugCode": "NEW-1",
            "ModuleCategory": "Editor Core",
        }]}},
        prior_close_build="100",
        current_close_build="101",
    )
    hypothesis = evaluation["hypotheses"][0]
    assert hypothesis["prediction_boundary_provenance"] == "schema7_validated"
    assert hypothesis["scope_boundary"] == recommendation["scope_boundary"]
    assert hypothesis["disconfirming_observation"] == recommendation[
        "disconfirming_observation"
    ]

    recommendation["disconfirming_observation"] = "Maybe investigate later"
    errors = process_tr.structured_prediction_errors(analysis)
    assert any("observable passing result" in error for error in errors)


def test_schema8_prediction_requires_consistent_grounding_audit_provenance():
    recommendation = {
        "target_category": "Editor Core",
        "area": "Timeline preview",
        "action": "Verify Timeline preview stability",
        "trigger": "Open Timeline preview after image generation",
        "expected_failure_mode": "Timeline preview crashes",
        **_prediction_boundaries("Timeline preview"),
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [],
        "grounding_audit_provenance": "draft_and_final_adversarial_passes",
    }
    recommendation["hypothesis_id"] = summarize_tr.stable_hypothesis_id(
        recommendation
    )
    analysis = {
        "schema_version": 8,
        "target_category_taxonomy": ["Editor Core"],
        "recommendations_with_historical_support_count": 0,
        "candidate_grounding_audit": {
            "schema_version": 1,
            "generated_candidate_count": 1,
            "duplicate_candidate_count": 0,
            "audited_candidate_count": 1,
            "supported_candidate_count": 1,
            "withheld_candidate_count": 0,
            "withheld_reason_counts": {},
            "withheld_candidates": [],
            "synthesized_recommendation_count": 1,
            "duplicate_final_recommendation_count": 0,
            "final_recommendation_audited_count": 1,
            "final_recommendation_supported_count": 1,
            "final_recommendation_withheld_count": 0,
                "final_withheld_reason_counts": {},
                "final_withheld_recommendations": [],
                "human_review_sampling": {
                    "policy": "deterministic",
                    "max_samples_per_run": 6,
                    "sampled_count": 0,
                    "sampled_refs": [],
                },
        },
        "recommendations": [recommendation],
    }
    assert process_tr.structured_prediction_errors(analysis) == []

    analysis["candidate_grounding_audit"]["supported_candidate_count"] = 0
    errors = process_tr.structured_prediction_errors(analysis)
    assert any("decision count mismatch" in error for error in errors)
    assert any("exceed supported candidates" in error for error in errors)

    analysis["candidate_grounding_audit"]["supported_candidate_count"] = 1
    recommendation.pop("grounding_audit_provenance")
    errors = process_tr.structured_prediction_errors(analysis)
    assert any("lack grounding-audit provenance" in error for error in errors)


def test_hypothesis_id_is_stable_and_target_sensitive():
    item = {
        "target_category": "Editor Core",
        "area": "Editor preview",
        "action": "Verify preview stability",
        "trigger": "Open the editor preview after generation",
        "expected_failure_mode": "The editor preview crashes",
        **_prediction_boundaries("Editor preview"),
        "source_bug_codes": ["B", "A"],
        "evidence_refs": [
            {"bug_code": "B", "signal_index": 1},
            {"bug_code": "A", "signal_index": 0},
        ],
    }
    first = summarize_tr.stable_hypothesis_id(item)
    reordered = {
        **item,
        "source_bug_codes": ["A", "B"],
        "evidence_refs": list(reversed(item["evidence_refs"])),
    }
    assert summarize_tr.stable_hypothesis_id(reordered) == first
    assert re.fullmatch(r"TRH-[A-F0-9]{12}", first)
    changed = {**item, "target_category": "AI Features"}
    assert summarize_tr.stable_hypothesis_id(changed) != first
    changed_boundary = {
        **item,
        "scope_boundary": (
            "Limit Editor preview to another path; independent paths are outside scope"
        ),
    }
    assert summarize_tr.stable_hypothesis_id(changed_boundary) != first


def test_prior_hypothesis_evaluation_is_coarse_category_overlap_only():
    prior = {
        "generated_at": "2026-08-01T00:00:00+00:00",
        "recommendations": [
            {
                "hypothesis_id": "TRH-000000000001",
                "target_category": "AI Features",
                "area": "AI preview",
            },
            {
                "hypothesis_id": "TRH-000000000002",
                "target_category": "Editor Core",
                "area": "Timeline preview",
            },
            {
                "hypothesis_id": "TRH-000000000003",
                "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
                "area": "Shared service",
            },
        ],
    }
    current = {"products": {"P": [
        {"BugCode": "NEW-1", "ModuleCategory": "AI Features"},
        {"BugCode": "NEW-2", "ModuleCategory": "AI Features"},
    ]}}
    result = process_tr.evaluate_prior_hypotheses(
        prior,
        current,
        prior_close_build="100",
        current_close_build="101",
    )
    assert result["scoreable_hypothesis_count"] == 2
    assert result["category_evaluable_hypothesis_count"] == 2
    assert result["scenario_evaluable_hypothesis_count"] == 0
    assert result["insufficient_prediction_detail_count"] == 2
    assert result["feature_candidate_rate"] is None
    assert result["category_overlap_count"] == 1
    assert result["category_overlap_rate"] == 0.5
    assert result["hypotheses"][0]["result"] == (
        "category_overlap_insufficient_prediction_detail"
    )
    assert result["hypotheses"][0]["scenario_evaluation_status"] == (
        "insufficient_prediction_detail"
    )
    assert result["hypotheses"][0]["scenario_evaluation_reasons"] == [
        "missing_trigger", "missing_failure_mode"
    ]
    assert result["hypotheses"][0]["matching_bug_codes"] == ["NEW-1", "NEW-2"]
    assert result["hypotheses"][1]["result"] == "not_observed_in_current_cohort"
    assert result["hypotheses"][2]["result"] == "unscorable_cross_cutting"
    assert result["hypotheses"][2]["scenario_evaluation_status"] == (
        "unscorable_target"
    )
    assert "does not prove" in result["limitation"]


def test_schema6_evaluation_preserves_historical_support_provenance():
    prior = {
        "schema_version": 6,
        "generated_at": "2026-08-01T00:00:00+00:00",
        "recommendations": [
            {
                "hypothesis_id": "TRH-000000000001",
                "target_category": "AI Features",
                "area": "AI preview",
                "historical_analog_refs": [{
                    "source_bug_code": "OLD-1",
                    "analog_bug_code": "HIST-1",
                    "similarity_score": 0.72,
                }],
            },
            {
                "hypothesis_id": "TRH-000000000002",
                "target_category": "AI Features",
                "area": "AI export",
                "historical_analog_refs": [],
            },
        ],
    }
    result = process_tr.evaluate_prior_hypotheses(
        prior,
        {"products": {"P": [{
            "BugCode": "NEW-1",
            "ModuleCategory": "AI Features",
        }]}},
        prior_close_build="100",
        current_close_build="101",
    )

    assert result["schema_version"] == 10
    assert result["historical_support_provenance_hypothesis_count"] == 2
    assert result["historical_support_hypothesis_count"] == 1
    supported, unsupported = result["hypotheses"]
    assert supported["historical_support"] is True
    assert supported["historical_analog_codes"] == ["HIST-1"]
    assert supported["historical_analog_max_similarity"] == 0.72
    assert unsupported["historical_support"] is False
    assert unsupported["historical_analog_codes"] == []


def test_prior_evaluation_preserves_exact_applied_guardrail_configuration():
    engine = "a" * 64
    prior = {
        "schema_version": 7,
        "model": "gemma4",
        "engine_sha256": engine,
        "analysis_guardrail_policy": {
            "schema_version": 1,
            "policy": "controlled",
            "active_guardrails": {
                "Editor Core": {
                    "guardrail": "require_stronger_shared_path_grounding",
                    "source_rejection_reason": "unsupported_shared_path",
                }
            },
        },
        "recommendations": [{
            "hypothesis_id": "TRH-000000000001",
            "target_category": "Editor Core",
            "area": "Editor preview",
            "scope_boundary": (
                "Limit Editor preview to shared state; independent paths are "
                "outside scope"
            ),
            "disconfirming_observation": (
                "Lower priority if Editor preview remains correct"
            ),
        }, {
            "hypothesis_id": "TRH-000000000002",
            "target_category": "Audio",
            "area": "Audio preview",
            "scope_boundary": (
                "Limit Audio preview to shared state; independent paths are "
                "outside scope"
            ),
            "disconfirming_observation": (
                "Lower priority if Audio preview remains correct"
            ),
        }],
    }
    result = process_tr.evaluate_prior_hypotheses(
        prior,
        {"products": {"P": []}},
        prior_close_build="100",
        current_close_build="101",
    )
    editor, audio = result["hypotheses"]

    assert result["schema_version"] == 10
    assert result["analysis_guardrail_provenance_hypothesis_count"] == 2
    assert result["applied_analysis_guardrail_hypothesis_count"] == 1
    assert editor["analysis_guardrail_provenance"] == "exact_policy"
    assert editor["analysis_guardrail"] == (
        "require_stronger_shared_path_grounding"
    )
    assert editor["analysis_guardrail_source_rejection_reason"] == (
        "unsupported_shared_path"
    )
    assert editor["analysis_model"] == "gemma4"
    assert editor["analysis_engine_sha256"] == engine
    assert audio["analysis_guardrail_provenance"] == "exact_policy"
    assert audio["analysis_guardrail"] is None


def test_prior_evaluation_preserves_only_valid_exact_synthesis_lineage():
    engine = "f" * 64
    source_id = "TRC-000000000001"
    prior = {
        "schema_version": 10,
        "model": "gemma4",
        "engine_sha256": engine,
        "recommendations": [{
            "hypothesis_id": "TRH-000000000001",
            "target_category": "Editor Core",
            "area": "Timeline preview",
            "prediction_origin": "recovered_draft_only",
            "synthesis_source_candidate_ids": [source_id],
            "recovered_source_candidate_ids": [source_id],
        }, {
            "hypothesis_id": "TRH-000000000002",
            "target_category": "Editor Core",
            "area": "Timeline export",
            "prediction_origin": "mixed_initial_and_recovered",
            "synthesis_source_candidate_ids": ["TRC-000000000002"],
            "recovered_source_candidate_ids": ["TRC-000000000002"],
        }],
    }
    result = process_tr.evaluate_prior_hypotheses(
        prior,
        {"products": {"P": []}},
        prior_close_build="100",
        current_close_build="101",
    )

    assert result["schema_version"] == 10
    assert result["synthesis_lineage_provenance_hypothesis_count"] == 1
    assert result["recovered_origin_hypothesis_count"] == 1
    exact, malformed = result["hypotheses"]
    assert exact["prediction_origin_provenance"] == (
        "schema10_exact_synthesis_lineage"
    )
    assert exact["prediction_origin"] == "recovered_draft_only"
    assert exact["synthesis_source_candidate_ids"] == [source_id]
    assert exact["recovered_source_candidate_ids"] == [source_id]
    assert exact["analysis_model"] == "gemma4"
    assert exact["analysis_engine_sha256"] == engine
    assert malformed["prediction_origin_provenance"] == "unavailable_legacy"
    assert malformed["prediction_origin"] == "legacy_unknown"
    assert malformed["synthesis_source_candidate_ids"] == []

    prior["schema_version"] = 11
    coherent = process_tr.evaluate_prior_hypotheses(
        prior,
        {"products": {"P": []}},
        prior_close_build="100",
        current_close_build="101",
    )["hypotheses"][0]
    assert coherent["prediction_origin_provenance"] == (
        "schema11_exact_claim_lineage"
    )


def test_prior_evaluation_does_not_guess_legacy_guardrail_provenance():
    result = process_tr.evaluate_prior_hypotheses(
        {
            "schema_version": 7,
            "model": "gemma4",
            "engine_sha256": "not-a-sha",
            "recommendations": [{
                "hypothesis_id": "TRH-000000000001",
                "target_category": "Editor Core",
                "area": "Editor preview",
            }],
        },
        {"products": {"P": []}},
        prior_close_build="100",
        current_close_build="101",
    )

    hypothesis = result["hypotheses"][0]
    assert hypothesis["analysis_guardrail_provenance"] == "legacy_unknown"
    assert hypothesis["analysis_guardrail"] is None
    assert hypothesis["analysis_model"] is None


def test_prior_hypothesis_feature_match_is_review_candidate_not_confirmation():
    prior = {
        "generated_at": "2026-08-01T00:00:00+00:00",
        "recommendations": [{
            "hypothesis_id": "TRH-000000000001",
            "target_category": "Editor Core",
            "area": "Editor preview",
            "trigger": "Open the editor preview after image generation",
            "expected_failure_mode": "The editor preview crashes or shows a black frame",
        }],
    }
    current = {"products": {"P": [
        {
            "BugCode": "NEW-1",
            "ModuleCategory": "Editor Core",
            "ParsedModule": "Editor Preview",
            "Short Description": "Editor preview crashes after image generation",
            "ReproSteps": "Generate an image and open the editor preview",
        },
        {
            "BugCode": "NEW-2",
            "ModuleCategory": "Editor Core",
            "ParsedModule": "Timeline",
            "Short Description": "Timeline title is clipped",
        },
        {
            "BugCode": "NEW-3",
            "ModuleCategory": "Editor Core",
            "ParsedModule": "Editor Preview",
            "Short Description": (
                "Editor preview subtitle overlay alignment position duration "
                "opacity handles colors fonts animation generation tracking "
                "blending masks shadows kerning"
            ),
        },
    ]}}
    result = process_tr.evaluate_prior_hypotheses(
        prior,
        current,
        prior_close_build="100",
        current_close_build="101",
    )
    hypothesis = result["hypotheses"][0]
    assert hypothesis["result"] == "feature_match_candidate"
    assert [item["bug_code"] for item in hypothesis["feature_match_candidates"]] == [
        "NEW-1"
    ]
    assert hypothesis["feature_match_candidates"][0]["review_status"] == "unreviewed"
    assert hypothesis["feature_match_candidates"][0]["candidate_kind"] == "feature_match"
    assert "preview" in hypothesis["feature_match_candidates"][0]["overlap_terms"]
    assert hypothesis["feature_match_candidates"][0]["matcher_policy"] == (
        prediction_reviews.FEATURE_MATCHER_POLICY
    )
    assert hypothesis["feature_match_candidates"][0]["overlap_facets"] == {
        "area": ["editor", "preview"],
        "trigger": ["generation", "image", "open"],
        "failure_mode": ["crashes"],
    }
    assert [
        item["bug_code"] for item in hypothesis["threshold_audit_candidates"]
    ] == ["NEW-3"]
    assert (
        hypothesis["threshold_audit_candidates"][0]["candidate_kind"]
        == "threshold_audit"
    )
    assert hypothesis["threshold_audit_candidates"][0]["match_score"] < 0.24
    assert result["feature_candidate_hypothesis_count"] == 1
    assert result["scenario_evaluable_hypothesis_count"] == 1
    assert result["scenario_evaluable_category_overlap_count"] == 1
    assert result["feature_candidate_rate"] == 1.0
    assert result["feature_candidate_rate_given_category_observed"] == 1.0
    assert result["threshold_audit_hypothesis_count"] == 1
    assert result["threshold_audit_candidate_count"] == 1
    assert "require human confirmation" in result["limitation"]
    assert "not predictions" in result["limitation"]

    suppressed = process_tr.evaluate_prior_hypotheses(
        prior,
        current,
        prior_close_build="100",
        current_close_build="101",
        feature_match_threshold=0.95,
        threshold_calibration={"status": "calibrated_higher", "threshold": 0.95},
    )
    assert suppressed["hypotheses"][0]["result"] == "category_overlap"
    assert suppressed["feature_candidate_hypothesis_count"] == 0
    assert suppressed["feature_match_threshold"] == 0.95
    assert suppressed["threshold_calibration"]["status"] == "calibrated_higher"

    category_suppressed = process_tr.evaluate_prior_hypotheses(
        prior,
        current,
        prior_close_build="100",
        current_close_build="101",
        feature_match_threshold=0.24,
        category_thresholds={
            "Editor Core": {
                "threshold": 0.95,
                "source": "category_reviewed",
                "status": "calibrated_higher",
            }
        },
    )
    category_hypothesis = category_suppressed["hypotheses"][0]
    assert category_hypothesis["result"] == "category_overlap"
    assert category_hypothesis["feature_match_threshold"] == 0.95
    assert category_hypothesis["threshold_source"] == "category_reviewed"


def test_scenario_match_rates_exclude_category_only_hypotheses():
    prior = {
        "recommendations": [{
            "hypothesis_id": "TRH-000000000001",
            "target_category": "Editor Core",
            "area": "Editor preview",
            "trigger": "Open Editor preview after image generation",
            "expected_failure_mode": "Editor preview crashes after generation",
        }, {
            "hypothesis_id": "TRH-000000000002",
            "target_category": "Editor Core",
            "area": "Timeline preview",
        }, {
            "hypothesis_id": "TRH-000000000003",
            "target_category": "Audio",
            "area": "Audio preview",
            "trigger": "Resume Audio preview after device interruption",
            "expected_failure_mode": "Audio preview loses synchronized playback",
        }],
    }
    current = {"products": {"P": [{
        "BugCode": "NEW-1",
        "ModuleCategory": "Editor Core",
        "ParsedModule": "Editor Preview",
        "Short Description": "Editor preview crashes after image generation",
        "ReproSteps": "Generate an image and open Editor preview",
    }]}}

    result = process_tr.evaluate_prior_hypotheses(
        prior,
        current,
        prior_close_build="100",
        current_close_build="101",
    )

    assert result["category_evaluable_hypothesis_count"] == 3
    assert result["scenario_evaluable_hypothesis_count"] == 2
    assert result["insufficient_prediction_detail_count"] == 1
    assert result["category_overlap_count"] == 2
    assert result["scenario_evaluable_category_overlap_count"] == 1
    assert result["feature_candidate_hypothesis_count"] == 1
    assert result["feature_candidate_rate"] == 0.5
    assert result["feature_candidate_rate_given_category_observed"] == 1.0


def test_facet_matcher_rejects_area_only_overlap_and_repeated_target_words():
    hypothesis = {
        "area": "Editor preview",
        "trigger": "Open the editor preview after image generation",
        "expected_failure_mode": "The editor preview crashes or shows a black frame",
    }
    matches = process_tr.ranked_feature_matches(hypothesis, [{
        "BugCode": "AREA-ONLY",
        "ParsedModule": "Editor Preview",
        "Short Description": (
            "Editor preview subtitle overlay alignment and typography are incorrect"
        ),
        "Description": (
            "The editor preview has an editor preview layout issue"
        ),
    }])

    assert matches == []


def test_scenario_evaluability_rejects_generic_restatements():
    reasons = process_tr.scenario_evaluation_reasons({
        "area": "Editor preview",
        "trigger": "Check the editor preview",
        "expected_failure_mode": "The editor preview result is wrong",
    })

    assert reasons == ["no_distinctive_scenario_terms"]
    assert reasons == summarize_tr.prediction_specificity_reasons(
        "Editor preview",
        "Check the editor preview",
        "The editor preview result is wrong",
    )


def test_prediction_review_ledger_upserts_atomically_and_summarizes(tmp_path):
    path = tmp_path / "prediction_reviews.json"
    ledger = prediction_reviews.load_review_ledger(path, "pdri")
    assert ledger["reviews"] == {}

    first = prediction_reviews.upsert_review(
        path,
        product="pdri",
        hypothesis_id="TRH-0123456789AB",
        prior_close_build="100",
        current_close_build="101",
        bug_code="NEW-1",
        target_category="Editor Core",
        predicted_area="Editor preview",
        decision="confirmed",
        reviewer="QA",
        note="Matches the predicted crash after generation.",
        match_score=0.763,
        overlap_terms=["editor", "preview", "crash"],
        trigger="Open the editor preview after generation",
        expected_failure_mode="The editor preview crashes",
        candidate_kind="feature_match",
        applied_threshold=0.24,
    )
    assert re.fullmatch(r"TRR-[A-F0-9]{16}", first["review_key"])
    assert first["decision"] == "confirmed"
    assert first["match_score"] == 0.763
    assert first["overlap_terms"] == ["editor", "preview", "crash"]
    assert first["candidate_kind"] == "feature_match"
    assert first["applied_threshold"] == 0.24
    created_at = first["created_at"]

    updated = prediction_reviews.upsert_review(
        path,
        product="pdri",
        hypothesis_id="TRH-0123456789AB",
        prior_close_build="100",
        current_close_build="101",
        bug_code="NEW-1",
        target_category="Editor Core",
        predicted_area="Editor preview",
        decision="unclear",
        reviewer="QA2",
        note="Needs RD confirmation.",
        match_score=0.763,
        overlap_terms=["editor", "preview", "crash"],
    )
    assert updated["review_key"] == first["review_key"]
    assert updated["created_at"] == created_at
    assert updated["decision"] == "unclear"
    assert updated["match_score"] == 0.763
    assert updated["trigger"] == "Open the editor preview after generation"
    assert [event["decision"] for event in updated["history"]] == [
        "confirmed", "unclear"
    ]
    assert [event["reviewer"] for event in updated["history"]] == ["QA", "QA2"]
    assert [event["candidate_kind"] for event in updated["history"]] == [
        "feature_match", "feature_match"
    ]
    loaded = prediction_reviews.load_review_ledger(path, "pdri")
    assert len(loaded["reviews"]) == 1
    summary = prediction_reviews.review_summary(loaded)
    assert summary == {
        "reviewed_count": 1,
        "confirmed_count": 0,
        "rejected_count": 0,
        "unclear_count": 1,
        "decisive_count": 0,
        "reviewed_confirmation_rate": None,
        "prediction_reviewed_count": 1,
        "prediction_confirmed_count": 0,
        "prediction_rejected_count": 0,
        "prediction_unclear_count": 1,
        "prediction_decisive_count": 0,
        "threshold_audit_reviewed_count": 0,
        "threshold_audit_confirmed_count": 0,
        "prediction_rejection_reason_counts": {},
        "threshold_audit_rejection_reason_counts": {},
        "rejected_without_reason_count": 0,
    }


def test_matcher_policy_versions_review_identity_and_calibration_population(tmp_path):
    path = tmp_path / "prediction_reviews.json"
    common = {
        "product": "pdri",
        "hypothesis_id": "TRH-0123456789AB",
        "prior_close_build": "100",
        "current_close_build": "101",
        "bug_code": "NEW-1",
        "target_category": "Editor Core",
        "predicted_area": "Editor preview",
        "decision": "confirmed",
        "match_score": 0.6,
        "overlap_terms": ["editor", "preview", "generation"],
    }
    legacy = prediction_reviews.upsert_review(path, **common)
    facet = prediction_reviews.upsert_review(
        path,
        **common,
        matcher_policy=prediction_reviews.FEATURE_MATCHER_POLICY,
        overlap_facets={
            "area": ["editor", "preview"],
            "trigger": ["generation"],
            "failure_mode": [],
        },
    )

    assert legacy["review_key"] != facet["review_key"]
    assert len(prediction_reviews.load_review_ledger(path, "pdri")["reviews"]) == 2

    reviews = {}
    for index in range(12):
        reviews[f"review-{index}"] = {
            "decision": "confirmed" if index % 2 == 0 else "rejected",
            "rejection_reason": (
                None if index % 2 == 0 else "lexical_overlap_only"
            ),
            "match_score": 0.3 + index / 100,
            "prior_close_build": f"{100 + index % 3}",
            "current_close_build": f"{200 + index % 3}",
            "hypothesis_id": (
                "TRH-AAAAAAAAAAAA" if index % 2 == 0
                else "TRH-BBBBBBBBBBBB"
            ),
            "matcher_policy": (
                prediction_reviews.FEATURE_MATCHER_POLICY
                if index < 6 else "legacy_union_v1"
            ),
        }
    calibration = prediction_reviews.calibrate_match_threshold(
        {"reviews": reviews},
        matcher_policy=prediction_reviews.FEATURE_MATCHER_POLICY,
        min_decisive=6,
        min_selected=3,
    )

    assert calibration["matcher_policy"] == (
        prediction_reviews.FEATURE_MATCHER_POLICY
    )
    assert calibration["decisive_review_count"] == 6
    assert calibration["excluded_matcher_policy_review_count"] == 6


def test_threshold_audit_review_overlays_without_mutating_evaluation(tmp_path):
    path = tmp_path / "prediction_reviews.json"
    evaluation = {
        "prior_close_build": "100",
        "current_close_build": "101",
        "hypotheses": [{
            "hypothesis_id": "TRH-0123456789AB",
            "feature_match_candidates": [],
            "threshold_audit_candidates": [{
                "bug_code": "NEAR-1",
                "match_score": 0.239999,
                "candidate_kind": "threshold_audit",
                "applied_threshold": 0.24,
                "review_status": "unreviewed",
            }],
        }],
    }
    prediction_reviews.upsert_review(
        path,
        product="pdri",
        hypothesis_id="TRH-0123456789AB",
        prior_close_build="100",
        current_close_build="101",
        bug_code="NEAR-1",
        target_category="Editor Core",
        predicted_area="Editor preview",
        decision="confirmed",
        match_score=0.239999,
        candidate_kind="threshold_audit",
        applied_threshold=0.24,
    )
    rendered, summary = prediction_reviews.overlay_reviews(
        evaluation, prediction_reviews.load_review_ledger(path, "pdri")
    )

    audit = rendered["hypotheses"][0]["threshold_audit_candidates"][0]
    assert audit["review_status"] == "confirmed"
    assert audit["match_score"] == 0.239999
    assert audit["match_score"] < audit["applied_threshold"]
    assert evaluation["hypotheses"][0]["threshold_audit_candidates"][0][
        "review_status"
    ] == "unreviewed"
    assert summary["threshold_audit_reviewed_count"] == 1
    assert summary["threshold_audit_confirmed_count"] == 1
    assert summary["reviewed_confirmation_rate"] is None
    assert summary["prediction_reviewed_count"] == 0


def test_prediction_review_overlay_does_not_mutate_immutable_evaluation(tmp_path):
    path = tmp_path / "prediction_reviews.json"
    evaluation = {
        "prior_close_build": "100",
        "current_close_build": "101",
        "hypotheses": [{
            "hypothesis_id": "TRH-0123456789AB",
            "feature_match_candidates": [{
                "bug_code": "NEW-1",
                "review_status": "unreviewed",
            }],
        }],
    }
    prediction_reviews.upsert_review(
        path,
        product="pdri",
        hypothesis_id="TRH-0123456789AB",
        prior_close_build="100",
        current_close_build="101",
        bug_code="NEW-1",
        target_category="Editor Core",
        predicted_area="Editor preview",
        decision="rejected",
        rejection_reason="wrong_target",
        note="Same category but unrelated title layout bug.",
    )
    ledger = prediction_reviews.load_review_ledger(path, "pdri")
    rendered, summary = prediction_reviews.overlay_reviews(evaluation, ledger)
    candidate = rendered["hypotheses"][0]["feature_match_candidates"][0]
    assert candidate["review_status"] == "rejected"
    assert candidate["review_rejection_reason"] == "wrong_target"
    assert candidate["review_note"].startswith("Same category")
    assert evaluation["hypotheses"][0]["feature_match_candidates"][0] == {
        "bug_code": "NEW-1",
        "review_status": "unreviewed",
    }
    assert summary["rejected_count"] == 1
    assert summary["reviewed_confirmation_rate"] == 0.0
    assert summary["prediction_rejection_reason_counts"] == {
        "wrong_target": 1
    }


def test_rejected_review_requires_a_supported_structured_reason(tmp_path):
    path = tmp_path / "prediction_reviews.json"
    kwargs = {
        "product": "pdri",
        "hypothesis_id": "TRH-0123456789AB",
        "prior_close_build": "100",
        "current_close_build": "101",
        "bug_code": "NEW-1",
        "target_category": "Editor Core",
        "predicted_area": "Editor preview",
    }
    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="require a supported rejection_reason",
    ):
        prediction_reviews.upsert_review(
            path, **kwargs, decision="rejected"
        )
    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="requires a note",
    ):
        prediction_reviews.upsert_review(
            path, **kwargs, decision="rejected", rejection_reason="other"
        )
    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="only valid for rejected",
    ):
        prediction_reviews.upsert_review(
            path, **kwargs, decision="confirmed", rejection_reason="wrong_target"
        )

    entry = prediction_reviews.upsert_review(
        path,
        **kwargs,
        decision="rejected",
        rejection_reason="wrong_target",
        note="Candidate targets another preview flow.",
    )
    assert entry["rejection_reason"] == "wrong_target"
    assert entry["history"][0]["rejection_reason"] == "wrong_target"

    contextual_path = tmp_path / "contextual_reviews.json"
    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="requires cited analog support",
    ):
        prediction_reviews.upsert_review(
            contextual_path,
            **kwargs,
            decision="rejected",
            rejection_reason="historical_corroboration_misleading",
        )
    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="requires validated prediction boundaries",
    ):
        prediction_reviews.upsert_review(
            contextual_path,
            **kwargs,
            decision="rejected",
            rejection_reason="outside_scope",
        )


def test_legacy_reasonless_rejection_loads_but_is_not_calibration_evidence(
    tmp_path,
):
    path = tmp_path / "prediction_reviews.json"
    key = prediction_reviews.review_key(
        "TRH-0123456789AB", "100", "101", "NEW-1"
    )
    path.write_text(json.dumps({
        "schema_version": 1,
        "product": "pdri",
        "updated_at": "2026-08-10T00:00:00+00:00",
        "reviews": {key: {
            "review_key": key,
            "hypothesis_id": "TRH-0123456789AB",
            "prior_close_build": "100",
            "current_close_build": "101",
            "bug_code": "NEW-1",
            "decision": "rejected",
            "match_score": 0.30,
            "candidate_kind": "feature_match",
            "reviewer": "",
            "note": "Legacy review",
        }},
    }), encoding="utf-8")

    ledger = prediction_reviews.load_review_ledger(path, "pdri")
    summary = prediction_reviews.review_summary(ledger)
    calibration = prediction_reviews.calibrate_match_threshold(ledger)
    assert summary["rejected_without_reason_count"] == 1
    assert calibration["excluded_unroutable_rejection_count"] == 1
    assert calibration["decisive_review_count"] == 0


def test_prediction_review_persists_historical_support_without_changing_identity(
    tmp_path,
):
    path = tmp_path / "prediction_reviews.json"
    first = prediction_reviews.upsert_review(
        path,
        product="pdri",
        hypothesis_id="TRH-0123456789AB",
        prior_close_build="100",
        current_close_build="101",
        bug_code="NEW-1",
        target_category="Editor Core",
        predicted_area="Editor preview",
        decision="confirmed",
        match_score=0.65,
        historical_analog_codes=["HIST-1", "HIST-2"],
        historical_analog_max_similarity=0.74,
        **_prediction_boundaries("Editor preview"),
    )
    updated = prediction_reviews.upsert_review(
        path,
        product="pdri",
        hypothesis_id="TRH-0123456789AB",
        prior_close_build="100",
        current_close_build="101",
        bug_code="NEW-1",
        target_category="Editor Core",
        predicted_area="Editor preview",
        decision="rejected",
        rejection_reason="historical_corroboration_misleading",
    )

    assert updated["review_key"] == first["review_key"]
    assert updated["historical_support"] is True
    assert updated["historical_analog_codes"] == ["HIST-1", "HIST-2"]
    assert updated["historical_analog_count"] == 2
    assert updated["historical_analog_max_similarity"] == 0.74
    assert updated["scope_boundary"].startswith("Limit Editor preview")
    assert "remains correct" in updated["disconfirming_observation"]
    comparison = prediction_reviews.review_summary(
        prediction_reviews.load_review_ledger(path, "pdri")
    )["historical_support_comparison"]
    assert comparison["status"] == "insufficient_reviews"
    assert comparison["supported"]["rejected_review_count"] == 1
    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="provenance cannot change",
    ):
        prediction_reviews.upsert_review(
            path,
            product="pdri",
            hypothesis_id="TRH-0123456789AB",
            prior_close_build="100",
            current_close_build="101",
            bug_code="NEW-1",
            target_category="Editor Core",
            predicted_area="Editor preview",
            decision="confirmed",
            historical_analog_codes=["DIFFERENT"],
            historical_analog_max_similarity=0.80,
        )
    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="prediction boundaries cannot change",
    ):
        prediction_reviews.upsert_review(
            path,
            product="pdri",
            hypothesis_id="TRH-0123456789AB",
            prior_close_build="100",
            current_close_build="101",
            bug_code="NEW-1",
            target_category="Editor Core",
            predicted_area="Editor preview",
            decision="confirmed",
            **_prediction_boundaries("Different preview"),
        )


def test_malformed_prediction_review_ledger_is_never_replaced(tmp_path):
    path = tmp_path / "prediction_reviews.json"
    original = '{"schema_version": 99, "reviews": {}}\n'
    path.write_text(original, encoding="utf-8")
    with pytest.raises(prediction_reviews.ReviewLedgerError, match="unsupported"):
        prediction_reviews.upsert_review(
            path,
            product="pdri",
            hypothesis_id="TRH-0123456789AB",
            prior_close_build="100",
            current_close_build="101",
            bug_code="NEW-1",
            target_category="Editor Core",
            predicted_area="Editor preview",
            decision="confirmed",
        )
    assert path.read_text(encoding="utf-8") == original


def _calibration_ledger(scored_decisions):
    return {
        "reviews": {
            f"review-{index}": {
                "decision": decision,
                "rejection_reason": (
                    "lexical_overlap_only" if decision == "rejected" else None
                ),
                "match_score": score,
                "prior_close_build": f"9{index % 3}",
                "current_close_build": f"10{index % 3}",
                "hypothesis_id": (
                    "TRH-AAAAAAAAAAAA" if index % 2 == 0
                    else "TRH-BBBBBBBBBBBB"
                ),
            }
            for index, (decision, score) in enumerate(scored_decisions)
        }
    }


def _categorized_calibration_ledger(category_decisions):
    reviews = {}
    for category_index, (category, scored_decisions) in enumerate(
        category_decisions.items()
    ):
        for index, (decision, score) in enumerate(scored_decisions):
            reviews[f"{category}-{index}"] = {
                "decision": decision,
                "rejection_reason": (
                    "lexical_overlap_only" if decision == "rejected" else None
                ),
                "match_score": score,
                "target_category": category,
                "prior_close_build": f"9{index % 3}",
                "current_close_build": f"10{index % 3}",
                "hypothesis_id": (
                    f"category-{category_index}-a" if index % 2 == 0
                    else f"category-{category_index}-b"
                ),
            }
    return {"reviews": reviews}


def test_match_threshold_calibration_requires_sufficient_mixed_reviews():
    ledger = _calibration_ledger([
        ("confirmed", 0.8), ("confirmed", 0.7),
        ("rejected", 0.3), ("unclear", 0.9),
    ])
    result = prediction_reviews.calibrate_match_threshold(ledger)
    assert result["status"] == "insufficient_reviews"
    assert result["threshold"] == 0.24
    assert result["decisive_review_count"] == 3


def test_rejection_reason_routes_only_matcher_errors_into_match_calibration():
    decisions = (
        [("confirmed", score) for score in (
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
        )]
        + [("rejected", score) for score in (0.25, 0.28, 0.30, 0.35)]
    )
    ledger = _calibration_ledger(decisions)
    for entry in ledger["reviews"].values():
        if entry["decision"] == "rejected":
            entry["rejection_reason"] = "unsupported_shared_path"
    model_rejections_excluded = prediction_reviews.calibrate_match_threshold(
        ledger
    )
    assert model_rejections_excluded["status"] == "insufficient_reviews"
    assert model_rejections_excluded[
        "excluded_non_matcher_rejection_count"
    ] == 4
    assert model_rejections_excluded["decisive_review_count"] == 8

    for entry in ledger["reviews"].values():
        if entry["decision"] == "rejected":
            entry.pop("rejection_reason", None)
    unroutable_rejections = prediction_reviews.calibrate_match_threshold(ledger)
    assert unroutable_rejections["excluded_unroutable_rejection_count"] == 4
    assert unroutable_rejections["decisive_review_count"] == 8

    for entry in ledger["reviews"].values():
        if entry["decision"] == "rejected":
            entry["rejection_reason"] = "lexical_overlap_only"
    matcher_rejections = prediction_reviews.calibrate_match_threshold(ledger)
    assert matcher_rejections["status"] == "calibrated_higher"
    assert matcher_rejections["excluded_non_matcher_rejection_count"] == 0


def test_historical_support_comparison_requires_diverse_two_group_evidence():
    reviews = {}
    for supported in (True, False):
        for index in range(8):
            reviews[f"{supported}-{index}"] = {
                "decision": "confirmed" if supported else "rejected",
                "candidate_kind": "feature_match",
                "historical_support": supported,
                "prior_close_build": f"9{index}",
                "current_close_build": f"10{index}",
                "hypothesis_id": f"TRH-{index:012X}",
            }
    result = prediction_reviews.historical_support_comparison({
        "reviews": reviews
    })

    assert result["status"] == "supported_higher"
    assert result["supported"]["confirmation_rate"] == 1.0
    assert result["unsupported"]["confirmation_rate"] == 0.0
    assert result["difference_confidence_interval"][0] > 0
    assert result["observational_only"] is True
    assert result["eligible_for_policy_change"] is False


def test_historical_analog_floor_calibration_can_only_raise_with_safe_evidence():
    decisions = (
        [("confirmed", score) for score in (
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
        )]
        + [("rejected", score) for score in (0.20, 0.24, 0.28, 0.35)]
    )
    ledger = _calibration_ledger(decisions)
    for entry, (_decision, score) in zip(
        ledger["reviews"].values(), decisions
    ):
        entry.update({
            "historical_support": True,
            "historical_analog_max_similarity": score,
            "target_category": "AI Features",
            "rejection_reason": (
                "historical_corroboration_misleading"
                if entry["decision"] == "rejected" else None
            ),
        })
    for entry in ledger["reviews"].values():
        if entry["decision"] == "rejected":
            entry["rejection_reason"] = "lexical_overlap_only"
    ambiguous_policy = prediction_reviews.calibrate_historical_analog_policy(
        ledger
    )
    assert ambiguous_policy["global"]["status"] == "insufficient_reviews"
    assert ambiguous_policy["global"][
        "excluded_ambiguous_rejection_count"
    ] == 4

    for entry in ledger["reviews"].values():
        if entry["decision"] == "rejected":
            entry["rejection_reason"] = (
                "historical_corroboration_misleading"
            )
    policy = prediction_reviews.calibrate_historical_analog_policy(ledger)
    resolved = (
        prediction_reviews.resolve_historical_analog_threshold_for_category(
            policy, "AI Features"
        )
    )

    assert policy["global"]["status"] == "calibrated_higher"
    assert policy["global"]["threshold"] > 0.18
    assert resolved["source"] == "category_reviewed"
    assert resolved["threshold"] >= 0.35


def test_match_threshold_keeps_default_when_reviewed_quality_is_good():
    ledger = _calibration_ledger(
        [("confirmed", 0.4 + index / 100) for index in range(12)]
        + [("rejected", 0.25 + index / 100) for index in range(3)]
    )
    result = prediction_reviews.calibrate_match_threshold(ledger)
    assert result["status"] == "default_performing_well"
    assert result["threshold"] == 0.24
    assert result["evidence"]["confirmation_rate"] >= 0.8


def test_match_threshold_raises_only_with_supported_precision_gain():
    ledger = _calibration_ledger(
        [("confirmed", score) for score in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)]
        + [("rejected", score) for score in (0.25, 0.28, 0.30, 0.35)]
    )
    result = prediction_reviews.calibrate_match_threshold(ledger)
    assert result["status"] == "calibrated_higher"
    assert 0.24 < result["threshold"] <= 0.45
    assert result["evidence"]["selected_count"] >= 8
    assert result["evidence"]["confirmation_rate"] >= 0.8
    assert result["evidence"]["wilson_lower_bound"] >= 0.5
    assert result["evidence"]["confirmed_retention_rate"] == 1.0


def test_match_threshold_stays_default_when_no_safe_cutoff_exists():
    ledger = _calibration_ledger(
        [
            ("confirmed" if index % 2 == 0 else "rejected", 0.50 + index / 100)
            for index in range(12)
        ]
    )
    result = prediction_reviews.calibrate_match_threshold(ledger)
    assert result["status"] == "no_safe_higher_threshold"
    assert result["threshold"] == 0.24


def test_match_threshold_cannot_gain_precision_by_dropping_confirmed_matches():
    decisions = (
        [("confirmed", 0.60 + index / 100) for index in range(8)]
        + [("confirmed", 0.25 + index / 100) for index in range(4)]
        + [("rejected", 0.25 + index / 100) for index in range(7)]
    )
    result = prediction_reviews.calibrate_match_threshold(
        _calibration_ledger(decisions),
        max_per_cohort=10,
        max_per_hypothesis_cohort=10,
    )

    assert result["status"] == "no_safe_higher_threshold"
    assert result["threshold"] == 0.24
    best_attempt = result["best_higher_threshold_attempt"]
    assert best_attempt["confirmation_rate"] == 1.0
    assert best_attempt["confirmed_retention_rate"] < 0.8


def test_confirmed_below_default_audit_blocks_threshold_increase():
    ledger = _calibration_ledger(
        [("confirmed", score) for score in (
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
        )]
        + [("rejected", score) for score in (0.25, 0.28, 0.30, 0.35)]
        + [("confirmed", 0.20)]
    )
    ledger["reviews"]["review-12"].update({
        "candidate_kind": "threshold_audit",
        "applied_threshold": 0.24,
    })
    result = prediction_reviews.calibrate_match_threshold(ledger)

    assert result["status"] == "default_recall_warning"
    assert result["eligible"] is True
    assert result["threshold"] == 0.24
    assert result["confirmed_below_default_audit_count"] == 1
    assert result["effective_confirmed_below_default_audit_count"] > 0


def test_match_threshold_rejects_correlated_single_cohort_evidence():
    reviews = {}
    decisions = (
        [("confirmed", 0.60 + index / 100) for index in range(18)]
        + [("rejected", 0.25 + index / 100) for index in range(6)]
    )
    for index, (decision, score) in enumerate(decisions):
            reviews[f"review-{index}"] = {
                "decision": decision,
                "rejection_reason": (
                    "lexical_overlap_only" if decision == "rejected" else None
                ),
                "match_score": score,
            "prior_close_build": "100",
            "current_close_build": "101",
            "hypothesis_id": "TRH-AAAAAAAAAAAA",
        }
    result = prediction_reviews.calibrate_match_threshold({"reviews": reviews})

    assert result["status"] == "insufficient_diversity"
    assert result["eligible"] is False
    assert result["threshold"] == 0.24
    assert result["decisive_review_count"] == 24
    assert result["effective_decisive_review_count"] == 2
    assert result["reviewed_cohort_count"] == 1
    assert result["reviewed_hypothesis_count"] == 1
    resolved = prediction_reviews.resolve_threshold_for_category({
        "global": {
            "status": "calibrated_higher",
            "eligible": True,
            "threshold": 0.45,
            "decisive_review_count": 16,
            "effective_decisive_review_count": 12,
            "reviewed_cohort_count": 3,
            "reviewed_hypothesis_count": 2,
        },
        "categories": {"Editor Core": result},
    }, "Editor Core")
    assert resolved["source"] == "global_reviewed"
    assert resolved["threshold"] == 0.45


def test_category_threshold_uses_own_evidence_then_global_fallback():
    calibrated = (
        [("confirmed", score) for score in (
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
        )]
        + [("rejected", score) for score in (0.25, 0.28, 0.30, 0.35)]
    )
    ledger = _categorized_calibration_ledger({
        "Editor Core": calibrated,
        "Audio": [("confirmed", 0.6), ("rejected", 0.3)],
    })
    policy = prediction_reviews.calibrate_threshold_policy(ledger)

    editor = prediction_reviews.resolve_threshold_for_category(
        policy, "Editor Core"
    )
    audio = prediction_reviews.resolve_threshold_for_category(policy, "Audio")
    unknown = prediction_reviews.resolve_threshold_for_category(policy, "Capture")

    assert policy["categories"]["Editor Core"]["status"] == "calibrated_higher"
    assert policy["categories"]["Audio"]["status"] == "insufficient_reviews"
    assert editor["source"] == "category_reviewed"
    assert audio["source"] == "global_reviewed"
    assert unknown["source"] == "global_reviewed"
    assert editor["threshold"] == audio["threshold"] == unknown["threshold"]


def test_sufficient_category_evidence_overrides_raised_global_threshold():
    calibrated = (
        [("confirmed", score) for score in (
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
        )]
        + [("rejected", score) for score in (0.25, 0.28, 0.30, 0.35)]
    )
    healthy_default = (
        [("confirmed", 0.40 + index / 100) for index in range(12)]
        + [("rejected", 0.25 + index / 100) for index in range(3)]
    )
    policy = prediction_reviews.calibrate_threshold_policy(
        _categorized_calibration_ledger({
            "Editor Core": calibrated,
            "Audio": healthy_default,
        })
    )
    audio = prediction_reviews.resolve_threshold_for_category(policy, "Audio")

    assert policy["global"]["status"] == "calibrated_higher"
    assert policy["categories"]["Audio"]["status"] == "default_performing_well"
    assert audio["threshold"] == 0.24
    assert audio["source"] == "category_reviewed"
    assert audio["status"] == "default_performing_well"
    assert audio["decisive_review_count"] == 15
    assert audio["effective_decisive_review_count"] == 12
    assert audio["reviewed_cohort_count"] == 3
    assert audio["reviewed_hypothesis_count"] == 2


def test_category_threshold_falls_back_to_default_without_global_evidence():
    policy = prediction_reviews.calibrate_threshold_policy(
        _categorized_calibration_ledger({
            "Editor Core": [("confirmed", 0.8), ("rejected", 0.3)],
        })
    )
    result = prediction_reviews.resolve_threshold_for_category(
        policy, "Editor Core"
    )
    assert result["threshold"] == 0.24
    assert result["source"] == "default"
    assert result["status"] == "insufficient_reviews"


def _analysis_guardrail_ledger(rejection_reason="unsupported_shared_path"):
    reviews = {}
    for index in range(12):
        rejected = index % 2 == 0
        reviews[f"guardrail-{index}"] = {
            "decision": "rejected" if rejected else "confirmed",
            "rejection_reason": rejection_reason if rejected else None,
            "candidate_kind": "feature_match",
            "target_category": "Editor Core",
            "prior_close_build": f"prior-{index // 4}",
            "current_close_build": f"current-{index // 4}",
            "hypothesis_id": f"TRH-{index:012X}",
        }
    return {"reviews": reviews}


def test_category_guardrail_requires_diverse_dominant_ai_failure_reason():
    policy = prediction_reviews.build_analysis_guardrail_policy(
        _analysis_guardrail_ledger()
    )
    editor = policy["categories"]["Editor Core"]

    assert editor["status"] == "active"
    assert editor["effective_decisive_review_count"] == 12
    assert editor["effective_structured_rejection_count"] == 6
    assert editor["rejected_cohort_count"] == 3
    assert editor["rejected_hypothesis_count"] == 6
    assert policy["active_guardrails"]["Editor Core"]["guardrail"] == (
        "require_stronger_shared_path_grounding"
    )

    matcher_policy = prediction_reviews.build_analysis_guardrail_policy(
        _analysis_guardrail_ledger("lexical_overlap_only")
    )
    assert matcher_policy["categories"]["Editor Core"]["status"] == (
        "no_dominant_ai_failure"
    )
    assert matcher_policy["active_guardrails"] == {}


def test_category_guardrail_rejects_correlated_single_cohort_reviews():
    ledger = _analysis_guardrail_ledger()
    for entry in ledger["reviews"].values():
        entry["prior_close_build"] = "prior"
        entry["current_close_build"] = "current"
        entry["hypothesis_id"] = "TRH-AAAAAAAAAAAA"
    policy = prediction_reviews.build_analysis_guardrail_policy(ledger)

    assert policy["categories"]["Editor Core"]["status"] == (
        "insufficient_reviews"
    )
    assert policy["categories"]["Editor Core"][
        "effective_decisive_review_count"
    ] == 2
    assert policy["active_guardrails"] == {}


def test_compact_guardrail_policy_and_prompt_exclude_notes_and_live_counts():
    full = prediction_reviews.build_analysis_guardrail_policy(
        _analysis_guardrail_ledger()
    )
    full["active_guardrails"]["Editor Core"]["reviewer_note"] = "SECRET NOTE"
    compact = prediction_reviews.compact_analysis_guardrail_policy(full)
    prompt = summarize_tr._candidate_prompt(
        [{"bug_code": "A"}],
        ["Editor Core", "Audio"],
        compact,
    )
    synthesis_prompt = summarize_tr._synthesis_prompt(
        [{"source_bug_codes": ["A"]}],
        {"A": {"bug_code": "A"}},
        ["Editor Core", "Audio"],
        compact,
    )
    critic_prompt = summarize_tr._candidate_grounding_audit_prompt(
        [{"source_bug_codes": ["A"]}],
        {"A": {"bug_code": "A"}},
        ["Editor Core", "Audio"],
        compact,
    )

    assert compact["active_guardrails"] == {
        "Editor Core": {
            "guardrail": "require_stronger_shared_path_grounding",
            "source_rejection_reason": "unsupported_shared_path",
        }
    }
    assert "require_stronger_shared_path_grounding" in prompt
    assert "concrete shared component or state" in prompt
    assert "SECRET NOTE" not in json.dumps(compact)
    assert "SECRET NOTE" not in prompt
    assert "require_stronger_shared_path_grounding" in synthesis_prompt
    assert "SECRET NOTE" not in synthesis_prompt
    assert "require_stronger_shared_path_grounding" in critic_prompt
    assert "SECRET NOTE" not in critic_prompt
    assert "effective_reason_count" not in json.dumps(compact)


def test_guardrail_policy_is_part_of_matching_analysis_source_hash(monkeypatch):
    policy = prediction_reviews.compact_analysis_guardrail_policy(
        prediction_reviews.build_analysis_guardrail_policy(
            _analysis_guardrail_ledger()
        )
    )
    data = {
        "meta": {"analysis_guardrail_policy": policy},
        "products": {"P": [{"BugCode": "A", "Short Description": "Issue"}]},
    }

    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
            return {"checklists": [{"bug_code": "A", "steps": ["Set up", "Verify"]}]}
        return {"candidates": []}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    source_hash, _schema = process_tr.analysis_source_hash(data)

    assert analysis["source_sha256"] == source_hash
    assert analysis["analysis_guardrail_policy"] == policy


def test_category_support_ordering_survives_case_variant_modules():
    """Case-only module variants must not tie, or the source hash goes unstable.

    ``sorted(..., key=str.casefold)`` leaves "PIP" and "PiP" tied, so their order
    came from set iteration order — randomized per process. The parent hashes the
    analysis source in-process and the summarizer re-hashes it in a subprocess, so
    the tie resolved differently on each side and Stage 8 aborted with
    "analysis source hash mismatch".
    """
    def support_for(modules):
        data = {
            "meta": {
                "module_category_taxonomy": [
                    {"category": "Editor Core", "modules": modules}
                ]
            },
            "products": {},
        }
        return summarize_tr.target_category_support(data, [])

    forward = support_for(["PIP", "PiP", "Opening Intro", "Opening intro"])
    reversed_ = support_for(["Opening intro", "Opening Intro", "PiP", "PIP"])

    assert forward == reversed_
    assert forward["taxonomy"][0]["modules"] == [
        "Opening Intro", "Opening intro", "PIP", "PiP",
    ]


def test_checklist_gate_accepts_imperatives_outside_the_verb_allowlist():
    """Real cohorts were failed for openers no allowlist happened to contain.

    Attempt / Monitor / Hover / Change / Leave each killed an entire phd or
    promeo analysis. One unexpected-but-valid verb must not fail the run.
    """
    for step in (
        "Attempt to upgrade to subscription from the Essential build.",
        "Monitor the progress indicator until completion.",
        "Hover on the thumbnail to check the Save/Edit button UI.",
        "Change the product template and click Generate More.",
        "Leave the app idle on the editing page for 30 minutes.",
        "Idle the app for 30 minutes.",
        # Conditional openers are legitimate steps, not narrative.
        "When playback starts, verify the overlay renders.",
        "If the dialog appears, tap OK.",
        # A conjugated verb further into the step is normal phrasing.
        "Confirm export is complete.",
        "Wait until playback is stable.",
        "Verify the overlay is not visible.",
        "Export the project to 4K.",
    ):
        assert summarize_tr.is_checklist_action(step), step


def test_checklist_gate_still_rejects_root_cause_narrative():
    """Widening the gate must not let non-executable prose through."""
    for step in (
        "The crash was caused by a null pointer.",
        "This regression affects playback.",
        "Fixed in build 1234.",
        "It fails when exporting.",
        "Users report a freeze.",
        "Root cause: race condition.",
        "Expected: video plays.",
        "There is a memory leak.",
        "A memory leak occurs on export.",
        "Issue occurs when tapping export.",
        # Narrative that opens on a product noun, so no opener denylist can
        # catch it — the conjugated verb right after the subject is the tell.
        "Watermark still shows on export",
        "Playback fails after the update.",
        # Openers that are both verb and noun read as subjects when a
        # conjugated verb follows immediately.
        "Export is broken on iPad.",
        "Preview remains black.",
        "",
    ):
        assert not summarize_tr.is_checklist_action(step), step


def test_boundary_patterns_accept_common_limit_and_passing_phrasings():
    """phd failed every run on wording these vocabulary gates did not cover."""
    assert summarize_tr.SCOPE_BOUNDARY_PATTERN.search(
        "The visibility of the Premium overlay on generated AIPC results "
        "within the AI Image Generator preview dialogs."
    )
    assert summarize_tr.DISCONFIRMING_PATTERN.search(
        "When generating results, the Premium overlay is not visible in the preview."
    )
    # The original vocabulary must keep working.
    assert summarize_tr.SCOPE_BOUNDARY_PATTERN.search("Limited to the export path only.")
    assert summarize_tr.DISCONFIRMING_PATTERN.search("Playback remains correct and passes.")
    # Prose with no limit or no observable result is still rejected.
    assert not summarize_tr.SCOPE_BOUNDARY_PATTERN.search(
        "The renderer handles overlays for generated results."
    )
    assert not summarize_tr.DISCONFIRMING_PATTERN.search(
        "The overlay renders during generation."
    )


def test_casefold_order_is_a_total_order_for_case_variants():
    """Every case-insensitive sort feeding hashed input needs a stable tiebreak.

    Sorting a set by ``str.casefold`` alone leaves distinct case variants tied,
    so the result depends on set iteration order, which Python randomizes per
    process. All three modules share this key, so pin the contract for each.
    """
    variants = {"PIP", "PiP", "Opening Intro", "Opening intro"}
    expected = ["Opening Intro", "Opening intro", "PIP", "PiP"]

    for module in (process_tr, summarize_tr, prediction_reviews):
        assert sorted(variants, key=module.casefold_order) == expected
        # Insertion order must not leak into the result.
        assert sorted(
            ["PiP", "PIP", "Opening intro", "Opening Intro"],
            key=module.casefold_order,
        ) == expected


def test_review_ledger_preserves_immutable_analysis_guardrail_provenance(tmp_path):
    path = tmp_path / "prediction_reviews.json"
    engine = "b" * 64
    kwargs = {
        "product": "pdri",
        "hypothesis_id": "TRH-0123456789AB",
        "prior_close_build": "100",
        "current_close_build": "101",
        "bug_code": "BUG-1",
        "target_category": "Editor Core",
        "predicted_area": "Editor preview",
        "decision": "confirmed",
        "analysis_guardrail_provenance": "exact_policy",
        "analysis_guardrail": "require_stronger_shared_path_grounding",
        "analysis_guardrail_source_rejection_reason": (
            "unsupported_shared_path"
        ),
        "analysis_model": "gemma4",
        "analysis_engine_sha256": engine,
    }
    entry = prediction_reviews.upsert_review(path, **kwargs)

    assert entry["analysis_guardrail_provenance"] == "exact_policy"
    assert entry["analysis_model"] == "gemma4"
    assert entry["analysis_engine_sha256"] == engine
    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="cannot change",
    ):
        prediction_reviews.upsert_review(
            path,
            **{**kwargs, "analysis_engine_sha256": "c" * 64},
        )
    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="requires model and engine",
    ):
        prediction_reviews.upsert_review(
            tmp_path / "invalid.json",
            **{
                **kwargs,
                "analysis_model": None,
                "analysis_engine_sha256": None,
            },
        )


def test_review_ledger_preserves_immutable_prediction_lineage(tmp_path):
    path = tmp_path / "prediction_reviews.json"
    engine = "9" * 64
    kwargs = {
        "product": "pdri",
        "hypothesis_id": "TRH-0123456789AB",
        "prior_close_build": "100",
        "current_close_build": "101",
        "bug_code": "BUG-1",
        "target_category": "Editor Core",
        "predicted_area": "Editor preview",
        "decision": "confirmed",
        "prediction_origin_provenance": "schema11_exact_claim_lineage",
        "prediction_origin": "mixed_initial_and_recovered",
        "synthesis_source_candidate_ids": [
            "TRC-000000000001", "TRC-000000000002",
        ],
        "recovered_source_candidate_ids": ["TRC-000000000002"],
        "analysis_model": "gemma4",
        "analysis_engine_sha256": engine,
    }
    entry = prediction_reviews.upsert_review(path, **kwargs)

    assert entry["prediction_origin"] == "mixed_initial_and_recovered"
    assert entry["recovered_source_candidate_ids"] == ["TRC-000000000002"]
    assert entry["analysis_model"] == "gemma4"
    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="cannot change",
    ):
        prediction_reviews.upsert_review(
            path,
            **{
                **kwargs,
                "prediction_origin": "initial_audit_pass_only",
                "recovered_source_candidate_ids": [],
            },
        )
    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="lacks an initial-pass source",
    ):
        prediction_reviews.upsert_review(
            tmp_path / "invalid-lineage.json",
            **{
                **kwargs,
                "recovered_source_candidate_ids": [
                    "TRC-000000000001", "TRC-000000000002",
                ],
            },
        )


def _guardrail_effectiveness_ledger(*, different_baseline_engine=False,
                                    correlated=False):
    reviews = {}
    guarded_engine = "d" * 64
    baseline_engine = "e" * 64 if different_baseline_engine else guarded_engine
    for guarded in (False, True):
        for index in range(16):
            cohort = 0 if correlated else index // 2
            hypothesis = 0 if correlated else index
            decision = "confirmed" if guarded else "rejected"
            reviews[f"{guarded}-{index}"] = {
                "decision": decision,
                "rejection_reason": (
                    None if guarded else "unsupported_shared_path"
                ),
                "candidate_kind": "feature_match",
                "target_category": "Editor Core",
                "prior_close_build": f"prior-{cohort}",
                "current_close_build": f"current-{cohort}",
                "hypothesis_id": f"TRH-{hypothesis:012X}",
                "analysis_guardrail_provenance": "exact_policy",
                "analysis_guardrail": (
                    "require_stronger_shared_path_grounding"
                    if guarded else None
                ),
                "analysis_guardrail_source_rejection_reason": (
                    "unsupported_shared_path" if guarded else None
                ),
                "analysis_model": "gemma4",
                "analysis_engine_sha256": (
                    guarded_engine if guarded else baseline_engine
                ),
            }
    reviews["legacy"] = {
        "decision": "confirmed",
        "candidate_kind": "feature_match",
        "target_category": "Editor Core",
        "prior_close_build": "legacy-prior",
        "current_close_build": "legacy-current",
        "hypothesis_id": "TRH-FFFFFFFFFFFF",
    }
    return {"reviews": reviews}


def test_guardrail_effectiveness_requires_same_model_engine_and_diversity():
    result = prediction_reviews.guardrail_effectiveness_comparison(
        _guardrail_effectiveness_ledger()
    )
    comparison = result["comparisons"][0]

    assert result["unknown_provenance_decisive_review_count"] == 1
    assert comparison["status"] == "improved"
    assert comparison["unguarded"]["target_failure_rate"] == 1.0
    assert comparison["guarded"]["target_failure_rate"] == 0.0
    assert comparison["target_failure_rate_reduction"] == 1.0
    assert comparison["difference_confidence_interval"][0] > 0
    assert comparison["observational_only"] is True
    assert comparison["eligible_for_policy_change"] is False

    confounded = prediction_reviews.guardrail_effectiveness_comparison(
        _guardrail_effectiveness_ledger(different_baseline_engine=True)
    )["comparisons"][0]
    assert confounded["status"] == "insufficient_reviews"
    assert confounded["unguarded"]["decisive_review_count"] == 0

    correlated = prediction_reviews.guardrail_effectiveness_comparison(
        _guardrail_effectiveness_ledger(correlated=True)
    )["comparisons"][0]
    assert correlated["status"] == "insufficient_reviews"
    assert correlated["guarded"]["effective_decisive_review_count"] == 1
    assert correlated["unguarded"]["effective_decisive_review_count"] == 1


def _repair_origin_effectiveness_ledger(*, different_initial_engine=False,
                                        different_initial_guardrail=False,
                                        correlated=False):
    reviews = {}
    repaired_engine = "7" * 64
    initial_engine = "8" * 64 if different_initial_engine else repaired_engine
    for repaired in (False, True):
        for index in range(16):
            cohort = 0 if correlated else index // 2
            hypothesis = 0 if correlated else index
            source_id = f"TRC-{(1000 if repaired else 0) + index:012X}"
            reviews[f"{repaired}-{index}"] = {
                "decision": "confirmed" if repaired else "rejected",
                "rejection_reason": None if repaired else "wrong_target",
                "candidate_kind": "feature_match",
                "target_category": "Editor Core",
                "prior_close_build": f"prior-{cohort}",
                "current_close_build": f"current-{cohort}",
                "hypothesis_id": f"TRH-{hypothesis:012X}",
                "prediction_origin_provenance": (
                    "schema11_exact_claim_lineage"
                ),
                "prediction_origin": (
                    "recovered_draft_only"
                    if repaired else "initial_audit_pass_only"
                ),
                "analysis_guardrail_provenance": "exact_policy",
                "analysis_guardrail": (
                    "require_stronger_shared_path_grounding"
                    if different_initial_guardrail and not repaired else None
                ),
                "analysis_guardrail_source_rejection_reason": (
                    "unsupported_shared_path"
                    if different_initial_guardrail and not repaired else None
                ),
                "synthesis_source_candidate_ids": [source_id],
                "recovered_source_candidate_ids": [source_id] if repaired else [],
                "analysis_model": "gemma4",
                "analysis_engine_sha256": (
                    repaired_engine if repaired else initial_engine
                ),
            }
    reviews["legacy"] = {
        "decision": "confirmed",
        "candidate_kind": "feature_match",
        "target_category": "Editor Core",
        "prior_close_build": "legacy-prior",
        "current_close_build": "legacy-current",
        "hypothesis_id": "TRH-FFFFFFFFFFFF",
    }
    return {"reviews": reviews}


def test_repair_origin_effectiveness_matches_configuration_and_diversity():
    ledger = _repair_origin_effectiveness_ledger()
    ledger["reviews"]["schema10"] = {
        "decision": "confirmed",
        "candidate_kind": "feature_match",
        "target_category": "Editor Core",
        "prior_close_build": "schema10-prior",
        "current_close_build": "schema10-current",
        "hypothesis_id": "TRH-EEEEEEEEEEEE",
        "prediction_origin_provenance": "schema10_exact_synthesis_lineage",
        "prediction_origin": "initial_audit_pass_only",
        "synthesis_source_candidate_ids": ["TRC-EEEEEEEEEEEE"],
        "recovered_source_candidate_ids": [],
        "analysis_guardrail_provenance": "exact_policy",
        "analysis_guardrail": None,
        "analysis_guardrail_source_rejection_reason": None,
        "analysis_model": "gemma4",
        "analysis_engine_sha256": "7" * 64,
    }
    result = prediction_reviews.repair_origin_effectiveness_comparison(
        ledger
    )
    comparison = result["comparisons"][0]

    assert result["unknown_provenance_decisive_review_count"] == 2
    assert comparison["status"] == "repaired_higher"
    assert comparison["initial_pass_only"]["confirmation_rate"] == 0.0
    assert comparison["repaired_origin"]["confirmation_rate"] == 1.0
    assert comparison["confirmation_rate_difference"] == 1.0
    assert comparison["difference_confidence_interval"][0] > 0
    assert comparison["observational_only"] is True
    assert comparison["eligible_for_policy_change"] is False

    confounded = prediction_reviews.repair_origin_effectiveness_comparison(
        _repair_origin_effectiveness_ledger(different_initial_engine=True)
    )["comparisons"]
    assert len(confounded) == 2
    assert all(item["status"] == "insufficient_reviews" for item in confounded)

    guardrail_confounded = (
        prediction_reviews.repair_origin_effectiveness_comparison(
            _repair_origin_effectiveness_ledger(
                different_initial_guardrail=True
            )
        )["comparisons"]
    )
    assert len(guardrail_confounded) == 2
    assert all(
        item["status"] == "insufficient_reviews"
        for item in guardrail_confounded
    )

    correlated = prediction_reviews.repair_origin_effectiveness_comparison(
        _repair_origin_effectiveness_ledger(correlated=True)
    )["comparisons"][0]
    assert correlated["status"] == "insufficient_reviews"
    assert correlated["initial_pass_only"][
        "effective_decisive_review_count"
    ] == 1
    assert correlated["repaired_origin"][
        "effective_decisive_review_count"
    ] == 1


def test_manifest_checklist_parser_keeps_atypical_codes_and_ignores_other_headings():
    markdown = (
        "## PHI2626060087\n1. Set up\n2. Verify\n\n"
        "## Invented Section\nIgnore\n\n"
        "## DRI261875-0030\n1. Set up\n2. Verify\n"
    )
    blocks = extract_manifest_checklists(
        markdown, ["PHI2626060087", "DRI261875-0030"]
    )
    assert [code for code, _block in blocks] == ["PHI2626060087", "DRI261875-0030"]
    assert "Invented Section" not in "\n".join(block for _code, block in blocks)


def test_manual_fetch_default_is_isolated_from_pipeline_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch_details, "post_json", lambda *args, **kwargs: {
        "meta": {"bug_count": 1},
        "products": {"PowerDirector Mobile for iOS": [{"BugCode": "A"}]},
    })
    monkeypatch.setattr(sys, "argv", [
        "fetch_tr_bug_details.py", "--product", "pdri", "--bug-codes", "A",
    ])
    fetch_details.main()
    outputs = list(Path("data/products/pdri/tr_reports/manual").glob("*/bug_details.json"))
    assert len(outputs) == 1
    assert not Path("data/products/pdri/tr_reports/latest.json").exists()


def test_grounding_recall_sample_is_deterministic_bounded_and_diverse():
    gate = [{
        "candidate_id": "TRC-0000000000AA",
        "reason": "adjacent_target_absent_from_current_signal",
    }]
    draft = [
        {
            "candidate_id": f"TRC-{index:012X}",
            "reason": reason,
        }
        for index, reason in enumerate((
            "target_not_supported", "target_not_supported",
            "trigger_not_supported", "failure_mode_not_supported",
        ), start=1)
    ]
    final = [
        {
            "candidate_id": f"TRC-{index:012X}",
            "reason": reason,
        }
        for index, reason in enumerate((
            "target_not_supported", "shared_path_not_concrete",
            "boundary_not_supported", "direct_fix_restatement",
        ), start=20)
    ]
    source_sha = "a" * 64

    first = summarize_tr._select_grounding_human_review_samples(
        draft, final, source_sha, gate_withheld=gate, limit=6
    )
    second = summarize_tr._select_grounding_human_review_samples(
        draft, final, source_sha, gate_withheld=gate, limit=6
    )

    assert first == second
    assert len(first) == 6
    selected_strata = {
        (stage, item["reason"])
        for stage, items in (
            ("deterministic_adjacency_gate", gate),
            ("draft_before_synthesis", draft),
            ("final_after_synthesis", final),
        )
        for item in items
        if (stage, item["candidate_id"]) in first
    }
    assert len(selected_strata) == 6


def test_grounding_review_overlay_includes_deterministic_gate_false_negative(
    tmp_path,
):
    ledger_path = tmp_path / "grounding_reviews.json"
    source_sha = "a" * 64
    engine_sha = "b" * 64
    candidate_id = "TRC-1234567890AB"
    prediction_reviews.upsert_grounding_review(
        ledger_path,
        product="pdri",
        analysis_source_sha256=source_sha,
        analysis_engine_sha256=engine_sha,
        analysis_model="model",
        close_build="100",
        candidate_id=candidate_id,
        audit_stage="deterministic_adjacency_gate",
        withheld_reason="adjacent_target_absent_from_current_signal",
        failed_checks=["target_supported"],
        target_category="Editor Core",
        area="Timeline preview",
        source_bug_codes=["A"],
        evidence_refs=[{"bug_code": "A", "signal_index": 0}],
        decision="false_negative",
        overturned_checks=["target_supported"],
    )
    ledger = prediction_reviews.load_grounding_review_ledger(
        ledger_path, "pdri"
    )
    audit = {
        "candidate_discovery_audit": {
            "adjacency_gate_withheld_candidates": [{
                "candidate_id": candidate_id,
                "audit_stage": "deterministic_adjacency_gate",
                "human_review_sample": True,
            }],
        },
        "withheld_candidates": [],
        "final_withheld_recommendations": [],
    }
    rendered, summary = prediction_reviews.overlay_grounding_reviews(
        audit, ledger, source_sha, engine_sha, "model", "100"
    )
    overlaid = rendered["candidate_discovery_audit"][
        "adjacency_gate_withheld_candidates"
    ][0]
    assert overlaid["grounding_review_status"] == "false_negative"
    assert overlaid["grounding_review_overturned_checks"] == [
        "target_supported"
    ]
    assert summary["false_negative_count"] == 1
    assert summary["by_stage"]["deterministic_adjacency_gate"][
        "false_negative_count"
    ] == 1
    assert summary["by_reason"][
        "adjacent_target_absent_from_current_signal"
    ]["false_negative_count"] == 1


def test_reviewed_target_alias_requires_explicit_diverse_false_negatives(
    tmp_path,
):
    ledger_path = tmp_path / "grounding_reviews.json"
    for index, close_build in enumerate(("100", "101", "102"), start=1):
        prediction_reviews.upsert_grounding_review(
            ledger_path,
            product="pdri",
            analysis_source_sha256=f"{index:x}" * 64,
            analysis_engine_sha256="b" * 64,
            analysis_model="model",
            close_build=close_build,
            candidate_id=f"TRC-{index:012X}",
            audit_stage="deterministic_adjacency_gate",
            withheld_reason="adjacent_target_absent_from_current_signal",
            failed_checks=["target_supported"],
            target_category="Text & Captions",
            area="Captions editor",
            source_bug_codes=["A"],
            evidence_refs=[{
                "bug_code": "A", "signal_index": 0,
                "quote": "Updated subtitle renderer service",
            }],
            decision="false_negative",
            overturned_checks=["target_supported"],
            target_alias_pair={
                "target_term": "captions",
                "evidence_term": "subtitle",
            },
        )
    ledger = prediction_reviews.load_grounding_review_ledger(
        ledger_path, "pdri"
    )
    sparse = prediction_reviews.build_target_alias_policy(ledger)
    assert sparse["active_aliases"] == {}
    assert sparse["candidate_aliases"][0]["status"] == (
        "insufficient_diverse_reviews"
    )

    prediction_reviews.upsert_grounding_review(
        ledger_path,
        product="pdri",
        analysis_source_sha256="d" * 64,
        analysis_engine_sha256="b" * 64,
        analysis_model="model",
        close_build="103",
        candidate_id="TRC-000000000004",
        audit_stage="deterministic_adjacency_gate",
        withheld_reason="adjacent_target_absent_from_current_signal",
        failed_checks=["target_supported"],
        target_category="Text & Captions",
        area="Captions editor",
        source_bug_codes=["D"],
        evidence_refs=[{
            "bug_code": "D", "signal_index": 0,
            "quote": "Updated subtitle rendering manager",
        }],
        decision="false_negative",
        overturned_checks=["target_supported"],
        target_alias_pair={
            "target_term": "captions",
            "evidence_term": "subtitle",
        },
        reviewer="QA",
        note="Product UI uses Captions for the subtitle subsystem.",
    )
    ledger = prediction_reviews.load_grounding_review_ledger(
        ledger_path, "pdri"
    )
    active = prediction_reviews.build_target_alias_policy(ledger)
    assert active["active_aliases"] == {"captions": ["subtitle"]}
    compact = prediction_reviews.compact_target_alias_policy(active)
    assert compact["active_aliases"] == {"captions": ["subtitle"]}
    assert "candidate_aliases" not in compact
    assert "QA" not in json.dumps(compact)
    assert prediction_reviews.compact_target_alias_policy({
        "active_aliases": {"captions": ["updated"]},
    })["active_aliases"] == {}

    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="evidence_term is absent",
    ):
        prediction_reviews.upsert_grounding_review(
            tmp_path / "invalid_grounding_reviews.json",
            product="pdri",
            analysis_source_sha256="e" * 64,
            analysis_engine_sha256="b" * 64,
            analysis_model="model",
            close_build="104",
            candidate_id="TRC-000000000005",
            audit_stage="deterministic_adjacency_gate",
            withheld_reason="adjacent_target_absent_from_current_signal",
            failed_checks=["target_supported"],
            target_category="Text & Captions",
            area="Captions editor",
            source_bug_codes=["E"],
            evidence_refs=[{
                "bug_code": "E", "signal_index": 0,
                "quote": "Updated audio renderer service",
            }],
            decision="false_negative",
            overturned_checks=["target_supported"],
            target_alias_pair={
                "target_term": "captions",
                "evidence_term": "subtitle",
            },
        )


def test_schema28_preserves_exact_target_alias_provenance_for_future_reviews():
    prior = {
        "schema_version": 28,
        "target_alias_policy": prediction_reviews.compact_target_alias_policy({
            "active_aliases": {"captions": ["subtitle"]},
        }),
        "recommendations": [{
            "hypothesis_id": "TRH-000000000001",
            "target_category": "Text & Captions",
            "area": "Captions editor",
            "adjacency_support_basis": {
                "reviewed_aliases_influence_support": True,
                "reviewed_alias_matches": [{
                    "bug_code": "A",
                    "signal_index": 0,
                    "target_term": "captions",
                    "evidence_alias_term": "subtitle",
                    "signal_terms": ["subtitle"],
                }],
            },
        }, {
            "hypothesis_id": "TRH-000000000002",
            "target_category": "Text & Captions",
            "area": "Title editor",
            "adjacency_support_basis": {
                "reviewed_aliases_influence_support": False,
                "reviewed_alias_matches": [],
            },
        }],
    }
    result = process_tr.evaluate_prior_hypotheses(
        prior, {"products": {"P": []}},
        prior_close_build="100", current_close_build="101",
    )

    assert result["schema_version"] == 10
    assert result["target_alias_provenance_hypothesis_count"] == 2
    assert result["alias_supported_hypothesis_count"] == 1
    assert result["hypotheses"][0]["target_alias_provenance"] == (
        "schema27_exact_basis"
    )
    assert result["hypotheses"][0]["target_alias_pairs"] == [{
        "target_term": "captions", "evidence_term": "subtitle",
    }]
    assert result["hypotheses"][1]["target_alias_pairs"] == []


def test_target_alias_retirement_uses_only_explicit_diverse_alias_failures(
    tmp_path,
):
    grounding_path = tmp_path / "grounding_reviews.json"
    alias_pair = {"target_term": "captions", "evidence_term": "subtitle"}
    for index, close_build in enumerate(("100", "101", "102", "103"), start=1):
        prediction_reviews.upsert_grounding_review(
            grounding_path,
            product="pdri",
            analysis_source_sha256=f"{index:x}" * 64,
            analysis_engine_sha256="b" * 64,
            analysis_model="model",
            close_build=close_build,
            candidate_id=f"TRC-{index:012X}",
            audit_stage="deterministic_adjacency_gate",
            withheld_reason="adjacent_target_absent_from_current_signal",
            failed_checks=["target_supported"],
            target_category="Text & Captions",
            area="Captions editor",
            source_bug_codes=[f"A-{index}"],
            evidence_refs=[{
                "bug_code": f"A-{index}", "signal_index": 0,
                "quote": "Updated subtitle subsystem",
            }],
            decision="false_negative",
            overturned_checks=["target_supported"],
            target_alias_pair=alias_pair,
        )
    grounding = prediction_reviews.load_grounding_review_ledger(
        grounding_path, "pdri"
    )
    prediction_path = tmp_path / "prediction_reviews.json"
    review_args = []
    for index in range(12):
        cohort = index // 4
        hypothesis = index % 4 + 1
        args = {
            "product": "pdri",
            "hypothesis_id": f"TRH-{hypothesis:012X}",
            "prior_close_build": f"20{cohort}",
            "current_close_build": f"21{cohort}",
            "bug_code": f"BUG-{index}",
            "target_category": "Text & Captions",
            "predicted_area": "Captions editor",
            "decision": "rejected",
            "rejection_reason": "wrong_target",
            "candidate_kind": "feature_match",
            "target_alias_provenance": "schema27_exact_basis",
            "target_alias_pairs": [alias_pair],
        }
        review_args.append(args)
        prediction_reviews.upsert_review(prediction_path, **args)
    prediction_ledger = prediction_reviews.load_review_ledger(
        prediction_path, "pdri"
    )
    matcher_rejections = prediction_reviews.build_target_alias_policy(
        grounding, prediction_ledger
    )
    matcher_outcome = matcher_rejections["candidate_aliases"][0][
        "outcome_evidence"
    ]
    assert matcher_rejections["active_aliases"] == {
        "captions": ["subtitle"]
    }
    assert matcher_outcome["decisive_review_count"] == 0
    assert matcher_outcome["excluded_other_rejection_count"] == 12

    for args in review_args[:6]:
        prediction_reviews.upsert_review(
            prediction_path,
            **{
                **args,
                "rejection_reason": "reviewed_target_alias_misleading",
            },
        )
    sparse_alias_failures = prediction_reviews.build_target_alias_policy(
        grounding,
        prediction_reviews.load_review_ledger(prediction_path, "pdri"),
    )
    assert sparse_alias_failures["active_aliases"] == {
        "captions": ["subtitle"]
    }
    assert sparse_alias_failures["candidate_aliases"][0][
        "outcome_evidence"
    ]["alias_misleading_review_count"] == 6

    for args in review_args[6:]:
        prediction_reviews.upsert_review(
            prediction_path,
            **{
                **args,
                "rejection_reason": "reviewed_target_alias_misleading",
            },
        )
    prediction_ledger = prediction_reviews.load_review_ledger(
        prediction_path, "pdri"
    )
    retired = prediction_reviews.build_target_alias_policy(
        grounding, prediction_ledger
    )
    outcome = retired["candidate_aliases"][0]["outcome_evidence"]
    assert retired["candidate_aliases"][0]["status"] == "retired_low_precision"
    assert outcome["alias_misleading_review_count"] == 12
    assert outcome["effective_decisive_review_count"] == 12
    assert outcome["distinct_cohort_count"] == 3
    assert outcome["distinct_hypothesis_count"] == 4
    assert retired["active_aliases"] == {}
    assert prediction_reviews.compact_target_alias_policy(retired)[
        "active_aliases"
    ] == {}


def test_alias_specific_rejection_requires_single_immutable_exact_pair(tmp_path):
    common = {
        "product": "pdri",
        "hypothesis_id": "TRH-000000000001",
        "prior_close_build": "100",
        "current_close_build": "101",
        "bug_code": "A",
        "target_category": "Text & Captions",
        "predicted_area": "Captions editor",
        "decision": "rejected",
        "rejection_reason": "reviewed_target_alias_misleading",
        "candidate_kind": "feature_match",
    }
    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="requires one exact alias",
    ):
        prediction_reviews.upsert_review(tmp_path / "reviews.json", **common)

    path = tmp_path / "reviews.json"
    entry = prediction_reviews.upsert_review(
        path,
        **common,
        target_alias_provenance="schema27_exact_basis",
        target_alias_pairs=[{
            "target_term": "captions", "evidence_term": "subtitle",
        }],
    )
    assert entry["target_alias_pairs"] == [{
        "target_term": "captions", "evidence_term": "subtitle",
    }]
    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="cannot change",
    ):
        prediction_reviews.upsert_review(
            path,
            **common,
            target_alias_provenance="schema27_exact_basis",
            target_alias_pairs=[{
                "target_term": "captions", "evidence_term": "transcript",
            }],
        )


def test_grounding_review_ledger_preserves_provenance_and_reports_false_negatives(
    tmp_path,
):
    ledger_path = tmp_path / "grounding_reviews.json"
    base = {
        "product": "pdri",
        "analysis_source_sha256": "a" * 64,
        "analysis_engine_sha256": "b" * 64,
        "analysis_model": "model",
        "close_build": "100",
        "candidate_id": "TRC-1234567890AB",
        "audit_stage": "draft_before_synthesis",
        "withheld_reason": "shared_path_not_concrete",
        "failed_checks": ["shared_path_concrete"],
        "target_category": "Editor Core",
        "area": "Timeline preview",
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
    }
    entry = prediction_reviews.upsert_grounding_review(
        ledger_path,
        **base,
        decision="false_negative",
        overturned_checks=["shared_path_concrete"],
        reviewer="QA",
        note="The shared renderer is explicit in the source resolution.",
    )
    assert entry["decision"] == "false_negative"

    ledger = prediction_reviews.load_grounding_review_ledger(
        ledger_path, "pdri"
    )
    summary = prediction_reviews.grounding_review_summary(ledger)
    assert summary["false_negative_count"] == 1
    assert summary["decisive_count"] == 1
    assert summary["false_negative_rate"] == 1.0
    assert summary["by_reason"]["shared_path_not_concrete"][
        "false_negative_count"
    ] == 1

    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="provenance cannot change",
    ):
        prediction_reviews.upsert_grounding_review(
            ledger_path,
            **{**base, "area": "Changed area"},
            decision="correctly_withheld",
        )


def test_grounding_review_false_negative_requires_overturned_failed_check(tmp_path):
    with pytest.raises(
        prediction_reviews.ReviewLedgerError,
        match="false negative needs overturned checks",
    ):
        prediction_reviews.upsert_grounding_review(
            tmp_path / "grounding_reviews.json",
            product="pdri",
            analysis_source_sha256="a" * 64,
            analysis_engine_sha256="b" * 64,
            analysis_model="model",
            close_build="100",
            candidate_id="TRC-1234567890AB",
            audit_stage="final_after_synthesis",
            withheld_reason="failure_mode_not_supported",
            failed_checks=["failure_mode_supported"],
            target_category="Editor Core",
            area="Timeline preview",
            source_bug_codes=["A"],
            evidence_refs=[{"bug_code": "A", "signal_index": 0}],
            decision="false_negative",
            overturned_checks=[],
        )


def test_checklist_accepts_direct_manipulation_verbs():
    """Natural mobile-QA openers must not be rejected as non-executable.

    A narrow allowlist previously failed every structured analysis for pdri,
    pdr, and phd with "contains a non-executable checklist step".
    """
    for step in (
        "Tap the timeline to add a clip",
        "Launch the app and open a project",
        "Navigate to the effects tab",
        "Add a transition between two clips",
        "Record a voiceover track",
        "Play back the preview",
        "Drag the clip to reorder it",
        "Install the build on the device",
        "Restart the app and reopen the project",
        "Delete the second clip",
        "Purchase a premium subscription",
        "Upgrade the account and restore purchases",
        "Sign in with the test account",
    ):
        assert summarize_tr.is_checklist_action(step), step

    for step in (
        "The root cause was a stale cache",
        "This bug occurs when the timeline reloads",
        "Regression risk is high for the editor",
    ):
        assert not summarize_tr.is_checklist_action(step), step


def test_non_executable_checklist_error_names_the_step():
    """The retry feedback and terminal failure must record the rejected text."""
    evidence = {
        "summary": "Watermark remains after purchase",
        "repro_or_description": "export still shows the watermark",
        "locale_or_os": "",
        "changes": [],
        "qa_verify_notes": [],
    }
    checklists, errors = summarize_tr._validate_verification({
        "checklists": [{
            "bug_code": "A",
            "steps": ["Watermark still shows on export", "Verify it is absent"],
        }],
    }, ["A"], {"A": evidence})

    assert checklists == []
    assert any("Watermark still shows on export" in error for error in errors)


def test_checklist_verbs_do_not_become_grounding_anchors():
    """Action verbs are instructions, not evidence anchors."""
    anchors = summarize_tr._checklist_anchor_terms(
        "Tap export and launch preview then purchase premium"
    )
    assert not (anchors & summarize_tr.UI_ACTION_VERBS)
    assert not (anchors & summarize_tr.COMMERCE_ACTION_VERBS)


def test_direct_fix_gate_record_passes_pipeline_validation(monkeypatch):
    """A withheld record must satisfy process_newest_tr's provenance audit."""
    data = {"products": {"P": [{
        "BugCode": "A",
        "Short Description": 'Tapping "Create another one" returns to Custom tab',
        "ReproSteps": "Generate from a template then tap Create another one",
        "Resolutions": [{
            "solution": (
                "Return to the existing Image to Video editor page after "
                "generation so its Custom/Template state is preserved"
            ),
        }],
        "ParsedModule": "Image to Video",
        "ModuleCategory": "AI Features",
    }]}}
    restatement = {
        "area": "Image to Video editor",
        "target_category": summarize_tr.UNKNOWN_TARGET_CATEGORY,
        "trigger": "Generating from a template in the Image to Video editor",
        "expected_failure_mode": (
            "Tapping Create another one returns to the Custom tab page"
        ),
        **_prediction_boundaries("Image to Video editor"),
        "action": "Verify the Image to Video editor after generating",
        "rationale": (
            "The fix returns to the existing Image to Video editor page, so the "
            "editor page state is the shared mechanism"
        ),
        "source_bug_codes": ["A"],
        "evidence_refs": [{"bug_code": "A", "signal_index": 0}],
        "historical_analog_refs": [],
        "confidence": "medium",
    }

    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
            return {"checklists": [{
                "bug_code": "A", "steps": ["Set up source", "Verify result"],
            }]}
        if "SECOND independent coverage-challenge" in prompt:
            return {"candidates": []}
        if "Infer only defensible ADJACENT" in prompt:
            return {"candidates": [restatement]}
        raise AssertionError("a restatement must not reach the model critic")

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    discovery = analysis["candidate_grounding_audit"]["candidate_discovery_audit"]

    assert discovery["direct_fix_gate_withheld_count"] == 1
    assert analysis["recommendations"] == []
    assert process_tr.structured_prediction_errors(analysis) == []


def test_one_unsatisfiable_checklist_does_not_fail_the_whole_analysis(monkeypatch):
    """A bug the model cannot describe must not discard the cohort's report."""
    data = {"products": {"P": [
        {
            "BugCode": "A",
            "Short Description": "Export stops at 50 percent",
            "ReproSteps": "Export a 4K project from the timeline",
            "Resolutions": [{"solution": "Fixed the export queue state"}],
        },
        {
            "BugCode": "B",
            "Short Description": "Sticker panel scrolls to the top",
            "ReproSteps": "Scroll the sticker panel and reopen it",
            "Resolutions": [{"solution": "Restored the sticker scroll offset"}],
        },
    ]}}
    verification_calls = []

    def fake_json(prompt, model, url, timeout, **kwargs):
        if "direct fix-verification" in prompt:
            evidence = _evidence_from_prompt(prompt)
            verification_calls.append([item["bug_code"] for item in evidence])
            checklists = []
            for item in evidence:
                if item["bug_code"] == "B":
                    # One step collapses to a single entry and can never
                    # satisfy the 2-4 step rule, however often it is retried.
                    checklists.append({"bug_code": "B", "steps": ["Verify it"]})
                else:
                    checklists.append({
                        "bug_code": item["bug_code"],
                        "steps": ["Open the timeline", "Verify the export completes"],
                    })
            return {"checklists": checklists}
        return {"candidates": []}

    monkeypatch.setattr(summarize_tr, "call_ollama_json", fake_json)
    analysis = generate_structured_analysis(data, "model", "url", 10)
    report = render_analysis_markdown(analysis)

    assert validate_report(report, data, 0) == []
    assert [item["bug_code"] for item in analysis["checklists"]] == ["A", "B"]

    # The healthy sibling keeps its model checklist; only B is degraded.
    healthy, degraded = analysis["checklists"]
    assert "checklist_source" not in healthy
    assert degraded["checklist_source"] == "deterministic_evidence_fallback"
    assert 2 <= len(degraded["steps"]) <= 4
    assert "sticker" in " ".join(degraded["steps"]).casefold()
    assert all(
        summarize_tr.is_checklist_action(step) for step in degraded["steps"]
    )

    generation = analysis["checklist_generation"]
    assert generation["deterministic_fallback_count"] == 1
    assert generation["deterministic_fallbacks"][0]["bug_code"] == "B"
    assert any(
        "must contain 2-4 steps" in error
        for error in generation["deterministic_fallbacks"][0]["errors"]
    )

    # The rejected batch is retried per bug so A is never punished for B.
    assert verification_calls[0] == ["A", "B"]
    assert ["A"] in verification_calls
    assert verification_calls.count(["B"]) == summarize_tr.MAX_MODEL_ATTEMPTS

    # QA must be told the steps are not model-authored.
    assert "evidence-only fallback" in report


def test_deterministic_checklist_fallback_only_restates_stored_evidence():
    evidence = summarize_tr.structured_bug_evidence({
        "BugCode": "C",
        "Short Description": "Audio ducking is inverted",
        "ReproSteps": "Enable ducking on a music clip",
        "VerifyComments": [{"comment": "Checked with the 30 second preview"}],
    })
    checklist = summarize_tr._deterministic_verification_checklist(
        evidence, ["C must contain 2-4 steps"]
    )

    assert checklist["bug_code"] == "C"
    assert len(checklist["steps"]) == 3
    assert checklist["model_rejection_errors"] == ["C must contain 2-4 steps"]
    for step in checklist["steps"]:
        assert summarize_tr.is_checklist_action(step)
        assert summarize_tr.checklist_unsupported_specificity_terms(
            step, evidence
        ) == []

    tracking_only = summarize_tr.structured_bug_evidence({
        "BugCode": "E",
        "Short Description": "Audio ducking is inverted",
        "ReproSteps": "Enable ducking on a music clip",
        "VerifyComments": [
            {"comment": "RD/PM: Taken. Target fixed in 16.0.10 https://docs/deck"}
        ],
    })
    tracking_checklist = summarize_tr._deterministic_verification_checklist(
        tracking_only, []
    )
    # A link is a tracking pointer, not a verification oracle.
    assert len(tracking_checklist["steps"]) == 2
    assert "http" not in " ".join(tracking_checklist["steps"])

    bare = summarize_tr.structured_bug_evidence({"BugCode": "D"})
    bare_checklist = summarize_tr._deterministic_verification_checklist(bare, [])
    assert len(bare_checklist["steps"]) == 2
    assert all(
        summarize_tr.is_checklist_action(step)
        for step in bare_checklist["steps"]
    )


def _drop_reason_validator():
    """The real candidate validator, bound to a one-bug evidence map."""
    return lambda payload: summarize_tr._validate_candidates(
        payload, ["DRI-1"],
        {"DRI-1": {"adjacency_signal_quotes": ["q"]}},
        ["AI Features"], None,
    )


def test_malformed_candidates_record_why_each_one_was_rejected():
    salvaged, drops = summarize_tr._salvage_valid_candidates(
        {"candidates": [
            {"area": 5, "source_bug_codes": ["DRI-1"]},
            "not-an-object",
        ]},
        _drop_reason_validator(),
    )
    assert salvaged == []
    assert [drop["candidate_index"] for drop in drops] == [1, 2]
    # The validator numbers every single-candidate check "candidate 1"; the
    # real position survives in candidate_index, so the prefix is stripped.
    assert drops[0]["rejection_reasons"] == ["text fields must be strings"]
    assert drops[1]["rejection_reasons"] == ["must be an object"]
    assert drops[0]["source_bug_codes"] == ["DRI-1"]
    assert drops[1]["payload_type"] == "str"


def test_unparseable_candidate_response_is_recorded_as_one_drop():
    salvaged, drops = summarize_tr._salvage_valid_candidates(
        {"candidates": "nope"}, _drop_reason_validator(),
    )
    assert salvaged == []
    assert len(drops) == 1
    assert drops[0]["candidate_index"] is None
    assert drops[0]["rejection_reasons"] == [
        "response had no 'candidates' array"
    ]


def test_candidate_drop_count_always_matches_the_recorded_reasons(monkeypatch):
    monkeypatch.setattr(
        summarize_tr, "call_ollama_json",
        lambda *args, **kwargs: {"candidates": [{"area": 5}, "nope"]},
    )
    candidates, dropped, records = summarize_tr._request_candidate_batch_json(
        "prompt", _drop_reason_validator(), "model", "url", 1, num_predict=10,
    )
    assert candidates == []
    assert dropped == len(records) == 2


def test_valid_candidate_batch_records_no_drops(monkeypatch):
    payload = {"candidates": [{"area": "Timeline"}]}
    monkeypatch.setattr(
        summarize_tr, "call_ollama_json", lambda *args, **kwargs: payload,
    )
    assert summarize_tr._request_candidate_batch_json(
        "prompt", lambda given: (list(given["candidates"]), []),
        "model", "url", 1, num_predict=10,
    ) == ([{"area": "Timeline"}], 0, [])
