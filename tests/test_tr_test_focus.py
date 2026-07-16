import json
from pathlib import Path

from scripts.process_newest_tr import (
    close_build_sort_key,
    filter_detail_data,
    local_fallback_bugs,
    merge_detail_data,
    select_newest_tr_cohort,
)
from scripts.fetch_from_n8n import save_json
from scripts.summarize_tr_bugs import (
    build_prompt,
    normalize_regression_bullets,
    regression_first,
    strip_chunk_wrappers,
    validate_report,
)


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
    ]


def test_report_validation_accepts_complete_output():
    data = {"products": {"PDR-I": [{"BugCode": "A"}, {"BugCode": "B"}]}}
    report = (
        "# Per-Bug Verification\n## A\n1. Test\n## B\n1. Test\n"
        "# Recommended Regression Areas\n## High\n- Test Export shared renderer"
    )
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
