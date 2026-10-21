import copy
import json

from streamlit.testing.v1 import AppTest

from scripts.tr_focus_view import (
    checklist_label, prior_comparison_context, recommendation_groups,
)


def cached_analysis():
    return {
        "schema_version": 36,
        "checklist_evidence_by_bug": {
            "A": {"summary": "Compare table shows 500 instead of 100 monthly free credits"},
            "B": {"summary": "Yearly offer shows 1800 instead of 1200 yearly free credits"},
        },
        "recommendations": [
            {
                "area": "IAP pages with credits", "target_category": "IAP",
                "trigger": "Open compare table",
                "expected_failure_mode": "Shows 500 instead of 100 monthly free credits",
                "action": "Verify 1200 Credits/year",
                "disconfirming_observation": "Lower priority if 100 monthly free credits display",
                "rationale": "The API refresh affects IAP pages with credits",
                "source_bug_codes": ["A"], "risk": "Medium", "confidence": "medium",
                "priority_rank": 1,
            },
            {
                "area": "IAP pages with credits", "target_category": "IAP",
                "trigger": "Select yearly offer",
                "expected_failure_mode": "Shows 1800 instead of 1200 yearly free credits",
                "action": "Verify 1200 yearly free credits",
                "disconfirming_observation": "Lower priority if 1200 yearly free credits display",
                "rationale": "The API refresh affects IAP pages with credits",
                "source_bug_codes": ["B"], "risk": "Medium", "confidence": "medium",
                "priority_rank": 2,
            },
        ],
    }


def test_cached_instruction_guard_preserves_each_scenario_and_original_analysis():
    analysis = cached_analysis()
    before = copy.deepcopy(analysis)
    groups = recommendation_groups(analysis)
    assert len(groups) == 1
    scenarios = groups[0]["scenarios"]
    assert [row["number"] for row in scenarios] == [1, 2]
    assert scenarios[0]["issues"]
    assert not scenarios[1]["issues"]
    assert [row["item"]["source_bug_codes"] for row in scenarios] == [["A"], ["B"]]
    assert analysis == before


def test_cached_contradiction_is_not_rendered_as_an_executable_action():
    payload = json.dumps(cached_analysis())
    app = AppTest.from_string(
        "import json\nimport streamlit as st\n"
        "from scripts.tr_focus_view import render_recommendation_groups\n"
        f"analysis = json.loads({payload!r})\n"
        "render_recommendation_groups(st, analysis, 'https://example.test/{bug_code}')\n"
    ).run()
    assert not app.exception
    visible = "\n".join(element.value for element in app.markdown)
    assert "Verify 1200 Credits/year" not in visible
    assert "Verify 1200 yearly free credits" in visible
    assert "[A](https://example.test/A)" in visible
    assert "[B](https://example.test/B)" in visible
    assert any("1 recommendation(s) need instruction review" in row.value for row in app.warning)


def test_checklist_labels_use_frozen_summary_with_legacy_fallback():
    analysis = cached_analysis()
    assert "A · Compare table shows" in checklist_label("A", analysis)
    assert checklist_label("UNKNOWN", analysis) == "🐛 UNKNOWN"
    assert checklist_label("A", {}) == "🐛 A"


def test_prior_comparison_does_not_claim_release_ancestry():
    selection, warning = prior_comparison_context({
        "prior_close_build": "16.8.0.188591",
        "current_close_build": "16.7.5.188662",
    })
    assert "does not establish release ancestry" in selection
    assert "16.8.0 → 16.7.5" in warning
    assert "comparability is unverified" in warning
    _, same_version_warning = prior_comparison_context({
        "prior_close_build": "16.7.5.188591",
        "current_close_build": "16.7.5.188662",
    })
    assert "comparability is unverified" in same_version_warning


# ------------------------------------------------- cohort provenance (IF3)

def details_meta_with_gap():
    return {
        "cohort_audit": {
            "candidates_considered": 3,
            "missing_from_cohort": [
                {"bug_code": "DRI261899-0042", "tr_code": "TR260910-067",
                 "short_description": "Timeline: crash on undo"},
            ],
            "undetermined": [],
            "detail_check_limit_reached": False,
            "errors": [],
        },
        "tr_context": {
            "testing_requests": [
                {
                    "tr_code": "TR260910-067", "short_description": "16.8.0 pass",
                    "status": "Testing", "tr_build": "189063",
                    "owner": "BALLY_HSU", "tester": "VIVIAN_HUANG",
                    "bugs_found_on_tr": [{"bug_code": "DRI261890-0001"}],
                    "bugs_attached_for_verify": [
                        {"bug_code": "DRI261880-0009", "status": "Close",
                         "short_description": "passed", "is_verified": True,
                         "bug_result": True, "verify_reason": "QA Verify [OK]"},
                        {"bug_code": "DRI261880-0006", "status": "Fail",
                         "short_description": "verified but failed",
                         "is_verified": True, "bug_result": False,
                         "verify_reason": "still repros on R8"},
                        {"bug_code": "DRI261880-0007", "status": "TRCreated",
                         "short_description": "outstanding one",
                         "is_verified": False, "bug_result": None,
                         "verify_reason": None},
                        {"bug_code": "DRI261880-0008", "status": "TRCreated",
                         "short_description": "flag absent", "is_verified": None,
                         "bug_result": None, "verify_reason": None},
                    ],
                },
            ],
            "errors": [],
        },
        "ecl_direct_backfill": {"recovered_bug_codes": ["DRI261882-0021"],
                                "errors": []},
    }


def test_cohort_provenance_surfaces_the_gap_and_the_backfill():
    from scripts.tr_focus_view import cohort_provenance

    provenance = cohort_provenance(details_meta_with_gap())
    assert provenance["checked"] is True
    assert [i["bug_code"] for i in provenance["missing_from_cohort"]] == [
        "DRI261899-0042"]
    assert provenance["recovered_bug_codes"] == ["DRI261882-0021"]
    assert len(provenance["testing_requests"]) == 1


def test_cohort_provenance_is_inert_when_if3_was_never_read():
    from scripts.tr_focus_view import cohort_provenance

    provenance = cohort_provenance({"bug_count": 20})
    assert provenance["checked"] is False
    assert provenance["testing_requests"] == []
    assert provenance["missing_from_cohort"] == []


def test_verification_tally_keeps_passed_and_failed_apart():
    """A QA Fail is checked work handed back to RD, not an unverified bug.

    The tally is the only place the three states stay distinct; the focus list
    reads it to decide what is QA's queue and what is already closed.
    """
    from scripts.tr_focus_view import verification_tally

    tally = verification_tally(
        details_meta_with_gap()["tr_context"]["testing_requests"])
    assert tally["failed"] == 1
    assert tally["not_verified"] == 1
    assert tally["passed"] == 1


def test_verification_tally_counts_this_rounds_outcomes():
    from scripts.tr_focus_view import verification_tally

    tally = verification_tally(
        details_meta_with_gap()["tr_context"]["testing_requests"])
    assert tally == {"attached": 4, "verified": 2, "passed": 1, "failed": 1,
                     "not_verified": 1, "unknown": 1, "attachments": 4}


# ------------------------------------------------------------- changesets

def details_with_changesets():
    return {"products": {"PowerDirector Mobile for iOS": [
        {"BugCode": "DRI261883-0048", "Resolutions": [
            {"user": "AUSTIN_HUANG", "trace_url":
             "https://github.com/CyberLink-Team/repo/pull/3081/changes/897fe8a"},
        ]},
        {"BugCode": "DRI261886-0057", "Resolutions": [
            {"user": "NEO_SHEN",
             "trace_url": "bf8f9941cd4943e2437aa9e328a0c202476ce525"},
        ]},
        {"BugCode": "DRI261880-0060", "Resolutions": [
            # RD writes this freehand; a label before the URL is common.
            {"user": "Z", "trace_url":
             "PR: https://github.com/CyberLink-Team/repo/pull/2967"},
        ]},
        {"BugCode": "DRI261890-0001", "Resolutions": [
            {"user": "X", "trace_url": "N/A"},          # boilerplate
            {"user": "Y", "trace_url": ""},             # empty
        ]},
        {"BugCode": "DRI261890-0002", "Resolutions": []},
    ]}}


def test_collect_changesets_keeps_urls_and_bare_hashes_apart():
    from scripts.tr_focus_view import collect_changesets

    links = collect_changesets(details_with_changesets())
    assert set(links) == {"DRI261883-0048", "DRI261886-0057", "DRI261880-0060"}
    assert links["DRI261883-0048"][0]["is_link"] is True
    # A bare commit hash is worth showing but must not be rendered as a link.
    assert links["DRI261886-0057"][0]["is_link"] is False


def test_collect_changesets_extracts_a_url_from_a_labelled_value():
    """"PR: https://..." must link, not render as a truncated hash."""
    from scripts.tr_focus_view import collect_changesets

    link = collect_changesets(details_with_changesets())["DRI261880-0060"][0]
    assert link["is_link"] is True
    assert link["url"] == "https://github.com/CyberLink-Team/repo/pull/2967"
    assert link["label"].startswith("PR: ")


def test_collect_changesets_drops_na_and_empty_placeholders():
    from scripts.tr_focus_view import collect_changesets

    links = collect_changesets(details_with_changesets())
    assert "DRI261890-0001" not in links
    assert "DRI261890-0002" not in links


def test_collect_changesets_tolerates_empty_details():
    from scripts.tr_focus_view import collect_changesets

    assert collect_changesets({}) == {}
    assert collect_changesets({"products": {}}) == {}


def test_render_changeset_links_is_silent_without_a_changeset():
    from scripts.tr_focus_view import render_changeset_links

    calls = []
    st = type("ST", (), {"markdown": lambda self, t: calls.append(t)})()
    render_changeset_links(st, ["DRI261890-0002"], {})
    assert calls == []


# ------------------------------------------------------------- focus list

def focus_details():
    return {
        "meta": {
            "close_build": "16.8.0.189063",
            # Where the cohort's bugs were reported: provenance, never the queue.
            "tr_context": {"testing_requests": [{
                "tr_code": "TR260910-067",
                "bugs_attached_for_verify": [
                    {"bug_code": "OLD-ROUND", "is_verified": False,
                     "short_description": "an earlier round's attachment"},
                ],
            }]},
            # The TR testing this build, and what it was created from.
            "build_trs": [{
                "tr_code": "TR260922-0013", "form_state": "Normal",
                "bugs_loaded": True,
                "bugs_attached_for_verify": [
                    {"bug_code": "A-PASS", "is_verified": True, "bug_result": True,
                     "short_description": "passed", "verify_reason": "OK"},
                    {"bug_code": "A-FAIL", "is_verified": True, "bug_result": False,
                     "short_description": "failed", "verify_reason": "still repros"},
                    {"bug_code": "A-TODO", "is_verified": False, "bug_result": None,
                     "status": "BuildResolved", "close_build": "16.8.0.189000",
                     "short_description": "not checked", "verify_reason": None},
                    {"bug_code": "B3", "is_verified": False, "bug_result": None,
                     "status": "TRCreated", "close_build": "16.8.0.189063",
                     "short_description": "in the cohort", "verify_reason": None},
                ],
                "bugs_found_on_tr": [
                    {"bug_code": "NEW-1", "status": "NewCreated",
                     "short_description": "found this round"},
                ],
            }],
        },
        "products": {"P": [
            {"BugCode": "B1", "Severity": "2", "ParsedModule": "AI Music Generator",
             "ModuleCategory": "Audio", "OSLanguage": "-9L+RUS",
             "Resolutions": [{"risk": "Please verify all languages."}],
             "HistoricalAnalogs": [{"summary": "old music bug"}]},
            {"BugCode": "B2", "Severity": "3", "ParsedModule": "AI Music Generator",
             "ModuleCategory": "Audio",
             "Resolutions": [{"risk": "none"}], "HistoricalAnalogs": []},
            {"BugCode": "B3", "Severity": "1", "ParsedModule": "Timeline",
             "ModuleCategory": "Editor Core", "Resolutions": [{"risk": "N/A"}],
             "DetailFetchError": "status log unavailable"},
        ]},
    }


def focus_analysis():
    return {
        "evidence_independence_by_bug": {
            "B1": {"adjacency_signal_quotes": ["rename and un-star rebuild the list"]},
            "B2": {"adjacency_signal_quotes": []},
        },
        "checklist_generation": {"deterministic_fallbacks": [{"bug_code": "B2"}]},
    }


def test_a_qa_fail_verdict_is_closed_work_not_a_todo():
    """QA marking Fail means it was checked and went back to RD.

    Putting it in the outstanding queue would tell QA to redo work it has
    already finished, on a bug it no longer owns.
    """
    from scripts.tr_focus_view import build_focus_list

    focus = build_focus_list(focus_details(), focus_analysis())
    assert [r["bug_code"] for r in focus["awaiting_verification"]] == ["A-TODO", "B3"]
    closed = focus["closed_no_action"]
    assert closed["passed"] == 1
    assert [r["bug_code"] for r in closed["failed_back_to_rd"]] == ["A-FAIL"]


def test_change_surface_ranks_by_how_much_changed():
    from scripts.tr_focus_view import build_focus_list

    surface = build_focus_list(focus_details(), focus_analysis())["change_surface"]
    assert [row["module"] for row in surface] == ["AI Music Generator", "Timeline"]
    assert surface[0]["bug_count"] == 2
    assert surface[0]["worst_severity"] == 2      # best (lowest) of 2 and 3
    assert surface[1]["worst_severity"] == 1


def test_rd_placeholder_risks_are_not_reported_as_instructions():
    """RD fills this field with "none" / "N/A" far more often than real text."""
    from scripts.tr_focus_view import build_focus_list

    instructions = build_focus_list(focus_details(), focus_analysis())["rd_instructions"]
    assert [row["bug_code"] for row in instructions] == ["B1"]
    assert instructions[0]["locales"] == "-9L+RUS"


def test_weak_evidence_covers_both_degraded_records_and_fallback_checklists():
    from scripts.tr_focus_view import build_focus_list

    weak = build_focus_list(focus_details(), focus_analysis())["weak_evidence"]
    assert {row["bug_code"] for row in weak} == {"B3", "B2"}


def test_repeat_offenders_aggregate_by_module_not_by_capped_count():
    """Per-bug analog counts cap at 3, so ranking bugs by them says nothing."""
    from scripts.tr_focus_view import build_focus_list

    repeats = build_focus_list(focus_details(), focus_analysis())["repeat_offenders"]
    assert repeats == [{"module": "AI Music Generator", "bugs_with_history": 1,
                        "bug_count": 2, "examples": ["old music bug"]}]


def test_shared_path_leads_come_only_from_bugs_with_a_signal():
    from scripts.tr_focus_view import build_focus_list

    leads = build_focus_list(focus_details(), focus_analysis())["shared_path_leads"]
    assert [row["bug_code"] for row in leads] == ["B1"]


def test_focus_list_works_without_an_analysis():
    """The list must still render if the analysis artifact is missing."""
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    focus = build_focus_list(focus_details())
    assert focus["shared_path_leads"] == []
    assert focus["change_surface"]
    assert "What to test" in focus_list_markdown(focus)


def test_markdown_leads_with_the_queue_and_ends_with_closed_items():
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    md = focus_list_markdown(build_focus_list(focus_details(), focus_analysis()))
    assert md.index("## 1. Waiting on you") < md.index("## 2. Where the changes are")
    assert md.index("## 2.") < md.index("## Already verified")
    assert "handed them back to RD to fix" in md
    assert "Not QA's queue" in md
    assert "A-TODO" in md.split("## 2.")[0]


def test_markdown_opens_with_a_headline_that_answers_where_to_start():
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    md = focus_list_markdown(build_focus_list(focus_details(), focus_analysis()))
    headline = md.split("\n\n")[2]           # after the format marker and title
    assert "**3 fixes**" in headline
    assert "AI Music Generator" in headline      # heaviest module
    assert "**2 waiting on you**" in headline


def test_markdown_carries_no_raw_html():
    """st.markdown escapes HTML by default, so tags would render literally."""
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    md = focus_list_markdown(build_focus_list(focus_details(), focus_analysis()),
                             "2026-09-17T01:00:00")
    for tag in ("<sub>", "<br>", "</sub>", "<span"):
        assert tag not in md


def test_identical_rd_instructions_are_grouped_not_repeated():
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    details = focus_details()
    details["products"]["P"][1]["Resolutions"] = [
        {"risk": "Please verify all languages."}]
    md = focus_list_markdown(build_focus_list(details, focus_analysis()))
    section = md.split("## 3.")[1].split("## 4.")[0]
    assert section.count("Please verify all languages.") == 1
    assert "`B1`" in section and "`B2`" in section


def test_long_fix_note_quotes_are_trimmed():
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    analysis = focus_analysis()
    analysis["evidence_independence_by_bug"]["B1"]["adjacency_signal_quotes"] = [
        "word " * 80
    ]
    md = focus_list_markdown(build_focus_list(focus_details(), analysis))
    lines = md.splitlines()
    # Layout is the linked "- [`B1`](…)" then the quote on the next line.
    header = next(i for i, line in enumerate(lines)
                  if line.startswith("- [`B1`](http"))
    quote = lines[header + 1].strip()
    assert len(quote) < 200
    assert quote.endswith("…")


def test_markdown_says_so_when_the_queue_is_empty():
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    details = focus_details()
    for row in details["meta"]["build_trs"][0]["bugs_attached_for_verify"]:
        row["is_verified"], row["bug_result"] = True, True
    md = focus_list_markdown(build_focus_list(details, focus_analysis()))
    assert "Nothing outstanding" in md


def test_dollar_signs_in_fix_notes_are_escaped():
    """Streamlit renders $...$ as LaTeX.

    DRI261880-0060's fix note quotes a format string — "Costs %1$d credit per
    second. Minimum %2$d credits." — and everything between the two dollars was
    taken as math and rendered one character per line.
    """
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    analysis = focus_analysis()
    analysis["evidence_independence_by_bug"]["B1"]["adjacency_signal_quotes"] = [
        'noun is hardcoded singular - "Costs %1$d credit per second. '
        'Minimum %2$d credits."'
    ]
    md = focus_list_markdown(build_focus_list(focus_details(), analysis))
    assert "%1$d" not in md
    assert r"%1\$d" in md


def test_quote_already_shown_as_an_rd_instruction_is_not_repeated():
    """A stated risk arrives in section 3 and again as its own quote."""
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    analysis = focus_analysis()
    analysis["evidence_independence_by_bug"]["B1"]["adjacency_signal_quotes"] = [
        "Please verify all languages."
    ]
    md = focus_list_markdown(build_focus_list(focus_details(), analysis))
    assert md.count("Please verify all languages.") == 1


def test_one_quote_covering_several_bugs_prints_once_with_all_codes():
    from scripts.ecl_links import bug_link
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    analysis = focus_analysis()
    shared = "both MasterVideoMenuData and MasterPhotoMenuData were changed"
    analysis["evidence_independence_by_bug"]["B1"]["adjacency_signal_quotes"] = [shared]
    analysis["evidence_independence_by_bug"]["B2"] = {
        "adjacency_signal_quotes": [shared]}
    md = focus_list_markdown(build_focus_list(focus_details(), analysis))
    assert md.count(shared) == 1
    section = md.split("## 4.")[1]
    assert f"{bug_link('B1')} {bug_link('B2')}" in section


def test_two_quotes_from_one_bug_share_a_single_bullet():
    """Keying on the quote alone printed the bug code twice, on two bullets."""
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    analysis = focus_analysis()
    analysis["evidence_independence_by_bug"]["B1"]["adjacency_signal_quotes"] = [
        "root cause names a shared manager", "fix touches the shared manager too"]
    md = focus_list_markdown(build_focus_list(focus_details(), analysis))
    section = md.split("## 4.")[1]
    assert section.count("`B1`") == 1
    assert "root cause names a shared manager" in section
    assert "  - fix touches the shared manager too" in section


def test_list_punctuation_pasted_into_a_fix_note_is_stripped():
    """RD pastes bullet lines verbatim: "- widthOfWaveCell(at:) returns…"."""
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    analysis = focus_analysis()
    analysis["evidence_independence_by_bug"]["B1"]["adjacency_signal_quotes"] = [
        "- widthOfWaveCell(at:) returns the aligned stride"]
    md = focus_list_markdown(build_focus_list(focus_details(), analysis))
    assert "  - widthOfWaveCell" not in md.split("## 4.")[1].split("\n  - ")[0]
    assert "widthOfWaveCell(at:) returns the aligned stride" in md


def test_attached_bugs_are_partitioned_with_none_lost_or_double_counted():
    """Queue + passed + failed must equal what eCL attached.

    The queue and the closed list are rendered in different places; a bug
    landing in both, or in neither, is how the page came to say "nothing
    outstanding" and "3 outstanding" at the same time.
    """
    from scripts.tr_focus_view import build_focus_list, verification_tally

    details = focus_details()
    focus = build_focus_list(details, focus_analysis())
    tally = verification_tally(details["meta"]["build_trs"])
    closed = focus["closed_no_action"]
    assert (
        len(focus["awaiting_verification"])
        + closed["passed"]
        + len(closed["failed_back_to_rd"])
    ) == tally["attached"]


def test_tr_expander_does_not_claim_the_trs_belong_to_this_close_build():
    """They test the build each bug was reported against, an earlier one.

    The Bug Creation TRs card says "not the TR for this Close Build"; this
    expander showed the same 9 codes under the opposite claim.
    """
    from scripts.tr_focus_view import render_cohort_provenance

    labels = []

    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    class _ST:
        def __getattr__(self, name):
            if name == "expander":
                return lambda *a, **k: (labels.append(str(a[0])) or _Ctx())
            if name == "container":
                return lambda *a, **k: _Ctx()
            return lambda *a, **k: None

    render_cohort_provenance(_ST(), details_meta_with_gap(), "x{bug_code}")
    assert labels, "the Testing Request expander should render"
    assert not any("in this Close Build" in label for label in labels)
    assert any("found on" in label for label in labels)


# ------------------------------------------------ verification, once per bug

def test_a_bug_attached_to_two_trs_is_one_bug_with_the_later_verdict():
    """DRI261884-0009 failed on TR260902-101 and passed on TR260904-097.

    Counted per row it landed in both "passed" and "failed back to RD" — the
    2026-09-23 page showed 71 + 9 = 80 for 78 bugs.
    """
    from scripts.tr_focus_view import attached_bug_outcomes, verification_tally

    requests = [
        {"tr_code": "TR260904-097", "bugs_attached_for_verify": [
            {"bug_code": "X-1", "status": "TRCreated", "is_verified": True,
             "bug_result": True}]},
        {"tr_code": "TR260902-101", "bugs_attached_for_verify": [
            {"bug_code": "X-1", "status": "TRCreated", "is_verified": True,
             "bug_result": False}]},
    ]
    [row] = attached_bug_outcomes(requests)
    assert row["outcome"] == "passed"
    assert row["tr_code"] == "TR260904-097"
    assert row["failed_a_round"] is True
    tally = verification_tally(requests)
    assert (tally["attached"], tally["passed"], tally["failed"],
            tally["attachments"]) == (1, 1, 0, 2)


def test_a_closed_bug_is_not_handed_back_to_rd_whatever_one_round_said():
    """Eight of the nine "failed" on 2026-09-23 were Close in eCL: re-fixed."""
    from scripts.tr_focus_view import attached_bug_outcomes

    rows = attached_bug_outcomes([{"tr_code": "TR1", "bugs_attached_for_verify": [
        {"bug_code": "C-1", "status": "Close", "is_verified": True,
         "bug_result": False},
        {"bug_code": "F-1", "status": "Fail", "is_verified": True,
         "bug_result": False},
    ]}])
    assert {r["bug_code"]: r["outcome"] for r in rows} == {
        "C-1": "passed", "F-1": "failed"}


def test_verified_bugs_are_listed_not_just_counted():
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    md = focus_list_markdown(build_focus_list(focus_details(), focus_analysis()))
    verified = md.split("## Already verified")[1]
    assert "A-PASS" in verified
    assert "A-FAIL" in verified and "Not QA's queue" in verified


# ------------------------------------------------ fixed earlier in version

def version_rows():
    def row(code, build, module="Timeline", status="Close"):
        return {"BugCode": code, "Close Build#": build, "parsed_module": module,
                "module_category": "Editor Core", "Severity": "3",
                "Status": status, "Short Description": f"{code} desc"}
    return [
        row("E-1", "16.8.0.188318"),
        row("E-2", " 16.8.0.189063 ", module="AI Music Generator",
            status="TRCreated"),
        row("NOW", "16.8.0.189304"),            # the build itself
        row("B1", "16.8.0.188318"),             # in the cohort already
        row("LATER", "16.8.0.189400"),          # not in the build tested
        row("OTHER", "16.7.5.188662"),          # another version
        row("MANUAL", "v0908 [abc1234]"),       # names no version
        row("NONE", ""),
    ]


def test_version_fixes_are_earlier_builds_of_the_same_version_only():
    from scripts.tr_focus_view import collect_version_fixes

    fixes = collect_version_fixes(version_rows(), "16.8.0.189304", ["B1"])
    assert fixes["version"] == "16.8.0"
    assert [b["bug_code"] for b in fixes["bugs"]] == ["E-2", "E-1"]


def test_build_version_reads_mobile_and_desktop_formats():
    from scripts.tr_focus_view import build_version

    assert build_version("16.8.0.189304") == "16.8.0"
    assert build_version("0916_25.0.7_365_Ultra_Ultimate_VL") == "25.0.7"
    assert build_version("v0908 [abc1234]") == ""
    assert build_version("2607013") == ""


def test_fixed_earlier_section_ranks_modules_changed_again_first():
    from scripts.tr_focus_view import (
        build_focus_list, collect_version_fixes, focus_list_markdown,
    )

    details = focus_details()
    details["meta"]["version_fixes"] = collect_version_fixes(
        version_rows(), "16.8.0.189304", ["B1", "B2", "B3"])
    focus = build_focus_list(details, focus_analysis())
    earlier = focus["fixed_earlier_in_version"]
    assert [(r["module"], r["changed_again"]) for r in earlier] == [
        ("AI Music Generator", True), ("Timeline", True)]
    md = focus_list_markdown(focus)
    section = md.split("## 7. Fixed earlier in 16.8.0")[1].split("## Already")[0]
    assert "_(TRCreated)_" in section          # not yet verified is called out
    assert "E-1" in section and "_(Close)_" not in section
    assert "**2** earlier 16.8.0 fixes" in md.split("\n\n")[2]


def test_no_fixed_earlier_section_without_the_evidence():
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    focus = build_focus_list(focus_details(), focus_analysis())
    assert focus["version_fixes_checked"] is False
    assert "## 7." not in focus_list_markdown(focus)


def test_stored_focus_list_from_an_older_builder_is_not_current():
    from scripts.tr_focus_view import (
        build_focus_list, focus_list_is_current, focus_list_markdown,
    )

    md = focus_list_markdown(build_focus_list(focus_details()))
    assert focus_list_is_current(md)
    assert not focus_list_is_current("# What to test — build 1\n")


def test_focus_list_links_the_tr_that_tests_this_build():
    """The parent TRs are provenance; the build's own TR is what QA opens."""
    from scripts.ecl_links import tr_url
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    details = focus_details()
    details["meta"]["build_trs"].append(
        {"tr_code": "TR260922-0099", "form_state": "Cancel"})
    md = focus_list_markdown(build_focus_list(details))
    head = md.split("---", 1)[0]
    assert f"[`TR260922-0013`]({tr_url('TR260922-0013')})" in head
    assert "TR260922-0099`]" in head and "(cancelled)" in head


def test_focus_list_says_nothing_about_a_build_tr_it_does_not_know():
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    details = focus_details()
    del details["meta"]["build_trs"]
    assert "Tested by" not in focus_list_markdown(build_focus_list(details))


def test_a_single_build_tr_links_the_build_number_in_the_heading():
    from scripts.ecl_links import tr_url
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    details = focus_details()
    md = focus_list_markdown(build_focus_list(details))
    assert (f"# What to test — build [16.8.0.189063]({tr_url('TR260922-0013')})"
            in md)
    # Unknown TR: the heading stays plain text.
    del details["meta"]["build_trs"]
    assert "# What to test — build 16.8.0.189063\n" in focus_list_markdown(
        build_focus_list(details))


# ------------------------------------------------ the build TR is the queue

def test_queue_is_the_build_trs_created_from_list_not_earlier_rounds():
    """Earlier rounds' TRs are always verified, so the queue read from them
    left "Waiting on you" blank on nearly every run."""
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    focus = build_focus_list(focus_details(), focus_analysis())
    codes = [r["bug_code"] for r in focus["awaiting_verification"]]
    assert "OLD-ROUND" not in codes
    md = focus_list_markdown(focus)
    section = md.split("## 1. Waiting on you")[1].split("## 2.")[0]
    assert "2 of the 4 bug(s) this TR was created from" in section
    # Outside the cohort: say which build it was fixed into.
    assert "_(fixed into 16.8.0.189000)_" in section
    assert "1 of these are not closed to this build" in section
    b3_line = next(line for line in section.splitlines() if "`B3`" in line)
    assert "fixed into" not in b3_line


def test_an_unread_build_tr_is_an_unknown_queue_not_an_empty_one():
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    for build_trs in (None, [{"tr_code": "TR260922-0013"}]):
        details = focus_details()
        if build_trs is None:
            del details["meta"]["build_trs"]
        else:
            details["meta"]["build_trs"] = build_trs
        focus = build_focus_list(details, focus_analysis())
        assert focus["queue_known"] is False
        md = focus_list_markdown(focus)
        assert "Nothing outstanding" not in md
        assert "nothing waiting on you" not in md
        assert "queue is unknown" in md
        assert "## Already verified" not in md


def test_a_cancelled_build_tr_contributes_no_queue():
    from scripts.tr_focus_view import build_focus_list

    details = focus_details()
    details["meta"]["build_trs"][0]["form_state"] = "Cancel"
    focus = build_focus_list(details, focus_analysis())
    assert focus["queue_known"] is False
    assert focus["awaiting_verification"] == []


def test_bugs_generated_on_the_build_tr_are_listed_separately():
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    focus = build_focus_list(focus_details(), focus_analysis())
    assert [r["bug_code"] for r in focus["found_on_build_tr"]] == ["NEW-1"]
    md = focus_list_markdown(focus)
    found = md.split("## Found on this TR so far")[1].split("## Already")[0]
    assert "NEW-1" in found
    assert "NEW-1" not in md.split("## 1. Waiting on you")[1].split("## 2.")[0]


def test_nothing_outstanding_only_when_the_build_tr_is_fully_verified():
    from scripts.tr_focus_view import build_focus_list, focus_list_markdown

    details = focus_details()
    for row in details["meta"]["build_trs"][0]["bugs_attached_for_verify"]:
        row["is_verified"], row["bug_result"] = True, True
    md = focus_list_markdown(build_focus_list(details, focus_analysis()))
    assert "all 4 bug(s) this TR was created from have been verified" in md
