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
