"""Test-focus presentation helpers; stored reports remain immutable."""
import re

try:
    from .summarize_tr_bugs import (
        _escape_markdown, deterministic_scenario_specificity_basis,
    )
except ImportError:
    from summarize_tr_bugs import (
        _escape_markdown, deterministic_scenario_specificity_basis,
    )


def recommendation_instruction_issues(item: dict, evidence_map: dict) -> list[str]:
    codes = item.get("source_bug_codes") or []
    if not codes or any(not evidence_map.get(code) for code in codes):
        return ["Stored source evidence is unavailable for checking these instructions."]
    basis = deterministic_scenario_specificity_basis(
        str(item.get("trigger") or ""),
        str(item.get("expected_failure_mode") or ""),
        codes, item.get("evidence_refs") or [], evidence_map,
        action=str(item.get("action") or ""),
        disconfirming_observation=str(item.get("disconfirming_observation") or ""),
    )
    issues = ([
        "The action, failure scenario, and passing observation disagree on the "
        "credit amount or billing period."
    ] if basis["instruction_consistency_errors"] else [])
    field_labels = {
        "action": "The test action", "trigger": "The trigger",
        "expected_failure_mode": "The failure description",
        "disconfirming_observation": "The passing observation",
    }
    issues.extend(
        f"{field_labels[row['claim_field']]} contains {row['term']!r}, "
        "which is absent from its cited current source evidence."
        for row in basis["unsupported_specificity"]
    )
    return list(dict.fromkeys(issues))


def recommendation_groups(analysis: dict) -> list[dict]:
    """Group matching area/category labels without merging scenario evidence."""
    groups = {}
    evidence = analysis.get("checklist_evidence_by_bug") or {}
    for index, item in enumerate(analysis.get("recommendations") or [], 1):
        if not isinstance(item, dict):
            continue
        area = str(item.get("area") or "Regression checks")
        category = str(item.get("target_category") or "Unclassified")
        key = (area.casefold().strip(), category.casefold().strip())
        group = groups.setdefault(key, {"area": area, "category": category, "scenarios": []})
        group["scenarios"].append({
            "item": item,
            "number": item.get("priority_rank") or index,
            "issues": recommendation_instruction_issues(item, evidence),
        })
    return list(groups.values())


def render_recommendation_groups(st, analysis: dict, bug_url: str) -> None:
    groups = recommendation_groups(analysis)
    flagged = sum(bool(row["issues"]) for group in groups for row in group["scenarios"])
    if flagged:
        st.warning(
            f"{flagged} recommendation(s) need instruction review. Their actions "
            "are withheld below; use the source bug's verification checklist. "
            "The stored report is preserved under Evidence & reasoning."
        )
    st.caption(
        "Scenario numbers retain the stored order. Equal risk and confidence "
        "labels do not imply different severity."
    )
    for group in groups:
        scenarios = group["scenarios"]
        with st.container(border=True):
            st.markdown(f"#### {_escape_markdown(group['area'])} — {len(scenarios)} scenario(s)")
            st.caption(f"Module: {group['category']}")
            for row in scenarios:
                item = row["item"]
                st.markdown(
                    f"- **Check {row['number']} · {_escape_markdown(item.get('trigger'))}**\n\n"
                    f"  Risk: **{_escape_markdown(item.get('risk'))}** · "
                    f"Confidence: **{_escape_markdown(str(item.get('confidence') or '').title())}**"
                )
                if row["issues"]:
                    st.warning("Needs review: " + " ".join(row["issues"]))
                else:
                    st.markdown(
                        f"**Action:** {_escape_markdown(item.get('action'))}\n\n"
                        f"**Watch for:** {_escape_markdown(item.get('expected_failure_mode'))}\n\n"
                        f"**Passing observation:** {_escape_markdown(item.get('disconfirming_observation'))}\n\n"
                        f"**Why test this:** {_escape_markdown(item.get('rationale'))}"
                    )
                links = " · ".join(
                    f"[{_escape_markdown(code)}]({bug_url.format(bug_code=code)})"
                    for code in item.get("source_bug_codes") or []
                )
                st.markdown(f"**Source bugs:** {links}")


def checklist_label(code: str, analysis: dict) -> str:
    source = (analysis.get("checklist_evidence_by_bug") or {}).get(code) or {}
    summary = " ".join(str(source.get("summary") or "").split())
    if len(summary) > 140:
        summary = summary[:137].rstrip() + "…"
    return f"🐛 {code}" + (f" · {_escape_markdown(summary)}" if summary else "")


def prior_comparison_context(evaluation: dict) -> tuple[str, str]:
    """Explain storage order without claiming known release-branch ancestry."""
    prior = str(evaluation.get("prior_close_build") or "")
    current = str(evaluation.get("current_close_build") or "")
    selection = (
        "Baseline selection uses a stored report from a different Close Build "
        "(the previous latest report, or the newest stored run for another build). "
        "It does not establish release ancestry."
    )
    versions = [re.fullmatch(r"(\d+\.\d+\.\d+)\.\d+", value) for value in (prior, current)]
    if all(versions) and versions[0][1] != versions[1][1]:
        warning = (
            f"Different version prefixes: {versions[0][1]} → {versions[1][1]}. "
            "Release-line comparability is unverified; treat these results as exploratory."
        )
    else:
        warning = "Release-line comparability is unverified; build labels alone do not establish it."
    return selection, warning
