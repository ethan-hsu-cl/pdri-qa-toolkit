"""Test-focus presentation helpers; stored reports remain immutable."""
import re

try:
    from .summarize_tr_bugs import (
        _escape_markdown, deterministic_scenario_specificity_basis,
    )
    from .ecl_links import bug_link, tr_link, tr_url
except ImportError:
    from summarize_tr_bugs import (
        _escape_markdown, deterministic_scenario_specificity_basis,
    )
    from ecl_links import bug_link, tr_link, tr_url


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


def collect_changesets(details: dict) -> dict[str, list[dict]]:
    """Map BugCode -> the RD changesets that carry its fix.

    ``TraceURL`` on an RDResolved status-log entry is the commit or PR the fix
    landed in. It reaches the model prompt already, but is deliberately absent
    from the grounding evidence map (a URL cannot support a textual claim), so
    without this it never reaches the reader either — leaving a QA told to
    check a behaviour with no way to see what actually changed.
    """
    links: dict[str, list[dict]] = {}
    for bugs in (details.get("products") or {}).values():
        for bug in bugs:
            code = str(bug.get("BugCode") or "").strip()
            if not code:
                continue
            for resolution in bug.get("Resolutions") or []:
                raw = str(resolution.get("trace_url") or "").strip()
                if not raw or raw.lower() in {"n/a", "n/a.", "na", "none"}:
                    continue
                # RD writes this field freehand: a bare URL, a bare commit
                # hash, or a label plus a URL ("PR: https://..."). Matching on a
                # leading "http" alone mislabels the labelled ones as hashes.
                match = re.search(r"https?://\S+", raw)
                links.setdefault(code, []).append({
                    "url": match.group(0).rstrip(".,);") if match else raw,
                    "label": raw,
                    "user": resolution.get("user"),
                    # A bare commit hash is worth showing, just not as a link.
                    "is_link": bool(match),
                })
    return links


def render_changeset_links(st, codes: list[str], changesets: dict) -> None:
    rows = [
        (code, link) for code in codes for link in changesets.get(code, [])
    ]
    if not rows:
        return
    rendered = " · ".join(
        f"[{code} diff]({link['url']})" if link["is_link"]
        else f"{code} `{_escape_markdown(link['url'][:12])}`"
        for code, link in rows
    )
    st.markdown(f"**Changed code:** {rendered}")


def render_recommendation_groups(st, analysis: dict, bug_url: str,
                                 changesets: dict | None = None) -> None:
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
                codes = item.get("source_bug_codes") or []
                links = " · ".join(
                    f"[{_escape_markdown(code)}]({bug_url.format(bug_code=code)})"
                    for code in codes
                )
                st.markdown(f"**Source bugs:** {links}")
                if changesets:
                    render_changeset_links(st, codes, changesets)


#: RD leaves this field filled in with a placeholder more often than not.
_EMPTY_RISK = {"", "-", "--", "n/a", "n/a.", "na", "none", "no", "nil", "null"}


def _severity(bug: dict) -> int:
    """Numeric severity; 1 is worst. Unknown sorts last."""
    value = str(bug.get("Severity") or "").strip()
    return int(value) if value.isdigit() else 99


#: Bump when the focus list's content changes, so the dashboard rebuilds a
#: stored file written by an older builder instead of serving it verbatim.
FOCUS_LIST_FORMAT = 5
#: A CommonMark link-reference definition: renders as nothing in Streamlit or
#: any markdown viewer, but survives in the downloaded file.
FOCUS_LIST_FORMAT_MARKER = f"[//]: # (focus-list format {FOCUS_LIST_FORMAT})"


def focus_list_is_current(markdown: str) -> bool:
    return str(markdown or "").startswith(FOCUS_LIST_FORMAT_MARKER)


def build_version(build: str) -> str:
    """The x.y.z release a build string belongs to; '' when it names none.

    Covers ``16.8.0.189304`` and desktop ``0916_25.0.7_365_Ultra`` alike. A
    bare numeric build or manual shorthand ("v0908 [abc1234]") names no
    version, so it can never be matched into one.
    """
    match = re.search(r"(?<![\d.])(\d+\.\d+\.\d+)(?=[._]|$)", str(build or ""))
    return match.group(1) if match else ""


def _build_numbers(build: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", str(build or "")))


def collect_version_fixes(rows: list[dict], close_build: str,
                          cohort_codes) -> dict:
    """Bugs fixed into earlier builds of this Close Build's version.

    They are in the build being certified too, and most of them were verified
    on an earlier round — which is exactly why they drop out of everyone's
    view. A verified fix is still changed code, so it belongs on the regression
    list. Builds after ``close_build`` are excluded: they are not in it.

    ``rows`` are parsed-CSV rows (``ecl_parsed.csv``).
    """
    version = build_version(close_build)
    if not version:
        return {}
    cohort = {str(code) for code in cohort_codes}
    ceiling = _build_numbers(close_build)
    bugs = []
    for row in rows:
        code = str(row.get("BugCode") or "").strip()
        build = str(row.get("Close Build#") or "").strip()
        if (not code or code in cohort or build == close_build
                or build_version(build) != version
                or _build_numbers(build) > ceiling):
            continue
        bugs.append({
            "bug_code": code,
            "module": str(row.get("parsed_module") or "").strip() or "(unmapped)",
            "category": str(row.get("module_category") or "").strip(),
            "severity": str(row.get("Severity") or "").strip(),
            "status": str(row.get("Status") or "").strip(),
            "close_build": build,
            "short_description": " ".join(
                str(row.get("Short Description") or "").split()),
        })
    bugs.sort(key=lambda bug: (bug["module"].casefold(), bug["bug_code"]))
    return {"schema_version": 1, "version": version,
            "close_build": close_build, "source": "ecl_parsed.csv",
            "bugs": bugs}


def build_focus_list(details: dict, analysis: dict | None = None) -> dict:
    """Everything worth testing for this TR, as recorded fact.

    Separate from Recommended Regression Areas on purpose. That section makes
    *predictions* — novel adjacent areas — and its gates exist to suppress
    anything that is merely a restatement of a fix. Correct for precision, and
    exactly wrong for a coverage list, where "re-test the path that was fixed"
    is a must-do item rather than noise. So nothing here is gated: every entry
    is something eCL or RD already recorded, and there is no claim about the
    future to ground.

    A QA "Fail" verdict is *finished* QA work: the bug was checked, it failed,
    and it went back to RD. It is reported as closed, never as a to-do — the
    outstanding queue is what nobody has verified yet.

    The queue is the build TR's "Bugs from which this Testing Request is
    created", never the attachments of the TRs the cohort's bugs were reported
    on. Those are earlier rounds, verified long ago, so reading the queue from
    them left "Waiting on you" empty on nearly every run. Without the build
    TR's list the queue is *unknown* (``queue_known`` False), not empty.
    """
    analysis = analysis or {}
    bugs = [b for v in (details.get("products") or {}).values() for b in v]
    meta = details.get("meta") or {}
    independence = analysis.get("evidence_independence_by_bug") or {}

    by_module: dict[str, list[dict]] = {}
    for bug in bugs:
        by_module.setdefault(
            str(bug.get("ParsedModule") or "(unmapped)"), []
        ).append(bug)
    change_surface = sorted(
        (
            {
                "module": module,
                "category": str(rows[0].get("ModuleCategory") or ""),
                "bug_count": len(rows),
                "worst_severity": min(_severity(b) for b in rows),
                "bug_codes": sorted(
                    str(b.get("BugCode")) for b in rows
                ),
            }
            for module, rows in by_module.items()
        ),
        key=lambda row: (-row["bug_count"], row["worst_severity"], row["module"]),
    )

    rd_instructions = []
    for bug in bugs:
        for resolution in bug.get("Resolutions") or []:
            risk = " ".join(str(resolution.get("risk") or "").split())
            if risk.casefold() in _EMPTY_RISK:
                continue
            rd_instructions.append({
                "bug_code": str(bug.get("BugCode")),
                "module": str(bug.get("ParsedModule") or ""),
                "instruction": risk,
                "locales": str(bug.get("OSLanguage") or "").strip(),
            })

    shared_path_leads = []
    for bug in bugs:
        quotes = (independence.get(str(bug.get("BugCode"))) or {}).get(
            "adjacency_signal_quotes"
        ) or []
        if quotes:
            shared_path_leads.append({
                "bug_code": str(bug.get("BugCode")),
                "module": str(bug.get("ParsedModule") or ""),
                "quotes": [" ".join(str(q).split()) for q in quotes],
            })
    shared_path_leads.sort(key=lambda row: (row["module"].casefold(),
                                            row["bug_code"]))

    weak_evidence = [
        {"bug_code": str(b.get("BugCode")),
         "module": str(b.get("ParsedModule") or ""),
         "reason": str(b.get("DetailFetchError"))}
        for b in bugs if b.get("DetailFetchError")
    ]
    for fallback in (analysis.get("checklist_generation") or {}).get(
            "deterministic_fallbacks") or []:
        weak_evidence.append({
            "bug_code": str(fallback.get("bug_code") or "?"),
            "module": "",
            "reason": "checklist fell back to restating stored evidence",
        })

    # Per-bug analog counts are capped (1 same-module + 2 adjacent), so ranking
    # bugs by that count just lists whoever hit the cap — 16 identical "3"s on
    # this build. The informative unit is the module: how much of what changed
    # here has failed here before.
    repeat_offenders = []
    for module, rows in by_module.items():
        with_history = [r for r in rows if r.get("HistoricalAnalogs")]
        if not with_history:
            continue
        repeat_offenders.append({
            "module": module,
            "bugs_with_history": len(with_history),
            "bug_count": len(rows),
            "examples": sorted(
                {
                    " ".join(str(analog.get("summary") or "").split())[:110]
                    for r in with_history
                    for analog in (r.get("HistoricalAnalogs") or [])[:1]
                }
            )[:2],
        })
    repeat_offenders.sort(
        key=lambda row: (-row["bugs_with_history"], row["module"].casefold())
    )

    # The build TR's created-from list split by who owns it now — once per
    # bug, with the eBug's current status overriding a stale round's verdict.
    queue_trs = build_tr_queue_sources(meta.get("build_trs"))
    cohort_codes = {str(b.get("BugCode")) for b in bugs}
    outcomes = attached_bug_outcomes(queue_trs)
    for row in outcomes:
        # The TR's scope and this report's cohort differ; say which is which.
        row["in_cohort"] = row["bug_code"] in cohort_codes
    awaiting = [row for row in outcomes if row["outcome"] == "awaiting"]
    failed_back_to_rd = [row for row in outcomes if row["outcome"] == "failed"]
    passed_bugs = [row for row in outcomes if row["outcome"] == "passed"]

    # Fixed earlier in this version: in the build too, and changed code even
    # when verified. Modules this build touches again are ranked first.
    version_meta = meta.get("version_fixes") or {}
    surface_modules = {row["module"] for row in change_surface}
    earlier_by_module: dict[str, list[dict]] = {}
    for bug in version_meta.get("bugs") or []:
        earlier_by_module.setdefault(bug.get("module") or "(unmapped)", []).append(bug)
    fixed_earlier = sorted(
        (
            {
                "module": module,
                "category": str(rows[0].get("category") or ""),
                "bug_count": len(rows),
                "changed_again": module in surface_modules,
                "bugs": [
                    {"bug_code": str(b.get("bug_code")),
                     "status": str(b.get("status") or "")}
                    for b in sorted(rows, key=lambda b: str(b.get("bug_code")))
                ],
            }
            for module, rows in earlier_by_module.items()
        ),
        key=lambda row: (not row["changed_again"], -row["bug_count"],
                         row["module"].casefold()),
    )

    return {
        "schema_version": 2,
        "close_build": str(meta.get("close_build") or ""),
        # The TR that tests this build — not the bugs' parent TRs.
        "build_trs": [
            {"tr_code": str(tr.get("tr_code")),
             "form_state": str(tr.get("form_state") or "")}
            for tr in (meta.get("build_trs") or []) if tr.get("tr_code")
        ],
        "queue_known": bool(queue_trs),
        "found_on_build_tr": sorted(
            (
                {"bug_code": str(row.get("bug_code")),
                 "tr_code": str(tr.get("tr_code")),
                 "status": str(row.get("status") or ""),
                 "short_description": row.get("short_description")}
                for tr in queue_trs
                for row in (tr.get("bugs_found_on_tr") or [])
                if row.get("bug_code")
            ),
            key=lambda row: row["bug_code"],
        ),
        "bug_count": len(bugs),
        "change_surface": change_surface,
        "awaiting_verification": awaiting,
        "rd_instructions": rd_instructions,
        "shared_path_leads": shared_path_leads,
        "weak_evidence": weak_evidence,
        "repeat_offenders": repeat_offenders,
        "version": str(version_meta.get("version") or ""),
        "version_fixes_checked": bool(version_meta),
        "fixed_earlier_in_version": fixed_earlier,
        "closed_no_action": {
            "passed": len(passed_bugs),
            "passed_bugs": passed_bugs,
            "failed_back_to_rd": failed_back_to_rd,
            "attachments": sum(len(row["tr_codes"]) for row in outcomes),
        },
    }


def _md_safe(text: str) -> str:
    """Neutralise markdown that fix notes contain as ordinary prose.

    Streamlit renders ``$...$`` as LaTeX, so an RD note quoting a format string
    — "Costs %1$d credit per second. Minimum %2$d credits." — has everything
    between the two dollars taken as math and rendered one character per line.
    Seen on DRI261880-0060. ``\\$`` renders as a plain dollar both in Streamlit
    and in any markdown viewer opening the saved file.
    """
    return str(text or "").replace("$", r"\$")


def _trim(text: str, limit: int = 155) -> str:
    """Shorten a quote at a word boundary; these run to 300+ characters.

    Also strips list punctuation RD leaves at the start of a pasted line, which
    otherwise reads as a stray bullet mid-sentence ("— - widthOfWaveCell(at:)").
    """
    text = " ".join(str(text or "").split()).lstrip("-–—•* ").strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:-") + "…"
    return _md_safe(text)


def focus_list_markdown(focus: dict, generated_at: str = "") -> str:
    """Render the focus list for reading cold, first thing in the morning.

    Scanning order, not interest order: the queue first, then where the changes
    landed, then what RD asked for. Anything already resolved sits at the
    bottom marked as needing nothing.

    Written to be skimmed in the thirty seconds before standup — a headline
    that answers "where do I start", a table for the ranking, and fix-note
    quotes trimmed rather than pasted whole.

    Every bug code is a link to its eCL eBug page, wrapped in backticks inside
    the link label so a run of codes still wraps instead of forming one long
    line. The links live in the file, not only on the page: the download button
    hands QA this same markdown, and a code they cannot click is a code they
    have to paste into eCL by hand.
    """
    surface = focus.get("change_surface") or []
    awaiting = focus.get("awaiting_verification") or []
    total = focus.get("bug_count", 0)

    earlier = focus.get("fixed_earlier_in_version") or []
    earlier_total = sum(row["bug_count"] for row in earlier)
    version = _md_safe(focus.get("version") or "")

    build_trs = focus.get("build_trs") or []
    build = focus.get("close_build") or "(unknown)"
    # The build number opens the TR that tests it, when there is exactly one;
    # with several, "Tested by" below names each rather than picking one.
    if len(build_trs) == 1 and focus.get("close_build"):
        build = f"[{build}]({tr_url(build_trs[0]['tr_code'])})"
    out = [FOCUS_LIST_FORMAT_MARKER, "", f"# What to test — build {build}", ""]

    # Headline: the one line someone reads if they read nothing else.
    headline = [f"**{total} fixes** in this build"]
    if surface:
        headline.append(
            f"heaviest: **{_md_safe(surface[0]['module'])}** "
            f"({surface[0]['bug_count']} of {total})"
        )
    if earlier_total:
        headline.append(f"**{earlier_total}** earlier {version} fixes to regression-check")
    if not focus.get("queue_known"):
        headline.append("queue unknown — this build's TR not read")
    else:
        headline.append(
            f"**{len(awaiting)} waiting on you**" if awaiting else "nothing waiting on you"
        )
    out += [" · ".join(headline), ""]
    if build_trs:
        out += ["Tested by " + ", ".join(
            tr_link(tr["tr_code"])
            + (" (cancelled)" if tr.get("form_state") == "Cancel" else "")
            for tr in build_trs
        ), ""]
    if generated_at:
        out += [f"_Generated {generated_at.replace('T', ' ')[:19]}_", ""]
    out += ["---", ""]

    closed = focus.get("closed_no_action") or {}
    failed = closed.get("failed_back_to_rd") or []
    passed_bugs = closed.get("passed_bugs") or []
    created_from = len(awaiting) + len(failed) + closed.get("passed", 0)

    out += ["## 1. Waiting on you", ""]
    if not focus.get("queue_known"):
        # Saying "nothing outstanding" here was the old failure: the queue was
        # read from earlier rounds' TRs, which are always fully verified.
        out += ["The Testing Request for this build has not been read from eCL "
                "yet, so its queue is unknown"
                + (" — open it above." if build_trs else
                   " — it may not exist yet, or no eCL token was available."), ""]
    elif awaiting:
        out += [f"{len(awaiting)} of the {created_from} bug(s) this TR was "
                "created from are not verified yet — the main scope of this round.",
                ""]
        for row in awaiting:
            note = "" if row.get("in_cohort") else (
                f" _(fixed into {_md_safe(row['close_build'])})_"
                if row.get("close_build") else " _(no Close Build# in eCL)_")
            out.append(f"- {bug_link(row['bug_code'])} · "
                       f"{_md_safe(row['status'] or 'status unknown')}{note}  \n"
                       f"  {_trim(row['short_description'], 120)}")
        outside = [row for row in awaiting if not row.get("in_cohort")]
        if outside:
            out += ["", f"_{len(outside)} of these are not closed to this "
                    "build, so their fix notes are not among the fixes analysed "
                    "below — read them on the ticket._"]
    elif created_from:
        out += [f"Nothing outstanding — all {created_from} bug(s) this TR was "
                "created from have been verified.", ""]
    else:
        out += ["This TR was created from no attached bugs.", ""]

    if surface:
        out += ["", "## 2. Where the changes are", "",
                "| Module | Fixes | Worst sev | Area |", "|---|---:|---:|---|"]
        for row in surface:
            severity = "?" if row["worst_severity"] == 99 else row["worst_severity"]
            out.append(f"| **{_md_safe(row['module'])}** | {row['bug_count']} | "
                       f"{severity} | {_md_safe(row['category']) or '—'} |")
        out += ["", "_Severity 1 is most severe._", ""]
        for row in surface:
            out.append(f"- **{_md_safe(row['module'])}** — "
                       + " ".join(bug_link(code) for code in row["bug_codes"]))

    # Identical instructions are common (a whole module's MUI fixes share one),
    # so group them rather than repeating the same sentence per bug.
    instructions = focus.get("rd_instructions") or []
    if instructions:
        # Group on the instruction alone. Keying on the locale string too split
        # one instruction across two bullets whenever a sibling ticket left the
        # language field empty — the same sentence printed twice.
        grouped: dict[str, dict] = {}
        for row in instructions:
            entry = grouped.setdefault(
                row["instruction"], {"codes": [], "locales": []}
            )
            entry["codes"].append(row["bug_code"])
            if row["locales"] and row["locales"] not in entry["locales"]:
                entry["locales"].append(row["locales"])
        out += ["", "## 3. RD asked for this specifically", ""]
        for instruction, entry in grouped.items():
            out.append(f"- **{_md_safe(instruction)}**  \n  "
                       + " ".join(
                           bug_link(code) for code in sorted(entry["codes"])))
            for locales in entry["locales"]:
                out.append(f"  _Languages on the ticket: {_md_safe(locales)}_")

    leads = focus.get("shared_path_leads") or []
    if leads:
        # Already-said filter. One RD note often covers several bugs, so the
        # same sentence arrives once per bug; and a stated risk shown in
        # section 3 arrives here again as its own quote. Print each distinct
        # sentence once, against every bug it covers.
        said = {row["instruction"].casefold() for row in instructions}
        grouped_quotes: dict[str, dict] = {}
        for row in leads:
            module = row["module"] or "(unmapped)"
            for quote in row["quotes"][:2]:
                trimmed = _trim(quote)
                key = trimmed.casefold()
                if not trimmed or key in said:
                    continue
                entry = grouped_quotes.setdefault(
                    key, {"text": trimmed, "module": module, "codes": []}
                )
                if row["bug_code"] not in entry["codes"]:
                    entry["codes"].append(row["bug_code"])

        # Invert to one bullet per set of bug codes. Keyed on the quote alone,
        # a bug whose root cause and solution both qualify printed its code
        # twice on two bullets — the grouping made the section longer, not
        # shorter.
        by_codes: dict[tuple, dict] = {}
        for entry in grouped_quotes.values():
            key = (entry["module"], tuple(sorted(entry["codes"])))
            by_codes.setdefault(
                key, {"module": entry["module"], "codes": key[1], "quotes": []}
            )["quotes"].append(entry["text"])

        if by_codes:
            out += ["", "## 4. Shared paths named in the fix notes", "",
                    "RD's own words for what else touches the changed code — "
                    "leads to judge, not predictions.", ""]
            by_module: dict[str, list[dict]] = {}
            for entry in by_codes.values():
                by_module.setdefault(entry["module"], []).append(entry)
            for module in sorted(by_module, key=lambda m: m.casefold()):
                out += [f"**{_md_safe(module)}**", ""]
                for entry in by_module[module]:
                    codes = " ".join(bug_link(code) for code in entry["codes"])
                    out.append(f"- {codes}  \n  {entry['quotes'][0]}")
                    out += [f"  - {quote}" for quote in entry["quotes"][1:]]
                out.append("")

    weak = focus.get("weak_evidence") or []
    if weak:
        out += ["", "## 5. Thin evidence — work from the ticket", ""]
        for row in weak:
            out.append(f"- {bug_link(row['bug_code'])}"
                       + (f" · {_md_safe(row['module'])}" if row["module"] else "")
                       + f" — {_md_safe(row['reason'])}")

    repeats = focus.get("repeat_offenders") or []
    if repeats:
        out += ["", "## 6. Seen before", "",
                "How much of what changed here has failed here before.", ""]
        for row in repeats:
            out.append(f"- **{_md_safe(row['module'])}** — {row['bugs_with_history']} of "
                       f"{row['bug_count']} fix(es) have prior similar defects")
            out += [f"  - _e.g. {_trim(example, 110)}_"
                    for example in row["examples"][:1]]

    if earlier:
        out += ["", f"## 7. Fixed earlier in {version} — regression check", "",
                f"{earlier_total} fix(es) closed into earlier {version} builds "
                "ship in this build too. Most were verified on their own round; "
                "they are still code that changed this version, so re-check "
                "the paths — modules this build changes again come first.", ""]
        for row in earlier:
            again = " · **changed again in this build**" if row["changed_again"] else ""
            codes = " ".join(
                bug_link(bug["bug_code"])
                + ("" if bug["status"].casefold() == "close"
                   else f" _({_md_safe(bug['status'] or 'status unknown')})_")
                for bug in row["bugs"]
            )
            out.append(f"- **{_md_safe(row['module'])}** ({row['bug_count']})"
                       f"{again}  \n  {codes}")

    found = focus.get("found_on_build_tr") or []
    if found:
        # "This TR Generated the following eBug(s)": found during this round,
        # already reported — listed so nobody files them twice.
        out += ["", "## Found on this TR so far", "",
                f"{len(found)} new bug(s) opened while testing this build — "
                "already reported, check here before filing a duplicate.", ""]
        out += [f"- {bug_link(row['bug_code'])} · "
                f"{_md_safe(row['status'] or 'status unknown')}  \n"
                f"  {_trim(row['short_description'], 120)}" for row in found]

    attached = len(failed) + closed.get("passed", 0)
    if not focus.get("queue_known"):
        out.append("")
        return "\n".join(out)
    out += ["", "---", "", "## Already verified — nothing left to verify", ""]
    if attached:
        out += [f"{attached} of the bug(s) this TR was created from already have "
                "a verdict. A passed fix is still a changed path — worth a glance "
                "when regression-testing its module.", ""]
    out.append(f"- **{closed.get('passed', 0)}** verified and passed"
               + (": " + " ".join(bug_link(row["bug_code"]) for row in passed_bugs)
                  if passed_bugs else "."))
    refixed = [row for row in passed_bugs if row.get("failed_a_round")]
    if refixed:
        out.append(f"  - {len(refixed)} of these failed a round first and were "
                   "closed after a re-fix — the riskiest of the passed: "
                   + " ".join(bug_link(row["bug_code"]) for row in refixed))
    if failed:
        out += ["- **{n}** verified and **failed** — a colleague already checked "
                "these and handed them back to RD to fix. Not QA's queue: "
                "{codes}".format(
                    n=len(failed),
                    codes=" ".join(bug_link(row["bug_code"]) for row in failed),
                )]
    out.append("")
    return "\n".join(out)


def cohort_provenance(details_meta: dict) -> dict:
    """Summarize where this report's cohort came from and what it may be missing.

    The cohort is reconstructed from the parsed CSV, which has no ParentTRCode
    column, so it is really a Close Build — and one Close Build bundles many
    Testing Requests. When IF3 has been read, ``meta.cohort_audit`` says whether
    that reconstruction matches eCL's own TR-to-eBug relation.

    A gap here qualifies every prediction on the page: the analysis ran over an
    incomplete cohort. That is a fact about the evidence, not a model output,
    so it is computed here rather than asked of the model.
    """
    audit = details_meta.get("cohort_audit") or {}
    context = details_meta.get("tr_context") or {}
    backfill = details_meta.get("ecl_direct_backfill") or {}
    missing = audit.get("missing_from_cohort") or []
    undetermined = audit.get("undetermined") or []
    return {
        "checked": bool(audit),
        "testing_requests": context.get("testing_requests") or [],
        "missing_from_cohort": missing,
        "undetermined": undetermined,
        "candidates_considered": int(audit.get("candidates_considered") or 0),
        "limit_reached": bool(audit.get("detail_check_limit_reached")),
        "recovered_bug_codes": backfill.get("recovered_bug_codes") or [],
        "errors": list(audit.get("errors") or []) + list(context.get("errors") or []),
    }


def build_tr_queue_sources(build_trs: list[dict] | None) -> list[dict]:
    """The build TRs whose "created from" list was read and still counts.

    Cancelled TRs are dropped: eCL still returns them, but their attachments
    are nobody's queue. A TR recorded before Stage 8 read the lists carries no
    ``bugs_loaded`` and is dropped too — its missing list means "not read".
    """
    return [
        tr for tr in (build_trs or [])
        if tr.get("bugs_loaded")
        and str(tr.get("form_state") or "").strip().casefold() != "cancel"
    ]


def attached_bug_outcomes(testing_requests: list[dict]) -> list[dict]:
    """One row per attached *bug*, with where it stands now.

    Two things made the per-row count wrong on the 2026-09-23 pdri page (71
    passed + 9 failed from 78 bugs):

    - A bug re-attached to a later TR after failing is one bug, not two.
      DRI261884-0009 failed on TR260902-101 and passed on TR260904-097, and
      was counted in both piles.
    - A TR verdict is a snapshot of that round. The eBug's own status is
      current: a bug that failed a round, was re-fixed and is now ``Close`` is
      not "handed back to RD" any more. Eight of the nine "failed" were Close.

    So the latest TR (TR codes sort by date) supplies the verdict, and the
    eBug status overrides it when it has settled the bug either way.
    """
    rows_by_bug: dict[str, list[tuple[str, dict]]] = {}
    for context in testing_requests:
        tr_code = str(context.get("tr_code") or "")
        for attached in context.get("bugs_attached_for_verify") or []:
            code = str(attached.get("bug_code") or "").strip()
            if code:
                rows_by_bug.setdefault(code, []).append((tr_code, attached))

    outcomes = []
    for code, rows in rows_by_bug.items():
        rows.sort(key=lambda item: item[0])
        tr_code, latest = rows[-1]
        status = str(latest.get("status") or "").strip()
        verified, result = latest.get("is_verified"), latest.get("bug_result")
        if status.casefold() == "close":
            outcome = "passed"
        elif status.casefold() == "fail":
            outcome = "failed"
        elif verified is True and result is True:
            outcome = "passed"
        elif verified is True and result is False:
            outcome = "failed"
        else:
            outcome = "awaiting"
        outcomes.append({
            "bug_code": code,
            "tr_code": tr_code,
            "tr_codes": [tr for tr, _ in rows],
            "status": status,
            "close_build": str(latest.get("close_build") or "").strip(),
            "short_description": latest.get("short_description"),
            "qa_note": latest.get("verify_reason"),
            "outcome": outcome,
            # Kept apart for the tally: "not checked" vs "flag missing".
            "verify_state": ("not_verified" if verified is False else
                             "unknown" if outcome == "awaiting" else "verified"),
            "failed_a_round": any(
                row.get("is_verified") is True and row.get("bug_result") is False
                for _, row in rows
            ),
        })
    outcomes.sort(key=lambda row: row["bug_code"])
    return outcomes


def verification_tally(testing_requests: list[dict]) -> dict:
    """Count attached-bug verification outcomes, once per bug.

    Per the ECL guide this relation is the only programmatic source for "how
    many were verified, and how many passed". ``attachments`` keeps the raw row
    count so a bug attached to two TRs is visible as such, not as two bugs.
    """
    outcomes = attached_bug_outcomes(testing_requests)
    tally = {"attached": len(outcomes), "verified": 0, "passed": 0, "failed": 0,
             "not_verified": 0, "unknown": 0,
             "attachments": sum(len(row["tr_codes"]) for row in outcomes)}
    for row in outcomes:
        if row["outcome"] in ("passed", "failed"):
            tally["verified"] += 1
            tally[row["outcome"]] += 1
        else:
            tally[row["verify_state"]] += 1
    return tally


def render_cohort_provenance(st, details_meta: dict, bug_url: str) -> None:
    """Render cohort completeness and TR scope. Read-only; no model output."""
    provenance = cohort_provenance(details_meta)
    if not provenance["checked"] and not provenance["testing_requests"]:
        return

    missing = provenance["missing_from_cohort"]
    if missing:
        st.error(
            f"Cohort gap: {len(missing)} bug(s) are TRCreated on this Close Build "
            "in eCL but are absent from this report's cohort. Every "
            "recommendation below was derived without them.\n\n"
            + "\n".join(
                f"- [`{item.get('bug_code')}`]"
                f"({bug_url.format(bug_code=item.get('bug_code'))}) "
                f"(on {item.get('tr_code')}) — "
                f"{_escape_markdown(item.get('short_description') or 'no summary')}"
                for item in missing
            )
            + "\n\nRe-run `process_newest_tr.py --ecl-complete-cohort` to include them."
        )
    elif provenance["checked"]:
        st.success(
            "Cohort verified against eCL's TR-to-eBug relation: "
            f"{provenance['candidates_considered']} additional candidate(s) "
            "checked and correctly out of scope."
        )
    if provenance["limit_reached"] or provenance["undetermined"]:
        st.caption(
            f"{len(provenance['undetermined'])} candidate(s) could not be "
            "classified, so the check is partial."
        )
    if provenance["recovered_bug_codes"]:
        st.caption(
            "Recovered over IF3 REST after the bug-detail webhook failed: "
            + ", ".join(
                f"[`{code}`]({bug_url.format(bug_code=code)})"
                for code in provenance["recovered_bug_codes"]
            )
        )
    for message in provenance["errors"]:
        st.caption(f"eCL read issue: {message}")

    requests = provenance["testing_requests"]
    # Same source as the focus list's queue: the build TR's created-from list.
    # Tallying the provenance TRs' attachments here reported earlier rounds as
    # "this round".
    tally = verification_tally(build_tr_queue_sources(details_meta.get("build_trs")))
    if tally["attached"]:
        st.caption(
            f"This build's TR verification — {tally['attached']} bug(s) it was "
            f"created from · {tally['passed']} passed · {tally['failed']} failed · "
            f"{tally['not_verified']} not yet verified · "
            f"{tally['unknown']} unknown"
        )
    # The attached-bug queue and the verified-failure list both live in the
    # focus list at the top of the page, which is the single place that encodes
    # what a QA "Fail" means: checked already, handed back to RD, not QA's work.
    # A second rendering here disagreed with it — the focus list reported
    # nothing outstanding while this block called the same 3 bugs "outstanding".

    # A cancelled TR still comes back from eCL's queries (guide 3-2), so it can
    # silently contribute scope to a report that certifies a build.
    cancelled = [
        str(context.get("tr_code") or "") for context in requests
        if str(context.get("form_state") or "").strip().casefold() == "cancel"
    ]
    if cancelled:
        st.warning(
            f"{len(cancelled)} of these Testing Requests are cancelled "
            f"(FormState = Cancel): {', '.join(cancelled)}. eCL still returns "
            "them, so their bugs may be contributing scope that no longer counts."
        )

    if requests:
        # These are the TRs the bugs were *found* on, each testing an earlier
        # build — none of the 9 on the 2026-09-17 pdri report tested the
        # cohort's own Close Build. Labelling them "in this Close Build"
        # contradicted the Bug Creation TRs card directly above.
        with st.expander(
            f"Testing Requests these bugs were found on ({len(requests)})"
        ):
            st.caption(
                "Provenance, not scope: each TR tested the build its bug was "
                "reported against, not the Close Build being certified here. "
                "These records come from eCL, not from the parsed bug list."
            )
            for context in requests:
                st.markdown(
                    f"**{_escape_markdown(context.get('tr_code'))}** — "
                    f"{_escape_markdown(context.get('short_description') or 'no title')}  \n"
                    f"Status: {_escape_markdown(context.get('status') or '—')} · "
                    # "Fail" is the normal verdict for a round that found bugs,
                    # so it is stated, not styled as an alarm.
                    f"Result: {_escape_markdown(context.get('result') or '—')} · "
                    f"Build: {_escape_markdown(context.get('tr_build') or '—')} · "
                    f"Owner: {_escape_markdown(context.get('owner') or '—')} · "
                    f"Tester: {_escape_markdown(context.get('tester') or '—')}  \n"
                    f"{len(context.get('bugs_found_on_tr') or [])} bug(s) found on "
                    f"this TR · "
                    f"{len(context.get('bugs_attached_for_verify') or [])} attached "
                    f"for verification"
                )


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
