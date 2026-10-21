#!/usr/bin/env python3
"""Direct IF3 REST access for pipeline stages — a second path to eCL.

Stage 8 gets its bug details from the n8n webhook, whose eCL node returns a
per-item transient error for roughly one detail fetch in 32. After two retry
rounds ``process_newest_tr`` gives up and enters the bug as a summary-only stub
(``local_fallback_bugs``) carrying ``DetailFetchError`` — no repro steps, no RD
root cause, no verify comments.

This module queries the same ``getOldeBug`` record over IF3 REST, a different
transport, so a *transport* failure no longer costs the report a bug's
evidence. Records come back in the shape the n8n Organize node produces, so
nothing downstream needs to know the source.

What it cannot fix: a failure inside eCL's own stored procedure. The bug the
2026-09-15 pdri report lost, DRI261882-0021, answers IF3 with ``HTTP 500
String or binary data would be truncated`` — the same server-side fault n8n
hit, reached by a second road. Verified against the live pool 2026-09-16. The
backfill is a second transport, not a second copy of the data.

The larger win turned out to be the cohort audit below, not this backfill.

Layering: ``ecl_if3.py`` and ``ecl_query.py`` are vendored upstream tools and
are not edited here. This module is the only place that knows both the IF3 row
shape and the toolkit's record shape; pipeline stages import this, never
``ecl_if3`` directly.

**Pipeline only — never a dashboard data source.** REST query tokens carry no
permission filtering (ECL guide 3-5): a valid token reads every ticket in
scope, including ones its holder could not open in the ECL web UI. The
dashboard is a shared process on port 8501, so these calls belong in batch
stages that write into immutable evidence snapshots, never behind a web
control.

Needs a token in ``$ECL_TOKEN`` or ``~/.ecl_token``. Without one every entry
point reports "unavailable" instead of raising, so Stage 8 behaves exactly as
it does today on a machine that has no token.
"""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

try:  # Direct script execution (scripts/ is first on sys.path)
    from ecl_if3 import ECLError, call, query, token
    from summarize_tr_bugs import _parse_json_field, enrich_from_raw
except ModuleNotFoundError:  # Imported as scripts.ecl_direct
    from scripts.ecl_if3 import ECLError, call, query, token
    from scripts.summarize_tr_bugs import _parse_json_field, enrich_from_raw

#: eCL's stored-procedure query model for a full eBug record. Same data the
#: n8n workflow's "getOldeBug" operation returns.
EBUG_DETAIL_MODEL = "TSR.SP.GetOldEbugData"

#: The Testing Request record itself, and the two eBug relations eCL keeps for
#: it. ``eBugByTR`` is what was found *on* the TR; ``AttachedEBug`` is what was
#: attached to it *for verification* — a distinct list, and the one that
#: describes the testing scope.
#:
#: On the eCL TR page these are two of its three bug lists. ``AttachedEBug`` is
#: "Bugs from which this Testing Request is created" — the TR's own scope, and
#: the same codes as the TR record's ``RefBugs`` (18 = 18 on TR260922-0013).
#: ``eBugByTR`` is "This TR Generated the following eBug(s)" — new bugs opened
#: while testing it. The third, "Known bugs" (open bugs that predate the TR),
#: is not in any IF3 query model or in ``TSR.TRFormData``; checked 2026-09-29.
TR_DETAIL_MODEL = "TR.SP.GetOldTRData"
TR_OWN_BUGS_MODEL = "TSR.eBugByTR"
TR_ATTACHED_BUGS_MODEL = "TSR.AttachedEBug"

#: Fallback for the TR record. ``TR.SP.GetOldTRData`` comes from the vendored
#: ecl_query tool and is absent from the ECL guide's model list, so it may not
#: exist on every deployment. ``TSR.TRSearch`` is documented (guide 3-2), spans
#: all ~229,000 TRs including old ASP ones, and returns the same field names —
#: this product's TRs are old-format (3-digit serial, e.g. TR260910-067), so
#: the fallback is the likelier path here, not an edge case.
TR_SEARCH_MODEL = "TSR.TRSearch"

#: All Query endpoints share one rate bucket, so space out a cohort's calls.
DEFAULT_CALL_DELAY = 0.2

#: Rows per page when walking a TR's eBug relations.
PAGE_SIZE = 200

#: (organized key, raw key). Verified against a stored bug_details.json record
#: and its embedded ``raw`` row — the two spellings differ in ways that are
#: easy to get wrong (``Build`` vs ``Build#``, ``EbugSeverity`` vs ``Severity``,
#: ``MainHandler`` vs ``Handler``), and a mismatch here reads downstream as a
#: missing field rather than as an error.
FIELD_MAP = (
    ("BugCode", "BugCode"),
    ("ProductName", "ProductName"),
    ("Status", "Status"),
    ("Short Description", "ShortDescription"),
    ("Priority", "EbugPriority"),
    ("Severity", "EbugSeverity"),
    ("Create Date", "CreateTime"),
    ("Closed Date", "CloseTime"),
    ("Build#", "Build"),
    ("Close Build#", "CloseToBuild"),
    ("Creator", "Creator"),
    ("Handler", "MainHandler"),
    ("ParentTRCode", "ParentTRCode"),
)


#: Repo-local token file. Git-ignored; `.ecl_token.example` is the tracked
#: template. Keeping it here rather than only in ``~`` means the Jenkins user
#: on the Mac Mini gets the same token as an interactive run without a second
#: copy in its own home directory.
REPO_TOKEN_FILE = Path(__file__).resolve().parent.parent / ".ecl_token"

#: The example file's body. A copy that was never filled in must fail loudly
#: here rather than as a puzzling 401 midway through a stage.
TOKEN_PLACEHOLDER = "PASTE_YOUR_ECL_TOKEN_HERE"

TOKEN_PAGE = "https://ecl.cyberlink.com/sso/mgm/tokenPage"


def read_token_file(path: Path) -> str | None:
    """Read a token file, ignoring blank lines and ``#`` comments.

    ``ecl_if3.token`` takes a whole file as the token, so a commented file
    would send the comments as the bearer credential. Parsing here lets the
    example file explain itself.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def token_status() -> dict:
    """Resolve the token and describe where it came from.

    Order: ``$ECL_TOKEN``, then ``$ECL_TOKEN_FILE``, then the repo-local
    ``.ecl_token``, then ``~/.ecl_token`` (the vendored client's own default).
    A repo-local token is exported into ``$ECL_TOKEN`` so ``ecl_if3.call``
    finds it without the vendored file being edited.

    Returns ``{"available", "source", "reason", "warnings"}``. ``warnings`` are
    advisory — a readable-by-others token file still works, but this is a
    credential that reads every ticket in scope, so say so.
    """
    warnings: list[str] = []

    if not os.environ.get("ECL_TOKEN", "").strip() and REPO_TOKEN_FILE.exists():
        value = read_token_file(REPO_TOKEN_FILE)
        if value == TOKEN_PLACEHOLDER:
            return {
                "available": False, "source": str(REPO_TOKEN_FILE),
                "warnings": warnings,
                "reason": (
                    f"{REPO_TOKEN_FILE.name} still contains the example "
                    f"placeholder. Replace it with a token from {TOKEN_PAGE}."
                ),
            }
        if value:
            mode = REPO_TOKEN_FILE.stat().st_mode
            if mode & (stat.S_IRWXG | stat.S_IRWXO):
                warnings.append(
                    f"{REPO_TOKEN_FILE.name} is readable by other users "
                    f"(mode {stat.filemode(mode)}); run "
                    f"chmod 600 {REPO_TOKEN_FILE.name}"
                )
            # Only fills a gap: an explicit ECL_TOKEN above always wins.
            os.environ["ECL_TOKEN"] = value

    try:
        token()
    except ECLError as exc:
        # The client's message spans several lines and names the token page and
        # both install paths; keep all of it, on one line for the stage log.
        return {"available": False, "source": None, "warnings": warnings,
                "reason": " ".join(str(exc).split())}

    if os.environ.get("ECL_TOKEN_FILE"):
        source = os.environ["ECL_TOKEN_FILE"]
    elif REPO_TOKEN_FILE.exists():
        source = str(REPO_TOKEN_FILE)
    else:
        source = "$ECL_TOKEN or ~/.ecl_token"
    return {"available": True, "source": source, "reason": None,
            "warnings": warnings}


def unavailable_reason() -> str | None:
    """Return why direct IF3 access cannot be used, or None when it can.

    ``token`` is bound at import rather than re-imported here on purpose. This
    module is reachable as both ``ecl_direct`` and ``scripts.ecl_direct``, and
    a function-local ``from ecl_if3 import token`` resolves against whichever
    spelling is on sys.path — which can be a *second* module object for the
    same file, with its own state.
    """
    return token_status()["reason"]


def available() -> bool:
    return unavailable_reason() is None


def status_timeline(raw: dict) -> list[dict]:
    """Flatten StatusLog into the {status, user, time} rows n8n produces.

    ``enrich_from_raw`` derives Resolutions/VerifyComments/ReproSteps from the
    same log but deliberately leaves StatusTimeline alone, because the n8n
    Organize node already supplied it. A record fetched here has no Organize
    node in front of it, so build it.
    """
    log = _parse_json_field(raw.get("StatusLog")) or []
    if not isinstance(log, list):
        return []
    return [
        {
            "status": entry.get("Status"),
            "user": entry.get("UserName"),
            "time": entry.get("StatusTime"),
        }
        for entry in log
        if isinstance(entry, dict)
    ]


def organize_record(raw: dict) -> dict:
    """Shape one IF3 row like an n8n-organized bug record.

    ``enrich_from_raw`` fills the derived fields (Resolutions, ReproSteps,
    VerifyComments, OSLanguage, the severity/priority labels, AttachmentCount)
    from ``raw``, so only the flat renames are done here. Running it now rather
    than leaving it to the summarizer matters, and not for the obvious reason:
    ``process_newest_tr`` scores the record with ``detail_quality_score``
    before the summarizer ever sees it, and a record carrying only the flat
    renames scores exactly 2 — it *passes* ``has_usable_full_detail`` on the
    strength of its raw blob alone, while holding no root cause, no fix and no
    repro steps. It would be counted as recovered and silently contribute
    nothing. Enriched, the same records score 7-8.
    """
    record = {
        organized: raw.get(source)
        for organized, source in FIELD_MAP
    }
    record["raw"] = raw
    record["StatusTimeline"] = status_timeline(raw)
    return enrich_from_raw(record)


def _paged(model: str, conditions: list[dict], *, limit: int = 2000) -> list[dict]:
    """Walk every page of a query result.

    ``ecl_if3.query`` reads page 1 only. A Testing Request's eBug list is
    normally well under one page, but a cohort-completeness check that silently
    stops at 200 rows would under-report exactly when it matters most.
    """
    rows: list[dict] = []
    page = 1
    while len(rows) < limit:
        size = min(PAGE_SIZE, limit - len(rows))
        _, body = call("POST", f"/IF3/tsr/api/public/Query/{model}", {
            "queryModel": model, "pageIndex": page, "pageSize": size,
            "conditions": conditions,
        })
        if not isinstance(body, dict):
            break
        data = body.get("data") if isinstance(body.get("data"), dict) else body
        page_rows = data.get("results") or []
        rows.extend(page_rows)
        total = data.get("total") or 0
        if not page_rows or len(rows) >= total:
            break
        page += 1
    return rows[:limit]


def _bug_relation(model: str, tr_code: str) -> list[dict]:
    rows = _paged(model, [{"field": "ParentTRCode", "term": {"query": tr_code}}])
    listed = [
        {
            "bug_code": str(row.get("BugCode") or "").strip(),
            "status": row.get("Status"),
            "short_description": row.get("ShortDescription"),
            # These three exist only on AttachedEBug, and they are the only
            # programmatic source of "how many did this round verify, and how
            # many passed". None means "not supplied", which is not the same as
            # a definite False. is_verified says the attachment was checked;
            # bug_result says whether it PASSED — a verified failure is both.
            "is_verified": row.get("IsVerified"),
            "bug_result": row.get("BugResult"),
            "verify_reason": row.get("Reason"),
            # Present on both relations and populated for anything fixed into a
            # build, which is what makes the cohort audit free of detail calls.
            # A null here is meaningful — not fixed into any build — and must
            # not be confused with the field being absent from the row shape,
            # which is the only case worth spending a detail call on.
            "close_build": row.get("CloseToBuild"),
            "close_build_known": "CloseToBuild" in row,
        }
        for row in rows
        if str(row.get("BugCode") or "").strip()
    ]
    listed.sort(key=lambda item: (item["bug_code"].casefold(), item["bug_code"]))
    return listed


def _tr_record(tr_code: str) -> list[dict]:
    """Read one TR record, falling back to the documented search model.

    A model this deployment does not have answers with an ECLError, which the
    caller would otherwise record as a hard failure for the TR. Both models
    return the same field names, so the fallback needs no separate mapping.
    """
    condition = [{"field": "TRCode", "term": {"query": tr_code}}]
    try:
        detail = query(TR_DETAIL_MODEL, condition, page_size=1)
    except ECLError as exc:
        # A spent rate bucket or a rejected token is not a missing model:
        # retrying on the fallback only burns another call and reports the
        # wrong cause. Let those through.
        if _rate_limited(exc):
            raise
        detail = []

    search = query(TR_SEARCH_MODEL, condition, page_size=1)
    if not detail:
        return search
    if not search:
        return detail
    # The two models are complementary, not redundant, so merge rather than
    # pick: GetOldTRData alone answers FormState with null (a cancelled TR then
    # looks Normal), while TRSearch has no Result, IsGMBuild or RefBugs. Verified
    # against the live pool 2026-09-16. Non-empty values win, detail first.
    merged = dict(search[0])
    for key, value in detail[0].items():
        if value not in (None, "", [], {}):
            merged[key] = value
    return [merged]


def _build_number(tr_build: object) -> str:
    """``"PowerDirector Mobile for iOS: 16.8.0.189304"`` -> ``"16.8.0.189304"``."""
    return str(tr_build or "").rsplit(":", 1)[-1].strip()


def find_build_trs(close_build: str, product_name: str, *,
                   with_bugs: bool = True, delay: float = DEFAULT_CALL_DELAY
                   ) -> tuple[list[dict], list[str]]:
    """The Testing Request(s) that test ``close_build`` itself.

    Not the same thing as ``fetch_tr_context``'s TRs: those are where each bug
    was *reported*, and none of them tests the Close Build (2026-09-17). The TR
    that does is the one SR_AUTO opens for the build, whose ``TRBuild`` is
    ``"<product>: <build>"`` — ``16.8.0.189304`` -> ``TR260922-0013``.

    ``match`` is a substring match, so ``16.8.0.18930`` would also hit
    ``16.8.0.189304``; rows are kept only on an exact build and product. One
    TSR.TRSearch call. Cancelled TRs are kept and carry ``form_state`` so the
    caller can say so rather than silently pick another.

    ``with_bugs`` also reads each live TR's two bug lists (two calls per TR):
    the bugs it was created from, which are what QA verifies on it and so the
    real queue, and the bugs it has generated so far. That queue is *not* this
    Close Build's cohort — TR260922-0013 was created from 18 bugs, 17 of them
    closed into 189251 / 189285, builds whose own TRs were cancelled.
    ``bugs_loaded`` separates "read, and empty" from "never read".
    """
    build = str(close_build or "").strip()
    if not build:
        return [], []
    reason = unavailable_reason()
    if reason:
        return [], [f"IF3 build TR lookup unavailable: {reason}"]
    conditions = [{"field": "TRBuild", "match": {"query": build}}]
    if product_name:
        conditions.append({"field": "ProductName", "match": {"query": product_name}})
    try:
        rows = query(TR_SEARCH_MODEL, conditions, page_size=50)
    except ECLError as exc:
        return [], [f"build TR lookup for {build}: {exc}"]
    found = [
        {
            "tr_code": str(row.get("TRCode") or "").strip(),
            "short_description": row.get("ShortDescription"),
            "status": row.get("Status"),
            "form_state": row.get("FormState"),
            "result": row.get("Result"),
            "tr_build": row.get("TRBuild"),
            "created_at": row.get("CreateTime"),
            "due_date": row.get("TRDueDate"),
            "tester": row.get("Tester"),
        }
        for row in rows
        if str(row.get("TRCode") or "").strip()
        and _build_number(row.get("TRBuild")) == build
        and (not product_name
             or str(row.get("ProductName") or "").strip() == product_name)
    ]
    found.sort(key=lambda item: (item["tr_code"].casefold(), item["tr_code"]))
    errors: list[str] = []
    if not with_bugs:
        return found, errors
    for index, tr in enumerate(found):
        # A cancelled TR's attachments are nobody's queue.
        if str(tr.get("form_state") or "").strip().casefold() == "cancel":
            continue
        if index and delay:
            time.sleep(delay)
        try:
            attached = _bug_relation(TR_ATTACHED_BUGS_MODEL, tr["tr_code"])
            generated = _bug_relation(TR_OWN_BUGS_MODEL, tr["tr_code"])
        except ECLError as exc:
            errors.append(f"bug lists for {tr['tr_code']}: {exc}")
            if _rate_limited(exc):
                break
            continue
        # Same keys as fetch_tr_context, so attached_bug_outcomes reads both.
        tr["bugs_attached_for_verify"] = attached
        tr["bugs_found_on_tr"] = generated
        tr["bugs_loaded"] = True
    return found, errors


def fetch_tr_context(tr_codes: list[str], *, delay: float = DEFAULT_CALL_DELAY
                     ) -> tuple[list[dict], list[str]]:
    """Fetch each Testing Request record and both of its eBug relations.

    The pipeline reconstructs its cohort from ``Status == TRCreated`` plus a
    shared ``Close Build#`` in the parsed CSV, because that CSV has no
    ParentTRCode column. That reconstruction is a Close Build, and one Close
    Build bundles many Testing Requests — the 2026-09-15 pdri report covered 20
    bugs spanning 9 of them. eCL knows the real TR-to-eBug relation, so this
    reads it rather than inferring it.

    Returns ``(contexts, errors)``, one context per TR code, ordered by TR code
    so the result is stable across runs.
    """
    reason = unavailable_reason()
    if reason:
        return [], [f"IF3 TR context unavailable: {reason}"]

    contexts: list[dict] = []
    errors: list[str] = []
    ordered = sorted(
        {str(code).strip() for code in tr_codes if str(code).strip()},
        key=lambda code: (code.casefold(), code),
    )
    for index, tr_code in enumerate(ordered):
        if index and delay:
            time.sleep(delay)
        try:
            records = _tr_record(tr_code)
            own = _bug_relation(TR_OWN_BUGS_MODEL, tr_code)
            attached = _bug_relation(TR_ATTACHED_BUGS_MODEL, tr_code)
        except ECLError as exc:
            errors.append(f"{tr_code}: {exc}")
            if _rate_limited(exc):
                remaining = ordered[index + 1:]
                if remaining:
                    errors.append(
                        f"stopped before {len(remaining)} remaining TR(s); the "
                        "token is rejected or its rate bucket is spent"
                    )
                break
            continue

        record = records[0] if records else {}
        if not records:
            errors.append(f"{tr_code}: TR record not found in IF3")
        contexts.append({
            "tr_code": tr_code,
            "short_description": record.get("ShortDescription"),
            "status": record.get("Status"),
            # Only TRSearch carries this; null here would read as "Normal".
            "form_state": record.get("FormState"),
            "product_name": record.get("ProductName"),
            "template_name": record.get("TemplateName"),
            "tr_build": record.get("TRBuild"),
            "owner": record.get("Owner"),
            "tester": record.get("Tester"),
            "due_date": record.get("TRDueDate"),
            # Only GetOldTRData carries these. `result` is this TR's overall
            # verdict ("Fail" on a live example) and the release flags say
            # whether the build it covers is a GM / shippable candidate — all
            # of it context a build-certification page should state outright.
            "result": record.get("Result"),
            "is_gm_build": record.get("IsGMBuild"),
            "is_shippable": record.get("IsShippable"),
            "closed_at": record.get("CloseTime"),
            "bugs_found_on_tr": own,
            "bugs_attached_for_verify": attached,
        })
    return contexts, errors


def audit_cohort_completeness(
    tr_contexts: list[dict],
    cohort_codes: list[str],
    close_build: str,
    expected_product: str | None = None,
    *,
    max_detail_checks: int = 60,
    delay: float = DEFAULT_CALL_DELAY,
) -> dict:
    """Check the reconstructed cohort against eCL's own TR-to-eBug relation.

    A bug sitting on one of these Testing Requests is *not* automatically a
    cohort omission: the same TR carries bugs closed to other builds and bugs
    never fixed into any build. The deciding test is Close Build#.

    **Close Build#, deliberately not status.** ``select_newest_tr_cohort``
    keeps only ``Status == TRCreated`` rows, but that status is a moving
    target: a bug goes TRCreated -> Close as QA verifies it, so the longer
    verification runs the more of the build's bugs drop out of the report.
    Measured against the live pool on 2026-09-16, build 16.8.0.189063 had 26
    bugs closed into it and the report covered 20 — the six it lost were all
    Close, and took the whole "Creator Music" module out of the report with
    them. Close Build# does not drift, so auditing on it catches exactly the
    bugs the status filter silently sheds.

    Costs no detail calls: both relations already return ``CloseToBuild``, and
    a row with none is not fixed into any build, so it cannot belong here.
    ``max_detail_checks`` remains as a bounded fallback for rows that arrive
    without the field at all.

    Reports; never edits the cohort. Expanding what a report certifies on the
    strength of a live query is the caller's decision, not this function's.
    """
    cohort = {str(code).strip() for code in cohort_codes}
    # Attached-for-verification bugs are candidates too: they usually belong to
    # other builds, and the Close Build# test costs nothing to apply to them.
    candidates: dict[str, dict] = {}
    for context in tr_contexts:
        tr_code = str(context.get("tr_code") or "")
        for key in ("bugs_found_on_tr", "bugs_attached_for_verify"):
            for row in context.get(key) or []:
                code = str(row.get("bug_code") or "").strip()
                if not code or code in cohort or code in candidates:
                    continue
                candidates[code] = {
                    "bug_code": code, "tr_code": tr_code,
                    "close_build": row.get("close_build"),
                    "close_build_known": row.get("close_build_known", False),
                    "status": row.get("status"),
                    "short_description": row.get("short_description"),
                }

    ordered = sorted(candidates, key=lambda code: (code.casefold(), code))
    missing, excluded, unresolved = [], [], []
    for code in ordered:
        entry = candidates[code]
        build = str(entry.get("close_build") or "").strip()
        if not entry.get("close_build_known"):
            # The row shape did not carry the field at all — the only case a
            # detail call can still settle.
            unresolved.append(entry)
        elif build == close_build:
            missing.append(entry)
        else:
            excluded.append({
                **entry,
                "reason": "not fixed into any build" if not build
                          else "closed to a different build",
            })

    # Only rows that arrived with no Close Build# field at all reach the API.
    errors: list[str] = []
    checked = unresolved[:max_detail_checks]
    if checked:
        records, errors = fetch_bug_records(
            [entry["bug_code"] for entry in checked], expected_product, delay=delay
        )
        by_code = {str(r.get("BugCode") or "").strip(): r for r in records}
        still_unresolved = []
        for entry in checked:
            record = by_code.get(entry["bug_code"])
            if record is None:
                still_unresolved.append({**entry, "reason": "detail lookup failed"})
                continue
            entry["close_build"] = str(record.get("Close Build#") or "").strip()
            entry["short_description"] = record.get("Short Description")
            if entry["close_build"] == close_build:
                missing.append(entry)
            else:
                excluded.append({**entry, "reason": "closed to a different build"})
        unresolved = still_unresolved + [
            {**entry, "reason": "not checked (limit reached)"}
            for entry in unresolved[max_detail_checks:]
        ]
        missing.sort(key=lambda item: (item["bug_code"].casefold(), item["bug_code"]))

    return {
        "schema_version": 2,
        "close_build": close_build,
        "cohort_size": len(cohort),
        "checked_tr_codes": [
            str(c.get("tr_code") or "") for c in tr_contexts if c.get("tr_code")
        ],
        "selection_rule": (
            "Close Build# equal to the cohort's build. Deliberately not the "
            "selector's Status == TRCreated filter: status drifts to Close as "
            "QA verifies, which is how bugs go missing from the report."
        ),
        "candidates_considered": len(ordered),
        "detail_checks_spent": len(checked),
        "detail_check_limit_reached": len(unresolved) > 0 and len(checked) >= max_detail_checks,
        "missing_from_cohort": missing,
        "excluded": excluded,
        "undetermined": unresolved,
        "errors": errors,
    }


def _rate_limited(exc: ECLError) -> bool:
    text = str(exc)
    return "429" in text or "Unauthorized" in text or "401" in text


def fetch_bug_records(
    bug_codes: list[str],
    expected_product: str | None = None,
    *,
    delay: float = DEFAULT_CALL_DELAY,
) -> tuple[list[dict], list[str]]:
    """Fetch full detail for each BugCode over IF3 REST.

    Returns ``(records, errors)``. A code that cannot be fetched produces an
    error string and no record — the caller keeps its own fallback for those,
    so a partial result is still worth having.

    A 401 or 429 stops the batch: the token is bad or the shared bucket is
    spent, and the remaining codes would only burn more of it. Other per-code
    failures are recorded and the walk continues.
    """
    reason = unavailable_reason()
    if reason:
        return [], [f"IF3 direct fetch unavailable: {reason}"]

    records: list[dict] = []
    errors: list[str] = []
    for index, code in enumerate(bug_codes):
        if index and delay:
            time.sleep(delay)
        try:
            rows = query(
                EBUG_DETAIL_MODEL,
                [{"field": "BugCode", "term": {"query": code}}],
                page_size=1,
            )
        except ECLError as exc:
            errors.append(f"{code}: {exc}")
            if _rate_limited(exc):
                remaining = bug_codes[index + 1:]
                if remaining:
                    errors.append(
                        f"stopped before {len(remaining)} remaining code(s); "
                        "the token is rejected or its rate bucket is spent"
                    )
                break
            continue

        if not rows:
            errors.append(f"{code}: not found in IF3")
            continue

        raw = rows[0]
        product = str(raw.get("ProductName") or "").strip()
        if expected_product and product != expected_product:
            # Never let a same-code record from another pool enter the cohort;
            # filter_detail_data would drop it later, but silently.
            errors.append(
                f"{code}: belongs to {product!r}, not {expected_product!r}"
            )
            continue
        records.append(organize_record(raw))

    return records, errors
