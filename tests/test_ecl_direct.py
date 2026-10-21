"""IF3 direct-fetch adapter: field mapping, degradation, and batch behaviour.

No test here touches the network. ``query`` is replaced wherever a call would
be made, and the token check is forced, so the suite runs on a machine with no
ECL token exactly as it does on one with a valid token.
"""

import json
import os
from pathlib import Path

import pytest

import scripts.ecl_direct as ecl_direct
from scripts.ecl_if3 import ECLError
from scripts.process_newest_tr import detail_quality_score, has_usable_full_detail

PRODUCT = "PowerDirector Mobile for iOS"


@pytest.fixture
def token_present(monkeypatch):
    monkeypatch.setattr(ecl_direct, "unavailable_reason", lambda: None)


def raw_row(**overrides):
    """A getOldeBug row with the JSON-encoded string fields eCL really sends."""
    row = {
        "BugCode": "DRI261890-0001",
        "ProductName": PRODUCT,
        "Status": "TRCreated",
        "ShortDescription": "PDR-I 16.8.0 - Timeline: crash on undo",
        "EbugPriority": 2,
        "EbugSeverity": 3,
        "CreateTime": "2026-09-01T10:00:00.000",
        "CloseTime": None,
        "Build": "188249",
        "CloseToBuild": "16.8.0.189063",
        "Creator": "BALLY_HSU",
        "MainHandler": None,
        "BugBelong": "RD",
        "ParentTRCode": "TR260910-067",
        "StatusLog": json.dumps([
            {"Status": "Assigned", "UserName": "WEI_LU",
             "StatusTime": "2026-09-02T09:00:00"},
            {"Status": "RDResolved", "UserName": "WEI_LU",
             "StatusTime": "2026-09-03T09:00:00",
             "RootCause": "undo stack not cleared",
             "Solution": "clear on project close",
             "TraceURL": "https://example.invalid/commit/abc"},
        ]),
        "Comments": json.dumps([
            {"UserName": "VIVIAN_HUANG", "Comment": "verified on 189063",
             "CommentTime": "2026-09-10T11:00:00"},
        ]),
        "DetailData": json.dumps([
            {"FieldCode": "Describe Issue", "FieldValue": "1. open 2. undo"},
            {"FieldCode": "OS Language", "FieldValue": "-9 Lang"},
            {"FieldCode": "priority", "FieldValue": "Must Fix"},
            {"FieldCode": "severity", "FieldValue": "3 Normal"},
        ]),
        "UploadFile": json.dumps([{"FileName": "a.png"}, {"FileName": "b.png"}]),
    }
    row.update(overrides)
    return row


# ----------------------------------------------------------------- mapping

def test_organize_record_matches_stored_n8n_records():
    """The adapter must reproduce the n8n Organize node, not approximate it.

    Downstream code cannot tell which transport a record arrived on, so any
    divergence here is a silent evidence difference between two bugs in the
    same report. The stored report is the ground truth for that contract.
    """
    manifest_path = Path("data/products/pdri/tr_reports/latest.json")
    if not manifest_path.exists():
        pytest.skip("no stored pdri TR report to compare against")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    details = json.loads(
        (Path("data/products/pdri") / manifest["details_path"]).read_text(
            encoding="utf-8")
    )
    organized = [
        bug for bugs in details["products"].values() for bug in bugs
        if bug.get("raw")
    ]
    if not organized:
        pytest.skip("stored report carries no raw eBug rows")

    # Attached by later pipeline stages, not by the fetcher.
    ignore = {"HistoricalAnalogs", "ParsedModule", "ModuleCategory"}
    for bug in organized:
        rebuilt = ecl_direct.organize_record(bug["raw"])
        for key in (set(bug) | set(rebuilt)) - ignore:
            assert rebuilt.get(key) == bug.get(key), (
                f"{bug['BugCode']} field {key!r} diverges from the n8n record"
            )


def test_organize_record_maps_the_easily_confused_names():
    record = ecl_direct.organize_record(raw_row())
    assert record["Build#"] == "188249"          # not raw "Build"
    assert record["Close Build#"] == "16.8.0.189063"
    assert record["Severity"] == 3               # not raw "EbugSeverity"
    assert record["Priority"] == 2
    assert record["Short Description"].startswith("PDR-I 16.8.0")
    assert record["Create Date"] == "2026-09-01T10:00:00.000"
    assert record["Closed Date"] is None
    assert record["Handler"] is None             # raw "MainHandler"
    assert record["ModuleOwner"] == "RD"         # derived from raw "BugBelong"


def test_organize_record_derives_evidence_fields():
    record = ecl_direct.organize_record(raw_row())
    assert record["Resolutions"][0]["root_cause"] == "undo stack not cleared"
    assert record["Resolutions"][0]["trace_url"].endswith("abc")
    assert record["ReproSteps"] == "1. open 2. undo"
    assert record["VerifyComments"][0]["comment"] == "verified on 189063"
    assert record["SeverityLabel"] == "3 Normal"
    assert record["AttachmentCount"] == 2


def test_status_timeline_is_built_here_not_by_the_summarizer():
    """enrich_from_raw leaves StatusTimeline alone — n8n used to supply it."""
    timeline = ecl_direct.organize_record(raw_row())["StatusTimeline"]
    assert timeline == [
        {"status": "Assigned", "user": "WEI_LU", "time": "2026-09-02T09:00:00"},
        {"status": "RDResolved", "user": "WEI_LU", "time": "2026-09-03T09:00:00"},
    ]


def test_organize_record_survives_unparseable_json_fields():
    record = ecl_direct.organize_record(
        raw_row(StatusLog="not json", Comments=None, DetailData="[")
    )
    assert record["StatusTimeline"] == []
    assert record["Resolutions"] == []
    assert record["VerifyComments"] == []


def test_enriched_record_clears_the_quality_gate_with_real_evidence():
    """A flat-rename-only record passes the gate while carrying nothing.

    detail_quality_score credits 2 points for a large ``raw`` blob alone, so an
    unenriched record would be counted as recovered and contribute no root
    cause, fix or repro steps to the report.
    """
    raw = raw_row()
    flat = {organized: raw.get(source) for organized, source in ecl_direct.FIELD_MAP}
    flat["raw"] = raw
    assert has_usable_full_detail(flat)
    assert detail_quality_score(flat) == 2

    enriched = ecl_direct.organize_record(raw)
    assert has_usable_full_detail(enriched)
    assert detail_quality_score(enriched) > detail_quality_score(flat)


# -------------------------------------------------------------- degradation

def test_no_token_reports_unavailable_instead_of_raising(monkeypatch):
    monkeypatch.setattr(
        ecl_direct, "unavailable_reason", lambda: "No token. Get one from ...")
    records, errors = ecl_direct.fetch_bug_records(["DRI261890-0001"], PRODUCT)
    assert records == []
    assert len(errors) == 1
    assert "unavailable" in errors[0]
    assert "No token" in errors[0]


def test_unavailable_reason_collapses_the_multiline_client_message(monkeypatch):
    def boom():
        raise ECLError("No token.\n    echo '<token>' > ~/.ecl_token\n    or: export")

    monkeypatch.setattr(ecl_direct, "token", boom)
    reason = ecl_direct.unavailable_reason()
    assert "\n" not in reason
    assert "~/.ecl_token" in reason


def test_unavailable_reason_is_none_when_a_token_resolves(monkeypatch):
    monkeypatch.setattr(ecl_direct, "token", lambda: "a-token")
    assert ecl_direct.unavailable_reason() is None
    assert ecl_direct.available()


# ------------------------------------------------------------ batch walking

def test_fetch_returns_records_and_per_code_errors(monkeypatch, token_present):
    def fake_query(model, conditions, page_size=200, **kwargs):
        code = conditions[0]["term"]["query"]
        if code == "DRI261890-0002":
            return []
        return [raw_row(BugCode=code)]

    monkeypatch.setattr(ecl_direct, "query", fake_query)
    records, errors = ecl_direct.fetch_bug_records(
        ["DRI261890-0001", "DRI261890-0002", "DRI261890-0003"], PRODUCT, delay=0)

    assert [r["BugCode"] for r in records] == ["DRI261890-0001", "DRI261890-0003"]
    assert errors == ["DRI261890-0002: not found in IF3"]


def test_wrong_product_record_is_rejected(monkeypatch, token_present):
    monkeypatch.setattr(
        ecl_direct, "query",
        lambda *a, **k: [raw_row(ProductName="PowerDirector Mobile for Android")])
    records, errors = ecl_direct.fetch_bug_records(
        ["DRI261890-0001"], PRODUCT, delay=0)
    assert records == []
    assert "Android" in errors[0]


def test_one_bad_code_does_not_abort_the_batch(monkeypatch, token_present):
    def fake_query(model, conditions, page_size=200, **kwargs):
        code = conditions[0]["term"]["query"]
        if code == "DRI261890-0001":
            raise ECLError("POST /IF3/... → HTTP 500")
        return [raw_row(BugCode=code)]

    monkeypatch.setattr(ecl_direct, "query", fake_query)
    records, errors = ecl_direct.fetch_bug_records(
        ["DRI261890-0001", "DRI261890-0002"], PRODUCT, delay=0)
    assert [r["BugCode"] for r in records] == ["DRI261890-0002"]
    assert "HTTP 500" in errors[0]


# --------------------------------------------------------------- TR context

def tr_relation_rows(codes_and_statuses, verified=None):
    """Raw IF3 relation rows, as the transport returns them."""
    return [
        {"BugCode": code, "Status": status, "ShortDescription": f"{code} desc",
         **({"IsVerified": verified} if verified is not None else {})}
        for code, status in codes_and_statuses
    ]


def normalized_relation_rows(rows, *, omit_close_build=False):
    """Relation rows after _bug_relation — the shape the audit consumes.

    Each entry is (bug_code, status, close_build). A None close_build is
    meaningful: eCL leaves it empty until the fix lands in a build.
    """
    out = []
    for code, status, close_build in rows:
        row = {"bug_code": code, "status": status,
               "short_description": f"{code} desc", "is_verified": None,
               "bug_result": None, "verify_reason": None}
        if not omit_close_build:
            row["close_build"] = close_build
            row["close_build_known"] = True
        out.append(row)
    return out


@pytest.fixture
def tr_transport(monkeypatch, token_present):
    """Stub ecl_if3's query/call so TR reads never leave the process."""
    state = {"own": {}, "attached": {}, "details": {}, "tr_records": {}}

    def fake_query(model, conditions, page_size=200, **kwargs):
        if model in (ecl_direct.TR_DETAIL_MODEL, ecl_direct.TR_SEARCH_MODEL):
            code = conditions[0]["term"]["query"]
            state.setdefault("tr_models_tried", []).append(model)
            record = state["tr_records"].get(code)
            return [record] if record else []
        if model == ecl_direct.EBUG_DETAIL_MODEL:
            code = conditions[0]["term"]["query"]
            row = state["details"].get(code)
            return [row] if row else []
        raise AssertionError(f"unexpected query model {model}")

    def fake_call(method, path, payload=None, expect=(200,)):
        model = payload["queryModel"]
        tr_code = payload["conditions"][0]["term"]["query"]
        key = "own" if model == ecl_direct.TR_OWN_BUGS_MODEL else "attached"
        rows = state[key].get(tr_code, [])
        return 200, {"data": {"results": rows, "total": len(rows)}}

    monkeypatch.setattr(ecl_direct, "query", fake_query)
    monkeypatch.setattr(ecl_direct, "call", fake_call)
    return state


def test_fetch_tr_context_reads_both_relations(tr_transport):
    tr_transport["tr_records"]["TR260910-067"] = {
        "ShortDescription": "16.8.0 regression pass", "Status": "Testing",
        "FormState": "Processing", "ProductName": PRODUCT,
        "TRBuild": "189063", "Owner": "BALLY_HSU", "Tester": "VIVIAN_HUANG",
        "TRDueDate": "2026-09-20T00:00:00",
    }
    tr_transport["own"]["TR260910-067"] = tr_relation_rows(
        [("DRI261890-0002", "TRCreated"), ("DRI261890-0001", "Close")])
    tr_transport["attached"]["TR260910-067"] = tr_relation_rows(
        [("DRI261880-0009", "Close")], verified=False)

    contexts, errors = ecl_direct.fetch_tr_context(["TR260910-067"], delay=0)
    assert errors == []
    context = contexts[0]
    assert context["tr_build"] == "189063"
    assert context["tester"] == "VIVIAN_HUANG"
    # Sorted, so a report diff is not noise from result ordering.
    assert [b["bug_code"] for b in context["bugs_found_on_tr"]] == [
        "DRI261890-0001", "DRI261890-0002"]
    attached = context["bugs_attached_for_verify"]
    assert attached[0]["bug_code"] == "DRI261880-0009"
    assert attached[0]["is_verified"] is False


def test_fetch_tr_context_orders_and_deduplicates_tr_codes(tr_transport):
    contexts, _ = ecl_direct.fetch_tr_context(
        ["TR260910-067", "TR260119-013", "TR260910-067", "  "], delay=0)
    assert [c["tr_code"] for c in contexts] == ["TR260119-013", "TR260910-067"]


def test_fetch_tr_context_records_a_missing_tr_without_dropping_it(tr_transport):
    tr_transport["own"]["TR260910-067"] = tr_relation_rows(
        [("DRI261890-0002", "TRCreated")])
    contexts, errors = ecl_direct.fetch_tr_context(["TR260910-067"], delay=0)
    assert len(contexts) == 1
    assert contexts[0]["status"] is None
    assert contexts[0]["bugs_found_on_tr"][0]["bug_code"] == "DRI261890-0002"
    assert "not found" in errors[0]


def test_tr_record_falls_back_to_the_documented_search_model(tr_transport):
    """TR.SP.GetOldTRData is absent from the ECL guide's model list.

    This product's TRs are old-format (3-digit serial), so the fallback to the
    documented TSR.TRSearch is the likely path, not an edge case.
    """
    tr_transport["tr_records"]["TR260910-067"] = {
        "ShortDescription": "found via search", "Status": "Testing",
        "TRBuild": "189063", "Owner": "BALLY_HSU", "Tester": "VIVIAN_HUANG",
    }
    calls = []
    real_query = ecl_direct.query

    def flaky(model, conditions, page_size=200, **kwargs):
        calls.append(model)
        if model == ecl_direct.TR_DETAIL_MODEL:
            raise ECLError("POST /IF3/... -> HTTP 404 model not found")
        return real_query(model, conditions, page_size=page_size, **kwargs)

    ecl_direct.query = flaky
    try:
        contexts, errors = ecl_direct.fetch_tr_context(["TR260910-067"], delay=0)
    finally:
        ecl_direct.query = real_query

    assert calls == [ecl_direct.TR_DETAIL_MODEL, ecl_direct.TR_SEARCH_MODEL]
    assert errors == []
    assert contexts[0]["short_description"] == "found via search"
    assert contexts[0]["tester"] == "VIVIAN_HUANG"


def test_tr_record_does_not_burn_the_fallback_on_a_rate_limit(tr_transport):
    """429/401 is not a missing model; retrying only spends another call."""
    calls = []

    def limited(model, conditions, page_size=200, **kwargs):
        calls.append(model)
        raise ECLError("429 rate limited - back off")

    ecl_direct.query = limited
    try:
        contexts, errors = ecl_direct.fetch_tr_context(["TR260910-067"], delay=0)
    finally:
        del ecl_direct.query

    assert calls == [ecl_direct.TR_DETAIL_MODEL]
    assert contexts == []
    assert any("429" in error for error in errors)


def test_bug_relation_keeps_the_verification_fields(tr_transport):
    """IsVerified / BugResult / Reason are the only source of pass-vs-fail."""
    tr_transport["attached"]["TR260910-067"] = [
        {"BugCode": "DRI261880-0009", "Status": "Close",
         "ShortDescription": "d", "IsVerified": True, "BugResult": False,
         "Reason": "QA Verify with R8\r\nword limit [NG]"},
    ]
    contexts, _ = ecl_direct.fetch_tr_context(["TR260910-067"], delay=0)
    row = contexts[0]["bugs_attached_for_verify"][0]
    assert row["is_verified"] is True
    assert row["bug_result"] is False
    assert "[NG]" in row["verify_reason"]


def test_paged_walks_past_the_first_page(monkeypatch, token_present):
    pages = []

    def fake_call(method, path, payload=None, expect=(200,)):
        pages.append(payload["pageIndex"])
        index = payload["pageIndex"]
        rows = [{"BugCode": f"DRI2618{index:02d}-{n:04d}", "Status": "Close"}
                for n in range(ecl_direct.PAGE_SIZE)] if index == 1 else \
               [{"BugCode": "DRI261899-0001", "Status": "Close"}]
        return 200, {"data": {"results": rows, "total": ecl_direct.PAGE_SIZE + 1}}

    monkeypatch.setattr(ecl_direct, "call", fake_call)
    rows = ecl_direct._bug_relation(ecl_direct.TR_OWN_BUGS_MODEL, "TR260910-067")
    assert pages == [1, 2]
    assert len(rows) == ecl_direct.PAGE_SIZE + 1


# ------------------------------------------------------------- cohort audit

BUILD = "16.8.0.189063"


def test_audit_flags_bugs_the_status_filter_silently_sheds(tr_transport):
    """The real defect, reproduced from live data (2026-09-16).

    select_newest_tr_cohort keeps only Status == TRCreated, but a bug moves
    TRCreated -> Close as QA verifies it. Build 16.8.0.189063 had 26 bugs
    closed into it and the report covered 20; every one of the six lost was
    Close. Auditing on Close Build# — which does not drift — catches them.
    """
    contexts = [{
        "tr_code": "TR260826-089",
        "bugs_found_on_tr": normalized_relation_rows([
            ("DRI261883-0049", "TRCreated", BUILD),   # already in the cohort
            ("DRI261883-0064", "Close", BUILD),       # shed by the status filter
            ("DRI261883-0065", "Close", BUILD),
        ]),
    }]
    audit = ecl_direct.audit_cohort_completeness(
        contexts, ["DRI261883-0049"], BUILD, PRODUCT, delay=0)

    assert [i["bug_code"] for i in audit["missing_from_cohort"]] == [
        "DRI261883-0064", "DRI261883-0065"]
    assert audit["detail_checks_spent"] == 0


def test_audit_costs_no_detail_calls_when_the_relation_carries_the_build(
        tr_transport):
    """Both relations return CloseToBuild, so classification is free."""
    contexts = [{
        "tr_code": "TR260826-089",
        "bugs_found_on_tr": normalized_relation_rows([
            ("DRI261889-0001", "Assigned", None),          # not in any build
            ("DRI261889-0003", "Close", BUILD),            # this build
            ("DRI261889-0009", "Close", "16.7.5.188405"),  # another build
            ("DRI261889-0005", "Propose Won't Fix", None),
        ]),
    }]
    audit = ecl_direct.audit_cohort_completeness(contexts, [], BUILD, PRODUCT,
                                                 delay=0)
    assert audit["detail_checks_spent"] == 0
    assert [i["bug_code"] for i in audit["missing_from_cohort"]] == [
        "DRI261889-0003"]
    reasons = {i["bug_code"]: i["reason"] for i in audit["excluded"]}
    assert reasons["DRI261889-0001"] == "not fixed into any build"
    assert reasons["DRI261889-0005"] == "not fixed into any build"
    assert reasons["DRI261889-0009"] == "closed to a different build"


def test_a_null_close_build_is_not_treated_as_unknown(tr_transport):
    """None means "not fixed into any build" — spending a call on it is waste.

    Conflating the two burned 60 detail calls against the live pool.
    """
    contexts = [{"tr_code": "TR260826-089",
                 "bugs_found_on_tr": normalized_relation_rows(
                     [("DRI261889-0001", "Assigned", None)])}]
    audit = ecl_direct.audit_cohort_completeness(contexts, [], BUILD, PRODUCT,
                                                 delay=0)
    assert audit["detail_checks_spent"] == 0
    assert audit["undetermined"] == []
    assert audit["excluded"][0]["reason"] == "not fixed into any build"


def test_a_row_without_the_field_at_all_still_gets_one_detail_call(tr_transport):
    """Absent field — not a null one — is the only case worth an API call."""
    tr_transport["details"]["DRI261889-0011"] = raw_row(
        BugCode="DRI261889-0011", CloseToBuild=BUILD)
    contexts = [{"tr_code": "TR260826-089",
                 "bugs_found_on_tr": normalized_relation_rows(
                     [("DRI261889-0011", "Close", None)], omit_close_build=True)}]
    audit = ecl_direct.audit_cohort_completeness(contexts, [], BUILD, PRODUCT,
                                                 delay=0)
    assert audit["detail_checks_spent"] == 1
    assert [i["bug_code"] for i in audit["missing_from_cohort"]] == [
        "DRI261889-0011"]


def test_attached_bugs_are_audited_too(tr_transport):
    """An attached bug closed to this build belongs in the cohort as well."""
    contexts = [{
        "tr_code": "TR260910-067",
        "bugs_found_on_tr": [],
        "bugs_attached_for_verify": normalized_relation_rows([
            ("DRI261880-0007", "Close", BUILD),
            ("DRI261880-0009", "Close", "16.7.5.188405"),
        ]),
    }]
    audit = ecl_direct.audit_cohort_completeness(contexts, [], BUILD, PRODUCT,
                                                 delay=0)
    assert [i["bug_code"] for i in audit["missing_from_cohort"]] == [
        "DRI261880-0007"]


def test_audit_never_mutates_the_cohort_it_was_given(tr_transport):
    cohort = ["DRI261883-0049"]
    contexts = [{"tr_code": "TR260826-089",
                 "bugs_found_on_tr": normalized_relation_rows(
                     [("DRI261883-0064", "Close", BUILD)])}]
    ecl_direct.audit_cohort_completeness(contexts, cohort, BUILD, PRODUCT,
                                         delay=0)
    assert cohort == ["DRI261883-0049"]


def test_audit_reports_its_own_limit_instead_of_silently_truncating(tr_transport):
    rows = [(f"DRI261890-{n:04d}", "Close", None) for n in range(5)]
    for code, _, _ in rows:
        tr_transport["details"][code] = raw_row(BugCode=code, CloseToBuild=BUILD)
    contexts = [{"tr_code": "TR260910-067",
                 "bugs_found_on_tr": normalized_relation_rows(
                     rows, omit_close_build=True)}]

    audit = ecl_direct.audit_cohort_completeness(
        contexts, [], BUILD, PRODUCT, max_detail_checks=2, delay=0)

    assert audit["detail_check_limit_reached"] is True
    assert audit["detail_checks_spent"] == 2
    assert len(audit["missing_from_cohort"]) == 2
    assert [i["reason"] for i in audit["undetermined"]] == [
        "not checked (limit reached)"] * 3


def test_audit_without_a_token_reports_rather_than_raising(monkeypatch):
    monkeypatch.setattr(ecl_direct, "unavailable_reason", lambda: "No token.")
    contexts = [{"tr_code": "TR260910-067",
                 "bugs_found_on_tr": normalized_relation_rows(
                     [("DRI261890-0009", "Close", None)], omit_close_build=True)}]
    audit = ecl_direct.audit_cohort_completeness(contexts, [], BUILD, PRODUCT,
                                                 delay=0)
    assert audit["missing_from_cohort"] == []
    assert audit["undetermined"][0]["reason"] == "detail lookup failed"
    assert "unavailable" in audit["errors"][0]


@pytest.mark.parametrize("message", ["429 rate limited — back off",
                                     "401 Request Unauthorized"])
def test_rate_limit_or_auth_failure_stops_the_batch(monkeypatch, token_present,
                                                    message):
    """Every Query endpoint shares one bucket; spending it helps nobody."""
    calls = []

    def fake_query(model, conditions, page_size=200, **kwargs):
        calls.append(conditions[0]["term"]["query"])
        raise ECLError(message)

    monkeypatch.setattr(ecl_direct, "query", fake_query)
    records, errors = ecl_direct.fetch_bug_records(
        ["DRI261890-0001", "DRI261890-0002", "DRI261890-0003"], PRODUCT, delay=0)

    assert records == []
    assert calls == ["DRI261890-0001"]
    assert any("2 remaining code(s)" in error for error in errors)


# ----------------------------------------------------- token file resolution

@pytest.fixture
def clean_token_env(monkeypatch, tmp_path):
    """No ambient token, and a repo-local file that cannot touch the real one."""
    monkeypatch.delenv("ECL_TOKEN", raising=False)
    monkeypatch.delenv("ECL_TOKEN_FILE", raising=False)
    monkeypatch.setattr(ecl_direct, "REPO_TOKEN_FILE", tmp_path / ".ecl_token")
    return tmp_path / ".ecl_token"


def test_repo_token_file_is_read_and_exported(clean_token_env, monkeypatch):
    clean_token_env.write_text("# a comment\n\nreal-token-value\n")
    clean_token_env.chmod(0o600)
    status = ecl_direct.token_status()
    assert status["available"] is True
    assert status["warnings"] == []
    # ecl_if3.call() reads the env var, so the file has to land there.
    assert os.environ["ECL_TOKEN"] == "real-token-value"


def test_unedited_example_fails_loudly_instead_of_401ing_later(clean_token_env):
    clean_token_env.write_text(
        f"# instructions\n{ecl_direct.TOKEN_PLACEHOLDER}\n")
    status = ecl_direct.token_status()
    assert status["available"] is False
    assert "placeholder" in status["reason"]
    assert ecl_direct.TOKEN_PAGE in status["reason"]
    assert "ECL_TOKEN" not in os.environ


def test_world_readable_token_warns_but_still_works(clean_token_env):
    clean_token_env.write_text("real-token-value\n")
    clean_token_env.chmod(0o644)
    status = ecl_direct.token_status()
    assert status["available"] is True
    assert "chmod 600" in status["warnings"][0]


def test_explicit_env_token_wins_over_the_repo_file(clean_token_env, monkeypatch):
    clean_token_env.write_text("file-token\n")
    clean_token_env.chmod(0o600)
    monkeypatch.setenv("ECL_TOKEN", "env-token")
    status = ecl_direct.token_status()
    assert status["available"] is True
    assert os.environ["ECL_TOKEN"] == "env-token"


def test_no_token_anywhere_reports_the_token_page(clean_token_env, monkeypatch):
    monkeypatch.setattr(ecl_direct, "token", _raise_no_token)
    status = ecl_direct.token_status()
    assert status["available"] is False
    assert status["source"] is None
    assert "tokenPage" in status["reason"]


def _raise_no_token():
    raise ECLError(
        f"No token. Get one from {ecl_direct.TOKEN_PAGE}\n    echo ... ")


def test_read_token_file_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "t"
    path.write_text("\n# lead comment\n\n  the-token  \n# trailing\n")
    assert ecl_direct.read_token_file(path) == "the-token"


def test_read_token_file_returns_none_for_a_comment_only_file(tmp_path):
    path = tmp_path / "t"
    path.write_text("# nothing but comments\n\n")
    assert ecl_direct.read_token_file(path) is None


def test_read_token_file_tolerates_a_missing_file(tmp_path):
    assert ecl_direct.read_token_file(tmp_path / "absent") is None


def test_tracked_example_file_matches_the_placeholder_constant():
    """The example must stay in sync or the placeholder guard goes silent."""
    example = Path(".ecl_token.example")
    if not example.exists():
        pytest.skip("example token file not present")
    assert ecl_direct.read_token_file(example) == ecl_direct.TOKEN_PLACEHOLDER


def _tr_row(code, build, product=PRODUCT, form_state="Normal"):
    return {"TRCode": code, "TRBuild": f"{product}: {build}", "ProductName": product,
            "FormState": form_state, "Status": "Done", "Result": "Fail",
            "ShortDescription": f"{build} (Deluxe)"}


def test_find_build_trs_keeps_only_an_exact_build_and_product(monkeypatch, token_present):
    """TRBuild ``match`` is a substring match; the filter has to be exact."""
    seen = []

    def fake_query(model, conditions, page_size=200, **kwargs):
        seen.append((model, conditions))
        return [
            _tr_row("TR260922-0013", "16.8.0.189304"),
            _tr_row("TR260923-0001", "16.8.0.1893041"),
            _tr_row("TR260922-0020", "16.8.0.189304",
                    product="PowerDirector Mobile for Android"),
            _tr_row("TR260921-0005", "16.8.0.189304", form_state="Cancel"),
        ]

    monkeypatch.setattr(ecl_direct, "query", fake_query)
    found, errors = ecl_direct.find_build_trs(
        "16.8.0.189304", PRODUCT, with_bugs=False)
    assert errors == []
    assert [tr["tr_code"] for tr in found] == ["TR260921-0005", "TR260922-0013"]
    assert found[0]["form_state"] == "Cancel"
    assert seen[0][0] == ecl_direct.TR_SEARCH_MODEL and len(seen) == 1


def test_find_build_trs_reads_the_live_trs_bug_lists(monkeypatch, token_present):
    """Created-from is the queue; a cancelled TR is not read at all."""
    monkeypatch.setattr(ecl_direct, "query", lambda *a, **k: [
        _tr_row("TR260922-0013", "16.8.0.189304"),
        _tr_row("TR260921-0005", "16.8.0.189304", form_state="Cancel"),
    ])
    reads = []

    def fake_relation(model, tr_code):
        reads.append((model, tr_code))
        return [{"bug_code": f"{model}-bug"}]

    monkeypatch.setattr(ecl_direct, "_bug_relation", fake_relation)
    found, errors = ecl_direct.find_build_trs("16.8.0.189304", PRODUCT, delay=0)
    assert errors == []
    by_code = {tr["tr_code"]: tr for tr in found}
    live, cancelled = by_code["TR260922-0013"], by_code["TR260921-0005"]
    assert live["bugs_loaded"] is True
    assert live["bugs_attached_for_verify"] == [
        {"bug_code": f"{ecl_direct.TR_ATTACHED_BUGS_MODEL}-bug"}]
    assert live["bugs_found_on_tr"] == [
        {"bug_code": f"{ecl_direct.TR_OWN_BUGS_MODEL}-bug"}]
    assert "bugs_loaded" not in cancelled
    assert {tr for _, tr in reads} == {"TR260922-0013"}


def test_find_build_trs_keeps_the_tr_when_its_bug_lists_fail(monkeypatch, token_present):
    monkeypatch.setattr(ecl_direct, "query", lambda *a, **k: [
        _tr_row("TR260922-0013", "16.8.0.189304")])

    def boom(model, tr_code):
        raise ECLError("HTTP 500")

    monkeypatch.setattr(ecl_direct, "_bug_relation", boom)
    found, errors = ecl_direct.find_build_trs("16.8.0.189304", PRODUCT, delay=0)
    assert [tr["tr_code"] for tr in found] == ["TR260922-0013"]
    assert "bugs_loaded" not in found[0]
    assert "HTTP 500" in errors[0]


def test_find_build_trs_degrades_without_a_token_or_a_build(monkeypatch):
    monkeypatch.setattr(ecl_direct, "unavailable_reason", lambda: "no token")
    assert ecl_direct.find_build_trs("16.8.0.189304", PRODUCT)[0] == []
    assert ecl_direct.find_build_trs("", PRODUCT) == ([], [])


def test_find_build_trs_reports_a_query_error(monkeypatch, token_present):
    def boom(*args, **kwargs):
        raise ECLError("HTTP 500")

    monkeypatch.setattr(ecl_direct, "query", boom)
    found, errors = ecl_direct.find_build_trs("16.8.0.189304", PRODUCT)
    assert found == [] and "HTTP 500" in errors[0]
