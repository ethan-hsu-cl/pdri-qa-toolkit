#!/usr/bin/env python3
"""IF3 REST client for ECL — the five action endpoints plus the queries they need.

Implements chapter 3-3 of ECL_MCP_RD_QA_Guide: Create eBug, Assign, Resolve,
Code Review, Build Resolve. Stdlib only.

This is NOT the MCP connector. The MCP is read-only (no BPMs on tsr); these are
plain HTTPS calls that need a bearer token from
https://ecl.cyberlink.com/sso/mgm/tokenPage

    export ECL_TOKEN="$(cat ~/.ecl_token)"     # never pass a token on argv

Commands:
    preflight   --product PowerDirector --template-name "PowerDirector iOS 16.8"
    create      bodies.json --tr TR260915-0007 --template-id 23156 \
                --creator ETHAN_HSU --version 16.8.0 --build 189063 [--dry-run]
    assign      DRI261890P0004 --assignees WEI_LU --belong RD --priority 4
    resolve     DRI261890P0004 --reviewer SEAN_KAO --root-cause "..." \
                --solution "..." --function "..." --changeset "..."
    codereview  DRI261890P0004
    buildresolve DRI261890P0004 --close-to-build 189100
    readback    DRI261890P0004 [DRI261890P0005 ...]

Create is guarded three ways, because the API has no idempotency of its own:
a local ledger, a live duplicate search, and a mandatory read-back.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("ECL_BASE", "https://ecl.cyberlink.com")

Q_PREFIX = "/IF3/tsr/api/public/Query"
TPL_PREFIX = "/IF3/templatesrv/api/public/Query"
CREATE_PATH = "/api/TSR/Ebug/Create"
ACTION_PREFIX = "/api/TSRPublic/ebug"

# The web form shows the left column; the API takes the right one. A decisions
# table signed off against the review page uses the web labels.
BELONG_WEB_TO_API = {
    "RD": "RD", "Setup Kit": "RDSK", "PC-Art": "PCArt", "PC-Doc": "PCDoc",
    "PM": "PM", "Marketing": "Marketing", "RD-ECL": "RD-ECL",
}
BELONG_API = set(BELONG_WEB_TO_API.values())

# ECL ProductCode (the BugCode prefix) -> the product token the Short Description
# format uses. These are NOT the same string and are easy to conflate.
CODE_TO_SHORT_PRODUCT = {
    "DRI": "PDR-I", "DRA": "PDR-A", "VDE": "PDR", "PDP": "PHD",
}


class ECLError(RuntimeError):
    pass


TOKEN_FILE = os.path.expanduser(os.environ.get("ECL_TOKEN_FILE", "~/.ecl_token"))


def token():
    """ECL_TOKEN, else ~/.ecl_token. Never accepted on argv, where it leaks to ps."""
    t = os.environ.get("ECL_TOKEN", "").strip()
    if not t and os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            t = f.read().strip()
    if not t:
        raise ECLError(
            f"No token. Get one from {BASE}/sso/mgm/tokenPage, then either\n"
            f"    echo '<token>' > {TOKEN_FILE} && chmod 600 {TOKEN_FILE}\n"
            '    or: export ECL_TOKEN="<token>"')
    return t


def call(method, path, payload=None, expect=(200,)):
    """One HTTPS call. Returns (status, parsed-body-or-text)."""
    url = BASE + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": "Bearer " + token(),
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            status, raw = r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        raise ECLError(f"{method} {path} — connection failed: {e.reason}")

    body = raw
    if raw.strip():
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            pass

    if status == 401:
        raise ECLError("401 Request Unauthorized — the token is invalid or expired.")
    if status == 429:
        raise ECLError("429 rate limited — all query endpoints share one bucket; back off.")
    if status not in expect:
        raise ECLError(f"{method} {path} → HTTP {status}\n{_fmt_err(body)}")

    # A failed query can still be HTTP 200 — check the body (guide 3-1).
    if isinstance(body, dict) and str(body.get("status", "")).lower() == "fail":
        raise ECLError(f"{method} {path} → 200 but status=fail: {body.get('message')}")
    if isinstance(body, dict) and body.get("Status") == "Error":
        raise ECLError(f"{method} {path} → {body.get('Message')}")
    return status, body


def _fmt_err(body):
    if isinstance(body, dict):
        if "errors" in body:
            return "\n".join(f"  {k}: {'; '.join(v)}"
                             for k, v in body["errors"].items())
        for k in ("Message", "message"):
            if body.get(k):
                tid = body.get("traceId")
                return f"  {body[k]}" + (f"  (traceId {tid})" if tid else "")
    return "  " + str(body)[:600]


def query(model, conditions, page_size=200, sort_by=None, desc=False, template=False):
    prefix = TPL_PREFIX if template else Q_PREFIX
    payload = {"queryModel": model, "pageIndex": 1, "pageSize": page_size,
               "conditions": conditions}
    if sort_by:
        payload["sortBy"] = sort_by
        payload["isDesc"] = desc
    _, body = call("POST", f"{prefix}/{model}", payload)
    if isinstance(body, dict):
        return body.get("results") or body.get("data", {}).get("results") or []
    return []


# ---------------------------------------------------------------- preflight

def cmd_preflight(a):
    conds = [{"field": "TemplateType", "term": {"query": "Ebug"}},
             {"field": "IsIF3Project", "term": {"query": "true"}}]
    if a.product:
        conds.append({"field": "ProductName", "match": {"query": a.product}})
    if a.template_name:
        conds.append({"field": "TemplateName", "match": {"query": a.template_name}})

    rows = query("TSR.ProjectTemplate", conds, page_size=100, sort_by="TemplateName")
    if not rows:
        print("No IF3 eBug template matched.\n"
              "That pool has NOT migrated — Create eBug will not work on it.\n"
              "File through the ECL web form instead (see SKILL.md → The form).")
        return 1

    print(f"{len(rows)} IF3 eBug template(s) — these are the only pools Create can open on:\n")
    for r in rows:
        print(f"  TemplateID   {r.get('TemplateID')}   (IF3TemplateID {r.get('IF3TemplateID')})")
        print(f"  TemplateName {r.get('TemplateName')}")
        print(f"  ProductID    {r.get('ProductID')}")
        print(f"  ProductName  {r.get('ProductName')}")
        print(f"  ProductCode  {r.get('ProductCode')}   <- BugCode prefix")
        print()

    if len(rows) == 1 and rows[0].get("IF3TemplateID"):
        _print_severity_map(rows[0]["IF3TemplateID"])
    return 0


def _print_severity_map(if3_template_id):
    """Severity/Priority are integers whose meaning is per-template."""
    try:
        rows = query("TemplateService.TemplateData",
                     [{"field": "IF3TemplateID",
                       "term": {"query": str(if3_template_id)}}],
                     page_size=100, template=True)
    except ECLError as e:
        print(f"(could not read template field definitions: {e})")
        return
    for r in rows:
        name = str(r.get("FieldName") or r.get("Name") or "")
        if re.search(r"severity|priority", name, re.I):
            print(f"  {name}: {json.dumps(r.get('Options') or r, ensure_ascii=False)[:400]}")


# ------------------------------------------------------------------- create

def _ledger_path(bodies_path):
    return os.path.join(os.path.dirname(os.path.abspath(bodies_path)),
                        "filed_formcodes.json")


def _load_ledger(p):
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def _save_ledger(p, led):
    with open(p, "w") as f:
        json.dump(led, f, indent=2, ensure_ascii=False)


def _find_duplicate(short):
    """Live search before every create. The API has no idempotency of its own."""
    rows = query("TSR.EbugSearch",
                 [{"field": "ShortDescription", "term": {"query": short}}],
                 page_size=5)
    return [r for r in rows if r.get("ShortDescription", "").strip() == short.strip()]


def cmd_create(a):
    with open(a.bodies) as f:
        bodies = json.load(f)

    tpl = query("TSR.ProjectTemplate",
                [{"field": "TemplateType", "term": {"query": "Ebug"}},
                 {"field": "IsIF3Project", "term": {"query": "true"}},
                 {"field": "TemplateID", "term": {"query": str(a.template_id)}}],
                page_size=5)
    if not tpl:
        raise ECLError(
            f"TemplateID {a.template_id} is not an IF3 eBug template.\n"
            "Create eBug only works on IF3 pools. A wrong TemplateID is NOT rejected "
            "by the server — it opens a broken ticket. Run `preflight` first.")
    t = tpl[0]

    # The format check is a GATE, not a separate step someone can forget. Filing is
    # irreversible and emails three people per ticket; a malformed Short Description
    # then has to be corrected one bug at a time through the web edit form.
    if not a.skip_format_check:
        short_product = a.short_product or CODE_TO_SHORT_PRODUCT.get(t["ProductCode"])
        if not short_product:
            raise ECLError(
                f"No Short Description product token known for ProductCode "
                f"{t['ProductCode']!r}. Pass --short-product (e.g. PDR-I).")
        checker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "check_ebug_format.py")
        cmd = [sys.executable, checker, a.bodies, "--product", short_product,
               "--version", a.version,
               "--upload-dir", os.path.dirname(os.path.abspath(a.bodies))]
        print(f"Format check: {short_product} {a.version}", flush=True)
        if subprocess.call(cmd) != 0:
            raise ECLError(
                "format check FAILED — nothing was sent.\n"
                "Fix the errors above and re-run. If the convention has genuinely moved "
                "on, fix check_ebug_format.py rather than the ticket; --skip-format-check "
                "exists but every batch that used it shipped malformed.")
        print()

    ledger_path = _ledger_path(a.bodies)
    ledger = _load_ledger(ledger_path)
    filed, skipped, failed = [], [], []

    for key in sorted(bodies):
        e = bodies[key]
        short = e["short"].strip()

        if key in ledger:
            print(f"[{key}] already filed as {ledger[key]['FormCode']} — skipping")
            skipped.append(key)
            continue

        belong = BELONG_WEB_TO_API.get(e["belong"], e["belong"])
        if belong not in BELONG_API:
            failed.append((key, f"BugBelong {e['belong']!r} is not valid for the API"))
            continue

        payload = {
            "Creator": a.creator,
            "ProductID": t["ProductID"],
            "ProductName": t["ProductName"],
            "ProductCode": t["ProductCode"],
            "TemplateID": t["TemplateID"],
            "TemplateName": t["TemplateName"],
            "Version": a.version,
            "BuildNo": str(a.build),
            # Value-typed fields are sent explicitly, every time: omitted ones are
            # not always caught by validation and can land as false / 0 / 0001-01-01.
            "IsOEM": False,
            "BugSource": a.bug_source,
            "ParentTRCode": a.tr,
            "ParentSRCode": "",
            "BugBelong": belong,
            "ReproduceOut": int(e["out"]),
            "ReproduceTimes": int(e["times"]),
            "IsHappened": e.get("happened", "Unknown"),
            "IssueType": "",
            "CustomerIssueID": "",
            "DueDate": a.due_date,
            "ShortDescription": short,
            "IssueReproducer": "QA",
            "Priority": int(e.get("priority", 5)),
            "Severity": int(e["sev"]),
            "DescribeIssue": e["desc"],
        }

        if a.dry_run:
            print(f"\n--- [{key}] would POST {CREATE_PATH} ---")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            continue

        dupes = _find_duplicate(short)
        if dupes and not a.allow_duplicate:
            print(f"[{key}] SKIPPED — identical Short Description already filed as "
                  f"{', '.join(d['BugCode'] for d in dupes)}")
            skipped.append(key)
            continue

        try:
            status, body = call("POST", CREATE_PATH, payload, expect=(200,))
        except ECLError as err:
            # Never blind-retry a create: a timed-out create may have succeeded.
            print(f"[{key}] FAILED — {err}")
            failed.append((key, str(err)))
            continue

        code = body.get("FormCode") if isinstance(body, dict) else None
        if not code:
            failed.append((key, f"no FormCode in response: {body}"))
            continue

        ledger[key] = {"FormCode": code, "Href": body.get("Href"),
                       "short": short, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _save_ledger(ledger_path, ledger)
        print(f"[{key}] created {code}")
        filed.append((key, code))
        time.sleep(a.delay)

    if a.dry_run:
        print(f"\nDry run — {len(bodies)} bug(s) would be created. Nothing was sent.")
        return 0

    print(f"\ncreated {len(filed)} · skipped {len(skipped)} · failed {len(failed)}")
    for key, why in failed:
        print(f"  FAILED {key}: {why}")
    if filed:
        print("\nRead-back (Status: OK is not proof — Version has been seen not to persist):")
        bad = _readback([c for _, c in filed], expect={
            "Version": a.version, "Build": str(a.build), "ParentTRCode": a.tr})
        if bad:
            print("\n*** Some bugs did not read back correctly. Fix them by hand "
                  "through the web edit form. ***")
            return 1
    print("\nAttachments are NOT part of create — attach screenshots through the "
          "web edit page (SKILL.md → Attaching screenshots).")
    return 1 if failed else 0


# ------------------------------------------------------------------ actions

def cmd_assign(a):
    # All three fields are required, and Assign OVERWRITES BugBelong and Priority.
    belong = BELONG_WEB_TO_API.get(a.belong, a.belong)
    if belong not in BELONG_API:
        raise ECLError(f"BugBelong {a.belong!r} is not valid; use {sorted(BELONG_API)}")
    if not 1 <= a.priority <= 5:
        raise ECLError("Priority must be an integer between 1 and 5.")
    payload = {"Assignees": [s.strip() for s in a.assignees.split(",") if s.strip()],
               "BugBelong": belong, "Priority": a.priority}
    if not payload["Assignees"]:
        raise ECLError("Assignees cannot be empty.")
    print(f"Assign overwrites BugBelong -> {belong} and Priority -> {a.priority} "
          f"on {a.bugcode}.")
    call("PUT", f"{ACTION_PREFIX}/{a.bugcode}/action/assign", payload, expect=(204,))
    print(f"{a.bugcode} assigned to {', '.join(payload['Assignees'])}")
    return 0


def cmd_resolve(a):
    if a.reviewer and a.reviewer == a.resolver:
        raise ECLError("Resolver could not be Code Reviewer.")
    payload = {
        "ResolveComment": {"RootCause": a.root_cause, "Solution": a.solution,
                           "Function": a.function, "Changeset": a.changeset},
        "CodeReviewer": a.reviewer or "",
        "NextRunHandler": a.next_handler or "",
        "ModuleOwner": a.module_owner or "NoNeed",
        "ModuleOwnerID": a.module_owner_id,
    }
    call("PUT", f"{ACTION_PREFIX}/{a.bugcode}/action/resolve", payload, expect=(204,))
    print(f"{a.bugcode} resolved; Handler now {a.reviewer or '(unchanged)'}")
    return 0


def cmd_codereview(a):
    # No body. Caller must be the current Handler (the reviewer named at Resolve) —
    # permission follows the ticket's current stage, not the token's role.
    call("PUT", f"{ACTION_PREFIX}/{a.bugcode}/action/codeReview", None, expect=(204,))
    print(f"{a.bugcode} code review passed")
    return 0


def cmd_buildresolve(a):
    if not str(a.close_to_build).strip():
        raise ECLError("Build number cannot be empty.")
    call("PUT", f"{ACTION_PREFIX}/{a.bugcode}/action/buildResolve",
         {"CloseToBuildNo": str(a.close_to_build)}, expect=(204,))
    print(f"{a.bugcode} build-resolved into {a.close_to_build}")
    return 0


# ----------------------------------------------------------------- readback

def _readback(codes, expect=None):
    rows = query("TSR.EbugSearch",
                 [{"field": "BugCode", "term": {"querys": list(codes)}}],
                 page_size=max(len(codes), 10))
    got = {r["BugCode"]: r for r in rows}
    bad = []
    for c in codes:
        r = got.get(c)
        if not r:
            print(f"  {c}  *** NOT FOUND on the server ***")
            bad.append(c)
            continue
        problems = []
        for field, want in (expect or {}).items():
            have = r.get(field)
            if str(have or "") != str(want):
                problems.append(f"{field}={have!r} (expected {want!r})")
        flag = "  <-- " + "; ".join(problems) if problems else ""
        print(f"  {c}  sev {r.get('EbugSeverity')}  {r.get('BugBelong')}  "
              f"{r.get('Status')}  v{r.get('Version')}/{r.get('Build')}{flag}")
        if problems:
            bad.append(c)
    return bad


def cmd_readback(a):
    return 1 if _readback(a.bugcodes) else 0


# --------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("preflight", help="find IF3 eBug templates you can file on")
    s.add_argument("--product")
    s.add_argument("--template-name")
    s.set_defaults(fn=cmd_preflight)

    s = sub.add_parser("create", help="create eBugs from a bodies.json")
    s.add_argument("bodies")
    s.add_argument("--tr", required=True)
    s.add_argument("--template-id", required=True, type=int)
    s.add_argument("--creator", required=True)
    s.add_argument("--version", required=True)
    s.add_argument("--build", required=True)
    s.add_argument("--due-date", required=True, help="ISO date, e.g. 2026-10-15")
    s.add_argument("--bug-source", default="QA")
    s.add_argument("--delay", type=float, default=1.0)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--allow-duplicate", action="store_true",
                   help="file even when an identical Short Description exists")
    s.add_argument("--short-product",
                   help="product token for the Short Description (PDR-I / PDR-A / PHD); "
                        "derived from the template's ProductCode when omitted")
    s.add_argument("--skip-format-check", action="store_true",
                   help="bypass check_ebug_format.py (every batch that did shipped malformed)")
    s.set_defaults(fn=cmd_create)

    s = sub.add_parser("assign")
    s.add_argument("bugcode")
    s.add_argument("--assignees", required=True, help="comma-separated ECL accounts")
    s.add_argument("--belong", required=True)
    s.add_argument("--priority", required=True, type=int)
    s.set_defaults(fn=cmd_assign)

    s = sub.add_parser("resolve")
    s.add_argument("bugcode")
    s.add_argument("--root-cause", required=True)
    s.add_argument("--solution", required=True)
    s.add_argument("--function", default="")
    s.add_argument("--changeset", default="")
    s.add_argument("--reviewer", default="")
    s.add_argument("--resolver", default="", help="your account, to catch self-review")
    s.add_argument("--next-handler", default="")
    s.add_argument("--module-owner", default="NoNeed")
    s.add_argument("--module-owner-id", type=int, default=0)
    s.set_defaults(fn=cmd_resolve)

    s = sub.add_parser("codereview")
    s.add_argument("bugcode")
    s.set_defaults(fn=cmd_codereview)

    s = sub.add_parser("buildresolve")
    s.add_argument("bugcode")
    s.add_argument("--close-to-build", required=True)
    s.set_defaults(fn=cmd_buildresolve)

    s = sub.add_parser("readback")
    s.add_argument("bugcodes", nargs="+")
    s.set_defaults(fn=cmd_readback)

    a = p.parse_args()
    try:
        return a.fn(a)
    except ECLError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
