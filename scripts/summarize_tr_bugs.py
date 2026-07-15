#!/usr/bin/env python3
"""summarize_tr_bugs.py — AI summary of a TR's fixed bugs: what to test + risk areas.

Reads the bug_details.json produced by fetch_tr_bug_details.py, feeds each
fixed bug's description + status-log text to a local Ollama model, and writes
a markdown test-focus report next to the input file:

  1. Per-bug verification — what to check to confirm each fix
  2. Recommended regression areas — other sections/modules worth re-testing

Usage:
    python scripts/summarize_tr_bugs.py data/products/pdri/tr_reports/<label>/bug_details.json
    python scripts/summarize_tr_bugs.py <bug_details.json> --model gemma4
    python scripts/summarize_tr_bugs.py <bug_details.json> --dry-run   # print prompt only
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

DEFAULT_MODEL = "gemma4"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Raw-detail keys that are noise, never status-log content
SKIP_LOG_KEYS = {"_detected_log_fields", "_status_changed", "_prev_status"}


def stringify(value, max_chars: int) -> str:
    """Render a raw field (string / list / dict) as readable text, truncated."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=None, default=str)
        except (TypeError, ValueError):
            text = str(value)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + " …[truncated]"
    return text


def _parse_json_field(v):
    """eBug stores StatusLog / Comments / DetailData as JSON-encoded strings."""
    if v is None:
        return None
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return None


def enrich_from_raw(bug: dict) -> dict:
    """Derive the structured fields (Resolutions / ReproSteps / VerifyComments /
    ParentTRCode / ModuleOwner) directly from the raw eBug record.

    This mirrors the workflow's Organize node so the AI step works even when the
    deployed n8n workflow is an older version that only saved `raw` +
    `_detected_log_fields`. If the structured fields are already present
    (newer workflow), the bug is returned unchanged.
    """
    if bug.get("Resolutions") is not None or bug.get("ReproSteps") is not None:
        return bug
    raw = bug.get("raw") or {}
    if not raw:
        return bug

    status_log = _parse_json_field(raw.get("StatusLog")) or []
    comments = _parse_json_field(raw.get("Comments")) or []
    detail = _parse_json_field(raw.get("DetailData")) or []
    uploads = _parse_json_field(raw.get("UploadFile")) or []

    repro = os_lang = prio_label = sev_label = None
    for f in detail if isinstance(detail, list) else []:
        key = str(f.get("FieldCode") or f.get("FieldName") or "").lower()
        if "describe" in key:
            repro = f.get("FieldValue")
        elif "os language" in key:
            os_lang = f.get("FieldValue")
        elif key == "priority":
            prio_label = f.get("FieldValue")
        elif key == "severity":
            sev_label = f.get("FieldValue")

    resolutions = [
        {
            "user": s.get("UserName"),
            "time": s.get("StatusTime"),
            "root_cause": s.get("RootCause"),
            "solution": s.get("Solution"),
            "risk": s.get("Risk"),
            "trace_url": s.get("TraceURL"),
        }
        for s in (status_log if isinstance(status_log, list) else [])
        if s and (s.get("RootCause") or s.get("Solution"))
    ]
    verify = [
        {"user": c.get("UserName"), "time": c.get("CommentTime"), "comment": c.get("Comment")}
        for c in (comments if isinstance(comments, list) else [])
    ]

    enriched = dict(bug)
    enriched.setdefault("ParentTRCode", raw.get("ParentTRCode"))
    enriched.setdefault("ModuleOwner", raw.get("BugBelong") or raw.get("IssueHandler"))
    enriched.setdefault("SeverityLabel", sev_label)
    enriched.setdefault("PriorityLabel", prio_label)
    enriched["ReproSteps"] = repro
    enriched["OSLanguage"] = os_lang
    enriched["Resolutions"] = resolutions
    enriched["VerifyComments"] = verify
    enriched["AttachmentCount"] = len(uploads) if isinstance(uploads, list) else 0
    return enriched


def build_bug_block(bug: dict, max_log_chars: int) -> str:
    """One prompt section per bug, favouring the structured fields produced by
    the workflow's Organize node (Resolutions / ReproSteps / VerifyComments).

    Falls back to the older raw-dump behaviour for bug_details.json files
    produced before the field-parsing upgrade (v1-001).
    """
    tr = bug.get("ParentTRCode")
    lines = [
        f"### {bug.get('BugCode', '?')}" + (f"  (TR {tr})" if tr else ""),
        f"- Module owner: {bug.get('ModuleOwner') or '(unknown)'} | "
        f"Severity: {bug.get('SeverityLabel') or bug.get('Severity')} | "
        f"Priority: {bug.get('PriorityLabel') or bug.get('Priority')} | "
        f"Close Build#: {bug.get('Close Build#')}",
        f"- Summary: {bug.get('Short Description') or '(none)'}",
    ]

    repro = stringify(bug.get("ReproSteps") or bug.get("Description"), max_log_chars)
    if repro:
        lines.append(f"- Repro / description: {repro}")
    if bug.get("OSLanguage"):
        lines.append(f"- Locale/OS coverage: {bug['OSLanguage']}")

    resolutions = bug.get("Resolutions") or []
    if resolutions:
        for r in resolutions:
            parts = []
            if r.get("root_cause"):
                parts.append(f"root cause: {r['root_cause']}")
            if r.get("solution"):
                parts.append(f"fix: {r['solution']}")
            if r.get("risk") and str(r["risk"]).strip() not in {"-", "N/A", "N/A."}:
                parts.append(f"stated risk: {r['risk']}")
            if r.get("trace_url") and str(r["trace_url"]).strip().lower() not in {"", "n/a", "n/a."}:
                parts.append(f"code: {r['trace_url']}")
            if parts:
                lines.append("- RD resolution — " + stringify("; ".join(parts), max_log_chars))

    verify = bug.get("VerifyComments") or []
    if verify:
        notes = "; ".join(
            str(v.get("comment", "")).strip() for v in verify if v.get("comment")
        )
        if notes:
            lines.append(f"- QA verify notes: {stringify(notes, max_log_chars)}")

    # Backward-compat: pre-parse files only have raw + _detected_log_fields
    if not resolutions and not repro and bug.get("raw"):
        raw = bug.get("raw", {}) or {}
        for key in bug.get("_detected_log_fields", []):
            if key in SKIP_LOG_KEYS:
                continue
            text = stringify(raw.get(key), max_log_chars)
            if text and len(text) > 40:
                lines.append(f"- {key}: {text}")
    return "\n".join(lines)


def build_prompt(data: dict, max_bugs: int, max_log_chars: int) -> tuple[str, int]:
    meta = data.get("meta", {})
    products = data.get("products", {})

    blocks = []
    total = 0
    truncated = False
    derived_tr_codes = []
    for product, bugs in products.items():
        blocks.append(f"\n## Product: {product}")
        for bug in bugs:
            if total >= max_bugs:
                truncated = True
                break
            bug = enrich_from_raw(bug)
            if bug.get("ParentTRCode"):
                derived_tr_codes.append(bug["ParentTRCode"])
            blocks.append(build_bug_block(bug, max_log_chars))
            total += 1

    build = meta.get("close_build") or "the newest TR"
    # meta.tr_codes is only present on newer-workflow files; derive otherwise.
    tr_codes = meta.get("tr_codes") or sorted(set(derived_tr_codes))
    tr_line = f" (TRs: {', '.join(tr_codes)})" if tr_codes else ""
    prompt = f"""You are a senior QA engineer for CyberLink's video/photo editing apps.
The following {total} bugs were fixed and closed in TR build {build}{tr_line}.
For each bug you get its summary, repro steps, the RD root-cause/fix, and QA verify notes.

Write a test-focus report in markdown with EXACTLY these two sections:

# Per-Bug Verification
For each bug (reference its BugCode): 2-4 concrete verification steps to confirm the fix,
including the specific feature/template/setting named in the bug. Note any device,
locale (e.g. ESM/PTB/JPN), or content preconditions mentioned.

# Recommended Regression Areas
Based on what the fixes touched (infer the affected modules/components from the
descriptions and fix logs): which OTHER sections of the app should be regression-tested
because they share code paths or were likely disturbed by these fixes. Group by module,
explain the reasoning in one sentence each, and rank by risk.

Be specific and actionable — a QA engineer will execute this directly.
{"NOTE: bug list was truncated to " + str(max_bugs) + " bugs." if truncated else ""}

Fixed bugs:
{chr(10).join(blocks)}

Begin the report now. Output ONLY the markdown report starting with the heading
"# Per-Bug Verification". Do not ask questions, do not add any preamble or offer
options — write the full report directly.
"""
    return prompt, total


SYSTEM_PROMPT = (
    "You are a senior QA engineer generating a test-focus report. You always "
    "respond with the finished markdown report only — never ask clarifying "
    "questions, never offer menus of options, never add conversational preamble. "
    "Follow the requested section structure exactly."
)


def call_ollama(prompt: str, model: str, ollama_url: str, timeout: int) -> str:
    req = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=json.dumps({
            "model": model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": 0.2},
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "")


def main():
    parser = argparse.ArgumentParser(description="Summarize a TR's fixed bugs into a test-focus report.")
    parser.add_argument("input", help="bug_details.json from fetch_tr_bug_details.py")
    parser.add_argument("--output", "-o", default=None,
                        help="Output markdown path (default: test_focus_summary.md next to input)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL,
                        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("--max-bugs", type=int, default=80,
                        help="Cap bugs included in the prompt (default: 80)")
    parser.add_argument("--max-log-chars", type=int, default=1500,
                        help="Truncate each description/log field to this many chars (default: 1500)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Ollama generation timeout in seconds (default: 600)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the prompt and exit without calling Ollama")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input not found: {input_path}")
        sys.exit(1)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt, bug_count = build_prompt(data, args.max_bugs, args.max_log_chars)
    if bug_count == 0:
        print("❌ No bugs found in input file.")
        sys.exit(1)

    if args.dry_run:
        print(prompt)
        print(f"\n--- dry run: {bug_count} bugs, prompt is {len(prompt):,} chars ---")
        return

    print(f"Summarizing {bug_count} fixed bug(s) with {args.model} "
          f"(prompt: {len(prompt):,} chars)…")
    try:
        report = call_ollama(prompt, args.model, args.ollama_url, args.timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"❌ Ollama call failed: {e}")
        print(f"   Is Ollama running at {args.ollama_url}? (ollama serve / ollama pull {args.model})")
        sys.exit(1)

    if not report.strip():
        print("❌ Ollama returned an empty response.")
        sys.exit(1)

    meta = data.get("meta", {})
    output_path = Path(args.output) if args.output else input_path.parent / "test_focus_summary.md"
    header = (
        f"<!-- Generated by summarize_tr_bugs.py on "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')} | model={args.model} | "
        f"close_build={meta.get('close_build')} | bugs={bug_count} -->\n\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + report.strip() + "\n")

    print(f"✅ Test-focus report → {output_path}")


if __name__ == "__main__":
    main()
