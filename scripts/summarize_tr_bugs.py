#!/usr/bin/env python3
"""summarize_tr_bugs.py — AI summary of a TR's fixed bugs: what to test + risk areas.

Reads the bug_details.json produced by fetch_tr_bug_details.py, feeds each
fixed bug's description + status-log text to a local Ollama model, and writes
a markdown test-focus report next to the input file:

  1. Recommended regression areas — other sections/modules worth re-testing
  2. Per-bug verification — what to check to confirm each fix

Usage:
    python scripts/summarize_tr_bugs.py data/products/pdri/tr_reports/<label>/bug_details.json
    python scripts/summarize_tr_bugs.py <bug_details.json> --model gemma4
    python scripts/summarize_tr_bugs.py <bug_details.json> --dry-run   # print prompt only
"""

import argparse
import json
import re
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
        f"### {bug.get('BugCode', '?')}" + (f"  (bug created from TR {tr})" if tr else ""),
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
    if bug.get("DetailFetchError"):
        lines.append(f"- Detail warning: {bug['DetailFetchError']}; verify from the summary "
                     "and flag the missing RD root-cause/status log in the report.")

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
    tr_line = (
        f" (bug creation/parent TRs: {', '.join(tr_codes)}; these are provenance, "
        "not the Testing Request for this Close Build)"
        if tr_codes else ""
    )
    prompt = f"""You are a senior QA engineer for CyberLink's video/photo editing apps.
The following {total} bugs are included in TR build {build}{tr_line}.
For each bug you get its summary, repro steps, the RD root-cause/fix, and QA verify notes.

Write a test-focus report in markdown with EXACTLY these two sections:

# Recommended Regression Areas
Based on what the fixes touched (infer the affected modules/components from the
descriptions and fix logs): which OTHER sections of the app should be regression-tested
because they share code paths or were likely disturbed by these fixes. Group by module,
explain the reasoning in one sentence each, and rank by risk.

# Per-Bug Verification
For each bug (reference its BugCode): 2-4 concrete verification steps to confirm the fix,
including the specific feature/template/setting named in the bug. Note any device,
locale (e.g. ESM/PTB/JPN), or content preconditions mentioned.

Be specific and actionable — a QA engineer will execute this directly.
Keep each bug concise: output only 2-4 numbered verification steps beneath its
BugCode. Do not repeat the supplied summary, root cause, code URL, or metadata.
{"NOTE: bug list was truncated to " + str(max_bugs) + " bugs." if truncated else ""}

Fixed bugs:
{chr(10).join(blocks)}

Begin the report now. Output ONLY the markdown report starting with the heading
"# Recommended Regression Areas". Do not ask questions, do not add any preamble or offer
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
            "think": False,
            # A large TR can contain dozens of bugs. The Ollama default output
            # limit truncates the report after only a few, so reserve enough
            # tokens for every per-bug checklist plus regression areas.
            "options": {
                "temperature": 0.2,
                "num_ctx": 8192,
                "num_predict": 1600,
            },
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "")


def expected_bug_codes(data: dict, max_bugs: int) -> list[str]:
    codes: list[str] = []
    for bugs in data.get("products", {}).values():
        for bug in bugs:
            if len(codes) >= max_bugs:
                return codes
            code = str(bug.get("BugCode") or "").strip()
            if code:
                codes.append(code)
    return codes


def validate_report(report: str, data: dict, max_bugs: int) -> list[str]:
    """Return completeness errors; an invalid report must not be published."""
    errors = []
    if report.count("# Per-Bug Verification") != 1:
        errors.append("missing or duplicated '# Per-Bug Verification' heading")
    if report.count("# Recommended Regression Areas") != 1:
        errors.append("missing or duplicated '# Recommended Regression Areas' heading")
    missing = [code for code in expected_bug_codes(data, max_bugs) if code not in report]
    if missing:
        errors.append(f"missing {len(missing)} BugCode(s): {', '.join(missing)}")
    regression_parts = re.split(r"^#\s+Recommended Regression Areas\s*$", report,
                                maxsplit=1, flags=re.I | re.M)
    if len(regression_parts) == 2:
        regression = regression_parts[1]
        if not re.search(r"\b(Critical|High|Medium|Low)\b", regression, flags=re.I):
            errors.append("regression section has no risk-ranked recommendations")
        if re.search(r"summary of (?:the )?(?:provided|design|bug)", regression, flags=re.I):
            errors.append("regression section summarizes inputs instead of recommending tests")
    return errors


def regression_first(report: str) -> str:
    """Normalize a complete report so recommendations are always the first section."""
    heading_re = re.compile(
        r"^#\s+(Recommended Regression Areas|Per-Bug Verification)\s*$",
        flags=re.I | re.M,
    )
    matches = list(heading_re.finditer(report))
    if len(matches) != 2:
        return report.strip()

    sections = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(report)
        sections[match.group(1).lower()] = report[match.end():end].strip()

    regression = sections.get("recommended regression areas")
    verification = sections.get("per-bug verification")
    if regression is None or verification is None:
        return report.strip()
    return (
        "# Recommended Regression Areas\n\n"
        + regression
        + "\n\n# Per-Bug Verification\n\n"
        + verification
    ).strip()


def normalize_regression_bullets(report: str) -> str:
    """Turn unbulleted bold regression recommendations into readable bullets."""
    parts = re.split(
        r"(^#\s+Recommended Regression Areas\s*$)",
        report,
        maxsplit=1,
        flags=re.I | re.M,
    )
    if len(parts) != 3:
        return report
    before, heading, remainder = parts
    regression_parts = re.split(
        r"(^#\s+Per-Bug Verification\s*$)",
        remainder,
        maxsplit=1,
        flags=re.I | re.M,
    )
    regression = regression_parts[0]
    regression = re.sub(
        r"(?m)^(?=\*\*[^*\n]+\*\*\s*[-–—:])",
        "- ",
        regression,
    )
    return before + heading + regression + "".join(regression_parts[1:])


def selected_bugs(data: dict, max_bugs: int) -> list[dict]:
    selected = []
    for bugs in data.get("products", {}).values():
        for bug in bugs:
            if len(selected) >= max_bugs:
                return selected
            selected.append(enrich_from_raw(bug))
    return selected


def build_verification_chunk_prompt(bugs: list[dict], max_log_chars: int) -> str:
    blocks = "\n\n".join(build_bug_block(bug, max_log_chars) for bug in bugs)
    return f"""You are a senior QA engineer. Write concise verification checklists for
the fixed bugs below. For EVERY bug, output a level-2 markdown heading containing
its exact BugCode, followed by 2-4 numbered, concrete verification steps. Include
device/locale/content preconditions when supplied. Do not repeat summaries,
metadata, root-cause text, or code URLs. Do not add a level-1 heading, preamble,
conclusion, or regression recommendations.

Fixed bugs:
{blocks}
"""


def build_regression_prompt(data: dict, max_bugs: int, max_log_chars: int) -> str:
    bugs = selected_bugs(data, max_bugs)
    # Use a compact cross-bug digest. Full repro logs crowd a local model's
    # context and cause it to summarize tickets instead of reasoning about
    # shared code paths and adjacent regression coverage.
    digest_lines = []
    for bug in bugs:
        resolutions = bug.get("Resolutions") or []
        root_fixes = "; ".join(
            " / ".join(filter(None, [
                stringify(r.get("root_cause"), 180),
                stringify(r.get("solution"), 220),
            ]))
            for r in resolutions
        )
        digest_lines.append(
            f"- {bug.get('BugCode')}: "
            f"{stringify(bug.get('Short Description'), 260)} | "
            f"root cause/fix: {root_fixes or 'detail unavailable'}"
        )
    blocks = "\n".join(digest_lines)
    return f"""You are a senior QA engineer. Based on the fixed bugs, root causes, and
solutions below, recommend OTHER app sections that should be regression-tested
because they share code paths or may have been disturbed. Output exactly one
level-1 heading: "# Recommended Regression Areas". Group recommendations by
module, rank them Critical/High/Medium/Low, and format every module/feature as a
separate markdown bullet. Give concrete test actions plus one-sentence reasoning.
Do not summarize the tickets, repeat bug details, create
a ticket table, or discuss design guidelines. Start directly with risk-ranked
adjacent test areas. Each item must name an app section outside the direct fix,
state what QA should do, and explain the shared state/component/code-path risk.

Compact fix digest:
{blocks}
"""


def strip_chunk_wrappers(text: str) -> str:
    text = re.sub(r"^\s*#\s+Per-Bug Verification\s*", "", text, flags=re.I)
    # Ignore any unsolicited regression tail; the dedicated synthesis call is
    # the only source for that section.
    text = re.split(r"^#\s+Recommended Regression Areas\s*$", text,
                    maxsplit=1, flags=re.I | re.M)[0]
    return text.strip()


def generate_chunked_report(data: dict, model: str, ollama_url: str,
                            timeout: int, max_bugs: int,
                            max_log_chars: int, chunk_size: int = 4) -> str:
    bugs = selected_bugs(data, max_bugs)
    verification_parts = []
    total_chunks = (len(bugs) + chunk_size - 1) // chunk_size
    for start in range(0, len(bugs), chunk_size):
        chunk = bugs[start:start + chunk_size]
        chunk_number = start // chunk_size + 1
        print(f"  Verification batch {chunk_number}/{total_chunks} "
              f"({len(chunk)} bugs)…", flush=True)
        response = call_ollama(
            build_verification_chunk_prompt(chunk, min(max_log_chars, 800)),
            model, ollama_url, timeout,
        )
        cleaned = strip_chunk_wrappers(response)
        # Small models occasionally omit one item even in a four-bug batch.
        # Repair only the missing items with a one-bug call instead of
        # restarting the entire TR generation.
        missing_from_chunk = [
            bug for bug in chunk
            if str(bug.get("BugCode") or "").strip() not in cleaned
        ]
        for missing_bug in missing_from_chunk:
            code = str(missing_bug.get("BugCode") or "").strip()
            print(f"    Repairing omitted bug {code}…", flush=True)
            repair = call_ollama(
                build_verification_chunk_prompt([missing_bug,], min(max_log_chars, 800)),
                model, ollama_url, timeout,
            )
            cleaned += "\n\n" + strip_chunk_wrappers(repair)
        verification_parts.append(cleaned)

    print("  Cross-bug regression synthesis…", flush=True)
    regression = call_ollama(
        build_regression_prompt(data, max_bugs, max_log_chars),
        model, ollama_url, timeout,
    ).strip()
    if not re.search(r"^#\s+Recommended Regression Areas\s*$", regression,
                     flags=re.I | re.M):
        regression = "# Recommended Regression Areas\n\n" + regression

    return regression + "\n\n# Per-Bug Verification\n\n" \
        + "\n\n".join(verification_parts)


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
          f"in bounded batches…")
    try:
        report = generate_chunked_report(
            data, args.model, args.ollama_url, args.timeout,
            args.max_bugs, args.max_log_chars,
        )
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"❌ Ollama call failed: {e}")
        print(f"   Is Ollama running at {args.ollama_url}? (ollama serve / ollama pull {args.model})")
        sys.exit(1)

    if not report.strip():
        print("❌ Ollama returned an empty response.")
        sys.exit(1)

    report = normalize_regression_bullets(regression_first(report))
    validation_errors = validate_report(report, data, args.max_bugs)
    if validation_errors:
        print("❌ Ollama returned an incomplete test-focus report:")
        for error in validation_errors:
            print(f"   - {error}")
        print("   The existing report was not replaced.")
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
