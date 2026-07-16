#!/usr/bin/env python3
"""fetch_from_n8n.py — Call the n8n eBug webhook and save raw JSON to disk.

Usage:
    python scripts/fetch_from_n8n.py
    python scripts/fetch_from_n8n.py --product pdri
    python scripts/fetch_from_n8n.py --product phdi --duration-months 1
    python scripts/fetch_from_n8n.py --output data/ecl_raw.json
    python scripts/fetch_from_n8n.py --webhook-url https://your-n8n-host/webhook/82746bb5-e140-4720-98a3-d1965900274d-v4

Then feed the output directly into the parser:
    python scripts/parse_ecl_export.py data/ecl_raw.json data/ecl_parsed.csv

Or run both steps in one go:
    python scripts/fetch_from_n8n.py --then-parse
"""

import argparse
import calendar
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_WEBHOOK_URL = (
    "https://ecl-agent.cyberlink.com/webhook/82746bb5-e140-4720-98a3-d1965900274d-v4"
)
DEFAULT_OUTPUT   = "data/ecl_raw.json"
DEFAULT_PARSED   = "data/ecl_parsed.csv"

# Product slug → full eBug ProductName
PRODUCT_MAP = {
    "pdri":   "PowerDirector Mobile for iOS",
    "phdi":   "PhotoDirector Mobile for iOS",
    "pdra":   "PowerDirector Mobile for Android",
    "phda":   "PhotoDirector Mobile for Android",
    "pdr":    "PowerDirector",
    "phd":    "PhotoDirector",
    "promeo": "Promeo",
    "vvg":    "Vivid Glam (iOS)",
}

DEFAULT_PRODUCT = "pdri"

# Fields that must be present for the parser to work correctly.
# Build# and Close Build# are nice-to-have; Short Description is critical.
REQUIRED_FIELDS  = {"Short Description", "BugCode", "Severity", "Status", "Create Date"}
OPTIONAL_FIELDS  = {"Build#", "Close Build#", "Version", "Creator"}

# ─────────────────────────────────────────────────────────────────────────────


class FetchError(Exception):
    """Raised when the webhook returns a non-retryable error."""


def fetch_bugs(
    webhook_url: str,
    payload: dict,
    timeout: int = 120,
    max_retries: int = 3,
    retry_delay: int = 15,
) -> list[dict]:
    """POST to the n8n webhook and return a flat list of bug records.

    Retries up to `max_retries` times on 5xx errors and network timeouts
    (exponential backoff starting at `retry_delay` seconds).
    Raises FetchError immediately on 4xx — those won't be fixed by retrying.
    """
    body = json.dumps(payload).encode("utf-8")

    for attempt in range(1, max_retries + 2):  # +2 so last pass shows attempt N
        req = urllib.request.Request(
            webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        label = f"(attempt {attempt}/{max_retries + 1})"
        print(f"→ POST {webhook_url}  {label}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            break  # success — exit retry loop

        except urllib.error.HTTPError as e:
            body_preview = e.read().decode("utf-8", errors="replace")[:300]
            if 400 <= e.code < 500:
                # Client error — retrying won't help
                raise FetchError(f"HTTP {e.code} {e.reason}: {body_preview}")
            # 5xx — may be transient
            print(f"  HTTP {e.code} {e.reason}: {body_preview}")

        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  Network error: {e}")

        if attempt > max_retries:
            raise FetchError(f"Webhook failed after {max_retries + 1} attempts.")

        wait = retry_delay * (2 ** (attempt - 1))
        print(f"  Retrying in {wait}s…")
        time.sleep(wait)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise FetchError(f"Failed to parse JSON response: {e}\nRaw (500 chars): {raw[:500]}")

    # n8n "Respond to Webhook → allIncomingItems" wraps each item as
    # {"json": {...}} OR returns a plain list of dicts depending on version.
    # Normalise both shapes into a flat list of bug dicts.
    if isinstance(data, list):
        records = []
        for item in data:
            if isinstance(item, dict) and "json" in item:
                records.append(item["json"])
            elif isinstance(item, dict):
                records.append(item)
        return records
    elif isinstance(data, dict):
        return [data.get("json", data)]
    else:
        raise FetchError(f"Unexpected response shape: {type(data)}")


def audit_fields(records: list[dict]) -> bool:
    """Warn about missing fields. Returns True if all required fields present."""
    if not records:
        return False
    sample = records[0]
    keys   = set(sample.keys())

    missing_required = REQUIRED_FIELDS - keys
    missing_optional = OPTIONAL_FIELDS - keys

    if missing_required:
        print(f"\n❌ MISSING REQUIRED fields: {sorted(missing_required)}")
        print("   These are needed for module/tag/severity parsing.")
        print("   Check the 'Get Columns_v4' Set node in your n8n workflow.")
        return False

    if missing_optional:
        print(f"\n⚠️  Missing optional fields: {sorted(missing_optional)}")
        print("   builds_to_fix will be unavailable — everything else will work.")

    print(f"  ✅ All required fields present")
    print(f"  Fields in response: {sorted(keys)}")
    return True


def save_json(records: list[dict], output_path: str) -> dict:
    """Merge incoming records with any existing file, deduplicating by BugCode.

    Returns a summary dict with keys: new, updated_status, unchanged, total.

    Change 4 — Status comparison / update
    ─────────────────────────────────────
    When a record already exists in the cache (same BugCode), its current Status
    is compared with the incoming Status.  If they differ the record is updated
    AND a `_status_changed` flag is set so downstream tools (parse, dashboard)
    can highlight freshly-transitioned bugs without re-processing everything.

    A record that is unchanged on every field still has its `_status_changed`
    flag cleared to False so it is never left incorrectly marked from a
    previous run.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    existing_map: dict = {}
    p = Path(output_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            try:
                for r in json.load(f):
                    if isinstance(r, dict) and r.get("BugCode"):
                        existing_map[r["BugCode"]] = r
            except json.JSONDecodeError:
                print("⚠️  Could not parse existing JSON — treating as empty cache.")

    # Flags describe transitions observed by *this* fetch. Clear every cached
    # record first, including records omitted by a narrow latest-version fetch;
    # otherwise an older True flag can repeatedly trigger newest-TR processing.
    for old in existing_map.values():
        old["_status_changed"] = False

    n_new = n_updated = n_unchanged = 0

    for rec in tqdm(records, desc="Merging records", unit="bug", file=sys.stderr):
        code = rec.get("BugCode")
        if not code:
            continue
        if code not in existing_map:
            rec["_status_changed"] = False
            existing_map[code] = rec
            n_new += 1
        else:
            old = existing_map[code]
            old_status = str(old.get("Status", "")).strip().lower()
            new_status = str(rec.get("Status", "")).strip().lower()
            if old_status != new_status:
                rec["_status_changed"] = True
                rec["_prev_status"]    = old.get("Status", "")
                existing_map[code]     = rec
                n_updated += 1
            else:
                # The pre-merge reset above also handles cached records that
                # were not included in this fetch.
                rec["_status_changed"] = False
                existing_map[code]     = rec
                n_unchanged += 1

    final = list(existing_map.values())

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    summary = dict(new=n_new, updated_status=n_updated, unchanged=n_unchanged, total=len(final))
    print(
        f"✅ Saved {summary['total']:,} bugs → {output_path}\n"
        f"   New: {n_new}  |  Status-changed: {n_updated}  |  Unchanged: {n_unchanged}"
    )
    return summary


def get_latest_version(output_path: str) -> str:
    """Read version_catalogue.csv (same directory as output_path) and return
    the most recent non-sparse version string.

    This is used when scope=latest so the n8n workflow receives the actual
    version string to filter on, not the bare word 'latest'.

    Returns an empty string if the catalogue is absent or unreadable — in
    that case the workflow falls back to its own 'latest' logic.
    """
    cat_path = Path(output_path).parent / "version_catalogue.csv"
    if not cat_path.exists():
        return ""
    try:
        cat = pd.read_csv(str(cat_path))
        real = cat[~cat["version_is_sparse"].astype(bool)].sort_values("version_rank")
        if not real.empty:
            ver = str(real.iloc[0]["parsed_version"])
            print(f"  Latest non-sparse version from catalogue: {ver}")
            return ver
    except Exception as e:
        print(f"  Could not read version catalogue ({e}) — n8n will decide 'latest'.")
    return ""


def resolve_scope(args) -> str:
    """Change 5 — Testing-schedule experiment.

    Weekdays (Mon–Fri): fetch only the latest version (faster, lighter).
    Weekends (Sat–Sun): fetch all versions to measure full-update duration.

    Override with --scope latest | all to force either mode regardless of day.
    """
    if args.scope == "latest":
        return "latest"
    if args.scope == "all":
        return "all"
    # auto: weekday → latest, weekend → all
    today = datetime.now().weekday()   # 0=Mon … 6=Sun
    if today >= 5:
        print(f"📅 Weekend detected (weekday={today}) → fetching ALL versions")
        return "all"
    else:
        print(f"📅 Weekday detected (weekday={today}) → fetching LATEST version only")
        return "latest"


def subtract_months(dt: datetime, months: int) -> datetime:
    """Return dt shifted back by `months` months, clamping to the last valid day."""
    month = dt.month - months
    year = dt.year
    while month <= 0:
        month += 12
        year -= 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def fetch_bugs_chunked(
    webhook_url: str,
    base_payload: dict,
    duration_months: int,
    chunk_months: int,
    timeout: int,
) -> list[dict]:
    """Split a large date range into `chunk_months`-sized windows and merge results.

    Sends `date_from` / `date_to` string fields in each payload so the n8n
    workflow uses explicit bounds instead of the `duration_months` fallback.
    Deduplicates by BugCode across all chunks.
    """
    now = datetime.now()
    chunks: list[tuple[str, str]] = []
    end = now
    remaining = duration_months
    while remaining > 0:
        size = min(chunk_months, remaining)
        start = subtract_months(end, size)
        chunks.append((
            start.strftime("%Y-%m-%d") + " 00:00:00",
            end.strftime("%Y-%m-%d %H:%M:%S"),
        ))
        end = start
        remaining -= size

    all_records: dict[str, dict] = {}
    for date_from, date_to in tqdm(chunks, desc="Fetching chunks", unit="chunk",
                                    file=sys.stderr):
        tqdm.write(f"  {date_from[:10]} → {date_to[:10]}", file=sys.stderr)
        payload = {**base_payload, "date_from": date_from, "date_to": date_to}
        records = fetch_bugs(webhook_url, payload, timeout=timeout)
        tqdm.write(f"    → {len(records):,} records", file=sys.stderr)
        for rec in records:
            code = rec.get("BugCode")
            if code:
                all_records[code] = rec
    merged = list(all_records.values())
    print(f"  Chunks complete — {len(merged):,} unique bugs total")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Fetch eBugs from n8n webhook and save as JSON."
    )
    parser.add_argument(
        "--webhook-url",
        default=DEFAULT_WEBHOOK_URL,
        help=f"n8n webhook POST URL (default: {DEFAULT_WEBHOOK_URL})",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Output JSON file path. When --product is used, defaults to "
            "data/products/<slug>/ecl_raw.json; otherwise data/ecl_raw.json"
        ),
    )
    parser.add_argument(
        "--parsed-output",
        default=None,
        help=(
            "Parsed CSV path used by --then-parse. When --product is used, "
            "defaults to data/products/<slug>/ecl_parsed.csv"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--then-parse",
        action="store_true",
        help="After saving, immediately run parse_ecl_export.py on the result",
    )
    # Change 5 — scope control
    parser.add_argument(
        "--scope",
        choices=["auto", "latest", "all"],
        default="auto",
        help=(
            "Fetch scope: 'latest' = newest version only (fast, weekday default); "
            "'all' = all versions (thorough, weekend default); "
            "'auto' = choose by day of week (default)."
        ),
    )
    # Multi-product support
    parser.add_argument(
        "--product",
        choices=list(PRODUCT_MAP.keys()),
        default=None,
        help=(
            f"Product slug ({', '.join(PRODUCT_MAP.keys())}). "
            f"When set, output paths default to data/products/<slug>/. "
            f"Default: {DEFAULT_PRODUCT} (backward-compatible legacy paths when omitted)."
        ),
    )
    parser.add_argument(
        "--duration-months",
        type=int,
        default=None,
        help="Duration in months for the date range filter (sent to n8n). Default: 36.",
    )
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=None,
        help=(
            "Split the date range into chunks of this many months to avoid "
            "504 timeouts on large products (e.g. --chunk-months 6). "
            "Omit to fetch in one request."
        ),
    )
    args = parser.parse_args()

    # Resolve product-aware output paths
    product_slug = args.product
    if product_slug:
        product_dir = f"data/products/{product_slug}"
        output_path = args.output or f"{product_dir}/ecl_raw.json"
        parsed_path = args.parsed_output or f"{product_dir}/ecl_parsed.csv"
        product_name = PRODUCT_MAP[product_slug]
    else:
        output_path = args.output or DEFAULT_OUTPUT
        parsed_path = args.parsed_output or DEFAULT_PARSED
        product_name = PRODUCT_MAP[DEFAULT_PRODUCT]

    duration_months = args.duration_months or 36

    scope = resolve_scope(args)
    start_time = datetime.now()
    product_label = f"{product_slug or DEFAULT_PRODUCT} ({product_name})"
    print(
        f"Fetching eBugs — {start_time.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"[product={product_label}  duration={duration_months}mo  scope={scope}]"
    )

    # Build the payload
    payload = {
        "product_name": product_name,
        "duration_months": duration_months,
        "scope": scope,
    }
    if scope == "latest":
        latest_ver = get_latest_version(output_path)
        if latest_ver:
            payload["latest_version"] = latest_ver
            print(f"  Sending latest_version={latest_ver!r} in webhook payload")

    try:
        if args.chunk_months and args.chunk_months < duration_months:
            print(f"  Chunked mode: {duration_months}mo ÷ {args.chunk_months}mo/chunk")
            records = fetch_bugs_chunked(
                args.webhook_url, payload, duration_months, args.chunk_months, timeout=args.timeout
            )
        else:
            records = fetch_bugs(args.webhook_url, payload, timeout=args.timeout)
    except FetchError as e:
        print(f"\n❌ Fetch failed: {e}")
        sys.exit(1)

    print(f"  Received {len(records):,} records")

    if not records:
        print("⚠️  No records returned. Check the n8n workflow and its date range filter.")
        sys.exit(1)

    ok = audit_fields(records)
    if not ok:
        sys.exit(1)

    summary = save_json(records, output_path)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"⏱  Fetch+save completed in {elapsed:.1f}s  (scope={scope})")

    if args.then_parse:
        print(f"\nRunning parser → {parsed_path}")
        result = subprocess.run(
            [
                sys.executable,
                "scripts/parse_ecl_export.py",
                output_path,
                parsed_path,
            ],
            check=False,
        )
        if result.returncode != 0:
            print("Parser exited with errors.")
            sys.exit(result.returncode)
    else:
        print(
            f"\nNext step:\n"
            f"  python scripts/parse_ecl_export.py {output_path} {parsed_path}\n"
            f"\nOr re-run with --then-parse to do both in one command."
        )


if __name__ == "__main__":
    main()
