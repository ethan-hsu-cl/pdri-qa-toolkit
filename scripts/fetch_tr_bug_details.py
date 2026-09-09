#!/usr/bin/env python3
"""fetch_tr_bug_details.py — Fetch full eBug details (incl. status log) for a TR's fixed bugs.

Calls the "(TR Report) Get Bug Details_v1" n8n workflow, which either:
  • queries closed bugs whose Close Build# matches --close-build, or
  • fetches an explicit --bug-codes list,
then pulls each bug's full record via the eBug "getOldeBug" operation and
returns everything grouped by product.

Default output is saved under
data/products/<slug>/tr_reports/manual/<timestamp>_<label>/bug_details.json so
manual fetches cannot overwrite immutable pipeline evidence.

Usage:
    python scripts/fetch_tr_bug_details.py --product pdri --close-build 16.5.0.185409
    python scripts/fetch_tr_bug_details.py --product pdri --close-build 185409 --duration-months 6
    python scripts/fetch_tr_bug_details.py --product pdri --bug-codes DRI261853-0005,DRI261853-0002

Next step (AI summary of what to test):
    python scripts/summarize_tr_bugs.py data/products/pdri/tr_reports/manual/<run>/bug_details.json
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

try:
    from fetch_from_n8n import PRODUCT_MAP
except ModuleNotFoundError:  # Imported as scripts.fetch_tr_bug_details
    from scripts.fetch_from_n8n import PRODUCT_MAP

DEFAULT_WEBHOOK_URL = (
    "https://ecl-agent.cyberlink.com/webhook/9f21c4d8-3b7a-4a05-8e2f-6d4c1b90a7e3-v1"
)


class FetchError(Exception):
    """Raised when the webhook returns a non-retryable error."""


def post_json(url: str, payload: dict, timeout: int = 300, max_retries: int = 2,
              retry_delay: int = 15):
    """POST JSON to the webhook, retrying on 5xx/network errors."""
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(1, max_retries + 2):
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        print(f"→ POST {url}  (attempt {attempt}/{max_retries + 1})")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            break
        except urllib.error.HTTPError as e:
            body_preview = e.read().decode("utf-8", errors="replace")[:500]
            if 400 <= e.code < 500:
                # Workflow-level errors (e.g. "No closed bugs matched close_build …")
                # come back as 4xx/500 with a message body — surface it.
                raise FetchError(f"HTTP {e.code} {e.reason}: {body_preview}")
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
    return data


def normalize_response(data) -> dict:
    """Unwrap the n8n response into the organized {meta, products} object."""
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict) and "json" in data and isinstance(data["json"], dict):
        data = data["json"]
    if not isinstance(data, dict) or "products" not in data:
        raise FetchError(
            f"Unexpected response shape (no 'products' key). "
            f"Got keys: {sorted(data.keys()) if isinstance(data, dict) else type(data)}"
        )
    if not isinstance(data.get("meta", {}), dict):
        raise FetchError("Unexpected response shape: 'meta' must be an object.")
    if not isinstance(data["products"], dict) or any(
        not isinstance(bugs, list) for bugs in data["products"].values()
    ):
        raise FetchError("Unexpected response shape: 'products' must map names to arrays.")
    return data


def sanitize_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("_")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch full bug details for a TR's fixed bugs via n8n."
    )
    parser.add_argument("--product", required=True, choices=list(PRODUCT_MAP.keys()),
                        help="Product slug (determines output dir and eBug ProductName).")
    parser.add_argument("--close-build", default=None,
                        help="TR build to match against Close Build# "
                             "(full '16.5.0.185409' or just '185409').")
    parser.add_argument("--bug-codes", default=None,
                        help="Comma-separated explicit BugCode list "
                             "(skips the Close Build# query).")
    parser.add_argument("--duration-months", type=int, default=12,
                        help="Search window for the closed-bug query (default: 12).")
    parser.add_argument("--webhook-url", default=DEFAULT_WEBHOOK_URL,
                        help=f"n8n webhook POST URL (default: {DEFAULT_WEBHOOK_URL})")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path. Default: "
                             "data/products/<slug>/tr_reports/manual/<run>/bug_details.json")
    parser.add_argument("--timeout", type=int, default=300,
                        help="HTTP timeout in seconds (default: 300 — detail fetch "
                             "loops one API call per bug, so this can be slow).")
    args = parser.parse_args()

    if not args.close_build and not args.bug_codes:
        parser.error("Provide --close-build and/or --bug-codes.")

    payload = {
        "product_name": PRODUCT_MAP[args.product],
        "duration_months": args.duration_months,
    }
    if args.close_build:
        payload["close_build"] = args.close_build
    if args.bug_codes:
        codes = [c.strip() for c in args.bug_codes.split(",") if c.strip()]
        payload["bug_codes"] = codes

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = sanitize_label(args.close_build or "bug_codes") or "request"
        output_path = Path(
            f"data/products/{args.product}/tr_reports/manual/"
            f"{timestamp}_{label}/bug_details.json"
        )

    print(f"Fetching TR bug details — product={args.product} "
          f"close_build={args.close_build or '-'} "
          f"bug_codes={len(payload.get('bug_codes', []))}")

    try:
        data = normalize_response(post_json(args.webhook_url, payload, timeout=args.timeout))
    except FetchError as e:
        print(f"\n❌ Fetch failed: {e}")
        sys.exit(1)

    meta = data.get("meta", {})
    products = data.get("products", {})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output_path)

    print(f"\n✅ Saved {meta.get('bug_count', '?')} bug detail(s) → {output_path}")
    for prod, bugs in products.items():
        print(f"   {prod}: {len(bugs)} bug(s)")
    if meta.get("tr_codes"):
        print(
            "   Bug creation/parent TRs (not the Close Build's Testing Request): "
            f"{', '.join(meta['tr_codes'])}"
        )

    # Coverage: how many bugs carry a parsed RD resolution (root cause / fix)
    all_bugs = [b for bugs in products.values() for b in bugs]
    with_fix = sum(1 for b in all_bugs if b.get("Resolutions"))
    if all_bugs:
        print(f"   RD root-cause/fix parsed for {with_fix}/{len(all_bugs)} bug(s).")

    if meta.get("errors"):
        print(f"⚠️  {len(meta['errors'])} bug(s) failed to fetch (transient tsr API errors "
              f"are auto-retried in the workflow):")
        for err in meta["errors"][:10]:
            print(f"   - {err}")

    print(f"\nNext step:\n  python scripts/summarize_tr_bugs.py {output_path}")


if __name__ == "__main__":
    main()
