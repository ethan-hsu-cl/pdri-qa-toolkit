#!/usr/bin/env python3
"""Pre-generate the Release Pulse "AI-Synthesized Test Scenarios" cache.

Replicates Tab 10's per-module aggregation and Ollama prompt so the dashboard
can render scenarios immediately on first load instead of waiting for the
user to trigger generation. Output is a single JSON file, version-tagged so
the dashboard can detect when a newer version requires regeneration.

Usage:
  python generate_release_pulse_scenarios.py <product_data_dir> <output_json>
      [--provider ollama|heuristic] [--model qwen3:4b] [--top-n 10]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

OLLAMA_BASE = "http://localhost:11434"

from ollama_compat import extract_json, response_text, with_thinking_disabled
PULSE_STATUSES = {"TRCreated", "RDResolved"}
RISK_ORDER = ["Low", "Medium", "High", "Critical"]


def normalise_module(name):
    if not isinstance(name, str):
        return name
    name = re.sub(r'\s*\([^)]*\)', '', name)
    name = re.sub(r'\s*\[[^\]]*\]\s*', ' ', name)
    return re.sub(r' +', ' ', name).strip()


def ollama_reachable(timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def call_ollama(model, module, quadrant, risk_level, risk_score,
                leading_signal, inflight_descs, hist_scenarios, version):
    inflight_bullets = "\n".join(f"- {d}" for d in inflight_descs[:12])
    hist_bullets = (
        "\n".join(f"- {s}" for s in hist_scenarios[:5])
        if hist_scenarios else "  (no historical predictions loaded)"
    )
    prompt = (
        f"You are a QA lead reviewing bugs currently in active development for a video editing iOS app.\n\n"
        f"Version: {version}\n"
        f"Module: {module}\n"
        f"ML-predicted risk level: {risk_level}\n"
        f"FMEA quadrant: {quadrant} (risk score: {risk_score:.3f})\n"
        f"Strongest historical signal: {leading_signal or 'N/A'}\n\n"
        f"Active in-progress bugs (TRCreated / RDResolved — not yet QA-verified):\n"
        f"{inflight_bullets}\n\n"
        f"ML-predicted historical risk patterns for this module:\n"
        f"{hist_bullets}\n\n"
        f"Given the current in-flight changes AND the historical risk patterns above, "
        f"predict 3-5 specific bug scenarios QA should test. Focus on:\n"
        f"1. Regressions introduced by the in-flight changes\n"
        f"2. Edge cases around the active bug fixes\n"
        f"3. Interaction effects with the known historical risk areas\n\n"
        f"Each scenario must be a concrete, testable statement (like a real bug title). "
        f"Do NOT write vague summaries. Do NOT mention counts.\n"
        # `format: json` makes small models mirror the input structure unless
        # the wrapper key is named explicitly, so ask for one named object.
        f"Return exactly this JSON object and nothing else. Do not echo the "
        f"context above:\n"
        f'{{"scenarios": [{{"scenario": "...", "confidence": "high|medium|low", '
        f'"based_on": "in_flight|historical|both", '
        f'"explanation": "**Why likely:** [reason]. **Steps to reproduce:** [steps]. '
        f'**What to verify:** [expected vs actual]."}}]}}'
    )
    payload = json.dumps(with_thinking_disabled({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.3, "num_predict": 2000},
    })).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    raw_text = ""
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw_text = response_text(json.loads(resp.read().decode()))
        parsed = extract_json(raw_text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("scenarios", "results", "predictions"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
        return []
    except Exception:
        try:
            salvaged = extract_json(raw_text, expect="array")
        except Exception:
            return []
        return salvaged if isinstance(salvaged, list) else []


def score_to_risk(score: float) -> str:
    if score > 0.7: return "Critical"
    if score > 0.4: return "High"
    if score > 0.2: return "Medium"
    return "Low"


def boost_risk(level: str, trc: int) -> str:
    if trc == 0:
        return level
    idx = RISK_ORDER.index(level) if level in RISK_ORDER else 0
    return RISK_ORDER[min(idx + 1, len(RISK_ORDER) - 1)]


def build_pulse_grp(product_dir: Path, closed_on: "date | None" = None):
    """Mirror Tab 10's aggregation: returns (newest_version, pulse_grp, pred_leading_map, hist_sc, closed_bugcodes).

    Includes bugs whose Status is in PULSE_STATUSES OR whose Closed Date == `closed_on`
    (defaults to yesterday). Returns the BugCodes of the closed-on cohort so the
    snapshot can record exactly which fixed bugs informed today's predictions.
    """
    parsed_fp = product_dir / "ecl_parsed.csv"
    if not parsed_fp.exists():
        raise FileNotFoundError(f"Missing {parsed_fp}")

    df = pd.read_csv(parsed_fp, low_memory=False)
    if "Create Date" in df.columns:
        df["Create Date"] = pd.to_datetime(df["Create Date"], errors="coerce")
    if "Closed Date" in df.columns:
        df["Closed Date"] = pd.to_datetime(df["Closed Date"], errors="coerce")
    if "parsed_module" in df.columns:
        df["parsed_module"] = df["parsed_module"].apply(
            lambda x: normalise_module(x) if pd.notna(x) else x
        )

    if "parsed_version" not in df.columns or "Create Date" not in df.columns:
        raise ValueError("ecl_parsed.csv missing parsed_version or Create Date columns")

    ver_dates = (df.dropna(subset=["parsed_version", "Create Date"])
                   .groupby("parsed_version")["Create Date"].max())
    if ver_dates.empty:
        raise ValueError("No version/date data in ecl_parsed.csv")
    newest_version = str(ver_dates.idxmax())

    desc_col = "Short Description" if "Short Description" in df.columns else "parsed_description"

    # Default to yesterday (matches the 3am cron's evaluation window).
    if closed_on is None:
        closed_on = date.today() - timedelta(days=1)

    status_mask = df["Status"].isin(PULSE_STATUSES)
    if "Closed Date" in df.columns:
        closed_mask = df["Closed Date"].dt.date == closed_on
    else:
        closed_mask = pd.Series(False, index=df.index)

    df_pulse = df[(status_mask | closed_mask) &
                  (df["parsed_version"] == newest_version)].copy()

    # BugCodes of the closed-on cohort (preserved in the snapshot for evaluation).
    closed_bugcodes: list[str] = []
    # Per-module BugCodes for in-flight (TRCreated/RDResolved) bugs that fed
    # the Ollama prompt. The dashboard's effectiveness panel excludes these
    # from "did predictions catch any bugs?" so the same bug can't both
    # inform a scenario and be counted as a successful match for it.
    inflight_bugcodes_by_module: dict[str, list[str]] = {}
    # Per-BugCode historical status (TRCreated vs RDResolved) at snapshot
    # time. The dashboard's time-travel uses this to restore each bug's
    # status as it was, since the live CSV will have moved on.
    inflight_status_by_bugcode: dict[str, str] = {}
    if "BugCode" in df_pulse.columns:
        closed_rows = df_pulse[df_pulse["Closed Date"].dt.date == closed_on] \
                      if "Closed Date" in df_pulse.columns else df_pulse.iloc[0:0]
        closed_bugcodes = [
            str(bc).strip() for bc in closed_rows["BugCode"].dropna().tolist()
            if str(bc).strip()
        ]
        if "parsed_module" in df_pulse.columns:
            inflight_rows = df_pulse[df_pulse["Status"].isin(PULSE_STATUSES)]
            for mod, grp in inflight_rows.groupby("parsed_module"):
                bcs = [
                    str(bc).strip() for bc in grp["BugCode"].dropna().tolist()
                    if str(bc).strip()
                ]
                if bcs:
                    inflight_bugcodes_by_module[str(mod)] = bcs
            for _, r in inflight_rows.iterrows():
                bc = str(r.get("BugCode", "")).strip()
                if bc:
                    inflight_status_by_bugcode[bc] = str(r["Status"])

    risk_fp = product_dir / "risk_register_scored_all.csv"
    if risk_fp.exists() and "parsed_module" in df_pulse.columns:
        risk_df = pd.read_csv(risk_fp, low_memory=False)
        risk_df = risk_df.drop_duplicates(subset=["parsed_module"])
        cols = [c for c in ["parsed_module", "quadrant", "risk_score_final"]
                if c in risk_df.columns]
        df_pulse = df_pulse.merge(risk_df[cols], on="parsed_module", how="left")
    df_pulse["quadrant"] = df_pulse.get("quadrant", pd.Series()).fillna("Unknown")
    df_pulse["risk_score_final"] = pd.to_numeric(
        df_pulse.get("risk_score_final", pd.Series()), errors="coerce"
    ).fillna(0.0)

    if df_pulse.empty:
        return (newest_version, df_pulse, {}, {}, closed_bugcodes, closed_on,
                inflight_bugcodes_by_module, inflight_status_by_bugcode)

    pulse_grp = (
        df_pulse.groupby("parsed_module")
        .agg(
            quadrant=("quadrant", "first"),
            risk_score=("risk_score_final", "first"),
            trc_count=("Status", lambda s: (s == "TRCreated").sum()),
            rdr_count=("Status", lambda s: (s == "RDResolved").sum()),
            total=("Status", "count"),
            sample_descs=(desc_col, lambda x: list(x.dropna().head(10))),
        )
        .reset_index()
        .sort_values(["risk_score", "total"], ascending=[False, False])
    )

    pred_rl_map: dict[str, str] = {}
    pred_leading_map: dict[str, str] = {}
    pred_fp = product_dir / "predictions" / "ecl_parsed_predictions.csv"
    if pred_fp.exists():
        pred_df = pd.read_csv(pred_fp, low_memory=False)
        if "module" in pred_df.columns and "risk_level" in pred_df.columns:
            for _, pr in pred_df.iterrows():
                pred_rl_map[str(pr["module"])] = str(pr["risk_level"])
                if "leading_signal" in pred_df.columns:
                    pred_leading_map[str(pr["module"])] = str(pr.get("leading_signal", ""))

    def module_risk(mod, score, trc):
        base = pred_rl_map.get(mod, score_to_risk(float(score)))
        return boost_risk(base, int(trc))

    pulse_grp["risk_level"] = pulse_grp.apply(
        lambda r: module_risk(r["parsed_module"], r["risk_score"], r["trc_count"]),
        axis=1,
    )
    pulse_grp = pulse_grp.sort_values(
        ["risk_level", "risk_score", "total"],
        key=lambda c: c.map(RISK_ORDER.index) if c.name == "risk_level" else c,
        ascending=[False, False, False],
    )

    hist_sc: dict[str, list[dict]] = {}
    sc_fp = product_dir / "predictions" / "ecl_parsed_predictions_by_scenario.csv"
    pulse_mods = set(pulse_grp["parsed_module"])
    if sc_fp.exists():
        sc_df = pd.read_csv(sc_fp, low_memory=False)
        if "module" in sc_df.columns:
            for mod, msc in sc_df.groupby("module"):
                if mod in pulse_mods:
                    hist_sc[mod] = msc.to_dict("records")

    return (newest_version, pulse_grp, pred_leading_map, hist_sc, closed_bugcodes,
            closed_on, inflight_bugcodes_by_module, inflight_status_by_bugcode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("product_dir", help="Path to data/products/<slug>/")
    ap.add_argument("output_json", help="Output JSON path")
    ap.add_argument("--provider", choices=["ollama", "heuristic"], default="ollama")
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument(
        "--history-dir",
        default=None,
        help="Directory for dated snapshots (default: <product_dir>/predictions/release_pulse_history). "
             "A file named <YYYY-MM-DD>.json (today's local date) is written alongside the canonical "
             "output_json so the dashboard can evaluate yesterday's predictions against bugs closed since.",
    )
    args = ap.parse_args()

    product_dir = Path(args.product_dir)
    if not product_dir.exists():
        print(f"  ERROR: product dir not found: {product_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        (newest_version, pulse_grp, leading_map, hist_sc,
         closed_bugcodes, closed_on, inflight_bugcodes_by_module,
         inflight_status_by_bugcode) = (
            build_pulse_grp(product_dir)
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Per-module list of historical source BugCodes — recorded in the snapshot
    # so the effectiveness panel can later check whether any closed bug is
    # mentioned in the source set of a predicted scenario.
    def _source_bugcodes_by_module(hist_sc):
        result: dict[str, list[str]] = {}
        for mod, recs in hist_sc.items():
            seen: set[str] = set()
            ordered: list[str] = []
            for r in recs:
                codes = str(r.get("source_bug_codes", "")).strip()
                if not codes or codes == "nan":
                    continue
                for bc in codes.split(" | "):
                    bc = bc.strip()
                    if bc and bc not in seen:
                        seen.add(bc)
                        ordered.append(bc)
            if ordered:
                result[mod] = ordered
        return result

    # Per-module risk classifications captured at snapshot time so the
    # dashboard's time-travel view can show the FMEA quadrant, ML-derived
    # risk level, leading signal, and risk score that were current that day.
    def _risk_classifications_by_module(pulse_grp, leading_map):
        result: dict[str, dict] = {}
        for _, r in pulse_grp.iterrows():
            mod = str(r["parsed_module"])
            result[mod] = {
                "quadrant":       str(r.get("quadrant", "Unknown")),
                "risk_score":     float(r.get("risk_score", 0.0) or 0.0),
                "risk_level":     str(r.get("risk_level", "Low")),
                "leading_signal": str(leading_map.get(mod, "") or ""),
            }
        return result

    # Per-module list of historical ML scenarios captured so the "What to
    # Test Right Now" section's "Historical pattern" lines render the same
    # text the day the snapshot was taken, not whatever predict_defects.py
    # produced on a later run.
    def _historical_scenarios_by_module(hist_sc):
        result: dict[str, list[dict]] = {}
        for mod, recs in hist_sc.items():
            cleaned: list[dict] = []
            for r in recs:
                rec: dict = {}
                for k, v in r.items():
                    if pd.isna(v):
                        continue
                    if hasattr(v, "item"):
                        v = v.item()
                    rec[str(k)] = v
                if rec:
                    cleaned.append(rec)
            if cleaned:
                result[str(mod)] = cleaned
        return result

    def _write_snapshots(out: dict):
        """Write canonical output + dated history snapshot (today's local date)."""
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, indent=2))
        print(f"  Wrote {output_path}")

        history_dir = (
            Path(args.history_dir)
            if args.history_dir
            else product_dir / "predictions" / "release_pulse_history"
        )
        history_dir.mkdir(parents=True, exist_ok=True)
        # Today's local date — same-day reruns overwrite. The dashboard reads
        # yesterday's file to evaluate against bugs closed yesterday.
        history_path = history_dir / f"{date.today().isoformat()}.json"
        history_path.write_text(json.dumps(out, indent=2))
        print(f"  Wrote history snapshot {history_path}")

    if pulse_grp.empty:
        print(
            f"  No in-flight or closed-on-{closed_on.isoformat()} bugs for version "
            f"{newest_version} — writing empty cache."
        )
        out = {
            "version": newest_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model if args.provider == "ollama" else "heuristic",
            "scenarios": {},
            "source_bug_codes_by_module": {},
            "closed_on": closed_on.isoformat(),
            "closed_bugcodes": closed_bugcodes,
            "inflight_bugcodes_by_module": inflight_bugcodes_by_module,
            "inflight_status_by_bugcode": inflight_status_by_bugcode,
            "risk_classifications_by_module": {},
            "historical_scenarios_by_module": {},
        }
        _write_snapshots(out)
        return

    if args.provider == "heuristic":
        print(f"  Heuristic mode — skipping AI synthesis (dashboard renders heuristic view directly).")
        return

    if not ollama_reachable():
        print(f"  Ollama not reachable — skipping AI synthesis.", file=sys.stderr)
        sys.exit(2)

    top = pulse_grp.head(args.top_n)
    scenarios: dict[str, list[dict]] = {}
    print(f"  Generating scenarios for {len(top)} modules (version {newest_version}, model {args.model})…")
    for i, (_, mr) in enumerate(top.iterrows(), 1):
        mod = mr["parsed_module"]
        hist_txts = [str(s.get("scenario_text", "")) for s in hist_sc.get(mod, [])]
        result = call_ollama(
            model=args.model,
            module=mod,
            quadrant=str(mr["quadrant"]),
            risk_level=str(mr["risk_level"]),
            risk_score=float(mr["risk_score"]),
            leading_signal=leading_map.get(mod, ""),
            inflight_descs=list(mr["sample_descs"]),
            hist_scenarios=hist_txts,
            version=newest_version,
        )
        scenarios[mod] = result
        print(f"    [{i}/{len(top)}] {mod} — {len(result)} scenario(s)")

    out = {
        "version": newest_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "scenarios": scenarios,
        "source_bug_codes_by_module": _source_bugcodes_by_module(hist_sc),
        "closed_on": closed_on.isoformat(),
        "closed_bugcodes": closed_bugcodes,
        "inflight_bugcodes_by_module": inflight_bugcodes_by_module,
        "inflight_status_by_bugcode": inflight_status_by_bugcode,
        "risk_classifications_by_module": _risk_classifications_by_module(pulse_grp, leading_map),
        "historical_scenarios_by_module": _historical_scenarios_by_module(hist_sc),
    }
    _write_snapshots(out)


if __name__ == "__main__":
    main()
