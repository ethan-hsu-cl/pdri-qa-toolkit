#!/usr/bin/env python3
"""PDR-I AI Risk Scorer v2.4

Behavior:

Given a combined risk register (ALL versions), this script will:

  - Score the combined file -> risk_register_scored_all.csv
  - Score each per-version risk register found in data/risk_register_versions/:
        risk_register_versions/risk_register_<version>.csv
    and save to the same subfolder:
        risk_register_versions/risk_register_scored_<version>.csv

No extra arguments needed.

Usage:
  python ai_risk_scorer.py data/risk_register_all.csv data/risk_register_scored_all.csv --provider heuristic
  python ai_risk_scorer.py data/risk_register_all.csv data/risk_register_scored_all.csv --provider ollama --model qwen3:4b
  python ai_risk_scorer.py data/risk_register_all.csv data/risk_register_scored_all.csv --provider ollama --verbose
"""
import warnings
warnings.filterwarnings("ignore", message="urllib3.*OpenSSL")
warnings.filterwarnings("ignore", message="urllib3.*LibreSSL")

import os, json, re, argparse, time, glob, sys

from ollama_compat import extract_json, with_thinking_disabled
import pandas as pd, numpy as np
import urllib.request
from tqdm import tqdm


IMPACT_OVERRIDES = {
    "export":5,"produced video":5,"ai storytelling":5,"auto edit":5,"project":5,
    "new project":5,"auto edit project":4,"image to video":4,"text to video":4,
    "ai art":4,"ai music generator":4,"auto captions":4,"video enhance":4,
    "camera":4,"trim":4,"split":3,"settings":1,"preference":1,
    "tutorials & tips":1,"credit":1,"more":1,"my artwork":1,
    "trending":2,"vip benefit page":3,
}


def clamp_detectability_by_coverage(row, detect: int, impact: int) -> tuple[int, str]:
    """Post-hoc sanity clamp: LLMs frequently invert the FMEA detectability scale.
    Enforce monotonicity with AutomationCatchRate so a fully-automated module can't
    end up rated as 'hard to detect' and vice versa."""
    at = float(row.get("automation_catch_rate", 0) or 0)
    crit = float(row.get("critical_count", 0) or 0)
    reg = float(row.get("regression_rate", 0) or 0)
    original = detect
    if at >= 0.5:
        detect = min(detect, 2)
    elif at >= 0.1:
        detect = min(detect, 3)
    elif at < 0.1 and (crit >= 5 or reg > 0.1 or impact >= 4):
        detect = max(detect, 4)
    note = "" if detect == original else f" [clamped D {original}->{detect} by AT={at:.2f}]"
    return detect, note


def score_heuristic(row):
    mod = str(row.get("module","")).lower()
    crit = float(row.get("critical_count",0))
    repro = float(row.get("avg_repro_rate",0.5)) if "avg_repro_rate" in row.index else 0.5
    at_rate = float(row.get("automation_catch_rate",0))
    reg_rate = float(row.get("regression_rate",0))
    impact = IMPACT_OVERRIDES.get(mod, 3)
    if crit >= 10: impact = min(impact+1, 5)
    if crit >= 5 and impact < 4: impact = 4
    detect = 3
    if at_rate > 0.1: detect -= 1
    if at_rate > 0.3: detect -= 1
    if at_rate == 0: detect += 1
    if repro < 0.5: detect += 1
    if reg_rate > 0.15: detect += 1
    detect = max(1, min(5, detect))
    reasoning = f"I={impact}(domain,crit={int(crit)}) D={detect}(AT={at_rate:.0%},repro={repro:.0%})"
    return impact, detect, "heuristic", reasoning


def score_ollama(row, model="qwen3:4b"):
    mod = row.get("module", "Unknown")
    crit = float(row.get('critical_count', 0) or 0)
    total = float(row.get('total_bugs', 0) or 0)
    crit_share = (crit / total) if total > 0 else 0.0
    prompt = (
        "You are a QA risk analyst producing FMEA static-impact scores.\n"
        "Return ONLY a JSON object, no extra text.\n"
        'Schema: {"impact": int (1-5), "detectability": int (1-5), "reasoning": string}\n\n'
        "PURPOSE: Your score answers 'IF this module fails, how bad would it be, and how "
        "hard would the failure be to catch?' This is a STATIC severity score — recency / "
        "trend is handled separately downstream. The COUNTS BELOW ARE LIFETIME AGGREGATES, "
        "not recent activity. Score the module's intrinsic risk, not whether it's currently noisy.\n\n"
        "SCALES (higher = worse for BOTH):\n"
        "  impact: 1=cosmetic only · 2=minor friction · 3=feature-level disruption · "
        "4=major workflow breakage (lost work, broken core feature) · "
        "5=blocks core workflow (export fails, project save corrupts, crash on launch, data loss).\n"
        "  detectability: 1=EASY to catch (high automation coverage, deterministic repro) · "
        "3=partial coverage · 5=HARD to catch (no automation, flaky/intermittent, escapes to production).\n"
        "  → AutomationCatchRate is the strongest detectability signal. High automation = LOW detectability.\n\n"
        "RUBRIC ANCHORS:\n"
        "  AutomationCatchRate >= 0.5         → detectability 1-2\n"
        "  AutomationCatchRate 0.1–0.5        → detectability 2-3\n"
        "  AutomationCatchRate <  0.1         → detectability 4-5 (especially if CriticalShare>=0.10 or RegressionRate>0.10)\n"
        "  Module is core workflow (export, produce, project save/load, crash/launch path) → impact >= 4\n"
        "  Critical share (Critical/TotalBugs) >= 0.15 → impact at least 4 (this module produces a lot of severe bugs proportionally)\n"
        "  Critical share < 0.05 AND module is non-core (settings, tutorials, more) → impact <= 2\n\n"
        f"Module: {mod}\n"
        f"Category: {row.get('category', '')}\n"
        f"TotalBugs (lifetime): {int(total)}\n"
        f"Critical (lifetime): {int(crit)}\n"
        f"Major (lifetime): {int(row.get('major_count', 0) or 0)}\n"
        f"CriticalShare: {crit_share:.3f}  (Critical / TotalBugs)\n"
        f"RegressionRate: {row.get('regression_rate', 0):.3f}\n"
        f"AutomationCatchRate: {row.get('automation_catch_rate', 0):.3f}\n"
    )
    payload = json.dumps(with_thinking_disabled({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    })).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        raw = resp.read().decode(errors="replace")
        envelope = json.loads(raw)
        text = envelope.get("response", "") or envelope.get("thinking", "")
        obj = extract_json(text, expect="object")
        impact = int(obj.get("impact", 3))
        detect = int(obj.get("detectability", 3))
        reason = str(obj.get("reasoning", ""))
        impact = max(1, min(5, impact))
        detect = max(1, min(5, detect))
        detect, clamp_note = clamp_detectability_by_coverage(row, detect, impact)
        return impact, detect, "ollama", reason + clamp_note
    except Exception as e:
        print(f"    Ollama failed for {mod} (model={model}): {e}")
        return score_heuristic(row)


def score_openai(row):
    api_key = os.environ.get("OPENAI_API_KEY","")
    if not api_key:
        return score_heuristic(row)
    mod = row.get("module","Unknown")
    crit = float(row.get('critical_count', 0) or 0)
    total = float(row.get('total_bugs', 0) or 0)
    crit_share = (crit / total) if total > 0 else 0.0
    msgs = [
        {"role":"system","content":(
            "You produce FMEA static-impact scores. Counts given are LIFETIME aggregates, "
            "not recent activity — recency is handled downstream. Score the module's "
            "intrinsic risk: IF it fails, how bad and how hard to catch?\n"
            "impact 1-5 (5=blocks core workflow: export, save, crash on launch, data loss). "
            "detectability 1-5: 5=HARD to catch (low AutomationCatchRate, escapes to prod), "
            "1=EASY to catch (high AutomationCatchRate). High coverage → LOW detectability. "
            "CriticalShare>=0.15 → impact ≥4. Return only JSON."
        )},
        {"role":"user","content":(
            f"Module:{mod} BugsLifetime:{int(total)} CritLifetime:{int(crit)} "
            f"Major:{int(row.get('major_count',0) or 0)} "
            f"CriticalShare:{crit_share:.3f} "
            f"RegressionRate:{row.get('regression_rate',0):.3f} "
            f"AutomationCatchRate:{row.get('automation_catch_rate',0):.3f}\n"
            'Return JSON: {"impact":<1-5>,"detectability":<1-5>,"reasoning":"..."}'
        )},
    ]
    payload = json.dumps({
        "model":"gpt-4o-mini",
        "messages":msgs,
        "temperature":0.2
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type":"application/json",
            "Authorization":f"Bearer {api_key}",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        text = json.loads(resp.read().decode())["choices"][0]["message"]["content"]
        jm = re.search(r"\{[^}]+\}", text)
        if jm:
            obj = json.loads(jm.group())
            impact = max(1, min(5, int(obj.get("impact", 3))))
            detect = max(1, min(5, int(obj.get("detectability", 3))))
            detect, clamp_note = clamp_detectability_by_coverage(row, detect, impact)
            return impact, detect, "openai", str(obj.get("reasoning", "")) + clamp_note
    except Exception as e:
        print(f"    OpenAI failed for {mod}: {e}")
    return score_heuristic(row)


def assign_quadrant(score):
    if score > 90: return "P1 - Critical"
    if score >= 70: return "P2 - High"
    if score >= 50: return "P3 - Medium"
    return "P4 - Low"


def score_file(in_csv: str, out_csv: str, provider: str, model: str, verbose: bool = False):
    df = pd.read_csv(in_csv)
    print(f"\nScoring {len(df)} modules from {in_csv}")

    if "parsed_module" not in df.columns and "module" in df.columns:
        df["parsed_module"] = df["module"]

    already_scored: set = set()
    if os.path.exists(out_csv):
        try:
            prev = pd.read_csv(out_csv)

            prev_modules = set(prev["module"].dropna())
            curr_modules = set(df["module"].dropna())
            dropped = prev_modules - curr_modules
            if dropped:
                print(f"  WARNING: {len(dropped)} modules from previous run are missing in current input:")
                for m in sorted(dropped):
                    print(f"    - {m}")
                print("  These will be dropped from output. Re-check your fetch/parse pipeline.")

            done = prev[
                (prev["impact_score"] > 0) & (prev["detectability_score"] > 0)
            ]["module"]
            already_scored = set(done.dropna().tolist())

            if already_scored:
                unscored_in_current = [m for m in df["module"] if m not in already_scored]
                print(f"  Resuming: {len(already_scored)} modules in cache, "
                      f"{len(unscored_in_current)} remaining.")
                # ADD THIS — skip everything if nothing left to score
                if not unscored_in_current:
                    print("  All modules already scored. Nothing to do.")
                    return  # ← early exit
                score_cols = ["module", "impact_score", "detectability_score",
                              "scoring_method", "ai_reasoning"]
                score_cols = [c for c in score_cols if c in prev.columns]
                df = df.merge(prev[score_cols], on="module", how="left",
                              suffixes=("", "_prev"))
                for col in ["impact_score", "detectability_score",
                            "scoring_method", "ai_reasoning"]:
                    if col + "_prev" in df.columns:
                        df[col] = df[col + "_prev"].combine_first(df.get(col))
                        df.drop(columns=[col + "_prev"], inplace=True)
        except Exception as e:
            print(f"  Warning: could not load previous output ({e}) -- scoring from scratch.")
            already_scored = set()

    for col in ["impact_score", "detectability_score", "scoring_method", "ai_reasoning"]:
        if col not in df.columns:
            df[col] = None

    total       = len(df)
    skipped     = len(already_scored)
    to_score    = max(0, total - skipped)
    done_so_far = 0

    # Build a stable version label from the input filename
    basename   = os.path.basename(in_csv)
    ver_raw    = basename.replace("risk_register_", "").replace(".csv", "")
    ver_label  = ver_raw.replace("_", ".") if re.match(r"^\d", ver_raw) else ver_raw
    if provider == "ollama":
        provider_label = f"ollama/{model}"
    else:
        provider_label = provider
    # Fixed-width desc so the bar width stays constant
    desc_str = f"v{ver_label:<8} [{provider_label:<12}]"

    print(f"  {chr(9472) * 78}")
    print(f"  {'#':>4}  {'Module':<30}  {'I':>2} {'P':>2} {'D':>2}  {'Risk':>6}  {'Quadrant':<14}  Reasoning")
    print(f"  {chr(9472) * 78}")

    pbar = tqdm(total=to_score, desc=desc_str, unit="mod",
                file=sys.stderr, dynamic_ncols=True)
    for idx, row in df.iterrows():
        module_name = str(row.get("module", ""))
        if module_name in already_scored:
            continue

        if provider == "heuristic":
            r = score_heuristic(row)
        elif provider == "ollama":
            r = score_ollama(row, model=model)
        else:
            r = score_openai(row)
            time.sleep(0.5)

        df.at[idx, "impact_score"]        = r[0]
        df.at[idx, "detectability_score"] = r[1]
        df.at[idx, "scoring_method"]      = r[2]
        df.at[idx, "ai_reasoning"]        = r[3]
        done_so_far += 1

        prob_val        = float(row.get("probability_score_auto", 3))
        risk_inline     = r[0] * prob_val * r[1]
        quadrant_inline = assign_quadrant(risk_inline)
        reason_display  = r[3][:55] + "..." if len(r[3]) > 55 else r[3]

        pbar.update(1)
        # Fixed-width postfix: module padded to 20, quadrant padded to 14
        pbar.set_postfix_str(f"{module_name[:20]:<20} → {quadrant_inline:<14}")

        print(
            f"  [{done_so_far:>3}/{to_score}]  {module_name:<30}"
            f"  {r[0]:>2} {int(prob_val):>2} {r[1]:>2}"
            f"  {risk_inline:>6.0f}  {quadrant_inline:<14}"
            f"  {reason_display}"
        )
        if verbose:
            print(f"           Full reasoning: {r[3]}")

        if done_so_far > 0 and done_so_far % 10 == 0:
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    pbar.close()
    prob = df["probability_score_auto"] if "probability_score_auto" in df.columns else 3
    df["risk_score_final"] = df["impact_score"].astype(float) * prob * df["detectability_score"].astype(float)
    df["quadrant"] = df["risk_score_final"].apply(assign_quadrant)
    df = df.sort_values("risk_score_final", ascending=False)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n  {'=' * 78}")
    print(f"  RISK SCORING SUMMARY  --  {os.path.basename(in_csv)}")
    print(f"  {'=' * 78}")

    for q_label in ["P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Low"]:
        qdf = df[df["quadrant"] == q_label].sort_values("risk_score_final", ascending=False)
        if qdf.empty:
            continue
        print(f"\n  {q_label}  ({len(qdf)} modules)")
        print(f"  {'Module':<30}  {'Score':>6}  I  P  D  Reasoning")
        print(f"  {'-' * 30}  {'------':>6}  -  -  -  ---------")
        for _, r in qdf.iterrows():
            p = int(r.get("probability_score_auto", 3))
            reason = str(r.get("ai_reasoning", ""))
            reason_short = reason[:55] + "..." if len(reason) > 55 else reason
            print(
                f"  {str(r['module']):<30}  {r['risk_score_final']:>6.0f}"
                f"  {int(r['impact_score'])}  {p}  {int(r['detectability_score'])}"
                f"  {reason_short}"
            )

    total_p1 = (df["quadrant"] == "P1 - Critical").sum()
    total_p2 = (df["quadrant"] == "P2 - High").sum()
    method_counts = df["scoring_method"].value_counts()
    method_str = "  |  ".join(f"{m}: {c}" for m, c in method_counts.items())
    print(f"\n  {'=' * 78}")
    print(f"  Modules needing immediate attention (P1+P2): {total_p1 + total_p2} / {total}")
    print(f"  Scoring method breakdown: {method_str}")
    if skipped:
        print(f"  Resumed {skipped} previously scored, scored {done_so_far} new.")
    print(f"  Saved to: {out_csv}")
    print(f"  {'=' * 78}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("output_csv")
    p.add_argument("--provider", choices=["heuristic", "ollama", "openai"], default="heuristic")
    p.add_argument("--model", default="qwen3:4b")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print full AI reasoning per module during scoring")
    a = p.parse_args()

    base_dir     = os.path.dirname(a.input_csv)  or "."
    base_out_dir = os.path.dirname(a.output_csv) or "."

    score_file(a.input_csv, a.output_csv, a.provider, a.model, verbose=a.verbose)

    ver_dir        = os.path.join(base_dir,     "risk_register_versions")
    scored_ver_dir = os.path.join(base_out_dir, "risk_register_versions")
    os.makedirs(scored_ver_dir, exist_ok=True)

    pattern   = os.path.join(ver_dir, "risk_register_*.csv")
    all_files = glob.glob(pattern)

    # FIX 1: exclude already-scored files from being used as inputs
    ver_files = sorted([
        f for f in all_files
        if not os.path.basename(f).startswith("risk_register_scored_")
    ])

    if ver_files:
        print(f"\nScoring {len(ver_files)} per-version files from {ver_dir}/")
        print(f"Scored outputs -> {scored_ver_dir}/")
    else:
        print(f"\nNo per-version files found in {ver_dir}/ -- skipping per-version scoring.")
        print("Re-run compute_risk_scores.py to generate them.")

    for path in ver_files:
        name = os.path.basename(path)
        # FIX 2: normalize version separator to underscores (1.0.0 → 1_0_0)
        ver_part = name.replace("risk_register_", "").replace(".csv", "").replace(".", "_")
        out_name = f"risk_register_scored_{ver_part}.csv"
        out_path = os.path.join(scored_ver_dir, out_name)
        score_file(path, out_path, a.provider, a.model, verbose=a.verbose)


if __name__ == "__main__":
    main()
