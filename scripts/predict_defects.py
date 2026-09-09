#!/usr/bin/env python3
"""PDR-I Defect Prediction v6.0

Changes from v5.0
─────────────────
Change 20 — Poisson HistGradientBoostingRegressor replaces GradientBoostingRegressor
• Bug counts are count data — Poisson loss is the correct generative model.
• HistGBR is faster, handles NaN natively, and compresses probability mass
  at zero for modules with rare bugs.

Change 21 — S1 class imbalance correction in risk classifier
• GradientBoostingClassifier now trains with sample_weight inversely proportional
  to class frequency, correcting the ~3.6% S1 positive rate bias.

Change 22 — Ollama semantic embeddings for scenario clustering
• _extract_description_patterns() now tries mxbai-embed-large via the Ollama
  /api/embed batch endpoint for cosine distances before falling back to TF-IDF.
• Semantic embeddings cluster "freezes/hangs/unresponsive" together; TF-IDF
  treats them as distinct.

Change 23 — Holt's double exponential smoothing trend forecast
• train_trend_forecast() adds a trend_forecast column using Holt's method
  (alpha=0.3 level, beta=0.1 trend).  No external deps; forecasts one
  version ahead per module.

Change 24 — Bayesian Beta-Binomial shrinkage on risk probability
• apply_bayesian_shrinkage() blends the ML classifier's P(S1) with a
  Beta-Binomial posterior, shrinking sparse modules toward the global S1
  rate (prior_strength=5 equivalent observations).

Changes from v4.0
─────────────────
Change 18 — Scenario-based bug forecasting
• NEW generate_bug_scenarios() orchestrator produces concrete, testable bug
  scenario predictions per module (e.g. "Crash when importing large .mov files
  on macOS Sonoma") instead of just counts.
• _extract_description_patterns() — heuristic (non-AI) scenario extractor that
  clusters recent bug descriptions with AgglomerativeClustering and picks
  representative descriptions as predicted scenarios.
• _generate_scenarios_ai_ollama() / _generate_scenarios_ai_claude() — AI-powered
  scenario generators that prompt the model for 3-5 concrete bug scenarios per
  module, grounded in historical patterns.
• Output: data/predictions/<stem>_predictions_by_scenario.csv with columns:
      module, risk_level, predicted_build, scenario_rank, scenario_text,
      confidence, source_bug_examples, supporting_categories, leading_signal
• generate_focus_summary() now appends top scenario predictions per module.

All v4.0 behaviour is unchanged.

Output paths:
  data/predictions/<stem>_predictions.csv
  data/predictions/<stem>_predictions_by_category.csv  (always)
  data/predictions/<stem>_predictions_by_scenario.csv  ← NEW (always)
  data/predictions/<stem>_predictions_by_cluster.csv   (when --cluster-csv given)
  data/predictions/<stem>_predictions_importance.csv
  data/predictions/<stem>_predictions_leading_indicators.csv
  data/predictions/<stem>_predictions_focus_summary.txt

Usage:
  python predict_defects.py <input_csv>
  python predict_defects.py <input_csv> --cluster-csv data/clusters/ecl_parsed_clustered.csv
  python predict_defects.py <input_csv> --provider ollama --model qwen3:4b
  python predict_defects.py <input_csv> --provider claude --api-key <key>
  python predict_defects.py <input_csv> --no-ai
  python predict_defects.py <input_csv> --scored-csv <path>
"""
import os
import re
import sys
import json
import urllib.request
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
warnings.filterwarnings("ignore")

try:
    from parse_ecl_export import (
        FIXED_BUG_INITIAL, FIXED_BUG_FLOOR, FIXED_BUG_DECAY_SPAN,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from parse_ecl_export import (
        FIXED_BUG_INITIAL, FIXED_BUG_FLOOR, FIXED_BUG_DECAY_SPAN,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PREDICTION_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           PDR-I DEFECT PREDICTION v5.0 — HOW TO READ THIS OUTPUT           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  WHAT THIS TOOL PREDICTS                                                    ║
║  ───────────────────────                                                    ║
║  For each module, the model predicts:                                       ║
║   1. A risk level (Low / Medium / High / Critical) for the NEXT version.   ║
║   2. WHAT TYPES of bugs are expected (by QA category).                     ║
║   3. CONCRETE BUG SCENARIOS — specific, testable predictions like real     ║
║      bug titles, grounded in recurring historical patterns.                 ║
║                                                                             ║
║  SCENARIO PREDICTION OUTPUT (_predictions_by_scenario.csv)                 ║
║  ─────────────────────────────────────────────────────────                  ║
║  module              Which module this scenario applies to                   ║
║  scenario_text       A concrete, testable bug prediction                    ║
║  confidence          high (3+ recurring bugs) / medium (2) / low (1)       ║
║  source_bug_examples Real bug descriptions this scenario is based on        ║
║  supporting_categories  QA categories that support this scenario            ║
║                                                                             ║
║  HOW TO ACT ON IT                                                           ║
║  ────────────────                                                           ║
║  Critical/High modules → add predicted scenarios to your test plan.        ║
║  Read '_by_scenario.csv' for WHAT SPECIFIC BUGS to test for.               ║
║  Read '_by_category.csv' for the category-level distribution.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

TFIDF_TOP_N  = 40
TFIDF_PREFIX = "tfidf_"
OLLAMA_BASE  = "http://localhost:11434"

from ollama_compat import (
    extract_json, response_text, strip_meta_preamble, with_thinking_disabled,
)

SCENARIO_TOP_MODULES    = 10
SCENARIOS_PER_MODULE    = 5
SCENARIO_HISTORY_BUILDS = 5

# Quadrant-aware scenario selection — keeps "What to Test" aligned with the
# FMEA heatmap. Any P1 module with enough data is always included, regardless
# of its ML priority_score rank, up to SCENARIO_HARD_CAP.
SCENARIO_HARD_CAP = 20
SCENARIO_FORCE_P1 = True
SCENARIO_FORCE_P2 = True
QUADRANT_BONUS = {
    "P1": 1.00,
    "P2": 0.60,
    "P3": 0.20,
    "P4": 0.00,
}

# Stability signal thresholds — a module is "stable_mature" only when it has
# enough history, has been quiet of recent S1s for a while, AND has no recent
# bug volume. Lets the classifier tell silent P1 cores apart from unknowns.
STABLE_HISTORY_MIN = 10  # builds of module history required
STABLE_CRIT_GAP    = 8   # builds since last S1 required

_FEATURE_LABELS = {
    "crit_1":  "critical-bug momentum (last version)",
    "crit_2":  "critical-bug momentum (last 2 versions)",
    "crit_3":  "critical-bug momentum (last 3 versions)",
    "bugs_1":  "recent bug-count momentum (last version)",
    "bugs_2":  "recent bug-count momentum (last 2 versions)",
    "bugs_3":  "recent bug-count momentum (last 3 versions)",
    "sev_1":   "severity-weighted momentum (last version)",
    "sev_2":   "severity-weighted momentum (last 2 versions)",
    "sev_3":   "severity-weighted momentum (last 3 versions)",
    "trend":   "upward bug-count trend",
    # Change 11 — new features
    "severity_escalation":     "severity escalation (negative = worsening toward S1)",
    "builds_since_last_crit":  "versions elapsed since last critical bug",
    # Change 12 — cluster features
    "cluster_entropy_2":       "bug-theme diversity index (last 2 versions)",
    "cluster_entropy_3":       "bug-theme diversity index (last 3 versions)",
    "top_cluster_velocity":    "fastest-growing bug theme velocity",
    # Change 19 — new features
    "crit_ratio":             "proportion of critical bugs (S1 / total)",
    "new_module":             "new module (first appeared in recent versions)",
    "cross_module_spike":     "correlated spike — related modules also spiking",
    "total_historical_bugs":  "total historical bug count (module maturity)",
    # Impact-aware features — make the classifier FMEA-aware
    "stable_mature":          "mature & quiet (stable core modules)",
    "impact_bug_ratio":       "impact per unit of recent activity",
    "impact_weighted_spike":  "cross-module spike weighted by FMEA impact",
    # existing optional features
    "repro_rate":             "high reproduce rate (consistently reproducible bugs)",
    "impact_score":           "domain impact score (I×P×D scorer)",
    "detectability_score":    "test detectability score (I×P×D scorer)",
    "probability_score_auto": "historical defect probability (I×P×D scorer)",
    "risk_score_final":       "composite risk score I×P×D",
}

# Cluster feature columns — excluded from per-module leading_signal search
# because they are more structural signals than per-version time-series signals.
_CLUSTER_FEATURE_COLS = {"cluster_entropy_2", "cluster_entropy_3", "top_cluster_velocity"}

_RISK_ADVICE = {
    "Critical": "Mandatory — add to test suite for every version. Focus on crash and data-loss scenarios.",
    "High":     "Priority — run core functional tests every version and watch for regressions.",
    "Medium":   "Standard — include in sprint regression pass.",
    "Low":      "Monitor — include in release-candidate pass.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Change 15 — QA bug category taxonomy
# ─────────────────────────────────────────────────────────────────────────────

# Each key is a human-readable QA category; values are lowercase keyword fragments
# that, when found in a bug description, score a point toward that category.
# Priority order (highest first) matters for tie-breaking.
BUG_CATEGORIES = {
    "Crash / Stability": [
        "crash", "exception", "fatal", "abort", "force close", "forcibly closed",
        "terminate", "stop working", "out of memory", "oom", "memory leak",
        "deadlock", "unresponsive", "hang", "freeze", "hung", "blue screen",
        "application error", "access violation",
    ],
    "Feature not working as intended": [
        "not work", "doesn't work", "does not work", "no longer work",
        "incorrect behavior", "wrong result", "not function", "broken",
        "fail", "failure", "unexpected", "not as expected", "supposed to",
        "incorrect", "mismatch", "should be", "should not", "not correct",
        "behaves", "regression", "stopped working",
    ],
    "UI / Display problem": [
        "display", "layout", "alignment", "visual", "render", "style",
        "color", "colour", "font", "icon", "button", "overlap", "cut off",
        "cutoff", "truncat", "clipping", "overflow", "ui ", " ui,", "interface",
        "appearance", "look and feel", "dark mode", "light mode", "theme",
        "blank screen", "blank page", "not show", "missing icon", "wrong icon",
        "z-order", "z order", "overlapping", "misaligned", "pixel",
    ],
    "UX / Usability problem": [
        "usability", "confusing", "confus", "navigation", "workflow",
        "hard to find", "not intuitive", "difficult to use", "frustrat",
        "response time", "slow", "lag ", "no feedback", "unclear", "unintuitive",
        "too many clicks", "hard to", "not obvious", "awkward",
    ],
    "Translation / Localization": [
        "translation", "localization", "localisation", "missing string",
        "wrong string", "language", "missing text", "garbled", "encoding",
        "locale", "region", "l10n", "i18n", "chinese", "japanese", "korean",
        "german", "french", "spanish", "traditional chinese", "simplified chinese",
        "zh-tw", "zh-cn", " de ", " fr ", " ja ", " ko ",
    ],
    "Data / File / Sync issue": [
        "data", "corrupt", "import", "export", "sync", "cannot save",
        "cannot load", "cannot open", "wrong data", "missing data",
        "incorrect data", "file", "read error", "write error", "database",
        "cloud", "upload", "download", "transfer", "backup",
    ],
}

_CATEGORY_ORDER = list(BUG_CATEGORIES.keys())  # priority: first wins on tie


def classify_bug_category(text: str) -> str:
    """Assign a bug description to the highest-scoring QA category.

    Scoring: count keyword hits per category; if tied, first in _CATEGORY_ORDER wins.
    Falls back to 'Feature not working as intended' when no keywords match.
    """
    if not isinstance(text, str) or len(text.strip()) < 5:
        return "Feature not working as intended"
    t = text.lower()
    scores: dict = {}
    for cat, keywords in BUG_CATEGORIES.items():
        scores[cat] = sum(1 for kw in keywords if kw in t)

    best_score = max(scores.values())
    if best_score == 0:
        return "Feature not working as intended"

    # Among tied categories keep the one that appears first in _CATEGORY_ORDER
    for cat in _CATEGORY_ORDER:
        if scores[cat] == best_score:
            return cat
    return "Feature not working as intended"


# ─────────────────────────────────────────────────────────────────────────────
# Risk feature loader  (unchanged from v2.6)
# ─────────────────────────────────────────────────────────────────────────────

def load_risk_features(scored_csv: str) -> "pd.DataFrame | None":
    try:
        rdf = pd.read_csv(scored_csv)
    except Exception as e:
        print(f"  WARNING: could not load scored CSV ({e}) — running without risk features.")
        return None

    mod_col = ("parsed_module" if "parsed_module" in rdf.columns
               else "module" if "module" in rdf.columns else None)
    if mod_col is None:
        print("  WARNING: scored CSV has no module/parsed_module column — skipping.")
        return None

    risk_cols = [c for c in [
        "impact_score", "detectability_score",
        "probability_score_auto", "risk_score_final",
    ] if c in rdf.columns]

    if not risk_cols:
        print("  WARNING: scored CSV has none of the expected risk columns — skipping.")
        return None

    out = rdf[[mod_col] + risk_cols].copy()
    out = out.rename(columns={mod_col: "module"})
    out = out.drop_duplicates(subset=["module"]).set_index("module")
    print(f"  Loaded risk features for {len(out)} modules from {scored_csv}")
    return out


def load_risk_metadata(scored_csv: str) -> "pd.DataFrame | None":
    """Load per-module metadata (quadrant label, regression rate, open count)
    for use in priority_score blending and the scenario explanation column.
    Kept separate from `load_risk_features` because these columns include
    non-numeric fields that must NOT flow into the regression feature matrix.
    """
    try:
        rdf = pd.read_csv(scored_csv)
    except Exception:
        return None
    mod_col = ("parsed_module" if "parsed_module" in rdf.columns
               else "module" if "module" in rdf.columns else None)
    if mod_col is None:
        return None
    meta_cols = [c for c in [
        "quadrant", "regression_rate", "open_count",
        "total_bugs", "critical_count",
    ] if c in rdf.columns]
    if not meta_cols:
        return None
    out = rdf[[mod_col] + meta_cols].copy()
    out = out.rename(columns={mod_col: "module"})
    out = out.drop_duplicates(subset=["module"]).set_index("module")
    return out


def load_module_cluster_velocity(cluster_csv: str,
                                  inp_dir: Path) -> "pd.DataFrame | None":
    """Compute per-module max cluster velocity from the cluster pipeline outputs.

    Joins cluster assignment (module, cluster_id from `*_clustered.csv`) with
    cluster velocity (`cluster_id → cluster_velocity_ratio` from
    `*_cluster_summary.csv`) to produce a DataFrame indexed by module with the
    module's hottest-cluster velocity. Rows where every joined cluster has an
    undefined velocity (insufficient history) stay NaN rather than being
    silently coerced to 1.0.
    """
    try:
        cdf = pd.read_csv(cluster_csv)
    except Exception as e:
        print(f"  WARNING: could not load cluster CSV ({e}) — skipping module velocity.")
        return None

    mod_col = ("parsed_module" if "parsed_module" in cdf.columns
               else "module" if "module" in cdf.columns else None)
    if mod_col is None or "cluster_id" not in cdf.columns:
        return None

    # Find summary file alongside the clustered CSV
    stem = Path(cluster_csv).stem  # e.g. ecl_parsed_clustered
    summary_candidates = [
        Path(cluster_csv).parent / (stem.replace("_clustered", "_cluster_summary") + ".csv"),
        inp_dir / "clusters" / (stem.replace("_clustered", "_cluster_summary") + ".csv"),
        inp_dir / "ecl_parsed_cluster_summary.csv",
    ]
    summary_path = next((p for p in summary_candidates if p.exists()), None)
    if summary_path is None:
        print("  NOTE: cluster summary CSV not found — module velocity unavailable.")
        return None

    try:
        sdf = pd.read_csv(summary_path)
    except Exception as e:
        print(f"  WARNING: could not load cluster summary ({e}).")
        return None

    if "cluster_velocity_ratio" not in sdf.columns or "cluster_id" not in sdf.columns:
        return None

    mapping = cdf[[mod_col, "cluster_id"]].dropna()
    mapping = mapping[mapping["cluster_id"] != -1]
    mapping["cluster_id"] = pd.to_numeric(mapping["cluster_id"], errors="coerce")
    sdf["cluster_id"] = pd.to_numeric(sdf["cluster_id"], errors="coerce")
    sdf["cluster_velocity_ratio"] = pd.to_numeric(sdf["cluster_velocity_ratio"], errors="coerce")

    merged = mapping.merge(sdf[["cluster_id", "cluster_velocity_ratio"]],
                            on="cluster_id", how="left")
    # Per module take the max velocity across its clusters (skipping NaN).
    # If every cluster is NaN the result is NaN, which is what we want.
    grouped = merged.groupby(mod_col)["cluster_velocity_ratio"].max(numeric_only=True)
    out = grouped.to_frame("module_cluster_velocity")
    out.index.name = "module"
    print(f"  Loaded module cluster velocity for {out['module_cluster_velocity'].notna().sum()}"
          f" / {len(out)} modules")
    return out


def ollama_is_reachable(timeout: float = 2.0) -> bool:
    """Ping Ollama's /api/tags endpoint. Returns True if reachable.

    Used to decide whether to attempt AI scenario generation or fall back to
    the heuristic path. Cached via module-level flag so we only hit the network
    once per run.
    """
    global _OLLAMA_REACHABLE
    if _OLLAMA_REACHABLE is not None:
        return _OLLAMA_REACHABLE
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags",
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _OLLAMA_REACHABLE = resp.status == 200
    except Exception:
        _OLLAMA_REACHABLE = False
    return _OLLAMA_REACHABLE


_OLLAMA_REACHABLE: "bool | None" = None


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF text features  (updated v6.0 — rolling version window)
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_features(orig_df: pd.DataFrame,
                          mod_col: str = "parsed_module",
                          text_col: str = "parsed_description",
                          version_col: str = "parsed_version",
                          top_n: int = TFIDF_TOP_N,
                          version_window: int = 3) -> "pd.DataFrame | None":
    """Build TF-IDF text features using only the last `version_window` versions.

    Each version spans 1-2 months, so 3 versions ≈ 3-6 months of recent bug
    descriptions.  This makes the text signal temporal: it captures what a
    module's bugs *currently* look like rather than aggregating all-time history.
    Falls back to all-time aggregation if version data is absent.
    """
    if text_col not in orig_df.columns or mod_col not in orig_df.columns:
        print(f"  NOTE: '{text_col}' column not found — skipping text features.")
        return None

    working = orig_df.copy()

    # Filter to recent versions if version column exists.
    # Sort by semantic version number (numeric tuple), filtering out
    # versions with <10 bugs to exclude typos/outliers (e.g. "152.0").
    if version_col in working.columns:
        ver_bug_counts = working[version_col].value_counts()
        valid_versions = ver_bug_counts[ver_bug_counts >= 10].index.tolist()

        def _version_key(v):
            try:
                return tuple(int(p) for p in str(v).split("."))
            except (ValueError, TypeError):
                return (0,)

        sorted_versions = sorted(valid_versions, key=_version_key)
        recent_versions = sorted_versions[-version_window:]
        before = len(working)
        working = working[working[version_col].isin(recent_versions)]
        print(f"  TF-IDF: windowed to last {version_window} versions "
              f"({', '.join(str(v) for v in recent_versions)}) — "
              f"{len(working)}/{before} bugs retained.")
    else:
        print(f"  TF-IDF: '{version_col}' not found — using all-time descriptions.")

    mod_docs = (
        working[[mod_col, text_col]]
        .dropna(subset=[mod_col, text_col])
        .groupby(mod_col)[text_col]
        .apply(lambda texts: " ".join(str(t) for t in texts if len(str(t)) > 5))
        .reset_index()
    )
    mod_docs = mod_docs[mod_docs[text_col].str.len() > 20]

    if len(mod_docs) < 5:
        print("  WARNING: too few modules with descriptions for TF-IDF — skipping.")
        return None

    for kwargs in [
        dict(max_features=top_n, stop_words="english", ngram_range=(1, 2),
             min_df=2, max_df=0.85, sublinear_tf=True),
        dict(max_features=top_n, stop_words="english", ngram_range=(1, 1),
             min_df=1, max_df=0.95, sublinear_tf=True),
    ]:
        try:
            vec = TfidfVectorizer(**kwargs)
            X = vec.fit_transform(mod_docs[text_col])
            terms = [f"{TFIDF_PREFIX}{t}" for t in vec.get_feature_names_out()]
            tfidf_df = pd.DataFrame(X.toarray(), columns=terms)
            tfidf_df.insert(0, "module", mod_docs[mod_col].values)
            print(f"  TF-IDF: {len(terms)} text features across {len(mod_docs)} modules.")
            return tfidf_df
        except ValueError:
            continue

    print("  WARNING: TF-IDF vectorization failed — skipping text features.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# NEW v3.0 — Cluster feature builder
# ─────────────────────────────────────────────────────────────────────────────

def load_cluster_features(cluster_csv: str,
                           orig_df: pd.DataFrame,
                           build_col: str = "Build#",
                           mod_col: str = "parsed_module") -> "pd.DataFrame | None":
    """
    Read the clustered bug CSV produced by cluster_bugs.py and build per-module
    per-version cluster-derived features:

      cluster_entropy_2   Shannon entropy of bug-theme distribution, last 2 versions
      cluster_entropy_3   Shannon entropy of bug-theme distribution, last 3 versions
      top_cluster_velocity  Growth ratio of the dominant theme over last 3 versions

    Returns a DataFrame keyed on (module, build) for merging into fdf,
    or None if the cluster CSV cannot be used.
    """
    try:
        cdf = pd.read_csv(cluster_csv)
    except Exception as e:
        print(f"  WARNING: could not load cluster CSV ({e}) — skipping cluster features.")
        return None

    if "cluster_id" not in cdf.columns:
        print("  NOTE: cluster CSV has no cluster_id column — skipping cluster features.")
        return None

    # Join cluster assignments back onto the raw bug data
    if "BugCode" in cdf.columns and "BugCode" in orig_df.columns:
        bug_cols = [c for c in [build_col, mod_col, "BugCode"] if c in orig_df.columns]
        merged = orig_df[bug_cols].merge(
            cdf[["BugCode", "cluster_id"]], on="BugCode", how="left"
        )
    elif mod_col in cdf.columns and build_col in cdf.columns:
        merged = orig_df[[build_col, mod_col]].merge(
            cdf[[build_col, mod_col, "cluster_id"]], on=[build_col, mod_col], how="left"
        )
    else:
        print("  NOTE: No shared key between cluster CSV and bug data — skipping cluster features.")
        return None

    merged[build_col] = pd.to_numeric(merged[build_col], errors="coerce")
    merged = merged.dropna(subset=[build_col, mod_col])
    merged[build_col] = merged[build_col].astype(int)
    merged["cluster_id"] = merged["cluster_id"].fillna(-1).astype(int)

    rows = []
    for mod, mod_bugs in merged.groupby(mod_col):
        builds_sorted = sorted(mod_bugs[build_col].unique())
        if len(builds_sorted) < 5:
            continue

        for idx in range(5, len(builds_sorted)):
            build_num = builds_sorted[idx]
            r = {"module": mod, "build": build_num}

            for lag in [2, 3]:
                prev_builds = builds_sorted[max(0, idx - lag):idx]
                window_bugs = mod_bugs[
                    mod_bugs[build_col].isin(prev_builds) & (mod_bugs["cluster_id"] != -1)
                ]
                if len(window_bugs) == 0:
                    r[f"cluster_entropy_{lag}"] = 0.0
                else:
                    counts = window_bugs["cluster_id"].value_counts().values.astype(float)
                    probs  = counts / counts.sum()
                    r[f"cluster_entropy_{lag}"] = round(
                        float(-np.sum(probs * np.log2(probs + 1e-12))), 3
                    )

            # Velocity of the dominant cluster: recent 3 versions vs prior 3
            if idx >= 6:
                recent_b = builds_sorted[idx - 3:idx]
                prior_b  = builds_sorted[idx - 6:idx - 3]
                r_bugs = mod_bugs[mod_bugs[build_col].isin(recent_b) & (mod_bugs["cluster_id"] != -1)]
                p_bugs = mod_bugs[mod_bugs[build_col].isin(prior_b)  & (mod_bugs["cluster_id"] != -1)]
                if len(r_bugs) > 0:
                    top_cid = r_bugs["cluster_id"].value_counts().index[0]
                    rc_n    = int((r_bugs["cluster_id"] == top_cid).sum())
                    pc_n    = int((p_bugs["cluster_id"] == top_cid).sum()) if len(p_bugs) > 0 else 0
                    r["top_cluster_velocity"] = round(rc_n / max(pc_n, 1), 2)
                else:
                    r["top_cluster_velocity"] = 1.0
            else:
                r["top_cluster_velocity"] = 1.0

            rows.append(r)

    if not rows:
        print("  NOTE: No cluster features could be built (insufficient history per module).")
        return None

    cf = pd.DataFrame(rows)
    print(f"  Cluster features: {len(cf)} rows  "
          f"(cluster_entropy_2/3, top_cluster_velocity)")
    return cf


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix builder  (updated for v3.0)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df, build_col="Build#", mod_col="parsed_module",
                   risk_features=None, tfidf_features=None,
                   cluster_features=None):
    df = df.copy()
    total_before = len(df)
    df[build_col] = pd.to_numeric(df[build_col], errors="coerce")
    non_numeric = df[build_col].isna().sum()
    df = df.dropna(subset=[build_col, mod_col])
    df[build_col] = df[build_col].astype(int)
    if non_numeric > 0:
        print(f"  WARNING: {non_numeric}/{total_before} rows dropped — "
              f"'{build_col}' could not be parsed as numbers.")

    # Status: drop invalid entirely; keep open + closed for per-target-build
    # time-decay below. status_weight == 0.5 marks a closed bug whose weight
    # decays linearly from FIXED_BUG_INITIAL down to FIXED_BUG_FLOOR over
    # FIXED_BUG_DECAY_SPAN builds since close, so old fixed bugs fade without
    # fully vanishing from the history.
    if "status_weight" in df.columns:
        df["status_weight"] = pd.to_numeric(df["status_weight"], errors="coerce").fillna(1.0)
        invalid_count = int((df["status_weight"] == 0.0).sum())
        closed_count  = int((df["status_weight"] == 0.5).sum())
        df = df[df["status_weight"] > 0].copy()
        df["_is_closed"] = (df["status_weight"] == 0.5).values
        print(f"  Status weighting: {invalid_count} invalid bugs excluded; "
              f"{closed_count} closed bugs time-decayed "
              f"(initial={FIXED_BUG_INITIAL}, floor={FIXED_BUG_FLOOR}, "
              f"span={FIXED_BUG_DECAY_SPAN} builds).")
    else:
        df["_is_closed"] = False

    # Close-build column for time-decay; absence falls back to flat weight.
    close_col = next(
        (c for c in df.columns if c.lower().strip() in ("close build#", "close build")),
        None,
    )
    if close_col:
        df["_close_build_num"] = pd.to_numeric(df[close_col], errors="coerce")
    else:
        df["_close_build_num"] = np.nan
        if df["_is_closed"].any():
            print(f"  NOTE: 'Close Build#' column missing — closed bugs use flat "
                  f"{FIXED_BUG_INITIAL}x weight (no decay).")

    if "severity_num" not in df.columns:
        df["severity_num"] = 3
    df["severity_num"] = pd.to_numeric(df["severity_num"], errors="coerce").fillna(3)
    if "severity_weight" not in df.columns:
        df["severity_weight"] = 1.0
    df["severity_weight"] = pd.to_numeric(df["severity_weight"], errors="coerce").fillna(1.0)

    total_distinct_builds = int(df[build_col].nunique())

    rows = []
    for mod, mdf in df.groupby(mod_col):
        mdf = mdf.sort_values(build_col).reset_index(drop=True)
        module_builds = np.sort(mdf[build_col].unique())
        if len(module_builds) < 6:
            continue

        bug_build    = mdf[build_col].values.astype(float)
        bug_close    = mdf["_close_build_num"].values.astype(float)
        bug_isclosed = mdf["_is_closed"].values.astype(bool)
        bug_sev_num  = mdf["severity_num"].values.astype(float)
        bug_sev_w    = mdf["severity_weight"].values.astype(float)
        bug_crit     = (bug_sev_num == 1).astype(float)
        bug_major    = (bug_sev_num == 2).astype(float)

        def weights_at(V: float) -> np.ndarray:
            w = np.ones(len(mdf), dtype=float)
            if not bug_isclosed.any():
                return w
            known = bug_isclosed & np.isfinite(bug_close) & (bug_close <= V)
            if known.any():
                age = V - bug_close[known]
                span = max(1, FIXED_BUG_DECAY_SPAN)
                dec = FIXED_BUG_INITIAL - (FIXED_BUG_INITIAL - FIXED_BUG_FLOOR) * (age / span)
                w[known] = np.maximum(dec, FIXED_BUG_FLOOR)
            fallback = bug_isclosed & ~np.isfinite(bug_close)
            w[fallback] = FIXED_BUG_INITIAL
            return w

        def per_build_sum(mask_by_bug: np.ndarray, w: np.ndarray, build: float) -> float:
            m = mask_by_bug & (bug_build == build)
            return float(w[m].sum())

        for i in range(5, len(module_builds)):
            V = float(module_builds[i])
            w = weights_at(V)

            target_mask = (bug_build == V)
            target_val  = float(w[target_mask].sum())

            r = {"module": mod, "build": int(V), "target": target_val}

            for lag in [1, 2, 3]:
                prev_builds = module_builds[max(0, i - lag):i]
                lag_mask = np.isin(bug_build, prev_builds)
                r[f"bugs_{lag}"] = float(w[lag_mask].sum())
                r[f"sev_{lag}"]  = float((w[lag_mask] * bug_sev_w[lag_mask]).sum())
                r[f"crit_{lag}"] = float((w[lag_mask] * bug_crit[lag_mask]).sum())

            three_prev = module_builds[max(0, i - 3):i]
            per_build_counts = [per_build_sum(np.ones_like(bug_build, dtype=bool), w, float(b))
                                for b in three_prev]
            r["trend"] = (per_build_counts[-1] - per_build_counts[0]) if len(per_build_counts) >= 2 else 0

            # avg_sev (unweighted) at last active build vs the three before that
            last_mask = (bug_build == float(module_builds[i - 1]))
            sv_last = float(bug_sev_num[last_mask].mean()) if last_mask.any() else 3.0
            prior_builds = module_builds[max(0, i - 4):i - 1]
            prior_mask = np.isin(bug_build, prior_builds)
            sv_prior = float(bug_sev_num[prior_mask].mean()) if prior_mask.any() else sv_last
            r["severity_escalation"] = round(float(sv_last - sv_prior), 3)

            last_crit_pos = None
            for p in range(i - 1, -1, -1):
                b = float(module_builds[p])
                b_mask = (bug_build == b)
                if float((w[b_mask] * bug_crit[b_mask]).sum()) > 0:
                    last_crit_pos = p
                    break
            r["builds_since_last_crit"] = (i - last_crit_pos - 1) if last_crit_pos is not None else i

            three_mask = np.isin(bug_build, module_builds[max(0, i - 3):i])
            total_w3 = float(w[three_mask].sum())
            crit_w3  = float((w[three_mask] * bug_crit[three_mask]).sum())
            r["crit_ratio"] = round(float(crit_w3 / max(total_w3, 1)), 3)

            r["new_module"] = 1 if len(module_builds) <= max(5, int(total_distinct_builds * 0.2)) else 0

            # stable_mature: mature module (enough history) that has been quiet
            # for a while (no recent S1s + no recent bugs). Lets the classifier
            # distinguish "stable core" from "dormant unknown" — both silent,
            # very different risk.
            r["stable_mature"] = int(
                i >= STABLE_HISTORY_MIN
                and r["builds_since_last_crit"] >= STABLE_CRIT_GAP
                and r["bugs_3"] == 0
            )

            hist_mask = (bug_build < V)
            r["total_historical_bugs"] = int(round(float(w[hist_mask].sum())))

            rows.append(r)

    fdf = pd.DataFrame(rows)

    # Change 19 — cross_module_spike: for each (module, version), compute
    # how many OTHER modules also spiked in that version.  A "spike" is a
    # version where bug_count > module's rolling 3-version mean.
    if not fdf.empty:
        # Pre-compute per-module rolling mean bug count
        fdf = fdf.sort_values(["module", "build"])
        fdf["_rolling_mean"] = (
            fdf.groupby("module")["target"]
            .transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
        )
        fdf["_is_spike"] = (fdf["target"] > fdf["_rolling_mean"]).astype(int)
        # Count how many modules spiked per version (excluding self)
        spikes_per_build = fdf.groupby("build")["_is_spike"].sum()
        fdf["cross_module_spike"] = (
            fdf["build"].map(spikes_per_build) - fdf["_is_spike"]
        ).clip(0).astype(int)
        fdf.drop(columns=["_rolling_mean", "_is_spike"], inplace=True)
        print(f"  New features: crit_ratio, new_module, total_historical_bugs, cross_module_spike")

    # Merge TF-IDF text features
    if tfidf_features is not None and not fdf.empty:
        fdf = fdf.merge(tfidf_features, on="module", how="left")
        tfidf_cols = [c for c in fdf.columns if c.startswith(TFIDF_PREFIX)]
        fdf[tfidf_cols] = fdf[tfidf_cols].fillna(0.0)
        print(f"  Text features merged: {len(tfidf_cols)} TF-IDF columns added.")

    # Merge risk features
    if risk_features is not None and not fdf.empty:
        fdf = fdf.merge(risk_features, on="module", how="left")
        # Conservative imputation: missing FMEA = low-risk (1), not neutral (3).
        # Forces the classifier to treat unknown-impact modules as low-risk rather
        # than letting pure activity signals dominate the probability output.
        for col, default in [
            ("impact_score", 1.0), ("detectability_score", 1.0),
            ("probability_score_auto", 1.0), ("risk_score_final", 1.0),
        ]:
            if col in fdf.columns:
                fdf[col] = fdf[col].fillna(default)
        n_matched = fdf["impact_score"].notna().sum() if "impact_score" in fdf.columns else 0
        print(f"  Risk features merged: {n_matched}/{len(fdf)} rows have scores.")

        # Impact-aware features — make the classifier FMEA-aware so P1 cores
        # with low recent activity don't get scored below churning P4 modules.
        if "impact_score" in fdf.columns:
            recent = fdf["bugs_3"] if "bugs_3" in fdf.columns else 0.0
            fdf["impact_bug_ratio"] = (
                fdf["impact_score"] / (1.0 + np.log1p(recent))
            ).round(3)
            if "cross_module_spike" in fdf.columns:
                max_imp = max(float(fdf["impact_score"].max()), 1.0)
                fdf["impact_weighted_spike"] = (
                    fdf["cross_module_spike"] * (fdf["impact_score"] / max_imp)
                ).round(3)

    # Change 12 — merge cluster features
    if cluster_features is not None and not fdf.empty:
        fdf = fdf.merge(cluster_features, on=["module", "build"], how="left")
        for col in ["cluster_entropy_2", "cluster_entropy_3", "top_cluster_velocity"]:
            if col in fdf.columns:
                fdf[col] = fdf[col].fillna(0.0)
        added = [c for c in _CLUSTER_FEATURE_COLS if c in fdf.columns]
        print(f"  Cluster features merged: {len(added)} columns added.")

    return fdf


# ─────────────────────────────────────────────────────────────────────────────
# Leading-indicator correlation  (unchanged from v2.6)
# ─────────────────────────────────────────────────────────────────────────────

def compute_leading_indicators(fdf: pd.DataFrame) -> pd.Series:
    fcols = [c for c in fdf.columns
             if c not in ["module", "build", "target"]
             and not c.startswith(TFIDF_PREFIX)]
    corr = {}
    for col in fcols:
        x = fdf[col].fillna(0)
        y = fdf["target"]
        corr[col] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else 0.0
    return pd.Series(corr).sort_values(key=abs, ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model training  (updated for v3.0)
# ─────────────────────────────────────────────────────────────────────────────

def train_predict(fdf: pd.DataFrame, orig_df: pd.DataFrame):
    fcols = [c for c in fdf.columns if c not in ["module", "build", "target"]]
    n_text    = sum(1 for c in fcols if c.startswith(TFIDF_PREFIX))
    n_cluster = sum(1 for c in fcols if c in _CLUSTER_FEATURE_COLS)
    n_numeric = len(fcols) - n_text - n_cluster
    X = fdf[fcols].fillna(0)
    y = fdf["target"]
    print(f"Training: {len(X)} samples, {len(fcols)} features "
          f"({n_text} text TF-IDF, {n_cluster} cluster, {n_numeric} numeric)")

    # Change 20 — Poisson HistGBR: correct generative model for count data
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=200, max_depth=4, learning_rate=0.1,
        random_state=42, min_samples_leaf=5,
    )
    y_fit = y.clip(lower=0)  # Poisson loss requires non-negative targets
    scores = cross_val_score(model, X, y_fit,
                             cv=TimeSeriesSplit(n_splits=3),
                             scoring="neg_mean_absolute_error")
    print(f"CV MAE (Poisson HGBR): {-scores.mean():.2f} (+/- {scores.std():.2f})")
    model.fit(X, y_fit)

    try:
        imp = pd.Series(model.feature_importances_, index=fcols).sort_values(ascending=False)
    except AttributeError:
        # HistGradientBoostingRegressor doesn't expose feature_importances_;
        # fall back to equal weights so leading-indicator logic still runs.
        imp = pd.Series(np.ones(len(fcols)) / len(fcols), index=fcols)
    print(f"\nTop features:\n{imp.head(10)}")

    leading = compute_leading_indicators(fdf)
    print(f"\nLeading indicators (Pearson r):")
    for feat, r in leading.head(8).items():
        print(f"  {feat:<25s} r = {r:+.3f}  ({_FEATURE_LABELS.get(feat, feat)})")

    latest = fdf.groupby("module").tail(1).copy()
    latest["predicted"] = model.predict(latest[fcols].fillna(0)).round(1)

    # Recency-weighted dominant_bug_type (unchanged from v2.6)
    sev_label_map = {1: "crash/Critical (S1)", 2: "Major functional (S2)",
                     3: "Normal (S3)", 4: "Minor/cosmetic (S4)"}
    dom_type = {}
    if "severity_num" in orig_df.columns and "parsed_module" in orig_df.columns:
        if "Build#" in orig_df.columns:
            orig_sorted = orig_df.copy()
            orig_sorted["_build_num"] = pd.to_numeric(orig_sorted["Build#"], errors="coerce")
        else:
            orig_sorted = orig_df.copy()
            orig_sorted["_build_num"] = np.nan

        for mod, grp in orig_sorted.groupby("parsed_module"):
            grp = grp.dropna(subset=["severity_num"])
            if grp.empty:
                dom_type[mod] = "mixed"
                continue
            if grp["_build_num"].notna().sum() >= 3:
                grp = grp.sort_values("_build_num")
                n = len(grp)
                weights = np.ones(n)
                cutoff_1 = grp["_build_num"].nlargest(1).min()
                cutoff_3 = grp["_build_num"].nlargest(max(1, int(n * 0.33))).min()
                weights[grp["_build_num"].values >= cutoff_3] = 3
                weights[grp["_build_num"].values >= cutoff_1] = 5
                sev_vals = grp["severity_num"].astype(int).values
                sev_weights: dict = {}
                for sv, w in zip(sev_vals, weights):
                    sev_weights[sv] = sev_weights.get(sv, 0.0) + w
                best_sev = min(sev_weights, key=lambda s: (-sev_weights[s], s))
            else:
                mode_sev = grp["severity_num"].mode()
                best_sev = int(mode_sev.iloc[0]) if len(mode_sev) else 3
            dom_type[mod] = sev_label_map.get(int(best_sev), "mixed")

    latest["dominant_bug_type"] = latest["module"].map(dom_type).fillna("mixed")

    # Per-module leading_signal (unchanged from v2.6)
    global_top_signal = leading.index[0] if len(leading) else ""
    global_top_label  = _FEATURE_LABELS.get(global_top_signal, global_top_signal)

    # Exclude TF-IDF, risk-score, and cluster-structural features from per-module signal
    signal_candidates = [
        c for c in fcols
        if not c.startswith(TFIDF_PREFIX)
        and c not in ("impact_score", "detectability_score",
                      "probability_score_auto", "risk_score_final")
        and c not in _CLUSTER_FEATURE_COLS
    ]

    def _per_module_signal(mod: str) -> str:
        mod_rows = fdf[fdf["module"] == mod]
        if len(mod_rows) < 4 or not signal_candidates:
            return global_top_label
        y_mod = mod_rows["target"].fillna(0)
        if y_mod.std() == 0:
            return global_top_label
        best_r, best_col = 0.0, global_top_signal
        for fc in signal_candidates:
            x_mod = mod_rows[fc].fillna(0)
            if x_mod.std() > 0:
                r = abs(float(np.corrcoef(x_mod, y_mod)[0, 1]))
                if r > best_r:
                    best_r, best_col = r, fc
        return _FEATURE_LABELS.get(best_col, best_col)

    latest["leading_signal"] = latest["module"].apply(_per_module_signal)

    return model, latest, imp, leading


# ─────────────────────────────────────────────────────────────────────────────
# Change 19 — Learned risk classifier (replaces hand-tuned composite)
# ─────────────────────────────────────────────────────────────────────────────

def train_risk_classifier(fdf: pd.DataFrame, orig_df: pd.DataFrame,
                          build_col: str = "Build#",
                          mod_col: str = "parsed_module") -> "tuple | None":
    """Train a calibrated GradientBoostingClassifier to predict probability
    of at least one Critical (S1) bug in the NEXT version for each module.

    The binary target is: did this module have any severity 1 (Critical) bug
    in this version?  S2 bugs are too common (~89% of all bugs) to be a useful
    signal — S1 is rare enough (~3.6%) to produce meaningful risk separation.

    Returns (classifier, latest_proba) or None if not enough data.
    """
    df = orig_df.copy()
    df[build_col] = pd.to_numeric(df[build_col], errors="coerce")
    df = df.dropna(subset=[build_col, mod_col])
    df[build_col] = df[build_col].astype(int)

    if "severity_num" not in df.columns:
        print("  Risk classifier: severity_num not found — skipping.")
        return None

    # Per (module, build): 1 if any S1 (Critical) bug, else 0
    has_severe = (
        df[df["severity_num"] == 1]
        .groupby([mod_col, build_col])
        .size()
        .reset_index(name="_severe_count")
    )
    has_severe["_had_severe"] = 1

    # Merge onto feature matrix
    clf_df = fdf.copy()
    clf_df = clf_df.merge(
        has_severe[[mod_col, build_col, "_had_severe"]].rename(
            columns={mod_col: "module", build_col: "build"}),
        on=["module", "build"], how="left"
    )
    clf_df["_had_severe"] = clf_df["_had_severe"].fillna(0).astype(int)

    fcols = [c for c in clf_df.columns
             if c not in ["module", "build", "target", "_had_severe"]]
    X = clf_df[fcols].fillna(0)
    y = clf_df["_had_severe"]

    pos_rate = y.mean()
    print(f"\n  Risk classifier: {len(X)} samples, {y.sum():.0f} positive "
          f"({pos_rate:.1%} had S1 critical bug)")

    if y.sum() < 10 or (1 - pos_rate) < 0.05:
        print("  Risk classifier: too few positive/negative samples — skipping.")
        return None

    base_clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1,
        random_state=42, min_samples_leaf=5,
    )

    # Change 21 — class imbalance correction: weight S1 samples inversely by
    # class frequency so the ~3.6% positive rate doesn't dominate toward Low.
    w_pos = 1.0 / max(float(pos_rate), 1e-6)
    w_neg = 1.0 / max(1.0 - float(pos_rate), 1e-6)
    sample_weight = np.where(y == 1, w_pos, w_neg).astype(float)
    sample_weight = (sample_weight / sample_weight.mean())  # normalize to mean=1

    tscv = TimeSeriesSplit(n_splits=3)
    try:
        cal_clf = CalibratedClassifierCV(base_clf, cv=tscv, method="isotonic")
        cal_clf.fit(X, y, sample_weight=sample_weight)
    except Exception as e:
        print(f"  Risk classifier: calibration failed ({e}), using uncalibrated.")
        base_clf.fit(X, y, sample_weight=sample_weight)
        cal_clf = base_clf

    # CV score for reporting
    from sklearn.metrics import brier_score_loss
    cv_proba = np.zeros(len(X))
    for train_idx, test_idx in tscv.split(X):
        fold_clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1,
            random_state=42, min_samples_leaf=5,
        )
        fold_clf.fit(X.iloc[train_idx], y.iloc[train_idx],
                     sample_weight=sample_weight[train_idx])
        cv_proba[test_idx] = fold_clf.predict_proba(X.iloc[test_idx])[:, 1]
    # Only score on test folds (non-zero entries)
    mask = cv_proba > 0
    if mask.sum() > 10:
        brier = brier_score_loss(y[mask], cv_proba[mask])
        print(f"  Risk classifier CV Brier score: {brier:.4f} "
              f"(lower is better, 0 = perfect, 0.25 = random)")

    # Predict on latest rows
    latest = fdf.groupby("module").tail(1).copy()
    proba = cal_clf.predict_proba(latest[fcols].fillna(0))
    # Handle binary vs single-class output
    if proba.shape[1] == 2:
        latest["risk_proba"] = proba[:, 1]
    else:
        latest["risk_proba"] = proba[:, 0]

    print(f"  Risk classifier: predicted probabilities for {len(latest)} modules")
    print(f"    Probability range: {latest['risk_proba'].min():.3f} – "
          f"{latest['risk_proba'].max():.3f}")

    return cal_clf, latest[["module", "risk_proba"]]


# ─────────────────────────────────────────────────────────────────────────────
# Change 19 — Stratified training (separate models for high/low activity)
# ─────────────────────────────────────────────────────────────────────────────

def train_predict_stratified(fdf: pd.DataFrame, orig_df: pd.DataFrame):
    """Train separate GBR models for high-activity and low-activity modules.

    Split point: modules with total historical bugs above the median get the
    high-activity model; below median get the low-activity model.
    Returns the same shape output as train_predict().
    """
    fcols = [c for c in fdf.columns if c not in ["module", "build", "target"]]
    X_all = fdf[fcols].fillna(0)
    y_all = fdf["target"]

    # Split by module historical activity
    mod_totals = fdf.groupby("module")["target"].sum()
    median_bugs = mod_totals.median()
    high_mods = set(mod_totals[mod_totals >= median_bugs].index)
    low_mods = set(mod_totals[mod_totals < median_bugs].index)

    print(f"\n  Stratified training: {len(high_mods)} high-activity modules "
          f"(≥{median_bugs:.0f} total bugs), {len(low_mods)} low-activity")

    all_latest = []
    all_imp = pd.Series(dtype=float)

    for label, mod_set in [("high-activity", high_mods), ("low-activity", low_mods)]:
        mask = fdf["module"].isin(mod_set)
        subset = fdf[mask]
        if len(subset) < 20:
            print(f"  {label}: too few samples ({len(subset)}) — skipping stratum.")
            continue

        X = subset[fcols].fillna(0)
        y = subset["target"]

        model = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=200, max_depth=4, learning_rate=0.1,
            random_state=42, min_samples_leaf=5,
        )
        y_fit = y.clip(lower=0)
        scores = cross_val_score(model, X, y_fit,
                                 cv=TimeSeriesSplit(n_splits=3),
                                 scoring="neg_mean_absolute_error")
        print(f"  {label} CV MAE (Poisson): {-scores.mean():.2f} (+/- {scores.std():.2f})")
        model.fit(X, y_fit)

        latest = subset.groupby("module").tail(1).copy()
        latest["predicted_stratified"] = model.predict(latest[fcols].fillna(0)).round(1)
        all_latest.append(latest[["module", "predicted_stratified"]])

        try:
            imp = pd.Series(model.feature_importances_, index=fcols)
        except AttributeError:
            imp = pd.Series(np.ones(len(fcols)) / len(fcols), index=fcols)
        all_imp = all_imp.add(imp * len(subset), fill_value=0)

    if not all_latest:
        return None

    stratified_preds = pd.concat(all_latest, ignore_index=True)
    # Normalize importance by total samples
    all_imp = (all_imp / len(fdf)).sort_values(ascending=False)

    return stratified_preds, all_imp


# ─────────────────────────────────────────────────────────────────────────────
# Change 23 — Holt trend forecast (v6.0)
# ─────────────────────────────────────────────────────────────────────────────

def _holt_forecast(values: np.ndarray, alpha: float = 0.3, beta: float = 0.1) -> float:
    """One-step-ahead forecast using Holt's double exponential smoothing.

    alpha controls level responsiveness; beta damps trend growth so a single
    spike doesn't produce a runaway forecast.  Returns 0 if the series is empty.
    """
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(values[0])
    s = float(values[0])
    b = float(values[1] - values[0])
    for i in range(1, n):
        s_prev, b_prev = s, b
        s = alpha * float(values[i]) + (1 - alpha) * (s_prev + b_prev)
        b = beta * (s - s_prev) + (1 - beta) * b_prev
    return max(0.0, round(s + b, 1))


def train_trend_forecast(fdf: pd.DataFrame) -> pd.DataFrame:
    """Apply Holt's double exponential smoothing per module.

    Returns DataFrame with [module, trend_forecast] for the latest build of
    each module.  alpha=0.3 / beta=0.1 give moderate responsiveness without
    overreacting to single-version spikes.
    """
    results = []
    for mod, grp in fdf.groupby("module"):
        counts = grp.sort_values("build")["target"].clip(lower=0).values
        results.append({
            "module": mod,
            "trend_forecast": _holt_forecast(counts),
        })
    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["module", "trend_forecast"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Change 24 — Bayesian Beta-Binomial shrinkage (v6.0)
# ─────────────────────────────────────────────────────────────────────────────

def apply_bayesian_shrinkage(
    preds: pd.DataFrame,
    orig_df: pd.DataFrame,
    mod_col: str = "parsed_module",
    prior_strength: float = 5.0,
) -> pd.Series:
    """Blend ML P(S1) with a Beta-Binomial posterior.

    Modules with sparse history are shrunk toward the global S1 rate;
    well-observed modules keep most of their ML estimate.

    prior_strength=5.0 means each module starts with 5 "virtual" observations
    at the global rate before any real data is counted.

    Returns a Series of adjusted probabilities indexed by preds.index.
    """
    if "risk_proba" not in preds.columns or "severity_num" not in orig_df.columns:
        return preds.get("risk_proba", pd.Series(0.0, index=preds.index))

    sev = pd.to_numeric(orig_df["severity_num"], errors="coerce")
    global_s1_rate = float((sev == 1).mean()) if len(sev) > 0 else 0.05
    s1_counts   = (orig_df[sev == 1]).groupby(mod_col).size()
    total_counts = orig_df.groupby(mod_col).size()

    alpha_prior = global_s1_rate * prior_strength
    beta_prior  = (1.0 - global_s1_rate) * prior_strength

    adjusted = []
    for _, row in preds.iterrows():
        mod     = row["module"]
        ml_p    = float(row["risk_proba"])
        n_s1    = int(s1_counts.get(mod, 0))
        n_total = int(total_counts.get(mod, 0))
        # Posterior mean: (alpha_prior + n_s1) / (alpha_prior + beta_prior + n_total)
        posterior = (alpha_prior + n_s1) / (alpha_prior + beta_prior + max(n_total, 1))
        # Weight ML estimate by data volume; shrink toward posterior when sparse
        trust = n_total / (n_total + prior_strength)
        adjusted.append(round(trust * ml_p + (1.0 - trust) * posterior, 4))

    return pd.Series(adjusted, index=preds.index)


# ─────────────────────────────────────────────────────────────────────────────
# NEW v3.0 — Bug-type (per-cluster) prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_bug_type(latest_preds: pd.DataFrame,
                     orig_df: pd.DataFrame,
                     cluster_csv: str,
                     build_col: str = "Build#",
                     mod_col: str = "parsed_module",
                     history_builds: int = 5) -> "pd.DataFrame | None":
    """
    For each module, predict the expected count per bug theme (cluster) for
    the next version. Strategy: scale the recent historical cluster distribution
    by the total predicted count from train_predict().

    Parameters
    ----------
    latest_preds   : DataFrame with [module, predicted] from train_predict()
    orig_df        : raw bug-level DataFrame
    cluster_csv    : path to the clustered output CSV from cluster_bugs.py
    history_builds : number of most recent versions to use for the distribution

    Returns
    -------
    DataFrame: module, cluster_id, cluster_label, historical_pct, predicted_count
    Sorted by module, then predicted_count descending.
    Returns None if the cluster CSV cannot be read or there is insufficient data.
    """
    try:
        cdf = pd.read_csv(cluster_csv)
    except Exception as e:
        print(f"  WARNING: cannot load cluster CSV for type prediction ({e}).")
        return None

    if "cluster_id" not in cdf.columns or "predicted" not in latest_preds.columns:
        return None

    # Join cluster assignments onto the raw bug data
    if "BugCode" in cdf.columns and "BugCode" in orig_df.columns:
        bug_cols = [c for c in [build_col, mod_col, "BugCode"] if c in orig_df.columns]
        merged = orig_df[bug_cols].merge(
            cdf[["BugCode", "cluster_id", "cluster_label"]], on="BugCode", how="left"
        )
    elif mod_col in cdf.columns and build_col in cdf.columns:
        merged = orig_df[[build_col, mod_col]].merge(
            cdf[[build_col, mod_col, "cluster_id", "cluster_label"]],
            on=[build_col, mod_col], how="left"
        )
    else:
        print("  NOTE: Cannot join cluster CSV for type prediction — no shared key.")
        return None

    merged[build_col] = pd.to_numeric(merged[build_col], errors="coerce")
    merged = merged.dropna(subset=[build_col, mod_col])
    merged[build_col] = merged[build_col].astype(int)
    merged["cluster_id"] = merged["cluster_id"].fillna(-1).astype(int)

    # Build cluster label lookup
    label_map = (
        cdf[cdf["cluster_id"] != -1][["cluster_id", "cluster_label"]]
        .drop_duplicates("cluster_id")
        .set_index("cluster_id")["cluster_label"]
        .to_dict()
        if "cluster_label" in cdf.columns else {}
    )

    results = []
    for _, pred_row in latest_preds.iterrows():
        mod        = pred_row["module"]
        pred_total = float(pred_row.get("predicted", 0))

        mod_data = merged[merged[mod_col] == mod]
        if len(mod_data) < 5:
            continue

        recent_builds = sorted(mod_data[build_col].unique())[-history_builds:]
        recent = mod_data[
            mod_data[build_col].isin(recent_builds) & (mod_data["cluster_id"] != -1)
        ]

        if len(recent) == 0:
            results.append({
                "module": mod, "cluster_id": -1,
                "cluster_label": "No cluster data",
                "historical_pct": 1.0,
                "predicted_count": round(pred_total, 1),
            })
            continue

        total_recent = len(recent)
        for cid, cnt in recent["cluster_id"].value_counts().items():
            pct = cnt / total_recent
            results.append({
                "module": mod,
                "cluster_id": int(cid),
                "cluster_label": label_map.get(int(cid), f"Cluster {cid}"),
                "historical_pct": round(float(pct), 3),
                "predicted_count": round(pred_total * pct, 1),
            })

    if not results:
        return None

    out = pd.DataFrame(results)
    out = out.sort_values(["module", "predicted_count"], ascending=[True, False])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Change 15 — Category breakdown prediction (no cluster CSV required)
# ─────────────────────────────────────────────────────────────────────────────

def predict_category_breakdown(latest_preds: pd.DataFrame,
                               orig_df: pd.DataFrame,
                               mod_col: str = "parsed_module",
                               text_col: str = "parsed_description",
                               build_col: str = "Build#",
                               history_builds: int = 5) -> "pd.DataFrame | None":
    """
    Classify each recent bug into a QA category (BUG_CATEGORIES) and scale
    the distribution by the ML-model total prediction to produce expected
    per-category counts for the next version.

    Parameters
    ----------
    latest_preds   : DataFrame with [module, predicted] from train_predict()
    orig_df        : raw bug-level DataFrame with parsed_description
    history_builds : number of most recent versions to use for the distribution

    Returns
    -------
    DataFrame: module, category, historical_count, historical_pct, expected_next_build,
               example_descriptions
    Sorted by module, then expected_next_build descending.
    Returns None if parsed_description is absent or no data can be built.
    """
    if text_col not in orig_df.columns or mod_col not in orig_df.columns:
        print(f"  NOTE: '{text_col}' not found — skipping category breakdown prediction.")
        return None

    working = orig_df.copy()
    working[build_col] = pd.to_numeric(working[build_col], errors="coerce")
    working = working.dropna(subset=[build_col, mod_col])
    working = working[working[text_col].notna()]
    working[build_col] = working[build_col].astype(int)
    if "severity_num" not in working.columns:
        working["severity_num"] = 3
    working["severity_num"] = pd.to_numeric(working["severity_num"], errors="coerce").fillna(3).astype(int)
    working["_bug_category"] = working[text_col].apply(classify_bug_category)

    has_bugcode = "BugCode" in working.columns

    results = []
    for _, pred_row in latest_preds.iterrows():
        mod        = pred_row["module"]
        pred_total = float(pred_row.get("predicted", 0))

        mod_data = working[working[mod_col] == mod]
        if len(mod_data) < 3:
            continue

        recent_builds = sorted(mod_data[build_col].unique())[-history_builds:]
        recent = mod_data[mod_data[build_col].isin(recent_builds)]
        if len(recent) == 0:
            continue

        total_recent = len(recent)
        latest_build = int(recent[build_col].max())
        # Use parsed_description (clean, prefix-stripped) for examples; fall back to Short Description
        _desc_col = text_col if text_col in recent.columns else (
            "Short Description" if "Short Description" in recent.columns else None
        )
        for cat, cnt in recent["_bug_category"].value_counts().items():
            pct = cnt / total_recent
            cat_bugs = recent[recent["_bug_category"] == cat]

            # Severity stratification — partition into S1 / S2 / S3+ tiers
            sev = cat_bugs["severity_num"]
            s1_count = int((sev == 1).sum())
            s2_count = int((sev == 2).sum())
            s3plus_count = int((sev >= 3).sum())
            total_predicted = pred_total * pct
            expected_s1     = round(total_predicted * (s1_count / cnt), 2) if cnt else 0.0
            expected_s2     = round(total_predicted * (s2_count / cnt), 2) if cnt else 0.0
            expected_s3plus = round(total_predicted * (s3plus_count / cnt), 2) if cnt else 0.0

            # Recency-weighted bug presence (more-recent bugs count more)
            recency_weight = float(
                (1.0 / (1.0 + (latest_build - cat_bugs[build_col].astype(int)))).sum()
            )

            # Severity weight: 1=S1 → 4×, 2=S2 → 2×, 3+ → 1×
            sev_weight = (s1_count * 4 + s2_count * 2 + s3plus_count * 1) / max(cnt, 1)
            raw_risk = float(pct * sev_weight * (recency_weight / max(cnt, 1)))

            # Confidence — share of recent bugs this category represents within
            # the module. A small module where one category dominates 30%+ of
            # recent bugs IS a high-confidence signal even at cnt=3.
            if pct >= 0.30:
                confidence = "high"
            elif pct >= 0.15:
                confidence = "medium"
            else:
                confidence = "low"

            # Examples — prefer most-severe-most-recent first
            example_descriptions = ""
            example_bug_codes = ""
            sort_cols = ["severity_num", build_col]
            sort_asc  = [True, False]
            cat_sorted = cat_bugs.sort_values(sort_cols, ascending=sort_asc)
            if _desc_col:
                raw_exs = cat_sorted[_desc_col].dropna().head(5).tolist()
                example_descriptions = " | ".join(
                    str(e).strip() for e in raw_exs if str(e).strip()
                )
            if has_bugcode:
                raw_codes = cat_sorted["BugCode"].dropna().head(5).tolist()
                example_bug_codes = " | ".join(
                    str(b).strip() for b in raw_codes if str(b).strip()
                )

            results.append({
                "module": mod,
                "category": cat,
                "historical_count": int(cnt),
                "historical_pct": round(float(pct), 3),
                "expected_next_build": round(total_predicted, 1),
                "expected_s1": expected_s1,
                "expected_s2": expected_s2,
                "expected_s3plus": expected_s3plus,
                "_raw_risk": raw_risk,
                "confidence": confidence,
                "latest_bug_build": latest_build,
                "example_descriptions": example_descriptions,
                "example_bug_codes": example_bug_codes,
            })

    if not results:
        print("  NOTE: No category breakdown could be built (insufficient description data).")
        return None

    out = pd.DataFrame(results)
    # Normalize raw_risk → 0-100 risk_score (per-run, max-row anchored)
    max_raw = float(out["_raw_risk"].max()) if len(out) else 0.0
    if max_raw > 0:
        out["risk_score"] = (100.0 * out["_raw_risk"] / max_raw).round(1)
    else:
        out["risk_score"] = 0.0
    out = out.drop(columns=["_raw_risk"])
    # Sort top-of-file = highest sub-feature risk (actionable ranking)
    out = out.sort_values(["risk_score", "module"], ascending=[False, True])
    print(f"  Category breakdown: {len(out)} module×category rows across "
          f"{out['module'].nunique()} modules. "
          f"Top risk_score: {out['risk_score'].iloc[0]:.1f}.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Bug-type AI description generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_category_ai_descriptions(
    category_df: pd.DataFrame,
    orig_df: pd.DataFrame,
    provider: str = "none",
    model: str = "qwen3:4b",
    max_ai_calls: int = 30,
    mod_col: str = "parsed_module",
    text_col: str = "parsed_description",
) -> pd.DataFrame:
    """Find the most similar existing bug for sparse (module, category) pairs.

    For rows where expected_next_build > 0.5 and fewer than 2 historical examples exist,
    uses TF-IDF cosine similarity to find the closest matching bug from other modules
    that shares the same QA category.

    Adds two columns to category_df:
    - generated_description: the matched bug's description text
    - generated_bug_code:    BugCode of the matched bug (for ECL linking)
    """
    if category_df is None or category_df.empty:
        return category_df
    if text_col not in orig_df.columns or mod_col not in orig_df.columns:
        return category_df.assign(generated_description="", generated_bug_code="")

    category_df = category_df.copy()
    category_df["generated_description"] = ""
    category_df["generated_bug_code"] = ""

    # Classify all bugs once; reset index so row position == TF-IDF matrix row
    _keep_cols = [mod_col, text_col] + (["BugCode"] if "BugCode" in orig_df.columns else [])
    working = orig_df[_keep_cols].copy().reset_index(drop=True)
    working["_bug_category"] = working[text_col].apply(classify_bug_category)

    # Build TF-IDF on all descriptions in one pass
    all_descs = working[text_col].fillna("").tolist()
    _vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=1)
    try:
        tfidf_all = _vec.fit_transform(all_descs)
    except Exception:
        return category_df

    def _example_count(s: str) -> int:
        return len([e for e in str(s).split(" | ") if e.strip()])

    needs_match = category_df[
        (pd.to_numeric(category_df["expected_next_build"], errors="coerce").fillna(0) > 0.5)
        & (category_df["example_descriptions"].apply(_example_count) < 2)
    ].sort_values("expected_next_build", ascending=False)

    found = 0
    for idx, row in needs_match.iterrows():
        mod = row["module"]
        cat = row["category"]

        # Candidate pool: same QA category, different module
        cand_indices = working.index[
            (working["_bug_category"] == cat) & (working[mod_col] != mod)
        ].tolist()
        if not cand_indices:
            continue

        # Use an existing example as the query if we have one; otherwise fall back to keywords
        existing_exs = [
            e.strip() for e in str(row.get("example_descriptions", "")).split(" | ")
            if e.strip()
        ]
        query = existing_exs[0] if existing_exs else f"{mod} {cat.replace('/', ' ')}"

        try:
            q_vec = _vec.transform([query])
            doc_vecs = tfidf_all[cand_indices]
            sims = 1.0 - cosine_distances(q_vec, doc_vecs)[0]
            best_pos = int(sims.argmax())
            best_row = working.iloc[cand_indices[best_pos]]

            best_desc = str(best_row.get(text_col, "")).strip()
            best_bc   = str(best_row.get("BugCode", "")).strip()
            if best_desc and best_desc not in ("nan", ""):
                category_df.at[idx, "generated_description"] = best_desc
                if best_bc and best_bc not in ("nan", ""):
                    category_df.at[idx, "generated_bug_code"] = best_bc
                found += 1
        except Exception:
            pass

    print(f"  Found {found} similar existing bugs for sparse category entries.")
    return category_df


# ─────────────────────────────────────────────────────────────────────────────
# Description sampler  (unchanged from v2.6)
# ─────────────────────────────────────────────────────────────────────────────

def _sample_descriptions(orig_df, module, mod_col="parsed_module",
                          text_col="parsed_description", n=12):
    if text_col not in orig_df.columns:
        return []
    rows = orig_df[orig_df[mod_col] == module][text_col].dropna()
    seen, out = set(), []
    for s in rows.tail(n * 2).tolist():
        s = str(s).strip()
        if s and s not in seen and len(s) > 10:
            seen.add(s); out.append(s)
        if len(out) >= n:
            break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Change 22 — Ollama semantic embedding helper (v6.0)
# ─────────────────────────────────────────────────────────────────────────────

def _get_ollama_embeddings(
    texts: list,
    model: str = "mxbai-embed-large",
) -> "np.ndarray | None":
    """Fetch embeddings for a list of texts from Ollama.

    Tries the batch /api/embed endpoint (Ollama >=0.3) first, then falls back
    to individual /api/embeddings calls.  Returns an (N, D) float32 array, or
    None if Ollama is unreachable or the model isn't available.
    """
    if not ollama_is_reachable() or not texts:
        return None

    # Batch endpoint (preferred — single round-trip)
    try:
        payload = json.dumps({"model": model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            if "embeddings" in data and len(data["embeddings"]) == len(texts):
                return np.array(data["embeddings"], dtype=np.float32)
    except Exception:
        pass

    # Per-text fallback
    embeddings = []
    for text in texts:
        try:
            payload = json.dumps({"model": model, "prompt": text}).encode()
            req = urllib.request.Request(
                f"{OLLAMA_BASE}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                if "embedding" in data:
                    embeddings.append(data["embedding"])
                else:
                    return None
        except Exception:
            return None

    if len(embeddings) == len(texts):
        return np.array(embeddings, dtype=np.float32)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Change 18 — Scenario generation pipeline (v5.0)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_description_patterns(
    orig_df: pd.DataFrame,
    module: str,
    mod_col: str = "parsed_module",
    text_col: str = "parsed_description",
    build_col: str = "Build#",
    history_builds: int = SCENARIO_HISTORY_BUILDS,
    top_n: int = SCENARIOS_PER_MODULE,
) -> list:
    """Heuristic (non-AI) scenario extractor.

    Clusters recent bug descriptions for a module using AgglomerativeClustering
    on TF-IDF cosine distances, then picks the description closest to each
    cluster centroid as a representative scenario.

    Returns list of dicts: {scenario_text, confidence, source_count, source_examples}
    """
    if text_col not in orig_df.columns or mod_col not in orig_df.columns:
        return []

    working = orig_df.copy()
    working[build_col] = pd.to_numeric(working[build_col], errors="coerce")
    working = working.dropna(subset=[build_col, mod_col])
    working[build_col] = working[build_col].astype(int)

    mod_data = working[working[mod_col] == module]
    if len(mod_data) < 2:
        return []

    recent_builds = sorted(mod_data[build_col].unique())[-history_builds:]
    recent = mod_data[mod_data[build_col].isin(recent_builds)]

    descriptions = recent[text_col].dropna().astype(str).tolist()
    # Deduplicate while preserving order
    seen = set()
    unique_descs = []
    for d in descriptions:
        d_stripped = d.strip()
        if d_stripped and len(d_stripped) > 10 and d_stripped not in seen:
            seen.add(d_stripped)
            unique_descs.append(d_stripped)

    # Keep aligned bookkeeping: for each unique description, remember the
    # original row index so we can recover severity / build info for the
    # explanation string later.
    row_by_desc: dict = {}
    for idx, d in zip(recent.index.tolist(), recent[text_col].dropna().astype(str).tolist()):
        s = d.strip()
        if s and len(s) > 10 and s not in row_by_desc:
            row_by_desc[s] = idx
    desc_rows = [row_by_desc.get(d, -1) for d in unique_descs]

    if len(unique_descs) < 2:
        if unique_descs:
            return [{
                "scenario_text": unique_descs[0],
                "confidence": "low",
                "source_count": 1,
                "source_examples": unique_descs[0],
                "cluster_top_terms": [],
                "member_row_indices": [desc_rows[0]] if desc_rows else [],
            }]
        return []

    # Build distance matrix — try Ollama semantic embeddings first (Change 22),
    # fall back to TF-IDF when Ollama is unavailable.
    X_tfidf = None
    feature_names: list = []

    emb = _get_ollama_embeddings(unique_descs)
    if emb is not None and len(emb) == len(unique_descs):
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        dist_matrix = cosine_distances(emb / norms)
    else:
        # TF-IDF fallback
        try:
            vec = TfidfVectorizer(
                max_features=100, stop_words="english",
                ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True,
            )
            X_tfidf = vec.fit_transform(unique_descs)
            feature_names = list(vec.get_feature_names_out())
        except ValueError:
            return [{
                "scenario_text": unique_descs[0],
                "confidence": "low",
                "source_count": 1,
                "source_examples": unique_descs[0],
                "cluster_top_terms": [],
                "member_row_indices": [desc_rows[0]] if desc_rows else [],
            }]
        dist_matrix = cosine_distances(X_tfidf)

    # Cluster
    n_samples = len(unique_descs)
    if n_samples <= 2:
        # Too few for clustering — return as-is
        results = []
        for i, d in enumerate(unique_descs[:top_n]):
            ri = desc_rows[i] if i < len(desc_rows) else -1
            results.append({
                "scenario_text": d,
                "confidence": "low",
                "source_count": 1,
                "source_examples": d,
                "cluster_top_terms": [],
                "member_row_indices": [ri] if ri != -1 else [],
            })
        return results

    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.7,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(dist_matrix)
    except Exception:
        return [{
            "scenario_text": unique_descs[0],
            "confidence": "low",
            "source_count": 1,
            "source_examples": unique_descs[0],
            "cluster_top_terms": [],
            "member_row_indices": [desc_rows[0]] if desc_rows else [],
        }]

    # Group descriptions by cluster, pick representative (closest to centroid)
    from collections import defaultdict
    clusters: dict = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    # Sort clusters by size descending
    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))

    # The latest two builds drive the "still happening" signal — a cluster
    # whose members all sit in older builds carries less weight even if it
    # was recurring back then.
    latest_two_builds = set(recent_builds[-2:]) if recent_builds else set()

    results = []
    for _, member_indices in sorted_clusters[:top_n]:
        member_descs = [unique_descs[i] for i in member_indices]
        cluster_size = len(member_indices)

        # Pick description closest to cluster centroid
        if cluster_size == 1:
            representative = member_descs[0]
        else:
            sub_dist = dist_matrix[np.ix_(member_indices, member_indices)]
            avg_dist = sub_dist.mean(axis=1)
            best_local_idx = int(np.argmin(avg_dist))
            representative = member_descs[best_local_idx]

        # Build-recency check — does any member appear in the last two builds?
        in_latest = False
        for mi in member_indices:
            ri = desc_rows[mi] if mi < len(desc_rows) else -1
            if ri != -1 and ri in recent.index:
                try:
                    if int(recent.at[ri, build_col]) in latest_two_builds:
                        in_latest = True
                        break
                except Exception:
                    pass

        # Confidence — relaxed thresholds: a recurring cluster (>=2) is high,
        # and a singleton that's still appearing in the latest builds is medium
        # (the bug is fresh, not a one-off from months ago).
        if cluster_size >= 2:
            confidence = "high"
        elif cluster_size == 1 and in_latest:
            confidence = "medium"
        else:
            confidence = "low"

        # Cluster top terms — only available on the TF-IDF path
        top_terms: list = []
        try:
            if X_tfidf is not None and feature_names:
                sub_tfidf = X_tfidf[member_indices].toarray().mean(axis=0)
                if sub_tfidf.size > 0:
                    order = sub_tfidf.argsort()[::-1]
                    top_terms = [feature_names[i] for i in order[:5] if sub_tfidf[i] > 0]
        except Exception:
            top_terms = []

        # Build member_rows in the SAME cluster order as member_descs so the
        # downstream `_bugcodes_for(pattern)` and `_descs_for(pattern)` helpers
        # produce parallel lists (BugCode N ↔ description N).
        member_rows = [desc_rows[i] for i in member_indices if desc_rows[i] != -1]
        # Source examples: parallel to the first 5 member_rows so they line up
        # 1:1 with the BugCodes produced by `_bugcodes_for`.
        example_descs = [unique_descs[i] for i in member_indices if desc_rows[i] != -1][:5]
        example_str = " | ".join(example_descs) if example_descs else representative

        results.append({
            "scenario_text": representative,
            "confidence": confidence,
            "source_count": cluster_size,
            "source_examples": example_str,
            "cluster_top_terms": top_terms,
            "member_row_indices": member_rows,
        })

    return results


def _generate_scenarios_ai_ollama(
    module: str,
    risk_level: str,
    descriptions: list,
    category_distribution: "dict | None" = None,
    cluster_labels: "list | None" = None,
    leading_signal: str = "",
    model: str = "qwen3:4b",
) -> list:
    """AI scenario generator using Ollama. Returns list of scenario dicts."""
    desc_block = "\n".join(f" - {d}" for d in descriptions[:15]) if descriptions \
        else " (no description samples available)"

    cat_block = ""
    if category_distribution:
        lines = [f"  • {cat} ({pct*100:.0f}%)" for cat, pct in list(category_distribution.items())[:6]]
        cat_block = "\nCategory distribution:\n" + "\n".join(lines)

    cluster_block = ""
    if cluster_labels:
        cluster_block = "\nBug theme clusters: " + ", ".join(cluster_labels[:5])

    prompt = (
        f"You are a QA lead predicting which specific failures will RECUR or extend in the next version.\n\n"
        f"Module: {module}\n"
        f"Current risk level: {risk_level}\n"
        f"Strongest predictive signal: {leading_signal}"
        f"{cat_block}\n{cluster_block}\n\n"
        f"Bug descriptions from RECENT versions of this module (most relevant signal):\n{desc_block}\n\n"
        f"TASK — predict 3-5 specific bug scenarios most likely to appear in the next version.\n"
        f"RULES:\n"
        f"  • Anchor each scenario in a specific recent description shown above. Quote the matching description verbatim into 'based_on'.\n"
        f"  • Prefer scenarios that match RECURRING patterns (similar wording across multiple descriptions) — these are most likely to break again.\n"
        f"  • Each 'scenario' must read like a real bug title: a concrete, testable statement of what goes wrong.\n"
        f"  • Do NOT invent failure modes that have no support in the descriptions above. If only N distinct patterns exist, return N scenarios.\n"
        f"  • Do NOT write vague summaries. Do NOT mention counts.\n"
        f"  • confidence: 'high' = matches 2+ recent descriptions; 'medium' = matches 1 recent description; 'low' = extrapolation.\n\n"
        f'Return exactly this JSON object and nothing else. Do not echo the context above:\n'
        f'{{"scenarios": [{{"scenario": "...", "confidence": "high|medium|low", "based_on": "<quoted description from list above>", "explanation": "**Why likely:** [reference the recent pattern]. **Steps to reproduce:** [concrete trigger]. **What to verify:** [expected vs actual]."}}]}}'
    )

    payload = json.dumps(with_thinking_disabled({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2, "num_predict": 2000},
    })).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode())
            text = response_text(raw)
            # Try parsing as JSON directly
            def _normalize_scenarios(lst: list) -> list:
                """Ensure every element is a dict with at least a 'scenario' key."""
                result = []
                for item in lst:
                    if isinstance(item, dict):
                        result.append(item)
                    elif isinstance(item, str) and item.strip():
                        result.append({"scenario": item.strip(), "confidence": "medium", "based_on": "", "explanation": ""})
                return result

            try:
                parsed = extract_json(text)
                if isinstance(parsed, list):
                    return _normalize_scenarios(parsed)
                # Some models wrap in a dict
                if isinstance(parsed, dict):
                    for key in ("scenarios", "predictions", "bugs", "results"):
                        if key in parsed and isinstance(parsed[key], list):
                            return _normalize_scenarios(parsed[key])
                    # …or return the array under some other single key.
                    for value in parsed.values():
                        if isinstance(value, list) and value:
                            return _normalize_scenarios(value)
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    return []


def _generate_scenarios_ai_claude(
    module: str,
    risk_level: str,
    descriptions: list,
    category_distribution: "dict | None" = None,
    cluster_labels: "list | None" = None,
    leading_signal: str = "",
    api_key: str = "",
) -> list:
    """AI scenario generator using Claude API. Returns list of scenario dicts."""
    desc_block = "\n".join(f" - {d}" for d in descriptions[:15]) if descriptions \
        else " (no description samples available)"

    cat_block = ""
    if category_distribution:
        lines = [f"  • {cat} ({pct*100:.0f}%)" for cat, pct in list(category_distribution.items())[:6]]
        cat_block = "\nCategory distribution:\n" + "\n".join(lines)

    cluster_block = ""
    if cluster_labels:
        cluster_block = "\nBug theme clusters: " + ", ".join(cluster_labels[:5])

    prompt = (
        f"You are a QA lead predicting which specific failures will RECUR or extend in the next version.\n\n"
        f"Module: {module}\n"
        f"Current risk level: {risk_level}\n"
        f"Strongest predictive signal: {leading_signal}"
        f"{cat_block}\n{cluster_block}\n\n"
        f"Bug descriptions from RECENT versions of this module (most relevant signal):\n{desc_block}\n\n"
        f"TASK — predict 3-5 specific bug scenarios most likely to appear in the next version.\n"
        f"RULES:\n"
        f"  • Anchor each scenario in a specific recent description shown above. Quote the matching description verbatim into 'based_on'.\n"
        f"  • Prefer scenarios that match RECURRING patterns (similar wording across multiple descriptions) — these are most likely to break again.\n"
        f"  • Each 'scenario' must read like a real bug title: a concrete, testable statement of what goes wrong.\n"
        f"  • Do NOT invent failure modes that have no support in the descriptions above. If only N distinct patterns exist, return N scenarios.\n"
        f"  • Do NOT write vague summaries. Do NOT mention counts.\n"
        f"  • confidence: 'high' = matches 2+ recent descriptions; 'medium' = matches 1 recent description; 'low' = extrapolation.\n\n"
        f'Return ONLY a JSON array: [{{"scenario": "...", "confidence": "high|medium|low", "based_on": "<quoted description from list above>", "explanation": "**Why likely:** [reference the recent pattern]. **Steps to reproduce:** [concrete trigger]. **What to verify:** [expected vs actual]."}}]'
    )

    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            text = data["content"][0]["text"].strip()
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    return []


def _build_scenario_explanation(
    *,
    module: str,
    mod_row: pd.Series,
    scenario: dict,
    orig_df: pd.DataFrame,
    mod_col: str,
    build_col: str,
    scenario_type: str,
) -> str:
    """Produce a multi-line explanation covering four dimensions:
    pattern evidence, module-selection rationale, cluster representation,
    and forward-looking risk. Used to populate the `explanation` column on
    `ecl_parsed_predictions_by_scenario.csv`."""

    # ── Gather raw signals ───────────────────────────────────────────────
    member_rows = scenario.get("member_row_indices") or []
    source_count = int(scenario.get("source_count", 0) or 0)
    critical_hits = 0
    most_recent_build = None
    if member_rows and not orig_df.empty:
        try:
            sub = orig_df.loc[[i for i in member_rows if i in orig_df.index]]
            if "severity_num" in sub.columns:
                sev = pd.to_numeric(sub["severity_num"], errors="coerce")
                critical_hits = int((sev <= 1).sum())
            if build_col in sub.columns:
                builds = pd.to_numeric(sub[build_col], errors="coerce").dropna()
                if len(builds) > 0:
                    most_recent_build = int(builds.max())
        except Exception:
            pass

    leading = str(mod_row.get("leading_signal") or "")
    reg_rate = mod_row.get("regression_rate")
    open_count = mod_row.get("open_count")
    vel = mod_row.get("module_cluster_velocity")
    top_terms = scenario.get("cluster_top_terms") or []
    scenario_text = str(scenario.get("scenario_text", ""))

    # ── Why likely ───────────────────────────────────────────────────────
    why_parts = [f"This pattern appeared in {source_count or 1} similar bug(s) in {module}."]
    if critical_hits:
        why_parts.append(f"{critical_hits} of those were Critical/S1 severity.")
    if most_recent_build:
        why_parts.append(f"Most recently seen in build {most_recent_build}.")
    if leading:
        why_parts.append(f"Strongest signal: {leading}.")
    if reg_rate is not None and pd.notna(reg_rate) and float(reg_rate) > 0:
        why_parts.append(f"Regression rate: {float(reg_rate) * 100:.0f}%.")
    if open_count is not None and pd.notna(open_count) and float(open_count) > 0:
        why_parts.append(f"{int(open_count)} related bug(s) still open.")
    if vel is not None and pd.notna(vel):
        tag = "growing" if float(vel) >= 1.5 else ("declining" if float(vel) <= 0.67 else "stable")
        why_parts.append(f"Cluster velocity is {tag} ({float(vel):.2f}×).")
    why_likely = "**Why likely:** " + " ".join(why_parts)

    # ── Steps to reproduce ───────────────────────────────────────────────
    if top_terms:
        keyword_hint = f" involving {', '.join(top_terms[:3])}"
    else:
        keyword_hint = ""
    steps = (
        f"**Steps to reproduce:** Navigate to the {module} area{keyword_hint}. "
        f"Perform the action described in the scenario and observe the result."
    )

    # ── What to verify ───────────────────────────────────────────────────
    verify = (
        f"**What to verify:** Confirm that {scenario_text.rstrip('.')} behaves as expected. "
        f"Check for incorrect UI state, missing feedback, or unexpected errors."
    )

    return "\n".join([why_likely, steps, verify])


def generate_bug_scenarios(
    preds: pd.DataFrame,
    orig_df: pd.DataFrame,
    category_preds_df: "pd.DataFrame | None",
    cluster_preds_df: "pd.DataFrame | None",
    provider: str = "none",
    model: str = "qwen3:4b",
    api_key: str = "",
    mod_col: str = "parsed_module",
    text_col: str = "parsed_description",
    build_col: str = "Build#",
    ai_scenarios: bool = False,
) -> "pd.DataFrame | None":
    """Orchestrate scenario generation for the top risk-ranked modules.

    For each module: gather context, call AI or heuristic scenario generator,
    build a DataFrame of predicted bug scenarios.
    """
    # Select top modules by blended priority_score if available, otherwise
    # fall back to the legacy risk_level + predicted ordering.
    ranked = preds.copy()
    if "priority_score" in ranked.columns and ranked["priority_score"].notna().any():
        ranked = ranked.sort_values("priority_score", ascending=False)
    else:
        risk_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        ranked["_risk_rank"] = ranked["risk_level"].astype(str).map(risk_order).fillna(3).astype(int)
        ranked = ranked.sort_values(["_risk_rank", "predicted"], ascending=[True, False])

    selected = ranked.head(SCENARIO_TOP_MODULES)

    # Force-include every P1 (and P2) module with data, guaranteeing heatmap
    # alignment. The forced set is unioned with the top-N by priority_score and
    # the result is capped at SCENARIO_HARD_CAP to bound AI call cost.
    forced_tiers: list[str] = []
    if SCENARIO_FORCE_P1:
        forced_tiers.append("P1")
    if SCENARIO_FORCE_P2:
        forced_tiers.append("P2")

    if "heatmap_quadrant" in ranked.columns and forced_tiers:
        q = (ranked["heatmap_quadrant"].astype("string").str.upper()
             .str.extract(r"(P[1-4])")[0])
        forced = ranked[q.isin(forced_tiers)]
        top_modules = (
            pd.concat([selected, forced])
              .drop_duplicates(subset=["module"], keep="first")
              .head(SCENARIO_HARD_CAP)
        )
    else:
        top_modules = selected.head(SCENARIO_HARD_CAP)

    print(f"  Scenario modules selected: {len(top_modules)} "
          f"(top-N={SCENARIO_TOP_MODULES}, cap={SCENARIO_HARD_CAP}, "
          f"forced={forced_tiers})")

    # Resolve AI availability once per run. Off by default — AI-synthesized
    # scenarios add cost but no real signal beyond the heuristic patterns and
    # break BugCode traceability. Opt in with --ai-scenarios.
    ai_active = False
    if not ai_scenarios:
        print("  Scenarios: historical-pattern only (AI synthesis off by default; pass --ai-scenarios to enable).")
    elif provider == "ollama":
        if ollama_is_reachable():
            ai_active = True
        else:
            print("  NOTE: Ollama unreachable — scenarios will use historical-pattern fallback.")
    elif provider == "claude" and api_key:
        ai_active = True

    # Determine predicted build number
    if build_col in orig_df.columns:
        build_nums = pd.to_numeric(orig_df[build_col], errors="coerce").dropna()
        predicted_build = int(build_nums.max()) + 1 if len(build_nums) > 0 else 0
    else:
        predicted_build = 0

    all_rows = []

    for _, mod_row in tqdm(top_modules.iterrows(), total=len(top_modules),
                           desc="Generating scenarios", unit="mod",
                           file=sys.stderr, dynamic_ncols=True):
        mod = mod_row["module"]
        risk_level = str(mod_row.get("risk_level", "Medium"))
        leading_signal = str(mod_row.get("leading_signal", ""))

        # Gather descriptions
        descriptions = _sample_descriptions(orig_df, mod, mod_col=mod_col,
                                            text_col=text_col, n=15)

        # Category distribution
        category_distribution = None
        if category_preds_df is not None and not category_preds_df.empty:
            mod_cats = category_preds_df[category_preds_df["module"] == mod].head(6)
            if not mod_cats.empty:
                category_distribution = dict(
                    zip(mod_cats["category"], mod_cats["historical_pct"]))

        # Cluster labels
        cluster_labels = None
        if cluster_preds_df is not None and not cluster_preds_df.empty:
            mod_cl = cluster_preds_df[cluster_preds_df["module"] == mod].head(5)
            if not mod_cl.empty and "cluster_label" in mod_cl.columns:
                cluster_labels = mod_cl["cluster_label"].tolist()

        # Supporting categories string
        supporting_cats = ""
        if category_distribution:
            supporting_cats = ", ".join(list(category_distribution.keys())[:3])

        # Try AI scenario generation (only when we've confirmed AI is usable)
        scenarios = []
        if ai_active and provider == "ollama":
            scenarios = _generate_scenarios_ai_ollama(
                module=mod, risk_level=risk_level,
                descriptions=descriptions,
                category_distribution=category_distribution,
                cluster_labels=cluster_labels,
                leading_signal=leading_signal,
                model=model,
            )
        elif ai_active and provider == "claude":
            scenarios = _generate_scenarios_ai_claude(
                module=mod, risk_level=risk_level,
                descriptions=descriptions,
                category_distribution=category_distribution,
                cluster_labels=cluster_labels,
                leading_signal=leading_signal,
                api_key=api_key,
            )

        # Heuristic results always computed — used as the fallback AND as the
        # substrate for the explanation column on AI-synthesized scenarios.
        heuristic = _extract_description_patterns(
            orig_df, mod, mod_col=mod_col, text_col=text_col,
            build_col=build_col,
        )

        # Resolve BugCodes once per heuristic-pattern row (reused by both branches
        # so AI-synthesized scenarios still carry traceable BugCodes from their substrate).
        has_bugcode = "BugCode" in orig_df.columns
        def _bugcodes_for(pattern: dict) -> str:
            if not has_bugcode:
                return ""
            member_rows = pattern.get("member_row_indices", []) or []
            codes = []
            for ri in member_rows[:5]:
                if ri in orig_df.index:
                    val = orig_df.at[ri, "BugCode"]
                    if pd.notna(val) and str(val).strip():
                        codes.append(str(val).strip())
            return " | ".join(codes)

        # Parallel descriptions — same row order, same slice as `_bugcodes_for`,
        # so position N of source_bug_examples corresponds to position N of
        # source_bug_codes. This is the source of truth for the "BugCode ↔
        # description" pairing the dashboard renders.
        def _descs_for(pattern: dict) -> str:
            member_rows = pattern.get("member_row_indices", []) or []
            descs = []
            for ri in member_rows[:5]:
                if ri in orig_df.index:
                    val = orig_df.at[ri, text_col]
                    if pd.notna(val) and str(val).strip():
                        descs.append(str(val).strip())
            return " | ".join(descs)

        # Fallback to heuristic if AI returned nothing or AI was disabled
        if not scenarios:
            for i, h in enumerate(heuristic):
                explanation = _build_scenario_explanation(
                    module=mod, mod_row=mod_row, scenario=h, orig_df=orig_df,
                    mod_col=mod_col, build_col=build_col,
                    scenario_type="historical_pattern",
                )
                # Prefer descriptions read directly from raw bugs (parallel to
                # source_bug_codes). Fall back to the heuristic's source_examples
                # only if the row-based lookup is empty (e.g., no BugCode column).
                _desc_field = _descs_for(h) or h.get("source_examples", "")
                all_rows.append({
                    "module": mod,
                    "risk_level": risk_level,
                    "predicted_build": predicted_build,
                    "scenario_rank": i + 1,
                    "scenario_type": "historical_pattern",
                    "scenario_text": h["scenario_text"],
                    "confidence": h["confidence"],
                    "source_bug_examples": _desc_field,
                    "source_bug_codes": _bugcodes_for(h),
                    "supporting_categories": supporting_cats,
                    "leading_signal": leading_signal,
                    "explanation": explanation,
                })
        else:
            for i, s in enumerate(scenarios[:SCENARIOS_PER_MODULE]):
                if isinstance(s, str):
                    s = {"scenario": s, "confidence": "medium", "based_on": "", "explanation": ""}
                scenario_text = s.get("scenario", s.get("scenario_text", ""))
                confidence = s.get("confidence", "medium")
                if confidence not in ("high", "medium", "low"):
                    confidence = "medium"
                based_on = s.get("based_on", "")
                substrate = heuristic[i] if i < len(heuristic) else {
                    "source_count": 0, "cluster_top_terms": [],
                    "member_row_indices": [],
                }
                # If AI returned its own explanation, prefer it; otherwise
                # synthesize one from the nearest heuristic pattern we have.
                ai_expl = str(s.get("explanation", "")).strip()
                if ai_expl:
                    explanation = ai_expl
                else:
                    explanation = _build_scenario_explanation(
                        module=mod, mod_row=mod_row, scenario=substrate,
                        orig_df=orig_df, mod_col=mod_col, build_col=build_col,
                        scenario_type="ai_synthesized",
                    )
                # Use descriptions parallel to the substrate's BugCodes, NOT the
                # AI's free-form `based_on` text — those have no row-level
                # correspondence and would mismatch the BugCode list.
                _desc_field = _descs_for(substrate) or based_on
                all_rows.append({
                    "module": mod,
                    "risk_level": risk_level,
                    "predicted_build": predicted_build,
                    "scenario_rank": i + 1,
                    "scenario_type": "ai_synthesized",
                    "scenario_text": scenario_text,
                    "confidence": confidence,
                    "source_bug_examples": _desc_field,
                    "source_bug_codes": _bugcodes_for(substrate),
                    "supporting_categories": supporting_cats,
                    "leading_signal": leading_signal,
                    "explanation": explanation,
                })

    if not all_rows:
        return None

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# AI narrative — Claude  (updated for v3.0 cluster_forecast param)
# ─────────────────────────────────────────────────────────────────────────────

def generate_ai_narrative(module, predicted, risk_level, dominant_bug_type,
                           leading_signal, descriptions, api_key,
                           cluster_forecast: "dict | None" = None,
                           category_forecast: "dict | None" = None) -> str:
    desc_block = "\n".join(f" - {d}" for d in descriptions[:12]) if descriptions \
        else " (no description samples available)"

    # Build category forecast block (preferred over cluster)
    type_block = ""
    if category_forecast:
        lines = []
        for cat, pct in list(category_forecast.items())[:6]:
            lines.append(f"  • {cat} ({pct*100:.0f}% of recent bugs)")
        type_block = (
            "\n\nExpected bug category distribution for next version "
            f"(risk level: {risk_level}):\n" + "\n".join(lines)
        )
    elif cluster_forecast:
        lines = []
        for label, count in list(cluster_forecast.items())[:5]:
            lines.append(f"  • '{label}'")
        type_block = (
            "\n\nExpected bug themes for next version:\n" + "\n".join(lines)
        )

    prompt = (
        f"You are a QA lead writing a pre-version risk briefing for your team. "
        f"Focus on what's been breaking RECENTLY — not lifetime history.\n\n"
        f"Module: {module}\n"
        f"Risk level: {risk_level}\n"
        f"Dominant bug type in recent versions: {dominant_bug_type}\n"
        f"Strongest predictive signal: {leading_signal}"
        f"{type_block}\n\n"
        f"Bug descriptions from RECENT versions of this module:\n{desc_block}\n\n"
        f"Write a concise 3-5 sentence paragraph that:\n"
        f"1. Describes WHAT specific parts of this feature are most likely to break next version "
        f"(name categories from the distribution above; reference concrete failure patterns from the descriptions).\n"
        f"2. Identifies which flows or sub-areas to prioritise testing — be specific (which button, dialog, "
        f"file format, or interaction), not generic ('test the feature').\n"
        f"3. Highlights any RECURRING patterns visible across multiple recent descriptions.\n\n"
        f"Do NOT state or estimate a total bug count. Do NOT speculate about modes not supported by the "
        f"descriptions above. Plain prose only. No bullet points. No headers."
    )
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 350,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data["content"][0]["text"].strip()
    except Exception as e:
        return f"[Claude narrative unavailable: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# AI narrative — Ollama  (updated for v3.0 cluster_forecast param)
# ─────────────────────────────────────────────────────────────────────────────

def generate_ai_narrative_ollama(module, predicted, risk_level, dominant_bug_type,
                                  leading_signal, descriptions, model="qwen3:4b",
                                  cluster_forecast: "dict | None" = None,
                                  category_forecast: "dict | None" = None) -> str:
    desc_block = "\n".join(f" - {d}" for d in descriptions[:12]) if descriptions \
        else " (no description samples available)"

    type_block = ""
    if category_forecast:
        lines = []
        for cat, pct in list(category_forecast.items())[:6]:
            lines.append(f"  • {cat} ({pct*100:.0f}% of recent bugs)")
        type_block = (
            "\n\nExpected bug category distribution for next version "
            f"(risk level: {risk_level}):\n" + "\n".join(lines)
        )
    elif cluster_forecast:
        lines = []
        for label, count in list(cluster_forecast.items())[:5]:
            lines.append(f"  • '{label}'")
        type_block = (
            "\n\nExpected bug themes for next version:\n" + "\n".join(lines)
        )

    prompt = (
        f"You are a QA lead writing a pre-version risk briefing for your team. "
        f"Focus on what's been breaking RECENTLY — not lifetime history.\n\n"
        f"Module: {module}\n"
        f"Risk level: {risk_level}\n"
        f"Dominant bug type in recent versions: {dominant_bug_type}\n"
        f"Strongest predictive signal: {leading_signal}"
        f"{type_block}\n\n"
        f"Bug descriptions from RECENT versions of this module:\n{desc_block}\n\n"
        f"Write a concise 3-5 sentence paragraph that:\n"
        f"1. Describes WHAT specific parts of this feature are most likely to break next version "
        f"(name categories from the distribution above; reference concrete failure patterns from the descriptions).\n"
        f"2. Identifies which flows or sub-areas to prioritise testing — be specific (which button, dialog, "
        f"file format, or interaction), not generic ('test the feature').\n"
        f"3. Highlights any RECURRING patterns visible across multiple recent descriptions.\n\n"
        f"Do NOT state or estimate a total bug count. Do NOT speculate about modes not supported by the "
        f"descriptions above. Plain prose only. No bullet points. No headers.\n"
        # Free-text tasks let reasoning models narrate their own instructions
        # into the answer; a single JSON field keeps the briefing clean.
        f'Return exactly this JSON object and nothing else:\n'
        f'{{"briefing": "<the 3-5 sentence paragraph>"}}'
    )
    payload = json.dumps(with_thinking_disabled({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2, "num_predict": 600},
    })).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode())
            text = response_text(raw)
            try:
                parsed = extract_json(text, expect="object")
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                for key in ("briefing", "narrative", "summary", "text"):
                    value = str(parsed.get(key) or "").strip()
                    if value:
                        return strip_meta_preamble(value)
            return strip_meta_preamble(text)
    except Exception as e:
        return f"[Ollama narrative unavailable: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# Focus summary  (updated to pass cluster_forecast to narratives)
# ─────────────────────────────────────────────────────────────────────────────

def generate_focus_summary(preds, leading, orig_df,
                            mod_col="parsed_module", top_n=8,
                            api_key="", provider="none", model="qwen3:4b",
                            cluster_preds_df: "pd.DataFrame | None" = None,
                            category_preds_df: "pd.DataFrame | None" = None,
                            scenario_df: "pd.DataFrame | None" = None) -> str:
    lines = [
        "=" * 78,
        " PDR-I DEFECT PREDICTION — FOCUS SUMMARY FOR NEXT BUILD",
        "=" * 78, "",
        " Modules ranked by risk level. Category breakdown shows WHAT bugs to expect.", "",
    ]

    dom_type_series = preds.set_index("module").get(
        "dominant_bug_type", pd.Series(dtype=str))

    top_feat       = leading.index[0] if len(leading) else "bug_count"
    top_feat_label = _FEATURE_LABELS.get(top_feat, top_feat)

    high_risk = preds[preds["risk_level"].isin(["Critical", "High"])].head(top_n)
    if high_risk.empty:
        high_risk = preds.head(min(top_n, len(preds)))

    for _, row in tqdm(high_risk.iterrows(), total=len(high_risk),
                       desc="Building focus summary", unit="mod",
                       file=sys.stderr, dynamic_ncols=True):
        mod   = row["module"]
        pred  = row["predicted"]
        rl    = str(row["risk_level"])
        dtype = str(row.get("dominant_bug_type",
                             dom_type_series.get(mod, "mixed")))

        if "leading_signal" in row and str(row["leading_signal"]) not in ("", "nan"):
            mod_feat_label = str(row["leading_signal"])
        else:
            fcols = [c for c in preds.columns
                     if c not in ["module", "build", "target", "predicted", "risk_level",
                                   "dominant_bug_type", "leading_signal", "ai_narrative"]
                     and not c.startswith(TFIDF_PREFIX)]
            mod_rows = preds[preds["module"] == mod]
            mod_feat_label = top_feat_label
            if fcols and len(mod_rows) > 1:
                best_corr, best_col = 0.0, fcols[0]
                y_mod = mod_rows["target"].fillna(0)
                for fc in fcols:
                    x_mod = mod_rows[fc].fillna(0)
                    if x_mod.std() > 0 and y_mod.std() > 0:
                        r = abs(float(np.corrcoef(x_mod, y_mod)[0, 1]))
                        if r > best_corr:
                            best_corr, best_col = r, fc
                mod_feat_label = _FEATURE_LABELS.get(best_col, best_col)

        # Build category_forecast dict for this module (preferred)
        category_forecast = None
        if category_preds_df is not None and not category_preds_df.empty:
            mod_cats = category_preds_df[category_preds_df["module"] == mod].head(6)
            if not mod_cats.empty:
                category_forecast = dict(
                    zip(mod_cats["category"], mod_cats["historical_pct"])
                )

        # Fall back to cluster_forecast if no category data
        cluster_forecast = None
        if category_forecast is None and cluster_preds_df is not None and not cluster_preds_df.empty:
            mod_clusters = cluster_preds_df[cluster_preds_df["module"] == mod].head(5)
            if not mod_clusters.empty:
                cluster_forecast = dict(
                    zip(mod_clusters["cluster_label"],
                        mod_clusters["predicted_count"])
                )

        lines.append(f"  {'─' * 74}")
        lines.append(f"  📍 {mod}  [{rl} risk]")
        lines.append(f"     Leading signal : {mod_feat_label}")

        if category_forecast:
            lines.append(f"     Expected bug categories (from recent history):")
            for cat, pct in list(category_forecast.items())[:4]:
                lines.append(f"       • {cat} ({pct*100:.0f}%)")
        elif cluster_forecast:
            lines.append(f"     Expected bug themes:")
            for lbl, cnt in list(cluster_forecast.items())[:4]:
                lines.append(f"       • {lbl}")

        # v5.0 — append top predicted scenarios
        if scenario_df is not None and not scenario_df.empty:
            mod_scenarios = scenario_df[scenario_df["module"] == mod].head(3)
            if not mod_scenarios.empty:
                lines.append(f"     Predicted bug scenarios:")
                for _, sc in mod_scenarios.iterrows():
                    conf = str(sc.get("confidence", ""))
                    text = str(sc.get("scenario_text", ""))
                    lines.append(f"       → [{conf}] {text}")

        if provider == "ollama":
            descriptions = _sample_descriptions(orig_df, mod, mod_col=mod_col)
            narrative = generate_ai_narrative_ollama(
                module=mod, predicted=pred, risk_level=rl,
                dominant_bug_type=dtype, leading_signal=mod_feat_label,
                descriptions=descriptions, model=model,
                cluster_forecast=cluster_forecast,
                category_forecast=category_forecast,
            )
            for narrative_line in narrative.splitlines():
                lines.append(f"  {narrative_line}")

        elif provider == "claude" and api_key:
            descriptions = _sample_descriptions(orig_df, mod, mod_col=mod_col)
            narrative = generate_ai_narrative(
                module=mod, predicted=pred, risk_level=rl,
                dominant_bug_type=dtype, leading_signal=mod_feat_label,
                descriptions=descriptions, api_key=api_key,
                cluster_forecast=cluster_forecast,
                category_forecast=category_forecast,
            )
            for narrative_line in narrative.splitlines():
                lines.append(f"  {narrative_line}")

        else:
            lines.append(f"  Bug type to expect : {dtype}")
            lines.append(f"  Leading signal     : {mod_feat_label}")
            lines.append(f"  Testing advice     : {_RISK_ADVICE.get(rl, 'Review.')}")

    lines += ["", "=" * 78, ""]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(PREDICTION_GUIDE)

    if len(sys.argv) < 2:
        print("Usage: python predict_defects.py <input_csv> "
              "[--provider claude|ollama|none] [--model <n>] "
              "[--no-ai] [--ai-scenarios] [--scored-csv <path>] [--api-key <key>] "
              "[--cluster-csv <path>]")
        sys.exit(1)

    args = sys.argv[1:]
    scored_csv_path  = None
    cluster_csv_path = None
    provider = "claude"
    model    = "qwen3:4b"
    api_key  = os.environ.get("ANTHROPIC_API_KEY", "")
    ai_scenarios = False

    if "--no-ai" in args:
        provider = "none"
        args.remove("--no-ai")

    if "--ai-scenarios" in args:
        ai_scenarios = True
        args.remove("--ai-scenarios")

    if "--provider" in args:
        idx = args.index("--provider")
        provider = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if "--model" in args:
        idx = args.index("--model")
        model = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if "--scored-csv" in args:
        idx = args.index("--scored-csv")
        scored_csv_path = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if "--api-key" in args:
        idx = args.index("--api-key")
        api_key = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    # Change 12 — new --cluster-csv argument
    if "--cluster-csv" in args:
        idx = args.index("--cluster-csv")
        cluster_csv_path = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    inp = args[0]
    inp_dir  = Path(inp).parent
    out_dir  = inp_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out      = str(out_dir / (Path(inp).stem + "_predictions.csv"))
    out_stem = out_dir / Path(inp).stem

    if provider == "claude" and not api_key:
        print("  NOTE: --provider claude but ANTHROPIC_API_KEY not set — falling back to none.")
        provider = "none"

    if provider not in ("claude", "ollama", "none"):
        print(f"  ERROR: unknown --provider '{provider}'. Choose: claude, ollama, none")
        sys.exit(1)

    print(f"AI provider: {provider}" + (f" (model={model})" if provider == "ollama" else ""))

    orig_df = pd.read_csv(inp)
    print(f"Loaded {len(orig_df)} bugs")

    # ── Load optional risk scores ─────────────────────────────────────────────
    risk_features = None
    risk_metadata = None
    _scored_path = scored_csv_path
    if _scored_path:
        print(f"\nLoading risk scores from: {_scored_path}")
    else:
        auto_path = inp_dir / "risk_register_scored_all.csv"
        if auto_path.exists():
            print(f"\nAuto-detected risk scores: {auto_path}")
            _scored_path = str(auto_path)
    if _scored_path:
        risk_features = load_risk_features(_scored_path)
        risk_metadata = load_risk_metadata(_scored_path)

    # ── TF-IDF text features ──────────────────────────────────────────────────
    print("\nBuilding TF-IDF text features from parsed_description...")
    tfidf_features = build_tfidf_features(orig_df)

    # ── Cluster features (Change 12) ──────────────────────────────────────────
    cluster_features = None
    if cluster_csv_path:
        print(f"\nBuilding cluster features from: {cluster_csv_path}")
        cluster_features = load_cluster_features(cluster_csv_path, orig_df)
    else:
        # Auto-detect: look in data/clusters/ next to the input CSV
        auto_cluster = inp_dir / "clusters" / (Path(inp).stem + "_clustered.csv")
        if auto_cluster.exists():
            print(f"\nAuto-detected cluster CSV: {auto_cluster}")
            cluster_features = load_cluster_features(str(auto_cluster), orig_df)

    # ── Feature matrix ────────────────────────────────────────────────────────
    fdf = build_features(orig_df,
                         risk_features=risk_features,
                         tfidf_features=tfidf_features,
                         cluster_features=cluster_features)
    print(f"Feature matrix: {fdf.shape}")

    if len(fdf) < 20:
        print("Not enough data for prediction. Need ≥5 versions per module.")
        sys.exit(0)

    model_obj, preds, imp, leading = train_predict(fdf, orig_df)

    preds = preds.sort_values("predicted", ascending=False)

    # ── Change 19 — Stratified training (show both global and stratified) ─
    print("\n" + "─" * 78)
    print(" STRATIFIED TRAINING (separate models for high/low activity modules)")
    print("─" * 78)
    strat_result = train_predict_stratified(fdf, orig_df)
    if strat_result is not None:
        strat_preds, strat_imp = strat_result
        preds = preds.merge(strat_preds, on="module", how="left")
        # Where stratified prediction is missing, fall back to global
        preds["predicted_stratified"] = preds["predicted_stratified"].fillna(preds["predicted"])
        print(f"\n  Stratified predictions merged for {strat_preds['module'].nunique()} modules.")
        print(f"  Compare: predicted (global) vs predicted_stratified in output CSV.")
    else:
        preds["predicted_stratified"] = preds["predicted"]
        print("  Stratified training could not run — using global model only.")

    # ── Change 23 — Holt trend forecast ───────────────────────────────────────
    print("\n" + "─" * 78)
    print(" TREND FORECAST (Holt double exponential smoothing, alpha=0.3 beta=0.1)")
    print("─" * 78)
    trend_result = train_trend_forecast(fdf)
    if not trend_result.empty:
        preds = preds.merge(trend_result, on="module", how="left")
        preds["trend_forecast"] = preds["trend_forecast"].fillna(preds["predicted"])
        r = trend_result["trend_forecast"]
        print(f"  Trend forecast: {r.min():.1f}–{r.max():.1f} "
              f"(mean {r.mean():.1f}) across {len(trend_result)} modules.")
    else:
        preds["trend_forecast"] = preds["predicted"]
        print("  Trend forecast could not run.")

    # ── Change 19 — Learned risk classifier (replaces hand-tuned composite) ─
    print("\n" + "─" * 78)
    print(" RISK CLASSIFIER (learned probability of S1 critical bug next version)")
    print("─" * 78)
    clf_result = train_risk_classifier(fdf, orig_df)

    if clf_result is not None:
        risk_clf, risk_proba = clf_result
        preds = preds.merge(risk_proba, on="module", how="left")
        preds["risk_proba"] = preds["risk_proba"].fillna(0.0)

        # Change 24 — Bayesian shrinkage: blend ML output with Beta-Binomial
        # posterior to stabilise sparse modules toward the global S1 rate.
        preds["risk_proba"] = apply_bayesian_shrinkage(preds, orig_df)
        print(f"  Bayesian shrinkage applied — range: "
              f"{preds['risk_proba'].min():.3f}–{preds['risk_proba'].max():.3f}")

        # Use learned probability as the primary risk score (0–100)
        preds["composite_risk"] = (preds["risk_proba"] * 100).round(2)
        risk_method = "learned classifier + Bayesian shrinkage (P(S1 critical bug))"
    else:
        # Fallback to weighted composite when classifier cannot train
        print("  Falling back to weighted composite scoring.")
        _rs = pd.to_numeric(preds.get("risk_score_final", 0), errors="coerce").fillna(0.0)
        _pred = pd.to_numeric(preds["predicted"], errors="coerce").fillna(0.0)
        _sev = pd.to_numeric(preds.get("severity_escalation", 0), errors="coerce").fillna(0.0)
        _trend = pd.to_numeric(preds.get("trend", 0), errors="coerce").fillna(0.0)

        rs_max = max(_rs.max(), 1.0)
        norm_rs = (_rs / rs_max).clip(0, 1)
        pred_max = max(_pred.max(), 1.0)
        norm_pred = (_pred / pred_max).clip(0, 1)
        norm_sev = (-_sev).clip(0, 1)
        trend_max = max(_trend.max(), 1.0)
        norm_trend = (_trend.clip(0, None) / trend_max).clip(0, 1)

        raw_composite = (
            0.50 * norm_rs +
            0.20 * norm_pred +
            0.20 * norm_sev +
            0.10 * norm_trend
        )
        preds["composite_risk"] = (raw_composite * 100).round(2)
        risk_method = "weighted composite (fallback)"

    # ── Blended priority_score — aligns Forecast ranking with Heatmap/Clusters ──
    # The Risk Heatmap sorts by risk_score_final (I×P×D). The forecast's own
    # ML-derived risk_proba and the cluster velocity add information, but the
    # heatmap signal must dominate so the top-N lists overlap. A structural
    # quadrant bonus guarantees a P1 module always outscores a P4 regardless
    # of ML/velocity signal strength.
    # Weights: 50% heatmap, 25% quadrant tier, 15% ML, 10% velocity.
    def _norm(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").fillna(0.0)
        max_v = float(s.max()) if len(s) else 0.0
        return (s / max_v).clip(0, 1) if max_v > 0 else s * 0.0

    # Merge module cluster velocity if we can find the cluster summary
    mod_velocity = None
    if cluster_features is not None and cluster_csv_path:
        mod_velocity = load_module_cluster_velocity(cluster_csv_path, inp_dir)
    elif cluster_features is not None:
        auto_cluster = inp_dir / "clusters" / (Path(inp).stem + "_clustered.csv")
        if auto_cluster.exists():
            mod_velocity = load_module_cluster_velocity(str(auto_cluster), inp_dir)
    if mod_velocity is not None and not mod_velocity.empty:
        preds = preds.merge(mod_velocity, left_on="module", right_index=True, how="left")
    else:
        preds["module_cluster_velocity"] = float("nan")

    # Pull heatmap_quadrant (P1–P4) from the risk register metadata if present
    if risk_metadata is not None and "quadrant" in risk_metadata.columns:
        preds = preds.merge(
            risk_metadata[["quadrant"]].rename(columns={"quadrant": "heatmap_quadrant"}),
            left_on="module", right_index=True, how="left",
        )
    else:
        preds["heatmap_quadrant"] = pd.NA

    # Also surface regression_rate / open_count for the explanation column
    for extra_col in ("regression_rate", "open_count"):
        if (risk_metadata is not None and extra_col in risk_metadata.columns
                and extra_col not in preds.columns):
            preds = preds.merge(
                risk_metadata[[extra_col]],
                left_on="module", right_index=True, how="left",
            )

    rs = pd.to_numeric(preds.get("risk_score_final", 0), errors="coerce").fillna(0.0)
    rp = pd.to_numeric(preds.get("risk_proba", 0), errors="coerce").fillna(0.0)
    vel = pd.to_numeric(preds.get("module_cluster_velocity", 1.0), errors="coerce")
    # For the blend we treat NaN velocity as neutral (=1.0) and cap positive
    # signal at 2×; undefined velocity shouldn't penalize a module.
    vel_signal = (vel.fillna(1.0) - 1.0).clip(lower=0, upper=1.0)

    q_raw = preds.get("heatmap_quadrant", pd.Series(pd.NA, index=preds.index))
    quadrant_bonus = (
        q_raw.astype("string").str.upper().str.extract(r"(P[1-4])")[0]
             .map(QUADRANT_BONUS).fillna(0.0)
    )

    preds["priority_score"] = (
        0.50 * _norm(rs) +
        0.25 * quadrant_bonus +
        0.15 * _norm(rp) +
        0.10 * vel_signal
    ) * 100
    preds["priority_score"] = preds["priority_score"].round(2)

    # Recency multiplier — continuous scaling by recent severity-weighted activity.
    # FMEA static signals (risk_score_final + quadrant_bonus = 75% of the blend)
    # don't decay, so a P1 module that's been quiet still scores high. We
    # multiply the final priority_score by a 0.4–1.0 factor driven by recent
    # bug volume × severity. Effect:
    #   • Modules with peak recent activity → multiplier = 1.0 (no change)
    #   • Modules with no recent activity   → multiplier = 0.4 (heavy damp)
    # The intensity score combines bug count, critical count, and severity
    # weight from the last 3 builds plus the latest target. It's normalized
    # per-product against the 90th percentile so a busy product doesn't
    # under-damp its quietest modules.
    bugs_recent = pd.to_numeric(preds.get("bugs_3", 0), errors="coerce").fillna(0)
    crit_recent = pd.to_numeric(preds.get("crit_3", 0), errors="coerce").fillna(0)
    sev_recent  = pd.to_numeric(preds.get("sev_3", 0), errors="coerce").fillna(0)
    target_last = pd.to_numeric(preds.get("target", 0), errors="coerce").fillna(0)

    intensity_raw = bugs_recent + 2.0 * crit_recent + 0.5 * sev_recent + 0.5 * target_last
    p90_intensity = float(intensity_raw.quantile(0.90))
    if p90_intensity > 0:
        intensity_norm = (intensity_raw / p90_intensity).clip(upper=1.0).clip(lower=0.0)
    else:
        intensity_norm = pd.Series(0.0, index=preds.index)
    recency_mult = 0.4 + 0.6 * intensity_norm
    preds["priority_score"] = (preds["priority_score"] * recency_mult).round(2)
    preds["recency_multiplier"] = recency_mult.round(3)
    n_damped = int((recency_mult < 0.7).sum())
    n_full   = int((recency_mult >= 0.95).sum())
    print(f"  Recency multiplier applied: {n_damped} module(s) damped <0.7×, "
          f"{n_full} unchanged at near-1.0× (intensity p90={p90_intensity:.2f}).")

    # ── risk_level from priority_score percentile thresholds ──────────────────
    # priority_score blends FMEA (50%), quadrant tier (25%), ML (15%), and
    # cluster velocity (10%). Using per-product relative percentiles prevents
    # the ML's compressed probability range (products with few S1 bugs) from
    # locking all modules into Medium/Low. When ML evidence is thin, FMEA
    # domain risk carries the classification.
    # Percentile bands: Critical = top 10%, High = 70–90th, Medium = 40–70th,
    # Low = bottom 40%.
    _ps = preds["priority_score"]
    _p40 = float(_ps.quantile(0.40))
    _p70 = float(_ps.quantile(0.70))
    _p90 = float(_ps.quantile(0.90))
    _rl = pd.Series("Low", index=preds.index, dtype=str)
    _rl[_ps >= _p40] = "Medium"
    _rl[_ps >= _p70] = "High"
    _rl[_ps >= _p90] = "Critical"
    # Absolute floor — break the percentile guarantee. A calm release with no
    # truly high-risk modules shouldn't auto-promote the top 10% to Critical.
    _rl[(_ps < 40.0) & (_rl == "High")] = "Medium"
    _rl[(_ps < 55.0) & (_rl == "Critical")] = "High"
    preds["risk_level"] = _rl
    risk_thresholds = (
        f"Critical ≥p90 (≥{_p90:.1f}) | High ≥p70 (≥{_p70:.1f}) | "
        f"Medium ≥p40 (≥{_p40:.1f}) | Low <{_p40:.1f}  [per-product priority_score percentiles]"
    )

    # Log the breakdown
    rl_counts = preds["risk_level"].value_counts()
    print(f"\n  Risk scoring method: {risk_method}")
    print(f"    ML composite_risk range: {preds['composite_risk'].min():.1f}–{preds['composite_risk'].max():.1f}")
    print(f"    risk_level source: priority_score percentiles")
    print(f"    Thresholds — {risk_thresholds}")
    for lvl in ["Critical", "High", "Medium", "Low"]:
        print(f"    {lvl}: {rl_counts.get(lvl, 0)} module(s)")

    print("\n" + "─" * 78)
    print(" PREDICTED BUG COUNT — next version estimate per module")
    print(" target = actual bugs in most recent version | predicted = next version forecast")
    print(f" risk scoring: {risk_method}")
    print(f" {risk_thresholds}")
    print("─" * 78)
    display_cols = ["module", "target", "predicted", "predicted_stratified",
                    "trend_forecast", "priority_score", "composite_risk", "risk_level",
                    "heatmap_quadrant", "module_cluster_velocity",
                    "dominant_bug_type", "leading_signal"]
    display_cols = [c for c in display_cols if c in preds.columns]
    sort_col = "priority_score" if "priority_score" in preds.columns else "composite_risk"
    preds = preds.sort_values(sort_col, ascending=False)
    print(preds[display_cols].head(20).to_string(index=False))

    # ── Bug-type prediction (Change 13 — cluster-based) ──────────────────────
    cluster_preds_df = None
    cluster_csv_for_type = cluster_csv_path
    if cluster_csv_for_type is None:
        auto_cluster = inp_dir / "clusters" / (Path(inp).stem + "_clustered.csv")
        if auto_cluster.exists():
            cluster_csv_for_type = str(auto_cluster)

    if cluster_csv_for_type:
        print("\nBuilding per-cluster bug-type predictions …")
        cluster_preds_df = predict_bug_type(preds, orig_df, cluster_csv_for_type)
        if cluster_preds_df is not None:
            by_cluster_path = str(out_stem) + "_predictions_by_cluster.csv"
            cluster_preds_df.to_csv(by_cluster_path, index=False, encoding="utf-8-sig")
            print(f"  Bug-type predictions saved → {by_cluster_path}")
            print(f"  ({len(cluster_preds_df)} module×cluster rows)")

    # ── Change 15 — QA category breakdown prediction (always runs) ───────────
    print("\nBuilding QA category breakdown predictions …")
    category_preds_df = predict_category_breakdown(preds, orig_df)
    if category_preds_df is not None:
        print("  Finding similar existing bugs for sparse bug-type categories …")
        category_preds_df = generate_category_ai_descriptions(
            category_preds_df, orig_df, provider=provider, model=model,
        )
        by_category_path = str(out_stem) + "_predictions_by_category.csv"
        category_preds_df.to_csv(by_category_path, index=False, encoding="utf-8-sig")
        print(f"  Category predictions saved → {by_category_path}")
        print(f"  ({len(category_preds_df)} module×category rows)")

    # ── Change 18 — Bug scenario predictions (always runs) ──────────────────
    print("\nGenerating bug scenario predictions ...")
    scenario_df = generate_bug_scenarios(
        preds, orig_df, category_preds_df, cluster_preds_df,
        provider=provider, model=model, api_key=api_key,
        ai_scenarios=ai_scenarios,
    )
    if scenario_df is not None:
        scenario_path = str(out_stem) + "_predictions_by_scenario.csv"
        scenario_df.to_csv(scenario_path, index=False, encoding="utf-8-sig")
        print(f"  Scenario predictions saved → {scenario_path}")
        print(f"  ({len(scenario_df)} scenario rows across {scenario_df['module'].nunique()} modules)")
    else:
        print("  NOTE: No scenario predictions could be generated.")

    # ── Focus summary ─────────────────────────────────────────────────────────
    use_ai = provider in ("claude", "ollama")
    print(f"\nGenerating focus summary (provider={provider})...")
    summary_text = generate_focus_summary(
        preds, leading, orig_df,
        top_n=8, api_key=api_key,
        provider=provider, model=model,
        cluster_preds_df=cluster_preds_df,
        category_preds_df=category_preds_df,
        scenario_df=scenario_df,
    )
    print("\n" + summary_text)

    # ── Per-module AI narratives ───────────────────────────────────────────────
    preds["ai_narrative"] = ""
    if use_ai:
        high_risk_mods = (
            preds[preds["risk_level"].isin(["Critical", "High"])]
            .head(8)["module"].tolist()
        )
        for mod in tqdm(high_risk_mods, desc="Generating narratives", unit="mod",
                        file=sys.stderr, dynamic_ncols=True):
            row = preds[preds["module"] == mod].iloc[0]
            descriptions = _sample_descriptions(orig_df, mod)

            # Build category_forecast for this module (preferred)
            category_forecast = None
            if category_preds_df is not None and not category_preds_df.empty:
                mod_cats = category_preds_df[category_preds_df["module"] == mod].head(6)
                if not mod_cats.empty:
                    category_forecast = dict(
                        zip(mod_cats["category"], mod_cats["historical_pct"]))

            # Fall back to cluster_forecast
            cluster_forecast = None
            if category_forecast is None and cluster_preds_df is not None and not cluster_preds_df.empty:
                mod_c = cluster_preds_df[cluster_preds_df["module"] == mod].head(5)
                if not mod_c.empty:
                    cluster_forecast = dict(
                        zip(mod_c["cluster_label"], mod_c["predicted_count"]))

            kwargs = dict(
                module=mod,
                predicted=float(row["predicted"]),
                risk_level=str(row["risk_level"]),
                dominant_bug_type=str(row.get("dominant_bug_type", "mixed")),
                leading_signal=str(row.get("leading_signal", "")),
                descriptions=descriptions,
                cluster_forecast=cluster_forecast,
                category_forecast=category_forecast,
            )
            if provider == "ollama":
                narrative = generate_ai_narrative_ollama(**kwargs, model=model)
            else:
                narrative = generate_ai_narrative(**kwargs, api_key=api_key)
            preds.loc[preds["module"] == mod, "ai_narrative"] = narrative

    # ── Save outputs ──────────────────────────────────────────────────────────
    summary_path = str(out_stem) + "_predictions_focus_summary.txt"
    Path(summary_path).write_text(summary_text, encoding="utf-8")

    save_cols = [c for c in preds.columns if not c.startswith(TFIDF_PREFIX)]
    preds[save_cols].to_csv(out, index=False, encoding="utf-8-sig")

    imp_path     = str(out_stem) + "_predictions_importance.csv"
    leading_path = str(out_stem) + "_predictions_leading_indicators.csv"
    imp_df = imp.rename_axis("feature").reset_index(name="importance")
    imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")

    leading_df = leading.reset_index()
    leading_df.columns = ["feature", "pearson_r"]
    leading_df["label"] = leading_df["feature"].map(_FEATURE_LABELS).fillna(
        leading_df["feature"])
    leading_df.to_csv(leading_path, index=False, encoding="utf-8-sig")

    print(f"\nOutputs:")
    print(f"  {out}              ← main predictions (includes ai_narrative)")
    if category_preds_df is not None:
        print(f"  {out_stem}_predictions_by_category.csv ← expected bug categories per module")
    if scenario_df is not None:
        print(f"  {out_stem}_predictions_by_scenario.csv ← concrete bug scenario predictions")
    if cluster_preds_df is not None:
        print(f"  {out_stem}_predictions_by_cluster.csv  ← bug-type forecast per cluster")
    print(f"  {imp_path}         ← model feature importances")
    print(f"  {leading_path}     ← leading indicator correlations")
    print(f"  {summary_path}     ← plain-English focus summary")


if __name__ == "__main__":
    main()