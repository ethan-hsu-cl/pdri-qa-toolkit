# PDR-I Pipeline — Integration Guide v3.0

This document describes the full run order for all PDR-I scripts, the files
each one reads and writes, and how the v3.0 clustering and prediction
improvements connect to the rest of the pipeline.

---

## Pipeline overview

```
fetch_from_n8n.py
       │  ecl_raw.json
       ▼
parse_ecl_export.py
       │  ecl_parsed.csv  +  version_catalogue.csv
       ▼
compute_risk_scores.py ──────────────────────────────────────────────┐
       │  risk_register_all.csv                                       │
       │  risk_register_versions/risk_register_<ver>.csv             │
       ▼                                                              │
ai_risk_scorer.py                                                     │
       │  risk_register_scored_all.csv                               │
       │  risk_register_versions/risk_register_scored_<ver>.csv      │
       ▼                                                              │
 ┌─────────────────────────────┐                                     │
 │     cluster_bugs.py (v3.0)  │◄────────────── ecl_parsed.csv       │
 │  clusters/                  │                                     │
 │    *_clustered.csv          │                                     │
 │    *_cluster_summary.csv    │   ← adds velocity, trend,           │
 │    *_module_entropy.csv     │     recurrence_rate                 │
 │    *_cluster_summary_s12    │   ← new: S1/S2 tier                 │
 │    *_cluster_summary_s34    │   ← new: S3/S4 tier                 │
 └─────────────┬───────────────┘                                     │
               │                                                     │
               ▼                                                     │
 ┌─────────────────────────────┐                                     │
 │   predict_defects.py (v3.0) │◄──── ecl_parsed.csv                 │
 │                             │◄──── *_clustered.csv    (optional)  │
 │                             │◄──── risk_register_scored_all.csv  ◄┘
 │  predictions/               │
 │    *_predictions.csv        │   ← total count + risk level
 │    *_predictions_by_cluster │   ← NEW: "what" bug per theme
 │    *_predictions_importance │
 │    *_predictions_leading_   │
 │    *_predictions_focus_sum  │
 └─────────────┬───────────────┘
               │
               ▼
 ┌─────────────────────────────┐
 │   auto_tag_tests.py (v3.0)  │◄──── risk_register_scored_all.csv
 │                             │◄──── *_predictions_by_cluster.csv
 │  tests/generated/           │   ← test_<module>.py
 │  data/quadrant_summary.md   │   ← theme column added for P1/P2
 │  data/cluster_test_plan.md  │   ← NEW: scenario-level test plan
 └─────────────────────────────┘

 bug_heatmap_dashboard.py  ← reads all outputs above; no step in the
                              run order, just start with streamlit run

 process_newest_tr.py      ← newly Close status transitions from ecl_parsed.csv
       │                     fetches full status logs through the TR n8n webhook
       └─ tr_reports/<build>/bug_details.json + test_focus_summary.md
                              shown in the dedicated TR Test Focus tab
```

---

## Step-by-step run order

### Step 1 — Fetch bugs

```bash
python scripts/fetch_from_n8n.py --output data/ecl_raw.json
# Or fetch + parse in one command:
python scripts/fetch_from_n8n.py --then-parse
```

**Inputs:** n8n webhook  
**Outputs:** `data/ecl_raw.json`

Run cadence: daily (weekdays fetch latest version only; weekends fetch all).

---

### Step 2 — Parse

```bash
python scripts/parse_ecl_export.py data/ecl_raw.json data/ecl_parsed.csv
```

**Inputs:** `data/ecl_raw.json`  
**Outputs:** `data/ecl_parsed.csv`, `data/version_catalogue.csv`

No changes in v3.0. The two new downstream features (`severity_escalation` and
`builds_since_last_crit`) are derived inside `predict_defects.py` from columns
that `parse_ecl_export.py` already produces (`severity_num`, `Build#`).

---

### Step 3 — Compute risk registers

```bash
python scripts/compute_risk_scores.py data/ecl_parsed.csv data/risk_register_all.csv
```

**Inputs:** `data/ecl_parsed.csv`  
**Outputs:**
- `data/risk_register_all.csv`
- `data/risk_register_versions/risk_register_<ver>.csv` (one per non-sparse version)

No changes in v3.0.

---

### Step 4 — Score risk registers

```bash
# Fast (heuristic, no AI needed):
python scripts/ai_risk_scorer.py data/risk_register_all.csv \
    data/risk_register_scored_all.csv --provider heuristic

# With Ollama (richer reasoning):
python scripts/ai_risk_scorer.py data/risk_register_all.csv \
    data/risk_register_scored_all.csv --provider ollama --model qwen3:7b
```

**Inputs:** `data/risk_register_all.csv` + per-version files from Step 3  
**Outputs:**
- `data/risk_register_scored_all.csv`
- `data/risk_register_versions/risk_register_scored_<ver>.csv`

No changes in v3.0. Resume support is built in — safe to re-run after a crash.

---

### Step 5 — Cluster bugs  *(updated in v3.0)*

```bash
# Minimum (TF-IDF, no Ollama):
python scripts/cluster_bugs.py data/ecl_parsed.csv \
    data/clusters/ecl_parsed_clustered.csv

# Recommended (Ollama + severity stratification):
python scripts/cluster_bugs.py data/ecl_parsed.csv \
    data/clusters/ecl_parsed_clustered.csv \
    --provider ollama --model qwen3:7b \
    --stratify-severity
```

**Inputs:** `data/ecl_parsed.csv`  
**Outputs (v3.0):**

| File | Description |
|------|-------------|
| `clusters/ecl_parsed_clustered.csv` | Bug-level file with `cluster_id`, `cluster_label`, and (if `--stratify-severity`) `cluster_id_s12`, `cluster_label_s12`, `cluster_id_s34`, `cluster_label_s34` |
| `clusters/ecl_parsed_cluster_summary.csv` | Per-cluster summary including `cluster_velocity_ratio`, `cluster_trend`, `recurrence_rate` |
| `clusters/ecl_parsed_module_entropy.csv` | Per-module `cluster_entropy` score (higher = broader instability) |
| `clusters/ecl_parsed_cluster_summary_s12.csv` | Summary for S1/S2 tier only (`--stratify-severity`) |
| `clusters/ecl_parsed_cluster_summary_s34.csv` | Summary for S3/S4 tier only (`--stratify-severity`) |

**What the new columns mean:**

`cluster_velocity_ratio` — ratio of bug count in the most recent 3 versions vs the prior 3 versions for each cluster. Values above 1.5 mean that theme is accelerating and should be prioritised in testing. Values below 0.67 mean the theme is declining (fixes may be holding).

`cluster_trend` — "growing", "stable", or "declining". A quick filter: sort the summary by `cluster_trend == "growing"` before the weekly team meeting.

`recurrence_rate` — fraction of recent bugs in this cluster whose source module also contributed to the same cluster in the prior version window. A rate above 0.5 means the root cause is not being fixed — escalate to RD.

`cluster_entropy` (per module) — Shannon entropy of a module's cluster distribution. Below 1.0 = all bugs in one theme (easy to target). Above 2.0 = bugs everywhere (needs comprehensive coverage).

Run cadence: every Friday, or immediately after parsing a new version batch.

---

### Step 6 — Predict defects  *(updated in v3.0)*

```bash
# Without cluster features (same as v2.6):
python scripts/predict_defects.py data/ecl_parsed.csv

# Recommended (with cluster features + bug-type predictions):
python scripts/predict_defects.py data/ecl_parsed.csv \
    --cluster-csv data/clusters/ecl_parsed_clustered.csv \
    --provider claude

# With Ollama narratives:
python scripts/predict_defects.py data/ecl_parsed.csv \
    --cluster-csv data/clusters/ecl_parsed_clustered.csv \
    --provider ollama --model qwen3:7b
```

**Inputs:**
- `data/ecl_parsed.csv`
- `data/clusters/ecl_parsed_clustered.csv` (auto-detected if present)
- `data/risk_register_scored_all.csv` (auto-detected if present)

**Outputs (v3.0):**

| File | Description |
|------|-------------|
| `predictions/ecl_parsed_predictions.csv` | Per-module total count forecast + `severity_escalation` and `builds_since_last_crit` now used as model features |
| `predictions/ecl_parsed_predictions_by_cluster.csv` | **NEW** — per-module, per-theme predicted counts: `module, cluster_id, cluster_label, historical_pct, predicted_count` |
| `predictions/ecl_parsed_predictions_importance.csv` | Feature importances (now includes cluster and severity-direction features) |
| `predictions/ecl_parsed_predictions_leading_indicators.csv` | Pearson-r correlations for all features |
| `predictions/ecl_parsed_predictions_focus_summary.txt` | Plain-English summary; now includes per-theme breakdowns in the text |

**New features added to the model:**

| Feature | What it captures |
|---------|-----------------|
| `severity_escalation` | Mean severity in the last version minus mean in the prior 3 versions. Negative = worsening toward S1. Catches modules about to hit a critical before it shows in counts. |
| `builds_since_last_crit` | How many versions have elapsed since the last critical bug. Modules that haven't had an S1 in a while but have historical S1s are flagged as potentially overdue. |
| `cluster_entropy_3 / _5` | Bug-theme diversity over the last 3 or 5 versions. Rising entropy = the module is accumulating new *kinds* of failures, not just more of the same. |
| `top_cluster_velocity` | Growth rate of the dominant bug theme. Doubles as an early-warning signal for theme-specific regressions. |

Run cadence: after every cluster run (weekly or per version batch).

---

### Step 7 — Generate tests and test plan  *(updated in v3.0)*

```bash
# Generate skeletons with theme-specific test methods:
python scripts/auto_tag_tests.py data/risk_register_scored_all.csv \
    --generate-skeletons tests/generated/ \
    --cluster-predictions data/predictions/ecl_parsed_predictions_by_cluster.csv

# Write the cluster-based test plan for the QA team:
python scripts/auto_tag_tests.py data/risk_register_scored_all.csv \
    --cluster-predictions data/predictions/ecl_parsed_predictions_by_cluster.csv \
    --cluster-plan

# Full run — skeletons + quadrant summary + cluster test plan:
python scripts/auto_tag_tests.py data/risk_register_scored_all.csv \
    --generate-skeletons tests/generated/ \
    --cluster-predictions data/predictions/ecl_parsed_predictions_by_cluster.csv \
    --summary --cluster-plan
```

**Inputs:**
- `data/risk_register_scored_all.csv`
- `data/predictions/ecl_parsed_predictions_by_cluster.csv` (optional but recommended)

**Outputs (v3.0):**

| File | Description |
|------|-------------|
| `tests/generated/test_<module>.py` | One file per module. Now contains one `test_<theme_name>()` method per predicted bug theme, with expected count in the TODO comment and allure severity inferred from the cluster label. |
| `data/quadrant_summary.md` | Same as v2.1 but P1/P2 rows now include a "Predicted bug themes" column showing the top 2 expected cluster labels. |
| `data/cluster_test_plan.md` | **NEW** — scenario-level test plan for P1/P2 modules. One table per module listing each predicted theme, its expected bug count, and a one-sentence test scenario description auto-generated from the cluster label keywords. |

---

### Step 8 — Dashboard

```bash
streamlit run scripts/bug_heatmap_dashboard.py \
    --server.address 0.0.0.0 --server.port 8501
```

No command-line changes. Load files from the sidebar. New in v3.0:

- **Step 3 (clusters):** adds an optional "Module entropy CSV" path field. Stratified summaries (`_s12.csv`, `_s34.csv`) load automatically from their default locations.
- **Step 4 (forecast):** adds an optional "Bug-type predictions CSV" path field (`_by_cluster.csv`).

---

### Newest TR fixed-bug test focus

`refresh_pipeline.sh` now runs this automatically after Release Pulse generation.
It selects all `TRCreated` bugs in the newest product version—the same cohort eCL
labels "RD Resolved Bug" in the Testing Request—then derives the dominant Close
Build# and fetches each bug's full status log through
`(TR Report) Get Bug Details_v1.json`. Ollama then produces:

The retained `ParentTRCode` values are displayed as **Bug Creation TRs**. They
identify the parent TRs where the bugs originated and must not be interpreted
as the Testing Request associated with the displayed Close Build.

- concrete per-bug verification steps;
- device, locale, content, and setting preconditions;
- adjacent modules/sections that share likely disturbed code paths, ranked by risk.

The organized source and report are stored per product under
`data/products/<slug>/tr_reports/<close_build>/`. `tr_reports/latest.json` points
the **TR Test Focus** dashboard tab to the latest successful report. When a later refresh
adds fixes to the same build, records are merged by BugCode before regenerating
the report.

Manual run:

```bash
python scripts/process_newest_tr.py \
  --product pdri \
  --product-dir data/products/pdri \
  --model gemma4:e2b-it-q4_K_M
```

Use `./refresh_pipeline.sh --skip-tr-summary` to disable this stage. If no bugs
changed to Close, the stage exits successfully and leaves the previous report in
place.
- **Tab 8 (Bug Clusters):** velocity chart, module entropy chart, and stratified S1/S2 vs S3/S4 tabs are shown when the corresponding files are loaded.
- **Tab 9 (Defect Forecast):** module forecast cards now show a per-theme breakdown ("~3 bugs — login crash / timeout") under "What to expect."

---

## File dependency map

```
ecl_raw.json
  └─► ecl_parsed.csv
        ├─► version_catalogue.csv
        ├─► risk_register_all.csv
        │     └─► risk_register_scored_all.csv ──────────────────────────────┐
        │                                                                     │
        ├─► clusters/ecl_parsed_clustered.csv ────────────────────────────┐  │
        │     ├─► clusters/ecl_parsed_cluster_summary.csv                 │  │
        │     ├─► clusters/ecl_parsed_module_entropy.csv                  │  │
        │     ├─► clusters/ecl_parsed_cluster_summary_s12.csv             │  │
        │     └─► clusters/ecl_parsed_cluster_summary_s34.csv             │  │
        │                                                                  │  │
        └─► predictions/  (consumes ecl_parsed + clusters + risk) ◄───────┘◄─┘
              ├─► ecl_parsed_predictions.csv
              ├─► ecl_parsed_predictions_by_cluster.csv ──────────────────┐
              ├─► ecl_parsed_predictions_importance.csv                   │
              ├─► ecl_parsed_predictions_leading_indicators.csv           │
              └─► ecl_parsed_predictions_focus_summary.txt                │
                                                                          │
auto_tag_tests.py  (consumes risk_register_scored_all + by_cluster) ◄────┘
  ├─► tests/generated/test_<module>.py
  ├─► data/quadrant_summary.md
  └─► data/cluster_test_plan.md
```

---

## Backward compatibility

All v3.0 changes are strictly additive:

| Script | Old command still works? |
|--------|--------------------------|
| `cluster_bugs.py` | ✅ Yes — `--stratify-severity` is optional; new output files are only written when the flag is passed |
| `predict_defects.py` | ✅ Yes — `--cluster-csv` is optional and auto-detected; new features fall back to zero when the cluster CSV is absent |
| `auto_tag_tests.py` | ✅ Yes — `--cluster-predictions` is optional; without it output is identical to v2.1 |
| `bug_heatmap_dashboard.py` | ✅ Yes — new sidebar fields accept the v3.0 files but are non-blocking; all new tab sections are conditional on their data being loaded |
| `compute_risk_scores.py` | ✅ Unchanged |
| `ai_risk_scorer.py` | ✅ Unchanged |
| `visual_regression.py` | ✅ Unchanged |
| `fetch_from_n8n.py` | Clears stale `_status_changed` flags for records omitted by narrow fetches |
| `parse_ecl_export.py` | ✅ Unchanged |
