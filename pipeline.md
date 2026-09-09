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
       └─ tr_reports/<build>/runs/<run>/ immutable evidence + analysis + report
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
- adjacent modules/sections that share likely disturbed code paths, ranked by risk;
- a normalized module-category target and stable hypothesis ID for future
  category-level outcome matching (not a calibrated failure probability).
- a concrete adjacent trigger and observable failure mode used for explainable,
  same-category feature-match candidates in later TR cohorts.
- a validated scope boundary plus a concrete passing observation that lowers
  the prediction's priority, preventing unbounded shared-path extrapolation.
- optional historical corroboration from resolved, older bugs in the same module
  or category, selected deterministically with lexical similarity and strict
  current-cohort/future leakage exclusion.

The organized source, structured analysis, and rendered report are stored in an
immutable run under `data/products/<slug>/tr_reports/<close_build>/runs/`.
`tr_reports/latest.json` points the **TR Test Focus** dashboard tab to the latest
successful run, while `status.json` records failed, unchanged, and no-cohort
attempts. Evidence is merged by BugCode without replacing rich RD details with a
degraded response, and `latest.json` is switched only after hash validation.
The evidence snapshot includes the current parsed module-category taxonomy, so
taxonomy changes invalidate stale AI analysis and prediction identities.
It also stores the historical-analog selection policy and selected analogs.
Analog selection is capped at one same-module and two adjacent-module records per
source bug, requires a resolved record created before the source, and rejects
module-name-only overlap. It also requires at least two meaningful shared
short-description terms. Analog BugCodes remain corroboration only: the model
must still cite exact current-fix adjacency evidence for every recommendation.
Human outcome reviews preserve whether a hypothesis had exact historical
support and its strongest cited similarity. The dashboard reports supported
versus unsupported confirmation rates as an observational comparison; legacy
reviews without this provenance are kept unknown. A global or independently
sufficient category-specific analog floor may be raised only after diverse
reviewed outcomes pass confirmation-rate, Wilson-confidence, and confirmed-
match-retention gates. It is never lowered below `0.18`.
From the second distinct schema-v4 cohort onward, the new immutable analysis
also records category overlap between prior hypotheses and current fixed bugs.
Schema-v5 runs additionally record lexical feature candidates with their shared
terms and scores. Both are retrospective review signals, never automatic
scenario accuracy; feature candidates remain unreviewed until a person confirms
or rejects them.
Retrospective evaluation schema v8 replaces pooled-token matching with
facet-aware matching. The current bug must overlap the predicted area and at
least one distinctive trigger or failure-mode term; area words repeated in the
scenario prose are removed before that check. Partial facet coverage receives a
deterministic score discount, and each candidate records its area, trigger, and
failure-mode overlap terms plus the `facet_coverage_v2` matcher policy.
Retrospective evaluation schema v9 records two distinct eligibility surfaces.
Category-evaluable hypotheses retain coarse category outcomes for historical
continuity. Scenario-evaluable hypotheses must additionally supply an area,
trigger, and observable failure mode with at least one distinctive area term and
one distinctive trigger/failure term. Category-only or generic-restatement
hypotheses remain visible with controlled insufficiency reasons but are excluded
from scenario match-rate denominators. The immutable evaluation reports the
scenario-evaluable population, its overall feature-candidate rate, and candidate
yield conditional on the target category being observed.
Schema-v7 predictions also preserve their scope boundary and disconfirming
observation through synthesis, stable identity, retrospective evaluation, and
the review ledger. The dashboard shows both beside the trigger and expected
failure so QA can actively falsify a recommendation instead of only looking for
confirmation.
Schema-v8 runs insert adversarial grounding passes before and after synthesis.
The configured model cannot rewrite the draft or final recommendation; it must
return controlled Boolean checks for the adjacent target, trigger, failure mode,
concrete shared path, boundary, and non-restatement of the direct fix. Only
drafts passing all six checks reach synthesis, and only synthesized results
passing the same checks are published. Exact duplicate drafts and synthesized
results are removed before their respective audits.
Withheld IDs, evidence references, failed checks, stage, and controlled reasons
are retained in `analysis.json` for dashboard audit. Historical analogs cannot
substitute for current-fix grounding. Because both passes use another invocation
of the same model, they are conservative second opinions rather than independent
ground-truth labels.
Schema-v9 analysis gives narrowly failed drafts one bounded repair attempt before
synthesis. Eligibility requires only one or two trigger, observable failure-mode,
or boundary wording failures. Missing target/shared-path evidence and direct-fix
restatements remain ineligible. A repair may change only fields routed to those
failed checks and must retain the exact category, BugCodes,
current-evidence references, historical references, and the same or lower
confidence; otherwise structural validation rejects it. `null` explicitly means
the frozen evidence cannot support a safe repair. A non-null repair receives a
fresh six-check adversarial audit, and only a full pass can enter synthesis.
Initial passes, declined or failed repairs, recovered identities, and changed
fields remain separate in immutable `analysis.json` provenance. If repair
generation or re-audit is unavailable after validated retries, the pipeline
records controlled unavailability, keeps the original draft withheld, and
continues the report.
Schema-v10 analysis requires every synthesized recommendation to cite the exact
accepted `TRC-...` draft identities that contributed to it. Validation confines
the output BugCodes, evidence and historical references, target category,
trigger/failure pair, and boundary to those selected drafts, and rejects unknown
or noncontributing IDs.
Each published hypothesis is deterministically labeled as initial-pass-only,
repaired-draft-only, or mixed, with exact recovered source IDs retained through
the later evaluation and immutable prediction-review ledger. The dashboard can
then compare repaired-origin against initial-pass confirmation outcomes only
within an identical category, model, analysis-engine SHA-256, and applied
category-guardrail state. Each side needs
eight raw/effective decisive reviews across three build-pair cohorts and three
hypotheses. This is an observational diagnostic: legacy/unknown lineage is
excluded and no result changes the bounded-repair policy automatically.
Schema-v11 synthesis additionally requires all cited draft IDs to share one exact
claim surface: area, category, trigger, observable failure mode, boundary,
disconfirming observation, and QA action. Those fields cannot be selected from
different drafts or rewritten during hierarchical consolidation. The accepted
registry stores the action and all other stable identity inputs, allowing
publication validation to recompute every `TRC-...` hash. Retrospective
repair-origin comparisons accept only this v11 coherent-claim provenance; schema
v10 and legacy review history remains readable but is excluded from that stricter
comparison.
Schema-v12 analysis shares one deterministic prediction-specificity policy with
retrospective scenario evaluation. A published area must have a distinctive
target term, and its trigger/failure pair must retain a distinctive state,
action, input, or observable symptom after removing generic QA terms and the
area vocabulary. The rule applies to initial candidates, repaired candidates,
synthesis output, immutable recommendations, and accepted lineage records.
Invalid model output receives controlled retry feedback and cannot be published;
older reports remain readable and are classified by evaluation schema v9 rather
than retroactively rejected.
Schema-v13 analysis adds a conservative coverage-diversity pass before the final
grounding audit. It computes an order-independent normalized term signature over
category, area, trigger, expected failure, scope boundary, disconfirming
observation, and QA action. Only complete signature matches are collapsed, so any
new facet term remains independently visible and auditable. Only articles and
word order are normalized away; negation, timing, quantities, and verbs remain
part of the signature. Risk-first,
confidence-first synthesis order chooses the retained form. Every suppressed
variant keeps its full claim, BugCodes, evidence references, synthesis lineage,
stable identity, retained identity, signature, and deterministic selection basis
in `analysis.json`; structured publication checks recompute identities and
signatures and reject inconsistent counts, missing retained counterparts, or
duplicate published coverage.
Schema-v14 moves the actual suppression decision until after the final grounding
audit. All exact synthesized variants are audited; only supported variants enter
the coverage-equivalence comparison. A preferred wording that fails can no
longer remove an equivalent wording that passes. Immutable counts separately
reconcile synthesized output, exact duplicates, final-audited claims, grounded
claims, final withholds, supported equivalents removed for diversity, and
distinct published recommendations. Suppressed records carry controlled
post-final-audit support provenance, and the dashboard exposes the grounded and
published totals separately.
Schema-v15 replaces model-selected confidence with deterministic current-evidence
confidence. High needs multiple independent current BugCodes with medium-or-
better source quality and at least one high-quality source; one high-quality
source or multiple medium-quality sources yields Medium; weaker support yields
Low. Extra quotes from one ticket do not create independence, and historical
analogs never increase the result. The immutable `confidence_basis` records the
controlled rule, source count, exact-reference count, quality mix, and analog
count. `evidence_quality_by_bug` allows structured publication validation to
recompute every published, coverage-suppressed, and accepted-lineage label. TR
Test Focus displays the same basis beside the exact evidence.
Schema-v16 replaces model-selected risk and impact basis with a deterministic
observable-failure classifier. Explicit data-loss, persistent saved-data
corruption, security/privacy, and billing harm map to Critical; crashes and
launch/startup blockers map to High; functional failures map to Medium; and
presentation-only symptoms map to Low. Low confidence caps severe labels at
Medium, and Critical requires High confidence. Model risk fields remain in the
response shape only for compatibility and are discarded. The immutable
`risk_basis` stores the matched signal, impact class, uncapped label, confidence
cap, and final label; structured publication checks recompute published and
coverage-suppressed claims, and TR Test Focus displays the rule beside its exact
evidence.
Schema-v17 makes the prediction category deterministic enough for retrospective
scoring. The parsed CSV contributes an immutable module-to-category vocabulary.
An unambiguous module named by the target claim wins; without one, the requested
category survives only when it matches the unique category of the cited current
BugCodes. Ambiguous or unsupported requests abstain as `Cross-cutting / unknown`
instead of becoming false category observations. `target_category_basis` stores
the model request, matched module/category, current-source categories, rule, and
final label. Structured publication checks recompute the basis for published,
coverage-suppressed, and accepted-lineage claims, and TR Test Focus exposes the
same provenance beside each prediction.
Schema-v18 makes evidence confidence correlation-aware. Two cited BugCodes count
as one evidence group when their exact cited signal terms are identical, or when
they come from the same parsed module and their meaningful signal terms have at
least 0.60 Jaccard overlap. High confidence requires two groups as well as the
existing quality rules; correlated medium-quality duplicates cannot manufacture
a Medium label. `evidence_independence_by_bug` freezes each source module and
adjacency quote, and `confidence_basis` records deterministic groups and links.
Structured publication checks reproduce grouping for published, suppressed, and
accepted-lineage claims and reject any quote that differs from its indexed source
signal. TR Test Focus displays group and correlated-BugCode counts beside the
exact evidence.
Schema-v19 adds deterministic adjacent-target anchoring. A correlation-aware High
evidence label remains High only when a distinctive term from the proposed target
area occurs in at least one exact cited current signal. Generic target words and
internal-component vocabulary are ignored; without an anchor the final confidence
is capped to Medium. The claim remains visible because a grounded shared-path
inference can still be useful, but it no longer appears highly certain.
`confidence_basis` stores the evidence label before the cap, anchor terms by
BugCode, aggregate terms, anchored-source count, rule, and cap. Publication checks
recompute this for published, coverage-suppressed, and accepted-lineage claims,
and TR Test Focus exposes it next to exact evidence.
Schema-v20 grounds direct-fix checklists rather than validating only their JSON
shape. Exact versions/numbers, quoted strings, platforms/devices, locale codes,
and internal-component names are rejected when absent from stored evidence. The
final checklist is deterministically anchored to the reproduction context and
reported symptom when available, while missing-evidence clarification remains
the first instruction. `checklist_evidence_by_bug` freezes the necessary source
fields and each checklist's `grounding_basis` records per-step evidence overlap,
anchor availability/presence, and unsupported specificity. Publication checks
recompute the final steps and basis, require exact cohort order, and reject any
unanchored or rewritten checklist before the immutable latest pointer changes.
Schema-v21 makes deterministic impact risk harm-specific and negation-aware.
Billing, subscription, entitlement, security, and privacy vocabulary alone no
longer elevates severity; Critical requires an explicit observable harm such as
a duplicate charge, lost paid entitlement, unauthorized access, sensitive-data
exposure, data loss, or persistent corruption. Negated signals before or after
the harm phrase are scoped to the local clause, stored in
`ignored_negated_signals`, and excluded from ranking, so
“does not crash, but data loss occurs” still selects the later Critical harm.
The immutable v2 `risk_basis` also declares that bare domain terms cannot raise
severity. Structured publication checks recompute it for published and
coverage-suppressed recommendations before the latest pointer changes, and TR
Test Focus displays both the policy and any ignored signal beside the evidence.
Schema-v22 makes the visible regression order deterministic rather than using
alphabetical category/area as the final tie-break. Impact risk and evidence
confidence remain primary; equal-label claims are then ordered by
target-anchored independent groups, independent-group quality, total independent
groups, and a stable hypothesis identity. Model response sequence, historical
analogs, and source-ticket severity cannot raise priority. Each claim stores the
full `priority_basis` and contiguous `priority_rank`, while the top-level
`recommendation_priority_policy` freezes the ordered hypothesis IDs. Structured
publication checks recompute the order and suppressed-equivalent selection and
reject rank, basis, sequence, or policy tampering before the latest pointer
changes.
Schema-v23 closes the selection gap before ranking. Because AI synthesis can
return at most eight recommendations, every accepted draft absent from its
lineage is treated as an explicit omission rather than an implicit rejection.
The strongest omitted drafts are selected deterministically with coverage
diversity first, promoted through the same final adversarial audit, and labeled
`deterministic_synthesis_omission_backfill`; model-selected claims remain labeled
`model_synthesis`. The immutable `synthesis_selection_safeguard` records all
accepted, model-selected, omitted, backfilled, and unbackfilled candidate IDs.
After final grounding, coverage-equivalent collapse and schema-v22 ordering, the
top eight claims form the visible portfolio. Additional supported claims retain
their full evidence and deterministic positions in
`portfolio_excluded_recommendations`. Publication checks reconstruct the
safeguard and validate the visible/overflow boundary before switching the latest
pointer.
Schema-v24 closes an earlier recall gap: a prediction never proposed by the first
candidate pass could not be restored by synthesis backfill. Each evidence batch
now receives a second independent coverage challenge over the same frozen input
and the primary pass's declared coverage. The challenge may return only a
materially distinct adjacent target, trigger, observable failure, boundary, or QA
action. Coverage-equivalent variants are deterministically filtered before draft
grounding; distinct additions enter the unchanged six-check critic, optional
bounded repair, synthesis safeguard, final critic, and portfolio ranking. The
immutable `candidate_discovery_audit` freezes raw and filtered IDs, coverage
signatures, full filtered claims and their retained coverage owners,
exact-duplicate counts, unique grounding inputs, and primary versus challenge
origin for every candidate. Publication checks reconcile that registry
with accepted, repaired, withheld, displayed, suppressed, and overflow lineage.
The origin flags are explicitly barred from confidence, risk, and priority
calculation, preventing correlated passes from masquerading as independent
evidence.
Schema-v25 adds a deterministic target-evidence boundary before model grounding.
Each primary or complementary draft must name an adjacent area with at least one
distinctive term present in an exact cited `adjacency_signal_quotes` entry.
Morphological variants and short acronyms are supported, but generic surface and
workflow terms do not count. Drafts that cite only a shared component while
inventing an unnamed target are recorded under
`adjacent_target_absent_from_current_signal` and removed before coverage
deduplication, repair, or either model critic. Their full normalized claim,
discovery origin, stable candidate identity, evidence references, and
`adjacency_support_basis` remain immutable. Supported gate records are also
preserved so publication can recompute both sides and reconcile primary and
challenge raw counts with coverage filtering and grounding inputs. Neither
historical analogs nor model verdicts can override the gate, and the gate itself
does not influence confidence, risk, or priority; it only decides whether the
target is evidentially eligible for those later stages.
Schema-v26 adds bounded deterministic evidence-reference completion before that
gate. If the model cites the right BugCode but selects a signal that omits the
proposed target, the normalizer checks only other exact
`adjacency_signal_quotes` from the same candidate BugCodes. The first stable
matching references are appended up to the existing four-reference limit, then
the original schema-v25 gate is recomputed. This cannot introduce BugCodes,
synonyms, historical evidence, or critic discretion. The immutable audit stores
the model's original references, every matching uncited reference, the references
actually added, the final reference set, and the before/after target-support
decision. Publication independently reconstructs the completion, and the
dashboard exposes every recovered target quote.
Schema-v27 handles the remaining reviewed vocabulary mismatch without granting
the model synonym authority. A reviewer adjudicating a sampled deterministic-
gate false negative may optionally select one directional
`prediction target term → exact cited-signal term` pair. Both terms are validated
against the immutable area and evidence quotes; generic, identical, free-form,
reverse, or uncited terms are rejected. The pair activates only after four
explicit target-gate false negatives from four source hashes and candidates,
spanning at least three Close Builds. Until then it is dashboard-only feedback.
The immutable analysis input receives only compact active pairs, never notes or
live counts. An active alias can satisfy target support in the reviewed direction
but cannot change evidence confidence, independence, risk, priority, or the
four-reference boundary. Candidate audits record the exact alias, BugCode, and
signal index used, and publication reconstructs every decision before switching
`latest.json`.
Schema-v28 adds longitudinal precision control for those aliases. When a prior
recommendation used a reviewed directional alias, its schema-v10 hypothesis
evaluation carries the exact normalized pair from the immutable adjacency basis
into the separate prediction-review ledger. A dedicated
`reviewed_target_alias_misleading` rejection is offered only for a feature-match
candidate with exactly one preserved pair, so a reviewer can attribute the
failure without guessing among aliases. That provenance is immutable. Confirmed
feature matches provide positive evidence; threshold audits and all other
rejection reasons—including matcher errors such as lexical overlap or wrong
target—are excluded. Reviews are correlation-capped per cohort and per
hypothesis/cohort. An active pair is retired only with 12 raw and eight effective
explicit outcomes across at least three cohorts and three hypotheses, at least six
alias-specific failures, a failure rate of at least 50%, and a 95% Wilson lower
bound of at least 30%. The next refresh omits a retired pair from compact analysis
input, producing a new source hash and immutable run; historical runs are never
mutated. The dashboard exposes activation evidence, outcome evidence, excluded
rejections, and retirement status.
Schema-v29 adds a deterministic shared-path claim boundary beside the target
gate. A candidate rationale must repeat at least one distinctive, non-target
mechanism from an exact cited current-fix signal. Target terms and reviewed target
aliases are excluded from the path vocabulary; generic claims built only from
words such as shared, changed, component, manager, service, state, system, or
path fail as `shared_path_not_concrete`. The immutable basis separates strong
from weak matches per evidence reference and explicitly records that model
critics, historical analogs, aliases, confidence, risk, and priority cannot alter
the result. Failing candidates stop before repair or model grounding. Candidate
audit identities bind the normalized rationale in schema-v29 records, preventing
different path claims from collapsing into one ID. Publication validation
recomputes target and path bases throughout discovery, accepted lineage, final
recommendations, and withheld samples; the dashboard exposes both failure types
and the exact matched terms.
Schema-v30 bounds deterministic severity with current evidence instead of
letting model-authored failure wording rank itself upward. Crash, launch/startup
blocking, data loss, persistent corruption, billing/entitlement harm, and
security/privacy harm must have a same-class, non-negated match in the cited
BugCodes' frozen summary, reproduction/description, or exact cited current-fix
signal. Without it, the hypothesis is preserved but capped to Medium before
priority sorting. Milder functional or presentation predictions need no harm
match, and evidence can only cap severity—it cannot promote a milder predicted
failure. The schema-v3 `risk_basis` records the exact supporting BugCode, source
field, and signal or the explicit unsupported-harm cap. Publication recomputes
the basis across displayed, overflow, and coverage-suppressed recommendations.
Historical analogs and both model critics are explicitly unable to create or
override current harm support, and the dashboard exposes the result beside each
prediction.
Schema-v31 makes the top-eight publication cutoff coverage-aware while preserving
risk and confidence precedence. All stronger risk/confidence tiers are selected
before a weaker tier is considered. If a single equal-risk, equal-confidence
tier straddles the remaining slots, its highest-priority claim is fixed as the
anchor; the other slots greedily add new exact shared-path terms, deterministic
target categories, and source BugCodes, using original priority as the final
tie-break. Selected recommendations are then displayed in their unchanged
priority order. Novelty therefore cannot promote a lower-risk or lower-confidence
claim. `recommendation_portfolio_policy` freezes the cutoff tier, slot counts,
selection sequence, original positions, novelty contribution, selected and
excluded hypothesis IDs, and controlled exclusion reasons. Publication
reconstructs membership and overflow provenance from the complete supported
portfolio. Model order, historical analogs, and source-ticket severity remain
ineligible inputs, and the dashboard shows the exact cutoff decisions.
Schema-v32 adds a deterministic exact-specificity boundary to triggers and
expected failures. Versions, numeric tokens, quoted UI literals, platforms or
devices, locales, lifecycle or network states, media formats, and internal
components are accepted only when the same detail exists in the candidate's
current source summary, reproduction or description, locale or OS metadata,
resolution changes, QA notes, or cited adjacency signal. Uncited BugCodes,
historical analogs, reviewed feedback, and model-critic approval cannot provide
support. The bounded JSON correction loop asks the candidate model to remove or
correct an unsupported detail before any grounding critic runs; accepted claims
retain their wording unchanged. `scenario_specificity_basis` freezes every
protected claim term, its trigger or failure field, and exact supporting BugCode
and evidence field, or the controlled unsupported set. The basis propagates
through repaired candidates, accepted lineage, synthesis, final grounding,
coverage suppression, and overflow. Publication recomputes it from frozen
current evidence, and TR Test Focus exposes the provenance beside each scenario.
Each run also marks a deterministic sample of at most six withheld items for a
human recall check, including drafts stopped by the deterministic evidence gate.
Those drafts use a distinct audit stage and a controlled `target_supported` or
`shared_path_concrete` failed check, so gate and critic false negatives remain
separately measurable.
The sampler preserves `(audit stage, controlled reason)`
diversity before filling remaining slots with a source-hash-stable ordering.
Reviewers can record Correctly withheld, False negative, or Unclear in the
separate atomic `tr_reports/grounding_reviews.json` ledger; a False negative must
name at least one failed critic check that the reviewer overturns. Immutable
source/model/engine/build/candidate/evidence provenance cannot change on a later
edit. The dashboard shows the raw false-negative rate and conservative 95% Wilson
upper bound, but these outcomes are observational and never feed the prediction
matcher, analog calibration, or AI guardrail activation automatically.
The evaluator also samples at most one same-category boundary audit per
hypothesis: the nearest lexical candidate within `0.12` below the applied
threshold, with an absolute floor of `0.12`. It is stored and displayed
separately as a false-negative audit, never counted or labeled as a prediction.
Human decisions are stored separately in
`tr_reports/prediction_reviews.json` as Confirmed, Rejected, or Unclear, with
optional reviewer initials and notes. Atomic locked writes prevent partial JSON;
the immutable analysis remains unchanged and the dashboard overlays the ledger
at read time. Prediction confirmation rates exclude Unclear, Unreviewed, and
boundary-audit samples; audit outcomes are reported separately and feed only
recall safeguards. Decision changes append audit-history events instead of deleting the
prior adjudication, and the refresh pipeline preserves this ledger.
Rejected reviews now require a controlled failure reason. Matcher errors
(lexical-only overlap, wrong target/trigger/failure, or outside-boundary matches)
feed lexical calibration; AI-analysis errors (a seen disconfirming observation
or unsupported shared path) are excluded from matcher calibration; and only an
explicit “historical corroboration was misleading” reason supplies negative
analog-calibration evidence. Reason counts are shown separately, while legacy
reasonless rejections remain valid but unroutable.
The dashboard additionally aggregates these structured outcomes by target
category as longitudinal AI-analysis feedback. An unsupported shared-path trend
can require stronger concrete shared-state grounding; a disconfirmation trend
can require a narrower boundary and earlier passing falsifier. A constraint is
activated only after the category has at least 12 raw and effective decisive
reviews, six raw and effective rejections, a 30% effective rejection rate,
three rejected cohorts, three rejected hypotheses, and a dominant AI reason
with at least four effective outcomes and 50% share. Matcher and analog reasons
cannot activate these prompt constraints. Only the controlled category/reason/
guardrail codes are copied into the immutable evidence input—never reviewer
notes or live aggregate counts—so source hashes change only when the applied
guardrail set changes.
Later retrospective evaluations copy the prior run's exact applied-guardrail
state, model identifier, and analysis-engine SHA-256 into each hypothesis; a
human review then preserves those fields immutably. The dashboard compares the
guardrail's targeted rejection rate against unguarded reviews only within the
same category, model identifier, and engine fingerprint. Both groups require at
least eight raw/effective decisive reviews, three build-pair cohorts, and three
hypotheses. Wilson intervals label the observed reduction as improved, worse,
or inconclusive. The comparison remains observational, excludes unknown legacy
provenance, and never changes policy automatically.
Reviewed match scores can calibrate later cohorts conservatively. The default
`0.24` threshold is retained until there are at least 12 scored decisive reviews
with both outcomes represented. The pipeline only raises it when the retained
sample has at least eight reviews, at least 80% confirmation, and a Wilson lower
bound of at least 0.50. A raised cutoff must retain at least 80% of effective
confirmed matches from the default cutoff, preventing a precision-only gain that
silently removes known-good predictions. A human-confirmed boundary audit below
the default `0.24` cutoff blocks threshold increases and records a recall
warning. It never lowers the cutoff from feedback, and every
prior/current-build cohort is capped at four effective reviews, with at most two
from any one hypothesis in that cohort. At least three cohorts and two distinct
hypotheses are required, so correlated candidates cannot masquerade as an
independent sample. Every module category must independently clear those same
gates before receiving its own cutoff. Sparse categories fall back to a
sufficiently reviewed global cutoff, or to `0.24` when global evidence is
insufficient. Every hypothesis
records the exact threshold, category/global/default source, calibration status,
sample counts, and bounds used for its future-cohort evaluation.
Matcher-policy provenance is immutable in the review ledger and participates in
the review identity. Calibration for `facet_coverage_v2` excludes every scored
review from older or unknown matcher policies, reports the excluded count, and
returns to the default cutoff until the new policy independently accumulates
sufficient diverse evidence.
Schema-v36 stops one bug from destroying a whole cohort's report. Verification
checklists are requested in batches and every supplied BugCode must come back,
so a single bug the model could not describe within the grounding rules failed
its batch, then failed the entire analysis: three of eight products lost their
TR report in one refresh over one rejected step. A rejected batch is now retried
one bug at a time, so healthy siblings keep their model checklists. A bug still
rejected alone receives an evidence-only checklist built solely from its stored
reproduction, reported symptom, and self-contained QA verification text — a
verify note carrying a link is an RD tracking pointer, not an oracle, and is
skipped. Nothing is invented, so the specificity and grounding rules are
unchanged; the checklist is weaker than a model-authored one, never silent.
`checklist_generation` records the policy, the fallback count, and each
degraded BugCode with the exact model errors that were rejected. The report and
TR Test Focus both label the affected checklists as evidence-only fallbacks
requiring review.

Manual run:

```bash
python scripts/process_newest_tr.py \
  --product pdri \
  --product-dir data/products/pdri \
  --model gemma4:e2b-it-q4_K_M
```

Use `./refresh_pipeline.sh --skip-tr-summary` to disable this stage. If the
newest parsed data has no `TRCreated` cohort, the stage records `no_cohort`, exits
successfully, and leaves the previous report in place.
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
