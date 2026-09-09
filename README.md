# PDR QA Toolkit

This repository refreshes eCL bug data, calculates QA risk signals, produces
defect forecasts and release-focused test recommendations, and serves the results
in a multi-product Streamlit dashboard.

The normal entry point is [`refresh_pipeline.sh`](refresh_pipeline.sh). You do
not need to run the individual Python scripts in sequence.

## Normal workflow

From the repository root:

```bash
./refresh_pipeline.sh --products pdri --duration-months 6
```

The pipeline activates `.venv`, starts Ollama when required, refreshes the
selected product, and restarts Streamlit on port `8501` with the new data.

Open the dashboard at:

```text
http://localhost:8501
```

Each run writes a timestamped log under `logs/`, for example:

```text
logs/refresh_20260716_115828.log
```

The pipeline writes intermediate files to `data/staging/` and promotes them only
after a stage succeeds. The dashboard therefore continues reading the last
complete dataset while a refresh is running.

## Pipeline stages

For each selected product, `refresh_pipeline.sh` runs these stages:

1. **Fetch and parse**
   - Calls the dashboard n8n webhook.
   - Merges records by BugCode so short refresh windows do not erase history.
   - Parses versions, modules, tags, status changes, severity, and build fields.
   - Updates the version catalogue and module-mapping caches.

2. **Risk register**
   - Aggregates bug history into module-level severity, recurrence, closure, and
     regression signals.

3. **AI risk scoring**
   - Uses the configured local Ollama model to assign impact and detectability.
   - `--skip-ollama` uses the heuristic scorer instead.

4. **Test skeletons and summary**
   - Generates product-scoped pytest skeletons under `tests/generated/<product>/`.
   - Generates the risk/quadrant summary.
   - These files are generated as planning aids; the refresh pipeline does not
     execute the tests.

5. **Bug clustering**
   - Uses Ollama embeddings and generated labels when Ollama is enabled.
   - Produces cluster membership, cluster summaries, and module entropy data.

6. **Defect predictions**
   - Builds module, category, theme, and scenario forecasts from historical bug
     patterns and current risk signals.

7. **Release Pulse scenarios**
   - Uses current `TRCreated`/`RDResolved` activity and historical signals to
     pre-generate actionable release test scenarios.
   - Saves dated history snapshots and retains the most recent 90 days.

8. **TR Test Focus**
   - Selects the newest version's current `TRCreated` bug cohort.
   - Uses the cohort's dominant `Close Build#` as the displayed build.
   - Fetches full bug details, RD root causes, solutions, and QA comments through
     the TR-detail n8n workflow.
   - Uses Ollama to generate prioritized **Recommended Regression Areas** and
     individual bug verification checklists.

After all selected products finish, Streamlit is restarted so the dashboard
loads the promoted files.

## TR Test Focus details

The **TR Test Focus** tab appears immediately after **Release Pulse**.

The main workflow shows regression scenarios grouped by area, followed by all
per-bug verification checklists with source summaries in their collapsed labels.
Evidence, generation diagnostics, and previous-build evaluation follow the
checklists. Full stored recommendation text remains under **Evidence & reasoning**.
Source evidence quality describes the available bug details; recommendation
confidence describes support for the proposed check. Neither measures release readiness.

Schema 37 checks exact details in recommendation actions and passing observations
as well as triggers and failure modes. It rejects explicit conflicting credit
amounts or billing periods; the model critic also checks broader instruction
consistency. The dashboard applies these checks to cached recommendations too,
flags affected instructions for review, and withholds their executable actions.
Use the linked source bug's verification checklist while reviewing the warning.
Cached artifacts are preserved; the next normal `./refresh_pipeline.sh` run
publishes a validated replacement using the existing immutable-run workflow.

Previous-build evaluation explains its stored-report selection and flags unverified
release-line comparability. Zero- and one-hypothesis samples show counts and an
insufficient-evidence note instead of highlighting an overlap percentage.

### Recommended Regression Areas

These are qualitative AI recommendations, not statistical failure
probabilities. For each selected bug, the summarizer supplies Ollama with:

- BugCode and short description;
- RD root cause;
- implemented solution;
- relevant device, locale, and content conditions when available.

Ollama returns validated structured JSON identifying other modules or features that may share state,
components, localization behavior, rendering paths, storage logic, or other code
paths with the fixes. Each recommendation must include a concrete test action,
one-sentence reasoning, the supporting BugCode(s), evidence confidence, and a
`Critical`, `High`, `Medium`, or `Low` impact ranking. `Critical` is reserved for credible adjacent crash,
data-loss, security/privacy, billing, or broadly shared release-blocking risk;
ticket count or source-ticket severity alone does not raise the ranking. Large TR
cohorts are analyzed in evidence batches and hierarchically deduplicated with
their original evidence so later bugs are not crowded out of model context.
Every prediction also carries machine-validated references to the exact
root-cause, solution, or stated-risk evidence slot that supports it. A model
cannot satisfy validation by citing the right BugCode with an unrelated
rationale or by inventing a new evidence reference.
Before analysis, the pipeline may attach up to three deterministic historical
analogs per source bug from the product's parsed bug history: at most one from
the same module and two from other modules in the same category. An analog must
be resolved, created before the source bug, exclude the current cohort, share at
least two meaningful short-description terms beyond module-name overlap, and
reach the configured similarity floor. The model may cite an exact analog as corroboration, but an
analog can never authorize a recommendation without validated current-fix
adjacency evidence. Future, active, cross-category, and invented analogs are
rejected or excluded.
When a later cohort is reviewed, the ledger records whether each prior
hypothesis cited an analog, the cited BugCodes, and the strongest similarity
score. TR Test Focus compares confirmation outcomes for analog-supported and
unsupported predictions, but labels the comparison observational and keeps
legacy reviews with missing provenance out of both groups. The selector's
similarity floor can only increase—never decrease—after sufficient diverse
confirmed/rejected reviews meet the existing confirmation, Wilson-confidence,
and confirmed-match-retention safeguards. Independently sufficient category
evidence takes precedence over global evidence; otherwise the default `0.18`
floor remains in effect.
The page shows the source-evidence quality mix and exposes those exact references
in a collapsed evidence panel for audit without cluttering the primary test list.
Each per-bug checklist is also labeled with deterministic source-evidence quality
and missing-field gaps (for example, unavailable full detail or reproduction
context), so fallback-based guidance is not presented with false certainty.
For longitudinal evaluation, every recommendation selects one normalized
`module_category` target from the parsed product taxonomy and receives a stable
`TRH-...` hypothesis ID. The free-form feature area remains visible for test
specificity. These fields enable later category-level outcome matching; they do
not turn qualitative recommendations into statistical probabilities.
Schema v5 also requires a concrete adjacent **trigger** and one observable,
falsifiable **expected failure mode**. Both participate in the stable hypothesis
identity and appear in the report, making each prediction testable instead of a
generic area label.
Schema v7 additionally requires a target-specific **scope boundary** and a
**disconfirming observation**. The boundary names where the evidence-grounded
inference stops, including independent paths that are outside scope. The
disconfirming observation states a concrete passing result under the trigger
that should lower the hypothesis's priority. Both fields are validated,
included in the stable hypothesis identity, preserved exactly through
hierarchical synthesis, displayed with the evidence, and retained in later
human reviews. This prevents a plausible shared-path rationale from silently
expanding into an unlimited regression claim.
Schema v8 adds adversarial grounding audits before and after synthesis. Separate
invocations of the configured Ollama model receive each immutable draft or final
recommendation with the original evidence but cannot rewrite it. An item advances
only if all six controlled checks pass: adjacent target, trigger, failure mode,
concrete shared path, scope boundary, and confirmation that it is not a direct-fix
restatement. Otherwise it is withheld under a controlled reason. Historical
analogs cannot rescue missing current-fix grounding. Exact duplicate drafts and
synthesized results are removed before their respective passes; every published recommendation records both-pass
audit provenance, and TR Test Focus shows aggregate pass/withhold counts plus the
controlled reasons for discarded drafts and synthesized results. Both passes use
the same model, so they are additional adversarial checks rather than independent
oracles.
Schema v9 adds one bounded recovery attempt before synthesis. A draft is eligible
only when the first audit fails one or two trigger, observable failure-mode, or
boundary wording checks. Missing target/shared-path evidence, direct-fix
restatements, and broad failures are never repaired. The repair model may alter
only fields routed to those failed checks and must preserve the exact target
category, source BugCodes, evidence references,
and historical references, and cannot increase confidence. It may return `null`
when the frozen evidence is insufficient. Every proposed repair receives a fresh
six-check adversarial audit and reaches synthesis only if all checks pass. The
immutable analysis records initial outcomes, repair attempts, changed fields,
recovered candidate identities, and permanent withholds separately so recovery
cannot be mistaken for an initial pass. Repair generation and re-audit are
optional fail-closed stages: an unavailable or invalid response retains the
original withheld draft and does not fail the otherwise valid report.
Schema v10 makes synthesis provenance exact. Every synthesized recommendation
must cite the accepted `TRC-...` draft identities that contributed to it. The
validator constrains its BugCodes, evidence references, historical references,
target category, trigger/failure pair, and boundary to those cited drafts, and rejects invented,
unknown, or noncontributing source identities. The immutable analysis therefore
labels each published prediction as initial-audit-only, repaired-draft-only, or a
mix of both, and retains the exact repaired source IDs. Later human prediction
reviews preserve this lineage together with the model and analysis-engine
fingerprint. TR Test Focus compares repaired-origin and initial-pass confirmation
outcomes only within the same target category, model, engine fingerprint, and
applied category-guardrail state,
with the same minimum raw/effective review and diversity gates used by the
guardrail diagnostic. The comparison is observational, excludes legacy reviews
without exact lineage, and never changes the bounded-repair policy automatically.
Schema v11 prevents cross-candidate claim splicing during hierarchical synthesis.
Every cited draft must carry the exact same area, normalized category, trigger,
observable failure mode, scope boundary, disconfirming observation, and executable
action as the published recommendation. Evidence from drafts with different claim
surfaces cannot be merged merely because individual fields look compatible. The
accepted-draft registry now contains enough immutable identity fields to recompute
and verify each `TRC-...` hash at publication. Later reviews label this stronger
provenance separately, and repaired-versus-initial comparisons use only v11
coherent-claim reviews so older union-validated runs cannot blur the result.
Schema v12 makes new recommendations scenario-scoreable by construction. One
deterministic specificity policy is shared by model-output validation and later
cohort matching. After generic QA words and the area vocabulary are removed,
the trigger or expected failure must retain a distinctive state, action, input,
or observable symptom. Generic restatements are rejected during initial
candidate generation, bounded repair, synthesis, immutable recommendation
validation, and accepted-lineage publication checks. The model receives the
controlled validation error and must correct the claim or omit it; a new report
can no longer publish a prediction that its own evaluator already knows it
cannot score.
Schema v13 prevents wording variants from overstating prediction coverage. After
exact duplicate removal, a deterministic signature compares the normalized term
sets for category, area, trigger, observable failure, scope boundary,
disconfirming observation, and QA action. Normalization ignores word order and
articles only; negation, timing, quantities, and action verbs remain semantic
terms. Only claims whose complete facet term sets are identical are collapsed;
one changed term preserves a separate
recommendation. Because synthesis output is already ordered by qualitative risk
and confidence, the strongest-ranked form is retained. The discarded wording,
BugCodes, exact evidence references, synthesis lineage, stable identity, retained
identity, signature, and selection policy remain in immutable audit provenance,
and publication validation recomputes those relationships before `latest.json`
can switch.
Schema v14 makes that diversity pass grounding-aware. Every exact synthesized
wording variant now receives the final six-check adversarial audit before any
coverage-equivalent claims are collapsed. Suppression considers only variants
that passed; therefore, if the highest-ranked wording is withheld but an
equivalent alternate passes, the supported alternate remains publishable. The
immutable audit separates exact duplicates, final-audited claims, grounded
claims, final withholds, supported coverage variants, and distinct published
recommendations. Each suppressed record explicitly states that selection
occurred after final grounding and that the discarded variant passed, and
publication validation recomputes all count relationships.
Schema v15 removes model self-assessment from the displayed evidence-confidence
label. Confidence is now derived deterministically from exact current-evidence
references, distinct source BugCodes, and each source's deterministic evidence
quality. **High** requires at least two current BugCodes, no low-quality source,
and at least one high-quality source. One high-quality source or multiple
medium-quality sources produce **Medium**; limited, single-medium, or low-quality
support produces **Low**. Multiple quotes from one ticket do not count as
independent corroboration, and historical analogs are reported but never raise
the label. Each recommendation records the complete controlled rule and counts
in `confidence_basis`, while `evidence_quality_by_bug` lets publication
validation recompute the label for published, suppressed, and accepted-lineage
claims before switching `latest.json`.
Schema v16 removes the remaining model self-assessment from qualitative risk.
The observable `expected_failure_mode` is classified by a controlled policy:
explicit data loss, persistent saved-data corruption, security/privacy harm, or
billing harm starts at **Critical**; crashes and launch/startup blockers start at
**High**; functional failures start at **Medium**; and presentation-only symptoms
start at **Low**. A Low evidence-confidence label caps severe impact at Medium,
while Critical requires High confidence and otherwise becomes High. Thus weak
evidence still caps both confidence and risk, ticket severity cannot inflate the
result, and model-provided `risk` or `impact_basis` values are compatibility-only
inputs that are discarded. The immutable `risk_basis` records the matched
observable signal, uncapped impact, applied confidence cap, and final label;
publication validation recomputes it for published and coverage-suppressed
recommendations.
Schema v17 prevents model-selected target categories from silently distorting
later prediction calibration. The parsed bug history now freezes the known
module-to-category mapping in each immutable input. A target whose area, trigger,
or action names one unambiguous known module receives that module's category;
otherwise a requested category is retained only when all cited current BugCodes
have one category and the request matches it. Ambiguous modules, mixed-source
categories, and unsupported cross-category requests become **Cross-cutting /
unknown** and are excluded from category outcome scoring. The original request,
matched module, cited-source categories, rule, and final label remain in
`target_category_basis`; publication validation recomputes this basis for
published, coverage-suppressed, and accepted-lineage claims.
Schema v18 closes a remaining confidence-inflation path: multiple BugCodes no
longer automatically mean independent corroboration. Exact cited signal terms
that are identical are grouped together; signals from the same parsed module are
also grouped when their meaningful-term Jaccard overlap is at least 0.60. High
confidence now requires at least two deterministic evidence groups, no low-
quality source, and at least one high-quality source. Correlated duplicate
medium-quality sources likewise cannot promote Low to Medium. The immutable
`confidence_basis` records every group, correlation link, reason, overlap terms,
and similarity, while `evidence_independence_by_bug` freezes module and exact
adjacency-signal provenance. Publication validation recomputes published,
coverage-suppressed, and accepted-lineage groups and rejects rewritten quotes.
Schema v19 separates strong source evidence from strong target inference. After
correlation-aware evidence confidence is calculated, a High label is capped to
Medium unless at least one distinctive token from the proposed adjacent `area`
appears in an exact cited current-fix signal. Generic words and inferred internal
components—including shared, feature, surface, renderer, service, pipeline, and
storage—cannot serve as target anchors. This preserves useful shared-path
hypotheses while preventing a model-selected but textually unanchored target from
looking highly certain. `confidence_basis` retains the pre-cap evidence label,
per-BugCode anchor terms, aggregate anchor terms, applied rule, and cap;
publication validation recomputes the same result across published, suppressed,
and accepted-lineage claims.
Schema v20 applies the same evidence-preserving standard to per-bug verification
checklists. Before publication, unsupported exact details—including versions,
numeric identifiers, quoted UI strings, platforms/devices, locale codes, and
internal components—are rejected unless present in the stored bug evidence. The
final first/last steps are then deterministically anchored to the supplied
reproduction context and reported symptom when those fields are available;
missing-detail clarification remains first. Each checklist carries an immutable
`grounding_basis` with per-step evidence fields, overlap terms, anchor status,
and unsupported-specificity results. `checklist_evidence_by_bug` freezes the
minimal evidence needed to recompute that basis, and publication validation
requires exact BugCode order, evidence metadata, anchored text, and zero invented
specificity before `latest.json` can switch.
Schema v21 reduces false severity inflation in deterministic impact risk. Bare
domain words such as payment, subscription, entitlement, security, or privacy no
longer imply Critical impact; the failure text must state explicit financial,
entitlement, security, privacy, data-loss, or persistent-corruption harm. Local
negations such as “does not crash,” “no data loss,” and “a security breach does
not occur” are recorded in
`ignored_negated_signals` and excluded, while punctuation and adversative clause
boundaries prevent a passing phrase from masking a later real failure. The v2
`risk_basis` freezes the matched harm, ignored signals, policy, and evidence cap.
Publication validation recomputes this provenance for both displayed and
coverage-suppressed recommendations before `latest.json` can switch.
Schema v22 removes alphabetical model-output order from recommendation
priority. Claims are sorted first by deterministic impact risk and evidence
confidence, then by target-anchored independent evidence groups, high-quality
independent groups, medium-or-better independent groups, total independent
groups, and finally a stable hypothesis identity. Historical analogs,
source-ticket severity, and model response order never move a claim upward.
Every recommendation stores this calculation in `priority_basis` plus a
contiguous `priority_rank`; `recommendation_priority_policy` freezes the complete
visible sequence. Publication validation recomputes displayed and
coverage-suppressed priority provenance, rejects reordered or renumbered output,
and verifies that the strongest coverage-equivalent wording was retained.
Schema v23 prevents AI synthesis capacity from silently deleting an already
grounded prediction. Accepted draft IDs absent from every model recommendation
are deterministically ordered with the schema-v22 priority basis. Up to eight
omitted drafts are promoted to the final adversarial audit, preserving distinct
coverage signatures first and using remaining capacity for fallback wording
variants. Each claim records whether it came from model synthesis or deterministic
omission backfill, while `synthesis_selection_safeguard` freezes the selected,
omitted, backfilled, and unbackfilled IDs. After final grounding and coverage
collapse, only the strongest eight distinct claims are displayed; additional
supported claims remain complete in `portfolio_excluded_recommendations` with
positions and evidence. Publication validation recomputes the safeguard, origins,
portfolio order, fixed limit, counts, and every overflow claim's confidence,
risk, category, and priority provenance.
Schema v24 improves candidate recall before grounding. For every frozen evidence
batch, a second independent coverage-challenge pass sees the primary coverage and
may propose only materially distinct missed adjacent targets, triggers, failure
modes, boundaries, or QA actions. Deterministic coverage signatures remove
paraphrases and equivalent variants before they can reach the critic; distinct
additions receive the same six-check draft audit, bounded repair rules, synthesis
omission safeguard, final audit, and portfolio limit as primary drafts. The
immutable `candidate_discovery_audit` reconciles every primary, raw challenge,
distinct, overlap-filtered, deduplicated, and grounded candidate ID and records
each surviving draft's discovery origin. Every filtered variant retains its full
claim, recomputable identity and coverage signature, and the grounding candidate
that already covers it. Publication validation reconstructs the
partition and origin lineage before `latest.json` can switch. Discovery origin is
provenance only: it never increases confidence, risk, or priority, so a second
model pass cannot manufacture corroboration.
Schema v25 closes the remaining model-only target-grounding gap. Before a draft
can reach the adversarial critic, at least one distinctive term from its proposed
adjacent `area` must appear in an exact cited current-fix signal. Morphological
variants and short target acronyms are matched deterministically, while generic
words such as preview, feature, shared, state, and workflow cannot anchor a
target. A generic shared-component quote therefore no longer authorizes an
arbitrary unnamed feature. Every evaluated draft stores a recomputable
`adjacency_support_basis`; unsupported primary and coverage-challenge drafts keep
their complete claim, discovery origin, evidence references, controlled reason,
and failed basis in `candidate_discovery_audit`, but never reach repair, critic,
synthesis, confidence, risk, or priority calculation. Historical analogs and the
same model's critic are explicitly unable to override the gate. Publication
validation reconstructs every decision and reconciles raw discovery, gate,
coverage filtering, deduplication, grounding input, accepted/withheld lineage,
and final recommendations before `latest.json` can switch.
Schema v26 prevents model reference selection from creating an avoidable gate
false negative. When a draft's selected signal does not name its target, the
normalizer scans only the other `adjacency_signal_quotes` belonging to that
draft's already-cited BugCodes. If an exact distinctive target term occurs, it
adds the first stable matching reference within the existing four-reference cap
and reruns the unchanged target gate. It never scans an uncited BugCode, uses a
synonym, changes source BugCodes, or lets the critic override the result. The
immutable completion basis retains the original, eligible, added, and final
references; publication validation recomputes the entire decision. TR Test Focus
shows how many drafts were recovered and exposes every added exact quote.
Schema v27 adds a deliberately narrow vocabulary bridge for product terms that
cannot be resolved by exact reference completion. On a sampled deterministic-
gate false negative, a reviewer may select one directional pair using a
distinctive term from the prediction area and an exact term present in the cited
current-fix quote—for example, `captions → subtitle`. Free-form reviewer notes,
model-proposed synonyms, generic words, reverse mappings, and uncited terms are
never eligible. A pair remains inactive until it has four explicit false-
negative reviews from four distinct immutable source runs and candidates across
at least three Close Builds. Only the compact active term pairs enter later
analysis input. Their use is preserved in each adjacency-support basis and
recomputed during publication; aliases may satisfy only the target gate and
never raise confidence, risk, priority, evidence independence, or historical
support. The dashboard shows inactive progress, the aliases active for the run,
and every candidate whose gate decision depended on one.
Schema v28 closes the feedback loop without conflating alias quality with the
future-cohort matcher. Each prior hypothesis now preserves the exact directional
alias pair(s) that its immutable adjacency basis actually used. Prediction reviews
may therefore choose a dedicated `Reviewed target alias was misleading` reason,
but only for a feature-match candidate backed by exactly one preserved pair; the
pair and provenance cannot be changed on later edits. Confirmed matches count as
positive alias outcomes. Lexical overlap, wrong-target matching, boundaries,
shared-path criticism, historical corroboration, threshold audits, unclear
reviews, and every other rejection are excluded from alias retirement. An active
pair retires only after at least 12 raw and eight correlation-capped effective
explicit outcomes across three cohorts and three hypotheses, including six
alias-specific failures, an observed failure rate of at least 50%, and a 95%
Wilson lower bound of at least 30%. Retirement removes the pair from the next
compact analysis input and therefore changes the immutable source hash; it never
rewrites an existing run. TR Test Focus shows positive outcomes, alias-specific
failures, excluded rejections, diversity, and retirement status.
Schema v29 makes the claimed shared path independently reproducible before any
model critic runs. In addition to an exact target anchor, the rationale must name
at least one distinctive non-target mechanism that also occurs in an exact cited
current-fix quote. Generic language such as shared, changed, component, manager,
service, state, system, or path cannot satisfy this check by itself, and target
terms or reviewed target aliases cannot be reused as path evidence. A failed
draft is frozen under `shared_path_not_concrete` with its complete rationale,
quotes, strong and weak term matches, and deterministic basis; it never reaches
repair, synthesis, confidence, risk, or priority. Schema-v29 candidate IDs bind
the exact rationale so two materially different path claims cannot share an audit
identity. Publication recomputes every basis, while historical analogs, reviewed
aliases, and model verdicts remain unable to override it. TR Test Focus reports
the target-only versus shared-path failures and shows the matched path terms for
every published recommendation.
Schema v30 prevents a model-authored failure mode from inflating its own
priority. Crash, launch/startup blocking, data loss, persistent saved-data
corruption, billing or entitlement harm, and security or privacy harm may retain
a High or Critical deterministic label only when the same non-negated harm class
appears in the cited BugCodes' frozen summary, reproduction/description, or exact
cited current-fix signal. Otherwise the prediction remains visible for QA but is
capped to Medium before priority sorting. Functional and presentation predictions
are unchanged, and stronger source harm can never raise a milder predicted
failure. The schema-v3 `risk_basis` preserves the exact BugCode, evidence field,
and matched signal or the explicit absence/cap decision. Publication reconstructs
this from frozen current evidence for displayed, overflow, and coverage-suppressed
recommendations; historical analogs and model critics cannot supply or override
harm support. TR Test Focus shows whether severe-harm evidence was found and
where.
Schema v31 improves the usefulness of the fixed eight-recommendation portfolio
without weakening severity order. Every stronger risk tier is exhausted first,
and confidence precedence is preserved within a risk tier. Diversity applies
only when one equal-risk, equal-confidence tier crosses the eighth slot: the
highest-priority claim in that tier is retained as an anchor, then remaining
slots favor previously uncovered exact shared-path mechanisms, deterministic
target categories, and source BugCodes. The selected rows remain displayed in
their original deterministic priority order. Lower-risk or lower-confidence
claims can never use novelty to displace a stronger claim. The immutable
`recommendation_portfolio_policy` freezes the cutoff tier, original positions,
selection rounds, new coverage contributed by every chosen item, and the reason
for every supported overflow item. Publication recomputes the full selection;
model order, historical analogs, and source-ticket severity cannot influence it.
TR Test Focus exposes both the equal-tier selection evidence and each excluded
claim's original priority and reason.
Schema v32 prevents invented precision in prediction scenarios. Exact versions,
numeric values, quoted UI strings, platforms or devices, locale codes, lifecycle
or network states, media formats, and internal-component terms used by a trigger
or expected failure must already occur in that candidate's frozen current
BugCode summary, reproduction or description, locale or OS metadata, resolution
changes, QA notes, or exact cited current-fix signal. Evidence from an uncited
BugCode, historical analog, reviewer feedback, or the later model critic cannot
satisfy the check. An invalid model response receives bounded correction feedback
before candidate grounding; unsupported precision is never silently stripped or
allowed to make a hypothesis look more specific than its evidence. Every
accepted draft, repair, final recommendation, suppressed equivalent, and
overflow claim carries a recomputable `scenario_specificity_basis` with the
protected term, claim field, and exact supporting BugCode and evidence field.
Publication validates the entire lineage, while TR Test Focus displays the
supporting source for every protected scenario detail or states that none was
claimed.
Gate-withheld drafts also enter the bounded deterministic recall sampler with
their own audit stage and either a `target_supported` or `shared_path_concrete`
failed check. This exposes strict gate false negatives separately from later
model-critic errors; reviewer outcomes remain observational and can neither
auto-relax either gate nor change its immutable decision.
To measure whether conservative evidence filtering suppresses useful recommendations,
each immutable run deterministically samples at most six permanently withheld
items. Sampling
first preserves audit-stage and controlled-reason diversity, then fills remaining
slots by a stable source-hash ordering. Reviewers mark the sample Correctly
withheld, False negative, or Unclear; false negatives must identify which failed
critic checks were actually supported by the frozen evidence. These judgments and
their immutable model, engine, source, build, candidate, BugCode, and evidence-ref
provenance are written atomically to `tr_reports/grounding_reviews.json`. The
dashboard reports the observed false-negative rate and its conservative 95% Wilson
upper bound. This ledger is observational, separate from
`prediction_reviews.json`, and never enters matcher, analog, or AI-guardrail
calibration automatically.
Actions must begin with an executable QA verb; implementation instructions such
as resetting an internal renderer are rejected. If a valid small-model response
uses an ambiguous phrase such as "its preview," the selected adjacent area is
inserted deterministically into the trigger and action before publication.
When two different schema-v4 TR cohorts exist, the newer analysis records
whether each prior target category appears among the newer fixed bugs. The page
labels this **category overlap** and preserves the matched BugCodes. It does not
call an absent category a false positive or a present category a confirmed
scenario match; both would require stronger feature-level adjudication.
Within an overlapping category, an explainable lexical matcher compares the
prior area, trigger, and failure mode with each new bug's summary, repro context,
and normalized module. Retrospective evaluation schema v8 keeps those three
facets separate: a candidate must overlap the predicted area plus at least one
distinctive trigger or failure-mode term. Target-area words repeated inside the
trigger or failure prose do not count as scenario evidence. Partial facet
coverage is score-discounted. Candidates display the exact terms supporting
each facet and remain `unreviewed`; human confirmation is required.
Retrospective evaluation schema v9 separates category observability from
scenario evaluability. A legacy hypothesis can still contribute to the coarse
category-overlap count, but it enters scenario matching only when it has an area,
trigger, and observable failure mode with distinctive target and scenario terms.
Missing or generic-restatement fields are retained with controlled insufficiency
reasons and excluded from feature-candidate rate denominators. The dashboard
shows both populations and separately reports candidate yield among evaluable
hypotheses whose category actually appeared in the current cohort.
For each hypothesis, the evaluator may also retain one deterministic
same-category **boundary audit**: the highest-scoring lexical candidate within
`0.12` below the applied cutoff (and never below `0.12`). Boundary audits are
clearly labeled as below-threshold samples, not predictions. Reviewing them
provides direct evidence about matcher false negatives instead of measuring only
the candidates the matcher already chose.
The dashboard lets a reviewer mark each candidate **Confirmed**, **Rejected**, or
**Unclear**, with optional initials and a note. Decisions are written atomically
to `tr_reports/prediction_reviews.json`, outside immutable evidence runs, and are
overlaid at display time. The prediction confirmation rate is calculated from
confirmed plus rejected feature-match decisions only; unclear, unreviewed, and
boundary-audit samples are excluded. Audit outcomes are reported separately and
used only for recall safeguards. Changing a decision appends an audit-history event instead of erasing
the prior reviewer, decision, note, score, or timestamp. A malformed or
incompatible ledger is reported and never replaced.
New rejected decisions require one controlled reason: lexical overlap only,
wrong target, wrong trigger, wrong failure mode, outside the prediction
boundary, disconfirming observation seen, unsupported shared-path inference,
misleading historical corroboration, or Other with a mandatory note. The page
aggregates these reasons and retains them in decision history. Calibration routes
them by responsibility: matcher-related rejections can tune the lexical cutoff;
model/evidence rejections are excluded from matcher tuning; and only explicitly
misleading historical corroboration can count as negative analog-calibration
evidence. Legacy rejections without a reason remain valid outcomes but are
labeled unroutable rather than guessed into a category.
TR Test Focus also shows longitudinal rejection trends for each target category.
Only the two controlled AI-analysis reasons—unsupported shared-path inference
and a seen disconfirming observation—can activate a constraint on a later
Ollama analysis. Activation requires at least 12 raw and effective decisive
reviews, six raw and effective structured rejections, a 30% effective rejection
rate, three rejected build-pair cohorts, three rejected hypotheses, and one AI
reason contributing at least four effective rejections and 50% of all effective
rejections in that category. Matcher and historical-analog reasons continue to
route only to their respective calibrators. The immutable model input contains
only the category, controlled reason, and guardrail code; live counts and
free-form reviewer notes never enter the prompt. Consequently, a new review
does not invalidate an analysis unless the active category guardrail actually
changes.
Each later hypothesis evaluation now carries exact provenance for the model
identifier, analysis-engine SHA-256, and whether that target category was
generated under a reviewed guardrail. Reviews preserve this provenance
immutably. The dashboard can therefore compare a guardrail's specific failure
reason before and after activation only within the same model identifier and
engine fingerprint. Each side needs at least eight raw and effective decisive
reviews across three build-pair cohorts and three hypotheses. Wilson intervals
classify the observed reduction as improved, worse, or inconclusive. This is a
prospective observational diagnostic—not a causal claim or an automatic policy
change—and legacy reviews without exact run provenance are excluded.
Each review also retains the candidate's match score and shared terms. For a
future distinct TR cohort, the pipeline may raise the default `0.24` lexical
threshold only after at least 12 scored Confirmed/Rejected reviews exist, with
at least three of each. A higher cutoff must retain at least eight reviewed
candidates, reach at least 80% confirmation, and have a Wilson lower bound of at
least 0.50. It must also retain at least 80% of the effective human-confirmed
matches available at the default threshold, so apparent precision cannot be
improved by discarding too much known-good coverage. A confirmed boundary audit
below the default `0.24` cutoff triggers a recall warning and prevents any
threshold increase. Sparse, one-sided, or inconclusive feedback keeps the
default; the calibrator never lowers the threshold. Correlated reviews are discounted: a
single prior/current-build cohort contributes at most four effective reviews,
and one hypothesis within that cohort contributes at most two. Calibration also
requires at least three distinct cohorts and two hypotheses, preventing a large
TR or repeated prediction from manufacturing confidence. A module category
uses its own cutoff only when its reviews independently meet those safeguards.
Otherwise it falls back to a sufficiently reviewed global cutoff, then to `0.24` when global
evidence is also sparse. The exact cutoff, evidence source, status, and counts
used for each hypothesis—including raw versus effective reviews and cohort
diversity—are stored in the new immutable evaluation and shown on the page.
Each review also stores the matcher-policy identity and per-facet overlap. A
policy change creates a separate review identity for the same hypothesis/build/
bug tuple. Threshold calibration uses only reviews produced by the current
matcher policy; older pooled-token scores remain readable but cannot tune the
facet-aware cutoff because their score distributions are not comparable.

The report is rejected instead of published when it is incomplete, omits or
duplicates a per-bug checklist, does not keep each checklist to 2–4 numbered
steps, contains ungrounded regression recommendations, lacks risk-ranked test
actions (unless it explicitly reports that the evidence supports no adjacent
area), uses an invalid target category or hypothesis ID, or only summarizes the
input tickets.

### Bug Creation TRs

The dashboard retains each bug's `ParentTRCode` as provenance and displays these
values as **Bug Creation TRs**. They are the parent TRs where the bugs originated.
They are not the Testing Request associated with the displayed Close Build.

The current cohort selection is based on the newest version's `TRCreated` state
and dominant Close Build. It does not yet query a build-level Testing Request
record directly.

TR report files are stored under:

```text
data/products/<product>/tr_reports/
├── latest.json
├── status.json
├── prediction_reviews.json        # created after the first human review
├── grounding_reviews.json         # created after the first critic-recall review
├── manual/
│   └── <timestamp>_<label>/bug_details.json
└── <close_build>/
    └── runs/
        └── <timestamp>_<source-hash>/
            ├── bug_details.json
            ├── analysis.json
            └── test_focus_summary.md
```

Each run is an immutable evidence snapshot. `latest.json` is switched only after
all files and hashes validate; `status.json` records the outcome of the latest
attempt. If fetching or analysis fails, the previous report remains available
with a dashboard freshness warning. Manual fetches are isolated from pipeline
runs unless an explicit `--output` path is supplied. The manifest fingerprints
both the source evidence and the analysis engine, so prompt, validation-policy,
or ranking changes invalidate a stale cached report even when the bug cohort is
unchanged. `./refresh_pipeline.sh` does not replace the separate prediction
review ledger.

## Common commands

Run PDRi with a six-month window:

```bash
./refresh_pipeline.sh --products pdri --duration-months 6
```

Run more than one product:

```bash
./refresh_pipeline.sh --products pdri,phdi --duration-months 6
```

Run all configured products with the same data window:

```bash
./refresh_pipeline.sh --products all --duration-months 6
```

Use the automatic weekday/weekend schedule:

```bash
./refresh_pipeline.sh
```

Force a schedule:

```bash
./refresh_pipeline.sh --weekday
./refresh_pipeline.sh --weekend
```

Preview what would run:

```bash
./refresh_pipeline.sh --dry-run --products pdri --duration-months 6
```

Useful optional flags:

| Flag | Effect |
|---|---|
| `--skip-ollama` | Use heuristic risk scoring and skip Ollama-only stages |
| `--skip-ollama-matcher` | Disable LLM module-name matching during parsing |
| `--skip-embedding-matcher` | Disable embedding-based module retrieval |
| `--skip-cluster` | Keep existing cluster outputs |
| `--skip-predict` | Keep existing prediction outputs |
| `--skip-pulse-scenarios` | Skip Release Pulse AI scenario generation |
| `--skip-tr-summary` | Skip the TR Test Focus refresh |
| `--force-relabel` | Regenerate cluster labels without re-embedding all bugs |
| `--ollama-model <model>` | Override the main Ollama model |
| `--ollama-matcher-model <model>` | Override the parser's matching model |
| `--embed-model <model>` | Override the embedding model |

The configured defaults are:

```text
Main AI model:       gemma4:e2b-it-q4_K_M
Embedding model:     nomic-embed-text
Cluster label model: gemma4:e2b-it-q4_K_M
```

## Automatic product schedule

Without `--products`, the script selects products according to the day:

| Schedule | Products and data window |
|---|---|
| Monday–Friday | `pdri` and `phdi`, one month |
| Saturday/Sunday automatic full schedule | All configured products, 36 months |

The configured product slugs are:

```text
pdri, phdi, pdra, phda, pdr, phd, promeo, vvg
```

The repository's cron example runs the limited schedule Monday through Friday
and the full schedule on Saturday:

```cron
0 3 * * 1-5 /path/to/pdri-qa-toolkit/refresh_pipeline.sh >> /path/to/pdri-qa-toolkit/logs/cron.log 2>&1
0 3 * * 6   /path/to/pdri-qa-toolkit/refresh_pipeline.sh >> /path/to/pdri-qa-toolkit/logs/cron.log 2>&1
```

## Dashboard

The dashboard automatically discovers products with
`data/products/<slug>/ecl_parsed.csv`. Use the product selector in the sidebar to
switch datasets.

Current tabs:

1. Module × Severity
2. Version Timeline
3. Tag Analysis
4. P/S Alignment
5. Team Coverage
6. KPI Dashboard
7. Risk Heatmap
8. Bug Clusters
9. Defect Forecast
10. Release Pulse
11. TR Test Focus

The pipeline manages Streamlit automatically. Starting it manually is normally
unnecessary, but the equivalent command is:

```bash
source .venv/bin/activate
streamlit run scripts/bug_heatmap_dashboard.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
```

## n8n workflows

The pipeline depends on these activated n8n workflows:

| File | Purpose |
|---|---|
| `(Dashboard) Query eBug List_v4.json` | Fetches the product bug list used by the main pipeline |
| `(TR Report) Get Bug Details_v1.json` | Fetches and organizes complete details for each TR Test Focus BugCode |

The default production webhook URLs are configured in:

- `scripts/fetch_from_n8n.py`
- `scripts/fetch_tr_bug_details.py`

If an n8n workflow is re-imported with a different webhook URL, update the
corresponding default before running the refresh.

## Main output layout

```text
data/products/<product>/
├── ecl_raw.json
├── ecl_parsed.csv
├── version_catalogue.csv
├── risk_register_all.csv
├── risk_register_scored_all.csv
├── module_mappings/
├── clusters/
│   ├── ecl_parsed_clustered.csv
│   ├── ecl_parsed_cluster_summary.csv
│   └── ecl_parsed_module_entropy.csv
├── predictions/
│   ├── ecl_parsed_predictions.csv
│   ├── ecl_parsed_predictions_by_cluster.csv
│   ├── release_pulse_scenarios.json
│   └── release_pulse_history/
└── tr_reports/
```

Generated test-planning files are stored under:

```text
tests/generated/<product>/
```

## Setup and prerequisites

For a new checkout:

```bash
chmod +x setup_mac_mini.sh refresh_pipeline.sh
./setup_mac_mini.sh
```

For the default AI configuration, ensure the required Ollama models are present:

```bash
ollama pull gemma4:e2b-it-q4_K_M
ollama pull nomic-embed-text
```

The refresh script expects a Python virtual environment at `.venv/` and uses the
dependencies in `requirements.txt`.

## Testing scope

This project flow does **not** run visual regression tests or Appium UI tests.
`refresh_pipeline.sh` performs data processing, AI analysis, report generation,
and dashboard refresh only. Stage 4 generates pytest skeletons as test-planning
artifacts but does not execute them.

Developer unit tests can be run separately when changing the implementation:

```bash
source .venv/bin/activate
pytest -q
```

## Troubleshooting

### Ollama embedding model is missing

If clustering reports an embedding `404` or falls back from Ollama, install the
configured embedding model and rerun:

```bash
ollama pull nomic-embed-text
./refresh_pipeline.sh --products pdri --duration-months 6
```

### A product fails while others continue

Failures are isolated by product. The final log prints a rerun command containing
the failed product slugs. Successful products keep their promoted output.

### TR Test Focus fails

Stage 8 does not discard the previous valid report. Check the timestamped refresh
log for missing BugCodes, n8n errors, or Ollama validation failures, then rerun:

```bash
./refresh_pipeline.sh --products pdri --duration-months 6
```

### Dashboard data looks stale

Check that the refresh completed and that Streamlit is listening on port `8501`:

```bash
lsof -i :8501
tail -n 100 logs/streamlit.log
```

For implementation-level details and file dependencies, see
[`pipeline.md`](pipeline.md).
