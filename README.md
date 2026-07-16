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

### Recommended Regression Areas

These are qualitative AI recommendations, not statistical failure
probabilities. For each selected bug, the summarizer supplies Ollama with:

- BugCode and short description;
- RD root cause;
- implemented solution;
- relevant device, locale, and content conditions when available.

Ollama is instructed to identify other modules or features that may share state,
components, localization behavior, rendering paths, storage logic, or other code
paths with the fixes. Each recommendation must include a concrete test action,
one-sentence reasoning, and a `Critical`, `High`, `Medium`, or `Low` ranking.

The report is rejected instead of published when it is incomplete, omits bugs,
does not contain risk-ranked regression recommendations, or only summarizes the
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
└── <close_build>/
    ├── bug_details.json
    └── test_focus_summary.md
```

`latest.json` points the dashboard to the most recent successfully validated
report. If a detail fetch or AI generation fails, the previous valid report
remains available.

## Common commands

Run PDRi with a six-month window:

```bash
./refresh_pipeline.sh --products pdri --duration-months 6
```

Run more than one product:

```bash
./refresh_pipeline.sh --products pdri,phdi --duration-months 6
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
