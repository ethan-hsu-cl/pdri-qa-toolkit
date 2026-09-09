#!/usr/bin/env bash
# =============================================================================
# QA Daily Refresh Pipeline — Multi-Product
# Keeps Streamlit running on port 8501 throughout the entire update process.
#
# Strategy: all scripts write to data/staging/, then files are atomically
# swapped into data/ with `mv` (rename syscall) only after each stage
# succeeds. Streamlit always reads a complete, self-consistent set of files.
#
# Product scheduling:
#   Weekdays (Mon–Fri): PDRi + PHDi with 1-month data window
#   Saturday only:      All products with 36-month data window (Sunday skipped)
#   Override with --products and --duration-months
#
# Usage:
#   ./refresh_pipeline.sh                                # auto schedule
#   ./refresh_pipeline.sh --dry-run                      # print steps, don't execute
#   ./refresh_pipeline.sh --products pdri,phdi           # specific products
#   ./refresh_pipeline.sh --products all                 # all configured products
#   ./refresh_pipeline.sh --products pdri --duration-months 1
#   ./refresh_pipeline.sh --weekday                      # force weekday schedule
#   ./refresh_pipeline.sh --weekend                      # force weekend (full) schedule
#   ./refresh_pipeline.sh --skip-ollama                  # use heuristic scorer
#   ./refresh_pipeline.sh --skip-cluster                 # skip clustering
#   ./refresh_pipeline.sh --skip-predict                 # skip predictions
#   ./refresh_pipeline.sh --skip-pulse-scenarios         # skip Release Pulse AI synthesis
#   ./refresh_pipeline.sh --skip-tr-summary              # skip newest-TR fixed-bug report
#   ./refresh_pipeline.sh --force-relabel                # re-run cluster labels only (no re-embedding)
#   ./refresh_pipeline.sh --ollama-model qwen3:4b                 # AI risk scorer model
#   ./refresh_pipeline.sh --tr-model qwen3:4b                     # TR Test Focus model
#   ./refresh_pipeline.sh --embed-model nomic-embed-text           # default embed model
#   ./refresh_pipeline.sh --embed-model mxbai-embed-large          # alternative embed model
#
# Cron schedule (weekdays at 03:00, Saturday only at 03:00 — Sunday skipped):
#   0 3 * * 1-5 /Users/yourname/pdri-qa-toolkit/refresh_pipeline.sh >> /Users/yourname/pdri-qa-toolkit/logs/refresh.log 2>&1
#   0 3 * * 6   /Users/yourname/pdri-qa-toolkit/refresh_pipeline.sh >> /Users/yourname/pdri-qa-toolkit/logs/refresh.log 2>&1
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv/bin/activate"
DATA="$SCRIPT_DIR/data"
STAGING="$SCRIPT_DIR/data/staging"
LOGS="$SCRIPT_DIR/logs"
LOG_FILE="$LOGS/refresh_$(date +%Y%m%d_%H%M%S).log"
LOCK_FILE="$SCRIPT_DIR/.pipeline.lock"
OLLAMA_MODEL="qwen3:4b"
# The TR Test Focus stage asks one model to propose grounded predictions and
# then adversarially audit them against long rulebook prompts, which a 2B-class
# model cannot hold. It is kept separate from OLLAMA_MODEL so the risk scorer
# and cluster labeller keep their tuned model.
TR_MODEL="qwen3:4b"
EMBED_MODEL="nomic-embed-text"
CLUSTER_LABEL_MODEL="qwen3:4b"
STREAMLIT_PORT=8501
export PYTHONUNBUFFERED=1   # force Python to flush stdout line-by-line when piped

# ── Product Schedule ─────────────────────────────────────────────────────────
# Format: "slug:duration_months slug:duration_months ..."
WEEKDAY_PRODUCTS="pdri:1 phdi:1"
WEEKEND_PRODUCTS="pdri:36 phdi:36 pdra:36 phda:36 pdr:36 phd:36 promeo:36 vvg:36"
ALL_PRODUCT_SLUGS="pdri phdi pdra phda pdr phd promeo vvg"

# ── Flags ─────────────────────────────────────────────────────────────────────
DRY_RUN=false
SKIP_OLLAMA=false
SKIP_OLLAMA_MATCHER=false
SKIP_EMBEDDING_MATCHER=false
SKIP_CLUSTER=false
SKIP_PREDICT=false
SKIP_PULSE_SCENARIOS=false
SKIP_TR_SUMMARY=false
FORCE_RELABEL=false
OVERRIDE_PRODUCTS=""
OVERRIDE_DURATION=""
FORCE_SCHEDULE=""
OLLAMA_MATCHER_MODEL=""

for arg in "$@"; do
  case $arg in
    --dry-run)                  DRY_RUN=true ;;
    --skip-ollama)              SKIP_OLLAMA=true ;;
    --skip-ollama-matcher)      SKIP_OLLAMA_MATCHER=true ;;
    --skip-embedding-matcher)   SKIP_EMBEDDING_MATCHER=true ;;
    --skip-cluster)             SKIP_CLUSTER=true ;;
    --skip-predict)             SKIP_PREDICT=true ;;
    --skip-pulse-scenarios)     SKIP_PULSE_SCENARIOS=true ;;
    --skip-tr-summary)           SKIP_TR_SUMMARY=true ;;
    --force-relabel)            FORCE_RELABEL=true ;;
    --weekday)                  FORCE_SCHEDULE=weekday ;;
    --weekend|--full)           FORCE_SCHEDULE=weekend ;;
    --products=*)               OVERRIDE_PRODUCTS="${arg#*=}" ;;
    --products)                 ;; # value comes in next arg
    --duration-months=*)        OVERRIDE_DURATION="${arg#*=}" ;;
    --duration-months)          ;; # value comes in next arg
    --ollama-model=*)           OLLAMA_MODEL="${arg#*=}" ;;
    --ollama-model)             ;; # value comes in next arg
    --tr-model=*)               TR_MODEL="${arg#*=}" ;;
    --tr-model)                 ;; # value comes in next arg
    --ollama-matcher-model=*)   OLLAMA_MATCHER_MODEL="${arg#*=}" ;;
    --ollama-matcher-model)     ;; # value comes in next arg
    --embed-model=*)            EMBED_MODEL="${arg#*=}" ;;
    --embed-model)              ;; # value comes in next arg
    *)
      # Handle --products VALUE and --duration-months VALUE (space-separated)
      ;;
  esac
done

# Parse positional-style --products VALUE and --duration-months VALUE
while [[ $# -gt 0 ]]; do
  case $1 in
    --products)
      if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
        OVERRIDE_PRODUCTS="$2"
        shift
      fi
      ;;
    --duration-months)
      if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
        OVERRIDE_DURATION="$2"
        shift
      fi
      ;;
    --ollama-model)
      if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
        OLLAMA_MODEL="$2"
        shift
      fi
      ;;
    --tr-model)
      if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
        TR_MODEL="$2"
        shift
      fi
      ;;
    --ollama-matcher-model)
      if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
        OLLAMA_MATCHER_MODEL="$2"
        shift
      fi
      ;;
    --embed-model)
      if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
        EMBED_MODEL="$2"
        shift
      fi
      ;;
  esac
  shift
done

# ── Helpers ───────────────────────────────────────────────────────────────────
mkdir -p "$STAGING" "$LOGS"

# ── Lock: prevent concurrent pipeline runs ────────────────────────────────────
# Writes PID + schedule to .pipeline.lock; cleans it up on exit (even on crash).
# If a lock exists and its PID is still alive, the new run exits immediately so
# a long weekend run cannot be clobbered by the next day's cron.
if [ -f "$LOCK_FILE" ]; then
  LOCK_PID=$(awk 'NR==1{print $1}' "$LOCK_FILE" 2>/dev/null || echo "")
  LOCK_INFO=$(cat "$LOCK_FILE" 2>/dev/null || echo "unknown")
  if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] ⚠️  Pipeline already running (PID $LOCK_PID — $LOCK_INFO). Exiting." \
      | tee -a "$LOG_FILE"
    exit 0
  else
    echo "[$(date '+%H:%M:%S')] Stale lock found (PID $LOCK_PID gone) — removing." \
      | tee -a "$LOG_FILE"
    rm -f "$LOCK_FILE"
  fi
fi
echo "$$ ${FORCE_SCHEDULE:-auto} $(date '+%Y-%m-%d %H:%M:%S')" > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# run CMD
#   Executes CMD and pipes both stdout and stderr through `tee -a` so output
#   appears in the terminal in real time AND is captured in the log file.
run() {
  if $DRY_RUN; then
    log "  [DRY-RUN] $*"
  else
    log "  Running: $*"
    # stdout (print lines) → log file only; stderr (tqdm bars + errors) → terminal
    eval "$*" >> "$LOG_FILE"
  fi
}

# run_silent CMD
#   Like run() but suppresses terminal output.
run_silent() {
  if $DRY_RUN; then
    log "  [DRY-RUN] $*"
  else
    eval "$*" >> "$LOG_FILE" 2>&1
    return $?
  fi
}

# Required commands run inside a per-product subshell. Because that subshell is
# itself evaluated by an `if`, Bash's `set -e` does not stop on nested failures.
# Exit the product subshell explicitly so later stages cannot run on missing or
# stale inputs and a failed product cannot be reported as successful.
run_required() {
  if ! run "$@"; then
    log "  ERROR: required command failed; stopping this product."
    exit 1
  fi
}

run_silent_required() {
  if ! run_silent "$@"; then
    log "  ERROR: required command failed; stopping this product."
    exit 1
  fi
}

promote_required() {
  if ! promote "$@"; then
    log "  ERROR: required output was not promoted; stopping this product."
    exit 1
  fi
}

promote_dir_required() {
  if ! promote_dir "$@"; then
    log "  ERROR: required output directory was not promoted; stopping this product."
    exit 1
  fi
}

# Atomically promote a staging file to live data.
promote() {
  local src="$1" dst="$2"
  if $DRY_RUN; then
    log "  [DRY-RUN] promote $src -> $dst"
    return
  fi
  if [ ! -f "$src" ]; then
    log "  ERROR: staging file not found: $src"
    return 1
  fi
  mkdir -p "$(dirname "$dst")"
  mv "$src" "$dst"
  log "  ✅ Promoted: $(basename "$dst")"
}

# Promote an entire staging sub-directory.
promote_dir() {
  local src_dir="$1" dst_dir="$2"
  if $DRY_RUN; then
    log "  [DRY-RUN] promote_dir $src_dir -> $dst_dir"
    return
  fi
  mkdir -p "$dst_dir"
  for f in "$src_dir"/*; do
    [ -f "$f" ] && mv "$f" "$dst_dir/$(basename "$f")"
  done
  log "  ✅ Promoted dir: $(basename "$dst_dir")"
}

ensure_streamlit_running() {
  if lsof -ti tcp:"$STREAMLIT_PORT" >/dev/null 2>&1; then
    log "  Streamlit already running on :$STREAMLIT_PORT — leaving it alone."
  else
    log "  Streamlit not running — starting it on :$STREAMLIT_PORT"
    source "$VENV"
    nohup streamlit run "$SCRIPT_DIR/scripts/bug_heatmap_dashboard.py" \
      --server.address 0.0.0.0 \
      --server.port "$STREAMLIT_PORT" \
      --server.headless true \
      >> "$LOGS/streamlit.log" 2>&1 &
    sleep 3
    log "  Streamlit PID: $!"
  fi
}

# ── Per-Product Pipeline ─────────────────────────────────────────────────────
# Runs Stages 1–6 for a single product slug with a given duration.
run_product_pipeline() {
  local SLUG="$1"
  local DURATION="$2"

  local DATA_PRODUCT="$DATA/products/$SLUG"
  local STAGING_PRODUCT="$STAGING/products/$SLUG"
  mkdir -p "$DATA_PRODUCT" "$STAGING_PRODUCT"

  log ""
  log "╔════════════════════════════════════════════════════════╗"
  log "║  Product: $SLUG  (duration: ${DURATION} months)"
  log "╚════════════════════════════════════════════════════════╝"

  # ── Stage 1: Fetch + Parse ──────────────────────────────────────────────
  log ""
  log "── [$SLUG] Stage 1: Fetch + Parse ───────────────────────"

  # Seed staging ecl_raw.json from live data so the save_json merge logic in
  # fetch_from_n8n.py can deduplicate by BugCode and preserve historical bugs.
  # Without this, a 1-month weekday run would overwrite all older version data.
  if [ -f "$DATA_PRODUCT/ecl_raw.json" ]; then
    cp "$DATA_PRODUCT/ecl_raw.json" "$STAGING_PRODUCT/ecl_raw.json"
    log "  Seeded staging ecl_raw.json from live data (merge cache active)."
  fi

  # Seed staging module_mappings from live data so previously confirmed/promoted
  # mappings are available to the parser (staging dir is empty on each run).
  if [ -d "$DATA_PRODUCT/module_mappings" ]; then
    mkdir -p "$STAGING_PRODUCT/module_mappings/versions" "$STAGING_PRODUCT/module_mappings/permanent"
    for f in "$DATA_PRODUCT"/module_mappings/permanent/*.json; do
      [ -f "$f" ] && cp "$f" "$STAGING_PRODUCT/module_mappings/permanent/$(basename "$f")"
    done
    for f in "$DATA_PRODUCT"/module_mappings/versions/*.json; do
      [ -f "$f" ] && cp "$f" "$STAGING_PRODUCT/module_mappings/versions/$(basename "$f")"
    done
    # Carry over caches so repeated raw strings skip Ollama / re-embedding.
    if [ -f "$DATA_PRODUCT/module_mappings/ollama_cache.json" ]; then
      cp "$DATA_PRODUCT/module_mappings/ollama_cache.json" \
         "$STAGING_PRODUCT/module_mappings/ollama_cache.json"
    fi
    if [ -f "$DATA_PRODUCT/module_mappings/canonical_embeddings.json" ]; then
      cp "$DATA_PRODUCT/module_mappings/canonical_embeddings.json" \
         "$STAGING_PRODUCT/module_mappings/canonical_embeddings.json"
    fi
    log "  Seeded staging module_mappings from live data."
  fi


  # Desktop products (pdr, phd) have far more bugs — chunk into 6-month windows
  # to avoid 504 Gateway Timeout on the n8n webhook.
  case "$SLUG" in
    pdr|phd) CHUNK_ARG="--chunk-months 6" ;;
    *)       CHUNK_ARG="" ;;
  esac

  # Enable Ollama semantic matcher in the parser subprocess unless the user opted
  # out. Env vars propagate through fetch_from_n8n.py's subprocess call.
  if ! $SKIP_OLLAMA && ! $SKIP_OLLAMA_MATCHER; then
    export PDRI_OLLAMA_MATCHER=1
    export PDRI_OLLAMA_MATCHER_MODEL="${OLLAMA_MATCHER_MODEL:-$OLLAMA_MODEL}"
  else
    unset PDRI_OLLAMA_MATCHER
    unset PDRI_OLLAMA_MATCHER_MODEL
  fi

  # Enable embedding-based canonical retrieval (default on when Ollama matcher
  # is active). Uses nomic-embed-text for cosine similarity top-K.
  if ! $SKIP_OLLAMA && ! $SKIP_EMBEDDING_MATCHER; then
    export PDRI_EMBEDDING_MATCHER=1
    export PDRI_EMBEDDING_MODEL="$EMBED_MODEL"
  else
    unset PDRI_EMBEDDING_MATCHER
    unset PDRI_EMBEDDING_MODEL
  fi

  run_required "python scripts/fetch_from_n8n.py \
    --product '$SLUG' \
    --duration-months '$DURATION' \
    --scope all \
    $CHUNK_ARG \
    --output '$STAGING_PRODUCT/ecl_raw.json' \
    --parsed-output '$STAGING_PRODUCT/ecl_parsed.csv' \
    --then-parse"

  promote_required "$STAGING_PRODUCT/ecl_raw.json"    "$DATA_PRODUCT/ecl_raw.json"
  promote_required "$STAGING_PRODUCT/ecl_parsed.csv"  "$DATA_PRODUCT/ecl_parsed.csv"

  [ -f "$STAGING_PRODUCT/version_catalogue.csv" ] && \
    promote "$STAGING_PRODUCT/version_catalogue.csv" "$DATA_PRODUCT/version_catalogue.csv"

  # Promote module mappings (pending/confirmed JSON + permanent store + LLM cache)
  if [ -d "$STAGING_PRODUCT/module_mappings" ]; then
    if [ -d "$STAGING_PRODUCT/module_mappings/versions" ]; then
      promote_dir "$STAGING_PRODUCT/module_mappings/versions" "$DATA_PRODUCT/module_mappings/versions"
    fi
    if [ -d "$STAGING_PRODUCT/module_mappings/permanent" ]; then
      promote_dir "$STAGING_PRODUCT/module_mappings/permanent" "$DATA_PRODUCT/module_mappings/permanent"
    fi
    if [ -f "$STAGING_PRODUCT/module_mappings/ollama_cache.json" ]; then
      promote "$STAGING_PRODUCT/module_mappings/ollama_cache.json" \
        "$DATA_PRODUCT/module_mappings/ollama_cache.json"
    fi
    if [ -f "$STAGING_PRODUCT/module_mappings/canonical_embeddings.json" ]; then
      promote "$STAGING_PRODUCT/module_mappings/canonical_embeddings.json" \
        "$DATA_PRODUCT/module_mappings/canonical_embeddings.json"
    fi
  fi

  log "[$SLUG] Stage 1 complete."

  # ── Stage 2: Risk Register ─────────────────────────────────────────────
  log ""
  log "── [$SLUG] Stage 2: Risk Register ───────────────────────"

  run_required "python scripts/compute_risk_scores.py \
    '$DATA_PRODUCT/ecl_parsed.csv' \
    '$STAGING_PRODUCT/risk_register_all.csv'"

  promote_required "$STAGING_PRODUCT/risk_register_all.csv" "$DATA_PRODUCT/risk_register_all.csv"

  if [ -d "$STAGING_PRODUCT/risk_register_versions" ]; then
    promote_dir "$STAGING_PRODUCT/risk_register_versions" "$DATA_PRODUCT/risk_register_versions"
  fi

  log "[$SLUG] Stage 2 complete."

  # ── Stage 3: AI Risk Scoring ───────────────────────────────────────────
  log ""
  log "── [$SLUG] Stage 3: AI Risk Scoring ─────────────────────"

  # Seed staging with the last good scored file so resume/checkpoint works.
  if [ -f "$DATA_PRODUCT/risk_register_scored_all.csv" ]; then
    cp "$DATA_PRODUCT/risk_register_scored_all.csv" "$STAGING_PRODUCT/risk_register_scored_all.csv"
    log "  Seeded staging with previous scored file (resume cache active)."
  fi
  if [ -d "$DATA_PRODUCT/risk_register_versions" ]; then
    mkdir -p "$STAGING_PRODUCT/risk_register_versions"
    for f in "$DATA_PRODUCT/risk_register_versions"/risk_register_scored_*.csv; do
      [ -f "$f" ] && cp "$f" "$STAGING_PRODUCT/risk_register_versions/$(basename "$f")"
    done
  fi

  if $SKIP_OLLAMA; then
    log "  Using heuristic scorer (--skip-ollama)"
    run_required "python scripts/ai_risk_scorer.py \
      '$DATA_PRODUCT/risk_register_all.csv' \
      '$STAGING_PRODUCT/risk_register_scored_all.csv' \
      --provider heuristic"
  else
    run_required "python scripts/ai_risk_scorer.py \
      '$DATA_PRODUCT/risk_register_all.csv' \
      '$STAGING_PRODUCT/risk_register_scored_all.csv' \
      --provider ollama \
      --model '$OLLAMA_MODEL'"
  fi

  promote_required "$STAGING_PRODUCT/risk_register_scored_all.csv" "$DATA_PRODUCT/risk_register_scored_all.csv"
  if [ -d "$STAGING_PRODUCT/risk_register_versions" ]; then
    promote_dir "$STAGING_PRODUCT/risk_register_versions" "$DATA_PRODUCT/risk_register_versions"
  fi

  log "[$SLUG] Stage 3 complete."

  # ── Stage 4: Test Skeletons ────────────────────────────────────────────
  log ""
  log "── [$SLUG] Stage 4: Test Skeletons + Summary ────────────"

  run_silent_required "python scripts/auto_tag_tests.py \
    '$DATA_PRODUCT/risk_register_scored_all.csv' \
    --generate-skeletons tests/generated/$SLUG/ \
    --summary"

  log "[$SLUG] Stage 4 complete."

  # ── Stage 5: Clustering ────────────────────────────────────────────────
  if $SKIP_CLUSTER; then
    log ""
    log "── [$SLUG] Stage 5: Clustering SKIPPED ─────────────────"
  else
    log ""
    log "── [$SLUG] Stage 5: Clustering ─────────────────────────"

    CLUSTER_OUT="$DATA_PRODUCT/clusters"
    mkdir -p "$CLUSTER_OUT"

    CLUSTER_CMD="python scripts/cluster_bugs.py \
        '$DATA_PRODUCT/ecl_parsed.csv' \
        '$CLUSTER_OUT/ecl_parsed_clustered.csv'"

    if ! $SKIP_OLLAMA; then
      CLUSTER_CMD="$CLUSTER_CMD \
        --provider ollama \
        --label-model '$CLUSTER_LABEL_MODEL' \
        --embed-model '$EMBED_MODEL'"
    fi

    if $FORCE_RELABEL; then
      CLUSTER_CMD="$CLUSTER_CMD --relabel"
    fi

    run_required "$CLUSTER_CMD"

    log "[$SLUG] Stage 5 complete."
  fi

  # ── Stage 6: Defect Predictions ────────────────────────────────────────
  # --force-relabel overrides --skip-predict because predictions consume cluster labels
  if $SKIP_PREDICT && ! $FORCE_RELABEL; then
    log ""
    log "── [$SLUG] Stage 6: Predictions SKIPPED ────────────────"
  else
    log ""
    log "── [$SLUG] Stage 6: Defect Predictions ─────────────────"

    PRED_STAGING="$STAGING_PRODUCT/predictions"
    mkdir -p "$PRED_STAGING"

    if $SKIP_OLLAMA; then
      run_required "python scripts/predict_defects.py \
        '$DATA_PRODUCT/ecl_parsed.csv' \
        '$PRED_STAGING/ecl_parsed_predictions.csv'"
    else
      run_required "python scripts/predict_defects.py \
        '$DATA_PRODUCT/ecl_parsed.csv' \
        '$PRED_STAGING/ecl_parsed_predictions.csv' \
        --provider ollama \
        --model '$OLLAMA_MODEL'"
    fi

    promote_dir_required "$PRED_STAGING" "$DATA_PRODUCT/predictions"

    log "[$SLUG] Stage 6 complete."
  fi

  # ── Stage 7: Release Pulse AI Scenarios ────────────────────────────────
  # Pre-generates the dashboard's "AI-Synthesized Test Scenarios" cache so
  # the Release Pulse tab renders immediately on first load. Skipped when
  # Ollama is disabled (the dashboard's heuristic fallback view does not
  # consume this cache).
  if $SKIP_OLLAMA || $SKIP_PULSE_SCENARIOS; then
    log ""
    log "── [$SLUG] Stage 7: Release Pulse Scenarios SKIPPED ────"
  else
    log ""
    log "── [$SLUG] Stage 7: Release Pulse Scenarios ────────────"

    PULSE_STAGING="$STAGING_PRODUCT/predictions"
    PULSE_HISTORY_STAGING="$PULSE_STAGING/release_pulse_history"
    PULSE_HISTORY_LIVE="$DATA_PRODUCT/predictions/release_pulse_history"
    mkdir -p "$PULSE_STAGING" "$PULSE_HISTORY_STAGING"

    run_required "python scripts/generate_release_pulse_scenarios.py \
      '$DATA_PRODUCT' \
      '$PULSE_STAGING/release_pulse_scenarios.json' \
      --provider ollama \
      --model '$OLLAMA_MODEL' \
      --history-dir '$PULSE_HISTORY_STAGING'"

    if [ -f "$PULSE_STAGING/release_pulse_scenarios.json" ]; then
      promote "$PULSE_STAGING/release_pulse_scenarios.json" \
        "$DATA_PRODUCT/predictions/release_pulse_scenarios.json"
    fi

    # Promote the dated history snapshot (today's run). Same-day reruns
    # overwrite the live file via mv. The Python script always writes exactly
    # one dated file per invocation.
    if $DRY_RUN; then
      log "  [DRY-RUN] promote dated snapshot from $PULSE_HISTORY_STAGING/ to $PULSE_HISTORY_LIVE/"
    else
      mkdir -p "$PULSE_HISTORY_LIVE"
      for snap in "$PULSE_HISTORY_STAGING"/*.json; do
        [ -f "$snap" ] || continue
        promote "$snap" "$PULSE_HISTORY_LIVE/$(basename "$snap")"
      done
    fi

    # Prune history files older than 90 days. Keeps the directory bounded
    # while preserving enough history for trend inspection.
    if $DRY_RUN; then
      log "  [DRY-RUN] prune $PULSE_HISTORY_LIVE/*.json older than 90 days"
    elif [ -d "$PULSE_HISTORY_LIVE" ]; then
      find "$PULSE_HISTORY_LIVE" -maxdepth 1 -type f -name '*.json' -mtime +90 -delete \
        2>/dev/null || true
    fi

    log "[$SLUG] Stage 7 complete."
  fi

  # ── Stage 8: Newest TR Fixed-Bug Test Focus ───────────────────────────
  # Selects the newest version's TRCreated (RD Resolved) cohort, derives its
  # dominant Close Build#, then asks Ollama for per-bug
  # verification steps and adjacent regression areas.
  if $SKIP_OLLAMA || $SKIP_TR_SUMMARY; then
    log ""
    log "── [$SLUG] Stage 8: Newest TR Test Focus SKIPPED ──────"
  else
    log ""
    log "── [$SLUG] Stage 8: Newest TR Test Focus ──────────────"
    if run "python scripts/process_newest_tr.py \\
      --product '$SLUG' \\
      --product-dir '$DATA_PRODUCT' \\
      --model '$TR_MODEL'"; then
      log "[$SLUG] Stage 8 complete."
    else
      # A detail-service outage should not discard the successful core refresh.
      log "[$SLUG] WARNING: Stage 8 failed; prior TR report remains available."
    fi
  fi

  log ""
  log "═══ [$SLUG] All stages complete ═════════════════════════"
}

# ── Determine Active Products ────────────────────────────────────────────────
resolve_product_schedule() {
  if [ -n "$OVERRIDE_PRODUCTS" ]; then
    # User override: parse comma-separated slugs, expand "all", validate names,
    # and apply the override duration (36 months when omitted).
    local dur="${OVERRIDE_DURATION:-36}"
    local products=""
    local requested="${OVERRIDE_PRODUCTS// /}"
    if [ "$requested" = "all" ]; then
      requested="${ALL_PRODUCT_SLUGS// /,}"
    elif [[ ",$requested," == *,all,* ]]; then
      log "ERROR: --products all cannot be combined with other product slugs." >&2
      return 1
    fi
    IFS=',' read -ra slugs <<< "$requested"
    for slug in "${slugs[@]}"; do
      slug="$(echo "$slug" | xargs)"  # trim whitespace
      if [[ " $ALL_PRODUCT_SLUGS " != *" $slug "* ]]; then
        log "ERROR: unsupported product '$slug'. Valid values: ${ALL_PRODUCT_SLUGS// /, } or all." >&2
        return 1
      fi
      products+="$slug:$dur "
    done
    echo "$products"
    return
  fi

  local schedule="$FORCE_SCHEDULE"
  if [ -z "$schedule" ]; then
    # Auto-detect from day of week
    local dow
    dow=$(date +%u)  # 1=Mon … 7=Sun
    if [ "$dow" -ge 6 ]; then
      schedule=weekend
    else
      schedule=weekday
    fi
  fi

  if [ "$schedule" = "weekend" ]; then
    log "📅 Full schedule (--weekend or auto-detected) → all products" >&2
    echo "$WEEKEND_PRODUCTS"
  else
    log "📅 Limited schedule (--weekday or auto-detected) → weekday products" >&2
    echo "$WEEKDAY_PRODUCTS"
  fi
}

# ── Main ──────────────────────────────────────────────────────────────────────

log "======================================================"
log "QA Daily Refresh  $(date '+%Y-%m-%d %H:%M:%S')"
log "dry_run=$DRY_RUN  skip_ollama=$SKIP_OLLAMA  skip_ollama_matcher=$SKIP_OLLAMA_MATCHER  skip_cluster=$SKIP_CLUSTER  skip_predict=$SKIP_PREDICT  skip_pulse_scenarios=$SKIP_PULSE_SCENARIOS  skip_tr_summary=$SKIP_TR_SUMMARY  force_relabel=$FORCE_RELABEL  force_schedule=$FORCE_SCHEDULE  ollama_model=$OLLAMA_MODEL  tr_model=$TR_MODEL  ollama_matcher_model=${OLLAMA_MATCHER_MODEL:-<inherit>}  embed_model=$EMBED_MODEL  cluster_label_model=$CLUSTER_LABEL_MODEL"
log "======================================================"

# 1. Make sure Streamlit is up before we do anything
ensure_streamlit_running

# Activate venv for all subsequent python calls
source "$VENV"

# Start Ollama once (shared across all products)
if ! $SKIP_OLLAMA; then
  if ! pgrep -x ollama >/dev/null 2>&1; then
    log "  Starting Ollama..."
    run_silent "ollama serve &>/dev/null &"
    run_silent "sleep 5"
  else
    log "  Ollama already running."
  fi
fi

# Resolve which products to run
ACTIVE_PRODUCTS=$(resolve_product_schedule)
log "Active products: $ACTIVE_PRODUCTS"

# Run pipeline for each product — failures are isolated per product.
# A failed product is logged and skipped; the rest continue.
FAILED_PRODUCTS=()
for entry in $ACTIVE_PRODUCTS; do
  SLUG="${entry%%:*}"
  DURATION="${entry##*:}"
  if ( run_product_pipeline "$SLUG" "$DURATION" ); then
    log "✅ [$SLUG] Pipeline succeeded."
  else
    log "❌ [$SLUG] Pipeline FAILED — skipping to next product."
    FAILED_PRODUCTS+=("$SLUG")
  fi
done

if [ ${#FAILED_PRODUCTS[@]} -gt 0 ]; then
  log ""
  log "⚠️  Products with failures: ${FAILED_PRODUCTS[*]}"
  log "   Re-run with: ./refresh_pipeline.sh --products $(IFS=,; echo "${FAILED_PRODUCTS[*]}") --duration-months $DURATION"
fi
PIPELINE_EXIT_CODE=0
if [ ${#FAILED_PRODUCTS[@]} -gt 0 ]; then
  PIPELINE_EXIT_CODE=1
fi

# ── Restart Streamlit ────────────────────────────────────────────────────────
log ""
log "── Restarting Streamlit ────────────────────────────────"

# Collect ALL pids listening on the Streamlit port (parent + children).
STREAMLIT_PIDS=$(lsof -ti tcp:"$STREAMLIT_PORT" 2>/dev/null | sort -u || true)
if [ -n "$STREAMLIT_PIDS" ]; then
  if $DRY_RUN; then
    log "  [DRY-RUN] kill PIDs: $STREAMLIT_PIDS"
  else
    # Graceful stop first (SIGTERM)
    for _pid in $STREAMLIT_PIDS; do
      kill "$_pid" 2>/dev/null || true
    done
    # Wait up to 10s for all processes to exit
    for _i in $(seq 1 10); do
      lsof -ti tcp:"$STREAMLIT_PORT" >/dev/null 2>&1 || break
      sleep 1
    done
    # Force-kill anything still lingering (SIGKILL)
    REMAINING=$(lsof -ti tcp:"$STREAMLIT_PORT" 2>/dev/null | sort -u || true)
    if [ -n "$REMAINING" ]; then
      log "  Streamlit still running after 10s — sending SIGKILL"
      for _pid in $REMAINING; do
        kill -9 "$_pid" 2>/dev/null || true
      done
      # Kill orphaned child processes by name (survive parent SIGKILL on macOS)
      pkill -9 -f "streamlit run" 2>/dev/null || true
      pkill -9 -f "bug_heatmap_dashboard" 2>/dev/null || true
      # Poll up to 15s for OS to release the port — fixed sleep is unreliable
      for _i in $(seq 1 15); do
        sleep 1
        lsof -ti tcp:"$STREAMLIT_PORT" >/dev/null 2>&1 || break
      done
    fi
    # Final verification
    if lsof -ti tcp:"$STREAMLIT_PORT" >/dev/null 2>&1; then
      log "  ERROR: port $STREAMLIT_PORT still in use after SIGKILL — cannot restart Streamlit"
    else
      log "  Stopped Streamlit (PIDs: $(echo $STREAMLIT_PIDS | tr '\n' ' '))"
    fi
  fi
else
  log "  Streamlit was not running — nothing to kill."
fi

# Clear any on-disk Streamlit cache to avoid stale data
if [ -d "$SCRIPT_DIR/.streamlit/cache" ]; then
  rm -rf "$SCRIPT_DIR/.streamlit/cache"
  log "  Cleared .streamlit/cache"
fi

STREAMLIT_OK=false
if $DRY_RUN; then
  log "  [DRY-RUN] streamlit run scripts/bug_heatmap_dashboard.py"
  STREAMLIT_OK=true
else
  # Only start if the port is actually free
  if lsof -ti tcp:"$STREAMLIT_PORT" >/dev/null 2>&1; then
    log "  ERROR: port $STREAMLIT_PORT still occupied — skipping Streamlit start"
  else
    # Brief pause to let any TIME_WAIT sockets fully clear before binding
    sleep 1
    nohup streamlit run "$SCRIPT_DIR/scripts/bug_heatmap_dashboard.py" \
      --server.address 0.0.0.0 \
      --server.port "$STREAMLIT_PORT" \
      --server.headless true \
      >> "$LOGS/streamlit.log" 2>&1 &
    NEW_PID=$!
    # Poll up to 15s — large dashboard takes longer than 3s to bind
    for _i in $(seq 1 15); do
      sleep 1
      if lsof -ti tcp:"$STREAMLIT_PORT" >/dev/null 2>&1; then
        log "  Streamlit restarted (PID $NEW_PID) on :$STREAMLIT_PORT — ready after ${_i}s"
        STREAMLIT_OK=true
        break
      fi
    done
    if ! $STREAMLIT_OK; then
      log "  WARNING: Streamlit process started but not yet listening on :$STREAMLIT_PORT after 15s"
      log "  Check $LOGS/streamlit.log for errors"
    fi
  fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
log ""
log "======================================================"
log "Refresh complete  $(date '+%Y-%m-%d %H:%M:%S')"
# Re-verify actual port state — STREAMLIT_OK can be stale if kill/start raced
if lsof -ti tcp:"$STREAMLIT_PORT" >/dev/null 2>&1; then
  log "Streamlit running on :$STREAMLIT_PORT — fresh data loaded."
else
  log "WARNING: Streamlit is NOT running on :$STREAMLIT_PORT — dashboard unavailable."
fi
log "Log: $LOG_FILE"
log "======================================================"

# Prune logs older than 30 days
find "$LOGS" -name "refresh_*.log" -mtime +30 -delete 2>/dev/null || true

# Preserve multi-product isolation (all products get a chance to run and the
# dashboard is restarted), but return failure to the shell/cron when any product
# failed so a partial refresh is never reported as fully successful.
exit "$PIPELINE_EXIT_CODE"
