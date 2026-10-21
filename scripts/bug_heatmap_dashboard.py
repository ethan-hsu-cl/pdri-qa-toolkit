#!/usr/bin/env python3
"""QA Bug Dashboard v3.0

Changes from v2.21:
  - Renamed from "PDR-I QA Dashboard" to "QA Bug Dashboard" for multi-product.
  - Added product selector dropdown in sidebar (scans data/products/ for
    available product directories containing ecl_parsed.csv).
  - All data paths are now dynamic based on selected product slug.
  - Falls back to legacy data/ paths when no product directories exist.
  - Version bumped to v3.0.

Changes from v2.20:
  - Tab 9 refactored to scenario-based bug forecasting layout.
    Primary view now shows "What to Test Next Build" with concrete bug
    scenario predictions per module, grouped by risk tier.
  - New "Predicted Bug Scenarios by Module" section with per-module
    expanders showing scenario text, confidence badges, and source examples.
  - Module forecast cards enhanced with top predicted scenarios.
  - Count-based sections (Predicted Bug Count chart, Actual vs Predicted,
    Module Signals Table, Full predictions table) demoted into a single
    collapsed "Advanced / Model Diagnostics" expander at the bottom.
  - Bug Categories stacked bar chart promoted to higher position.
  - Loads new _predictions_by_scenario.csv (v5.0) from predict_defects.py.
  - Headline metrics replace MAE with scenario count.

Changes from v2.19:

Changes from v2.18:
  - Sidebar Steps 1–4 consolidated into a single "📁 Data Sources" expander.
    Previously four separate expanders; now one panel with all four steps
    separated by st.divider() lines inside it.  The expander label shows a
    live badge for each dataset (e.g. "✅ bugs · ✅ risk · ✅ clusters · ✅
    forecast") so status is readable from the collapsed state.
    Expander auto-opens only when the required bug data file is not found.

Changes from v2.17:
  - Tab 9 forecast bar chart: fixed "High" risk bars disappearing from chart
    when the visible top-N slice contains no High-risk modules initially.
    Root cause: pd.Categorical with ordered=True caused Plotly to silently
    drop absent categories from both bars and legend.
    Fix: risk_level cast to plain str; sort order driven by a _risk_rank
    helper column instead; invisible zero-height add_bar() traces inserted for
    any of the four risk levels absent from the current slice, keeping all four
    legend colours always visible.

Changes from v2.16:
  - Sidebar Steps 1–4 are now collapsible expanders (collapsed by default).
    Each expander header shows a live ✅ badge when its default file path
    exists on disk, so you can confirm all data is loaded at a glance without
    opening any panel.  Step 1 auto-expands on first run (when the default
    path doesn't exist yet) and collapses itself once the file is found.
  - Compact data-status strip added below the expanders, always visible:
    shows bug count and which optional datasets are active
    (e.g. "📊 9,654 bugs  ·  🔥 risk  ·  🔬 clusters  ·  🔮 forecast").
  - Removed the sidebar success/info boxes that duplicated information now
    shown by the status strip and expander badges.
  - All sidebar widget calls inside expanders now use st.* instead of
    st.sidebar.* (Streamlit renders them in the correct context automatically).

Changes from v2.15:
  - Tab 8 (Bug Clusters): Expanded "📖 How to read this tab" expander into a
    full reference guide covering headline metrics, colour coding, bar chart
    interpretation, issue group detail cards, investigation workflow, Ollama vs
    TF-IDF mode explanation, and re-run cadence guidance.
  - Tab 9 (Defect Forecast): Expanded "📖 How to read this tab" expander into
    a full reference guide covering the prediction target, all six model
    features in plain English, risk level thresholds, headline metric meanings,
    module card fields (dominant_bug_type, leading_signal), actual-vs-predicted
    comparison guidance, leading indicators chart interpretation, model
    limitations, and data requirements.

Changes from v2.14:
  - Heatmap color scale: Module×Severity cells now use a blue gradient
    (HEATMAP_COLOR_SCALE) so continuous cell shading never clashes with the
    red/orange/yellow P1/P2 priority badge text on the same rows.
  - Module×Severity heatmap now sorted by total weighted count (all severities
    combined) so the busiest modules always float to the top.
  - Risk Score vs Probability scatter: added ±0.35 x-jitter, reduced bubble
    size_max from 30→18, added 0.75 opacity so overlapping dots are legible.
  - predict_defects.py companion: target variable documented, leading-indicator
    correlation added, plain-English focus summary generated per top-risk module.
  - fetch_from_n8n.py: status comparison now flags changed records with
    _status_changed=True; weekday/weekend scope auto-scheduling added.
  - parse_ecl_export.py: parsed_version now read from the dedicated Version
    column rather than regex-scraped from Short Description.

Changes from v2.13:
  - Reverted responsive layout — always uses left/right [6,4] split.
  - Removed JS width-detection snippet (components.html import gone).
  - Removed wide_layout / vertical logic and render_detail_panel helper.

Changes from v2.12:
  - dynamic treemap height (min 700, max 1100, scales with module count).
  - right panel scrollable div capped at treemap height.

Usage:
  streamlit run scripts/bug_heatmap_dashboard.py \\
    --server.address 0.0.0.0 --server.port 8501
"""
import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

try:
    from summarize_tr_bugs import NO_GROUNDED_REGRESSION
except ImportError:
    from scripts.summarize_tr_bugs import NO_GROUNDED_REGRESSION

try:
    from ecl_links import (
        ECL_BUG_URL, bug_link, link_column_values, linkify_bug_codes, tr_url,
    )
except ImportError:
    from scripts.ecl_links import (
        ECL_BUG_URL, bug_link, link_column_values, linkify_bug_codes, tr_url,
    )

try:
    from tr_focus_view import (
        checklist_label, collect_changesets, collect_version_fixes,
        build_focus_list, focus_list_is_current, focus_list_markdown,
        prior_comparison_context, render_cohort_provenance,
        render_recommendation_groups,
    )
    from tr_at_cases import (
        AT_REPOS, at_cases_for_focus, at_repo_dir, configured_at_dir,
        index_at_suites, pytest_commands, save_at_repo_dir,
    )
except ImportError:
    from scripts.tr_focus_view import (
        checklist_label, collect_changesets, collect_version_fixes,
        build_focus_list, focus_list_is_current, focus_list_markdown,
        prior_comparison_context, render_cohort_provenance,
        render_recommendation_groups,
    )
    from scripts.tr_at_cases import (
        AT_REPOS, at_cases_for_focus, at_repo_dir, configured_at_dir,
        index_at_suites, pytest_commands, save_at_repo_dir,
    )

try:
    from tr_history import (
        DEFAULT_HISTORY_DAYS, current_run_dir, history_digest, run_focus_list,
        run_history,
    )
except ImportError:
    from scripts.tr_history import (
        DEFAULT_HISTORY_DAYS, current_run_dir, history_digest, run_focus_list,
        run_history,
    )


st.set_page_config(page_title="QA Bug Dashboard", layout="wide", page_icon="🔥")
st.sidebar.title("🔥 QA Bug Dashboard v3.0")

# ── Product selector ─────────────────────────────────────────────────────
PRODUCT_MAP = {
    "pdri":   "PowerDirector Mobile (iOS)",
    "phdi":   "PhotoDirector Mobile (iOS)",
    "pdra":   "PowerDirector Mobile (Android)",
    "phda":   "PhotoDirector Mobile (Android)",
    "pdr":    "PowerDirector",
    "phd":    "PhotoDirector",
    "vvg":    "Vivid Glam (iOS)",
    "promeo": "Promeo",
}

_products_dir = Path("data/products")
_PRODUCT_ORDER = ["pdri", "pdra", "phdi", "phda", "pdr", "phd", "vvg", "promeo"]
_available_products = []
if _products_dir.exists():
    _existing_slugs = {d.name for d in _products_dir.iterdir() if d.is_dir() and (d / "ecl_parsed.csv").exists()}
    for slug in _PRODUCT_ORDER:
        if slug in _existing_slugs:
            _available_products.append((slug, PRODUCT_MAP.get(slug, slug)))
    # Append any unknown slugs not in the fixed order
    for slug in sorted(_existing_slugs - set(_PRODUCT_ORDER)):
        _available_products.append((slug, PRODUCT_MAP.get(slug, slug)))

if _available_products:
    _product_labels = [f"{label} ({slug})" for slug, label in _available_products]
    _selected_idx = st.sidebar.selectbox(
        "Product",
        range(len(_product_labels)),
        format_func=lambda i: _product_labels[i],
        key="product_selector",
    )
    selected_slug = _available_products[_selected_idx][0]
    st.session_state["selected_product"] = selected_slug
    _product_dir = f"data/products/{selected_slug}"
else:
    # Legacy fallback: no product directories, use flat data/
    selected_slug = None
    _product_dir = "data"

# Reset product-scoped widget/session state when switching products.
# Streamlit widget keys keep their prior values across reruns, so a changed
# default path alone will not update text_input/radio values unless we clear
# the old keys first.
_product_state_key = selected_slug or "__legacy__"
_prev_product_state_key = st.session_state.get("_last_product_state_key")
if _prev_product_state_key != _product_state_key:
    _prev_slug = None if _prev_product_state_key == "__legacy__" else _prev_product_state_key
    _new_slug = None if _product_state_key == "__legacy__" else _product_state_key
    # Preserve the currently selected navigation tab across the product
    # switch. Without this explicit re-write, the radio can desync after
    # `st.rerun()`: the sidebar keeps visually showing the previous tab
    # while `active_tab` evaluates to the default (index 0), rendering
    # Module × Severity instead of the user's selected tab.
    _preserved_active_tab = st.session_state.get("active_tab")
    # Explicitly set every product-scoped widget key to the new product's
    # defaults.  Simply popping keys is unreliable: Streamlit's client-side
    # widget state can re-inject old values on the next rerun.  By writing
    # the correct values here we guarantee the widgets render with the new
    # product's paths regardless of what the frontend sends.
    _new_dir = f"data/products/{selected_slug}" if selected_slug else "data"
    st.session_state["fp_bugs"]           = f"{_new_dir}/ecl_parsed.csv"
    st.session_state["fp_risk"]           = f"{_new_dir}/risk_register_scored_all.csv"
    st.session_state["fp_cluster"]        = f"{_new_dir}/clusters/ecl_parsed_clustered.csv"
    st.session_state["fp_cluster_sum"]    = f"{_new_dir}/clusters/ecl_parsed_cluster_summary.csv"
    st.session_state["fp_cluster_ent"]    = f"{_new_dir}/clusters/ecl_parsed_module_entropy.csv"
    st.session_state["fp_pred"]           = f"{_new_dir}/predictions/ecl_parsed_predictions.csv"
    st.session_state["fp_pred_sum"]       = f"{_new_dir}/predictions/ecl_parsed_predictions_focus_summary.txt"
    st.session_state["fp_pred_li"]        = f"{_new_dir}/predictions/ecl_parsed_predictions_leading_indicators.csv"
    st.session_state["fp_pred_cluster"]   = f"{_new_dir}/predictions/ecl_parsed_predictions_by_cluster.csv"
    st.session_state["fp_pred_category"]  = f"{_new_dir}/predictions/ecl_parsed_predictions_by_category.csv"
    st.session_state["fp_pred_scenario"]  = f"{_new_dir}/predictions/ecl_parsed_predictions_by_scenario.csv"
    # Reset radio/upload widgets and version selection to defaults.
    for _k in ["ds_bugs", "up_bugs", "ds_risk", "ds_cluster", "ds_pred",
               "version_multiselect"]:
        st.session_state.pop(_k, None)
    st.session_state["_product_switched_notice"] = (
        f"Product changed: {(_prev_slug or 'legacy data')} -> {(_new_slug or 'legacy data')}. "
        "Data source paths were reset to this product's defaults."
    )
    st.session_state["_last_product_state_key"] = _product_state_key
    # Re-write the preserved navigation tab after the pops above, so the
    # radio renders on the selected tab after the rerun (see comment near
    # `_preserved_active_tab` for why this is necessary).
    if _preserved_active_tab:
        st.session_state["active_tab"] = _preserved_active_tab
    # Force a clean rerun so widgets pick up the new session state values.
    st.rerun()

if st.session_state.get("_product_switched_notice"):
    st.toast(st.session_state.pop("_product_switched_notice"), icon="🔄")


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

QUADRANT_COLORS = {
    "P1 - Critical": "#ef4444",
    "P2 - High":     "#f97316",
    "P3 - Medium":   "#eab308",
    "P4 - Low":      "#22c55e",
}

# Separate color scale for continuous heatmap cells (severity-weighted counts).
# Uses blue→purple so it never clashes with the red/orange/yellow priority badge
# colors that appear as text labels on the same chart rows.
HEATMAP_COLOR_SCALE = "Blues"

PRIORITY_LABEL_MAP = {
    1: "1-Fix Now",
    2: "2-Must Fix",
    3: "3-Better Fix",
    4: "4-No Matter",
    5: "5-N/A",
}
PRIORITY_ORDER = ["1-Fix Now", "2-Must Fix", "3-Better Fix", "4-No Matter", "5-N/A"]

BARE_TO_FULL_PRIORITY = {
    "Fix Now":    "1-Fix Now",
    "Must Fix":   "2-Must Fix",
    "Better Fix": "3-Better Fix",
    "No Matter":  "4-No Matter",
    "N/A":        "5-N/A",
}

INACTIVE_STATUSES = {
    "close", "need more info", "nab", "propose nab",
    "wont fix", "won't fix", "propose wont fix", "qa propose wont fix",
    "not reproducible", "notreproducible", "not a bug",
    "new feature", "external issue", "hqqa close", "fae close",
}

TAB_NAMES = [
    "🗺️ Module × Severity",
    "📅 Version Timeline",
    "🏷️ Tag Analysis",
    "⚖️ P/S Alignment",
    "👥 Team Coverage",
    "📊 KPI Dashboard",
    "🔥 Risk Heatmap",
    "🔬 Bug Clusters",
    "🔮 Defect Forecast",
    "🎯 Release Pulse",
    "🧪 TR Test Focus",
]

SEV_OPTIONS = ["All", "1-Critical", "2-Major", "3-Normal", "4-Minor"]
SEV_NUM_MAP  = {"1-Critical": 1, "2-Major": 2, "3-Normal": 3, "4-Minor": 4}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def is_active(status_val) -> bool:
    if pd.isna(status_val):
        return True
    return str(status_val).strip().lower() not in INACTIVE_STATUSES


def _file_mtime(fp: str) -> float:
    """Return file mtime for cache-busting; 0 if missing."""
    try:
        return os.path.getmtime(fp)
    except OSError:
        return 0.0


@st.cache_data
def _load_csv_cached(fp: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(fp, low_memory=False)
    for dc in ["Create Date", "Closed Date"]:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
    return df


def load_csv(fp: str) -> pd.DataFrame:
    """Load CSV with automatic cache-busting when the file changes on disk."""
    return _load_csv_cached(fp, _file_mtime(fp))


def normalise_module(name: str) -> str:
    """Strip parenthetical and bracketed sub-variants from module names.
    e.g. "Auto Edit(Pet 02)"    -> "Auto Edit"
         "Transition[Portrait]" -> "Transition"
    Also collapses multiple spaces and strips whitespace.
    """
    import re
    if not isinstance(name, str):
        return name
    name = re.sub(r'\s*\([^)]*\)', '', name)
    name = re.sub(r'\s*\[[^\]]*\]\s*', ' ', name)
    name = re.sub(r' +', ' ', name).strip()
    return name


@st.cache_data(ttl=600, show_spinner=False)
def _load_at_index(repo: str) -> list:
    """AT_PDRi's files and cases; the SFT files run to 8,000 lines each."""
    return index_at_suites(Path(repo))


def show_dataframe(data, **kwargs):
    """``st.dataframe``, with any bug-code column rendered as an eCL link.

    Bug codes reach tables from a dozen call sites — risk drill-downs, cohort
    listings, prediction scoring, the Stage 8 diagnostics — and asking each one
    to remember a ``LinkColumn`` is how most of them ended up plain text. This
    decides from the data instead: a column is linked when every value in it is
    a single BugCode, so ``Source bug`` and ``analog_bug_code`` link without
    being named here, and a free-text column that merely mentions a code is
    left alone.

    Cells holding *several* codes (``"A · B · C"``) stay text — a dataframe
    cell carries one URL, not markdown. Those live in diagnostics expanders;
    prose renderings of the same lists go through ``linkify_bug_codes``.

    A caller's own ``column_config`` wins, so a table that already knows what
    it wants keeps it.
    """
    try:
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    except (TypeError, ValueError):
        st.dataframe(data, **kwargs)
        return

    caller_config = kwargs.pop("column_config", None) or {}
    config, linked = dict(caller_config), {}
    for column in frame.columns:
        if column in caller_config or list(frame.columns).count(column) > 1:
            continue
        urls = link_column_values(frame[column])
        if urls is None:
            continue
        linked[column] = urls
        config[column] = st.column_config.LinkColumn(
            str(column), display_text=r"BugCode=([^&]+)"
        )

    if linked:
        frame = frame.copy()
        for column, urls in linked.items():
            frame[column] = urls
    st.dataframe(frame, column_config=config or None, **kwargs)


def render_bug_table(frame: pd.DataFrame) -> None:
    display_cols_wanted = [
        "BugCode", "Short Description", "Status", "severity_label",
        "priority_label", "parsed_version", "Creator", "parsed_module",
    ]
    display_cols = [c for c in display_cols_wanted if c in frame.columns]
    disp = frame[display_cols].copy()
    bug_code_col = next((c for c in frame.columns if "bugcode" in c.lower()), None)
    if bug_code_col and bug_code_col in disp.columns:
        # Replace BugCode cell values with the full ECL URL.
        # display_text regex extracts the BugCode from the URL so the cell
        # shows e.g. "DRI261841-0001" but is clickable and opens ECL.
        disp[bug_code_col] = frame[bug_code_col].apply(
            lambda x: ECL_BUG_URL.format(bug_code=x) if pd.notna(x) else ""
        )
        st.dataframe(
            disp,
            column_config={
                bug_code_col: st.column_config.LinkColumn(
                    "BugCode",
                    display_text=r"BugCode=([^&]+)",
                )
            },
            width='stretch',
            hide_index=True,
        )
    else:
        st.dataframe(disp, width='stretch', hide_index=True)


# ---------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------

if "tm_selected_module"   not in st.session_state:
    st.session_state["tm_selected_module"]   = None
if "tm_selected_category" not in st.session_state:
    st.session_state["tm_selected_category"] = None


# ---------------------------------------------------------------------
# Sidebar – Navigation
# ---------------------------------------------------------------------

st.sidebar.subheader("📑 Navigation")
active_tab = st.sidebar.radio(
    "View", TAB_NAMES, key="active_tab", label_visibility="collapsed",
)


# ---------------------------------------------------------------------
# Sidebar – load data  (Steps 1–4 inside one collapsible expander)
# ---------------------------------------------------------------------
# All four data-loading steps live inside a single expander so the sidebar
# stays compact once paths are configured.  The expander label shows a live
# status badge so you know what is loaded without opening it.
# Step 1 is the only blocking load; Steps 2–4 are all optional.
#
# NOTE: derived-field computation (severity_weight, priority_label, etc.)
# runs OUTSIDE the expander block — those lines produce no sidebar widgets
# and must execute unconditionally regardless of whether the expander is open.
# ---------------------------------------------------------------------

# Probe all default paths up front so the expander label can show a
# meaningful status badge even before the user opens it.
_bugs_default         = f"{_product_dir}/ecl_parsed.csv"
_risk_default         = f"{_product_dir}/risk_register_scored_all.csv"
_cluster_default      = f"{_product_dir}/clusters/ecl_parsed_clustered.csv"
_cluster_sum_default  = f"{_product_dir}/clusters/ecl_parsed_cluster_summary.csv"
_cluster_ent_default  = f"{_product_dir}/clusters/ecl_parsed_module_entropy.csv"
_cluster_s12_default  = f"{_product_dir}/clusters/ecl_parsed_cluster_summary_s12.csv"
_cluster_s34_default  = f"{_product_dir}/clusters/ecl_parsed_cluster_summary_s34.csv"
_pred_default         = f"{_product_dir}/predictions/ecl_parsed_predictions.csv"
_pred_sum_def         = f"{_product_dir}/predictions/ecl_parsed_predictions_focus_summary.txt"
_pred_li_def          = f"{_product_dir}/predictions/ecl_parsed_predictions_leading_indicators.csv"
_pred_cluster_def     = f"{_product_dir}/predictions/ecl_parsed_predictions_by_cluster.csv"
_pred_category_def    = f"{_product_dir}/predictions/ecl_parsed_predictions_by_category.csv"
_pred_scenario_def    = f"{_product_dir}/predictions/ecl_parsed_predictions_by_scenario.csv"
_pred_imp_def         = f"{_product_dir}/predictions/ecl_parsed_predictions_importance.csv"

_bugs_probe    = Path(_bugs_default).exists()
_risk_probe    = Path(_risk_default).exists()
_cluster_probe = Path(_cluster_default).exists()
_pred_probe    = Path(_pred_default).exists()

# Build a compact badge: ✅ for each path that exists
_badge_parts = ["✅ bugs" if _bugs_probe else "⚠️ bugs"]
if _risk_probe:    _badge_parts.append("✅ risk")
if _cluster_probe: _badge_parts.append("✅ clusters")
if _pred_probe:    _badge_parts.append("✅ forecast")
_datasources_label = f"📁 Data Sources — {' · '.join(_badge_parts)}"

# Auto-open only on first run (when the required bug data file isn't found yet)
with st.sidebar.expander(_datasources_label, expanded=not _bugs_probe):

    # ── Step 1 — Bug data (required) ────────────────────────────────────────
    st.markdown("**Step 1 — Bug data** (required)")
    ds = st.radio("Bug data source", ["Upload CSV", "File Path"], key="ds_bugs", index=1)

    if ds == "Upload CSV":
        up = st.file_uploader("Upload ecl_parsed.csv", type="csv", key="up_bugs")
        if up:
            df = pd.read_csv(up, low_memory=False)
            for dc in ["Create Date", "Closed Date"]:
                if dc in df.columns:
                    df[dc] = pd.to_datetime(df[dc], errors="coerce")
        else:
            st.info("📂 Upload **ecl_parsed.csv** (from `parse_ecl_export.py`) to begin.")
            st.stop()
    else:
        fp = st.text_input("CSV path", value=_bugs_default, key="fp_bugs")
        if Path(fp).exists():
            df = load_csv(fp)
            st.caption(f"✅ {len(df):,} bugs loaded")
        else:
            st.error(f"File not found: {fp}")
            st.stop()

    st.divider()

    # ── Step 2 — Risk scores (optional) ─────────────────────────────────────
    # Source info stored here; actual merge happens after the version filter.
    st.markdown("**Step 2 — Risk scores** (optional, enriches all charts)")

    risk_base_path: str = ""
    risk_ver_dir:   str = ""
    risk_uploaded       = None

    rds = st.radio(
        "Risk data source", ["File Path", "None"], key="ds_risk", index=0
    )
    if rds == "File Path":
        rfp = st.text_input(
            "Risk CSV path", value=_risk_default, key="fp_risk",
        )
        if Path(rfp).exists():
            risk_base_path = rfp
            risk_ver_dir   = str(Path(rfp).parent / "risk_register_versions")
            st.caption("✅ Risk scores found")
        elif rfp:
            st.caption("⚠️ File not found — run ai_risk_scorer.py first.")

    st.divider()

    # ── Step 3 — Bug clusters (optional) ────────────────────────────────────
    st.markdown("**Step 3 — Bug clusters** (optional, unlocks Tab 8)")

    cluster_df      = None
    cluster_sum_df  = None
    cluster_ent_df  = None   # v3.0: module entropy
    cluster_s12_df  = None   # v3.0: S1/S2 stratified summary
    cluster_s34_df  = None   # v3.0: S3/S4 stratified summary

    cds = st.radio(
        "Cluster data source", ["File Path", "None"], key="ds_cluster", index=0
    )
    if cds == "File Path":
        cfp = st.text_input(
            "Clustered CSV path", value=_cluster_default, key="fp_cluster",
        )
        csfp = st.text_input(
            "Cluster summary CSV", value=_cluster_sum_default, key="fp_cluster_sum",
        )
        cefp = st.text_input(
            "Module entropy CSV (v3.0)", value=_cluster_ent_default, key="fp_cluster_ent",
        )
        if Path(cfp).exists():
            cluster_df = load_csv(cfp)
            st.caption(f"✅ {len(cluster_df):,} bugs loaded")
        elif cfp:
            st.caption("⚠️ File not found — run cluster_bugs.py first.")
        if Path(csfp).exists():
            cluster_sum_df = load_csv(csfp)
        if Path(cefp).exists():
            cluster_ent_df = load_csv(cefp)
        # Optional stratified summaries (auto-loaded if present, no UI input needed)
        if Path(_cluster_s12_default).exists():
            cluster_s12_df = load_csv(_cluster_s12_default)
        if Path(_cluster_s34_default).exists():
            cluster_s34_df = load_csv(_cluster_s34_default)

    st.divider()

    # ── Step 4 — Defect forecast (optional) ─────────────────────────────────
    st.markdown("**Step 4 — Defect forecast** (optional, unlocks Tab 9)")

    pred_df             = None
    pred_summary_txt    = ""
    pred_leading_df     = None
    pred_cluster_df     = None   # v3.0: per-cluster bug-type predictions
    pred_category_df    = None   # v4.0: per-category bug-type predictions
    pred_scenario_df    = None   # v5.0: per-module bug scenario predictions
    pred_importance_df  = None   # feature importances from trained model

    pds = st.radio(
        "Prediction data source", ["File Path", "None"], key="ds_pred", index=0
    )
    if pds == "File Path":
        pfp = st.text_input(
            "Predictions CSV path", value=_pred_default, key="fp_pred",
        )
        psfp = st.text_input(
            "Focus summary .txt", value=_pred_sum_def, key="fp_pred_sum",
        )
        plfp = st.text_input(
            "Leading indicators CSV", value=_pred_li_def, key="fp_pred_li",
        )
        pcfp = st.text_input(
            "Bug-type predictions CSV (v3.0)", value=_pred_cluster_def, key="fp_pred_cluster",
        )
        pcatfp = st.text_input(
            "Bug category predictions CSV (v4.0)", value=_pred_category_def, key="fp_pred_category",
        )
        pscfp = st.text_input(
            "Bug scenario predictions CSV (v5.0)", value=_pred_scenario_def, key="fp_pred_scenario",
        )
        pimpfp = st.text_input(
            "Feature importance CSV", value=_pred_imp_def, key="fp_pred_imp",
        )
        if Path(pfp).exists():
            pred_df = load_csv(pfp)
            st.caption(f"✅ {len(pred_df):,} modules loaded")
        elif pfp:
            st.caption("⚠️ File not found — run predict_defects.py first.")
        if Path(psfp).exists():
            pred_summary_txt = Path(psfp).read_text(encoding="utf-8")
        if Path(plfp).exists():
            pred_leading_df = load_csv(plfp)
        if Path(pcfp).exists():
            pred_cluster_df = load_csv(pcfp)
        if Path(pcatfp).exists():
            pred_category_df = load_csv(pcatfp)
        if Path(pscfp).exists():
            pred_scenario_df = load_csv(pscfp)
        if Path(pimpfp).exists():
            pred_importance_df = load_csv(pimpfp)

# ── End of expander ────────────────────────────────────────────────────────
# Derived-field computation runs here, outside the expander, so it always
# executes regardless of whether the panel is open or closed.

required = ["parsed_module", "severity_num"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("❌ Wrong file! Upload **ecl_parsed.csv** (bug-level), not risk_register_scored.csv")
    st.warning(f"Missing columns: {', '.join(missing)}")
    st.stop()

if "risk_score_final" in df.columns and "quadrant" in df.columns and "bug_count" in df.columns:
    st.error("❌ This looks like risk_register_scored.csv. Please upload ecl_parsed.csv instead.")
    st.stop()

if "severity_weight" not in df.columns:
    df["severity_weight"] = df["severity_num"].map({1: 10, 2: 5, 3: 2, 4: 1}).fillna(2)
if "module_category" not in df.columns:
    df["module_category"] = "Uncategorized"

if "priority_label" not in df.columns and "Priority" in df.columns:
    df["priority_num"] = pd.to_numeric(df["Priority"], errors="coerce").fillna(5).astype(int)
    df["priority_label"] = df["priority_num"].map(PRIORITY_LABEL_MAP).fillna("5-N/A")
elif "priority_label" not in df.columns:
    df["priority_label"] = "5-N/A"

df["priority_label"] = df["priority_label"].apply(
    lambda v: BARE_TO_FULL_PRIORITY.get(str(v).strip(), str(v).strip())
    if pd.notna(v) else "5-N/A"
)

if "status_active" not in df.columns:
    if "Status" in df.columns:
        df["status_active"] = df["Status"].apply(is_active)
    else:
        df["status_active"] = True

if "parsed_module" in df.columns:
    df["parsed_module"] = df["parsed_module"].apply(
        lambda x: normalise_module(x) if pd.notna(x) else x
    )

sl_map = {1: "1-Critical", 2: "2-Major", 3: "3-Normal", 4: "4-Minor"}
if "severity_label" not in df.columns:
    df["severity_label"] = df["severity_num"].map(sl_map)


# ---------------------------------------------------------------------
# Sidebar – filters
# ---------------------------------------------------------------------

st.sidebar.subheader("🔍 Filters")

include_inactive = st.sidebar.checkbox(
    "Include closed/inactive bugs (Close, NAB, Won't Fix, etc.)",
    value=False,
    help="Inactive = Close, NAB, Won't Fix, Not Reproducible, New Feature, Need More Info",
)
if not include_inactive:
    df = df[df["status_active"] == True]

only_s1_s2 = st.sidebar.checkbox(
    "Only show S1/S2 bugs (Critical & Major)",
    value=True,
    help="S1 = Critical, S2 = Major. Uncheck to include S3 (Normal) and S4 (Minor) bugs.",
)
if only_s1_s2 and "severity_num" in df.columns:
    df = df[df["severity_num"].isin([1, 2])]

if "parsed_version" in df.columns:
    # ── Version ordering: recency-first, sparse versions last ─────────────
    # We read version_catalogue.csv (produced by parse_ecl_export.py) when
    # available because it contains recency order based on max(Create Date)
    # per version — robust against version strings that don't sort
    # lexicographically and against typo/sparse versions.
    #
    # Fallback: if the catalogue isn't present, derive the same ordering
    # directly from the loaded dataframe.

    _cat_path = Path(fp if ds != "Upload CSV" else "").parent / "version_catalogue.csv" \
        if ds != "Upload CSV" else None

    def _build_version_order_from_df(vdf):
        """Derive recency-ordered version list directly from bug data."""
        VERSION_SPARSE_THRESHOLD = 5
        rows = []
        for ver, grp in vdf.groupby("parsed_version", dropna=True):
            n = len(grp)
            latest = pd.to_datetime(grp["Create Date"], errors="coerce").max() \
                if "Create Date" in grp.columns else pd.NaT
            rows.append({"v": ver, "n": n, "d": latest,
                         "sparse": n < VERSION_SPARSE_THRESHOLD})
        if not rows:
            return [], []
        import pandas as _pd
        tmp = _pd.DataFrame(rows)
        tmp["_k"] = tmp["d"].fillna(_pd.Timestamp.min)
        real   = tmp[~tmp["sparse"]].sort_values("_k", ascending=False)
        sparse = tmp[ tmp["sparse"]].sort_values("_k", ascending=False)
        ordered = _pd.concat([real, sparse], ignore_index=True)
        all_vers  = ordered["v"].tolist()
        real_vers = ordered[~ordered["sparse"]]["v"].tolist()
        return all_vers, real_vers

    vers_all   = []   # all versions, recency order
    vers_real  = []   # non-sparse only, for default selection

    if _cat_path and _cat_path.exists():
        try:
            _cat = load_csv(str(_cat_path))
            if {"parsed_version", "version_rank", "version_is_sparse"}.issubset(_cat.columns):
                _cat = _cat.sort_values("version_rank")
                vers_all  = [str(r) for r in _cat["parsed_version"].dropna()]
                vers_real = [
                    str(r["parsed_version"])
                    for _, r in _cat.iterrows()
                    if pd.notna(r["parsed_version"]) and not r["version_is_sparse"]
                ]
                n_sparse = int(_cat["version_is_sparse"].sum())
                if n_sparse:
                    st.sidebar.caption(
                        f"ℹ️ {n_sparse} sparse/typo version(s) hidden from default selection "
                        f"(shown in full list below)."
                    )
        except Exception:
            pass  # fall through to dataframe derivation

    if not vers_all:
        vers_all, vers_real = _build_version_order_from_df(df)

    # Keep only versions that actually exist in the current (possibly filtered) df
    present = set(df["parsed_version"].dropna().unique())
    vers_all  = [v for v in vers_all  if v in present]
    vers_real = [v for v in vers_real if v in present]

    # Default: all non-sparse versions; fall back to all versions.
    # Session state key "version_multiselect" is seeded on first load only,
    # so "All versions" / "Clear" buttons can override it without being
    # overwritten by the default= argument on subsequent renders.
    default_vers = vers_real[:3] if vers_real else vers_all[:3]

    # Reset version selection when the data file changes on disk (e.g., mid-pipeline
    # refresh where pdri completes before Streamlit is restarted). Without this, the
    # old selection (e.g., 2 weekday versions) persists even after new 36-month data
    # is promoted because the pruning below only removes versions that no longer exist.
    _cur_mtime = _file_mtime(fp) if ds != "Upload CSV" else 0.0
    if st.session_state.get("_version_data_mtime") != _cur_mtime:
        st.session_state["_version_data_mtime"] = _cur_mtime
        st.session_state.pop("version_multiselect", None)

    if "version_multiselect" not in st.session_state:
        st.session_state["version_multiselect"] = default_vers
    else:
        # Prune stale selections that no longer exist in the current product's
        # version list (safety net for product switches).
        _valid = set(vers_all)
        _cur = st.session_state["version_multiselect"]
        _pruned = [v for v in _cur if v in _valid]
        if len(_pruned) != len(_cur):
            st.session_state["version_multiselect"] = _pruned if _pruned else default_vers

    # Quick-select buttons above the multiselect
    _btn_col1, _btn_col2 = st.sidebar.columns(2)
    if _btn_col1.button("All versions", use_container_width=True):
        st.session_state["version_multiselect"] = vers_all
    if _btn_col2.button("Clear", use_container_width=True):
        st.session_state["version_multiselect"] = []

    sel_v = st.sidebar.multiselect(
        "Version", vers_all, key="version_multiselect",
    )
    if sel_v:
        df = df[df["parsed_version"].isin(sel_v)]
else:
    sel_v = []

# Version-aware risk file selection
# Rule: 1 version selected  -> try per-version scored file first
#       0 or 2+ selected    -> use combined _all file
def _load_risk_df(base_path, ver_dir, sel_versions, uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded, low_memory=False)
    if not base_path:
        return None
    if len(sel_versions) == 1:
        ver_safe = str(sel_versions[0]).replace(" ", "_")
        ver_path = Path(ver_dir) / f"risk_register_scored_{ver_safe}.csv"
        if ver_path.exists():
            return load_csv(str(ver_path))
        st.sidebar.caption(
            f"No per-version score file for {sel_versions[0]} — using combined scores."
        )
    return load_csv(base_path)

risk_df = _load_risk_df(risk_base_path, risk_ver_dir, sel_v, risk_uploaded)

if risk_df is not None:
    risk_ok = all(c in risk_df.columns for c in ["parsed_module", "risk_score_final", "quadrant"])
    if not risk_ok:
        st.sidebar.warning("Risk CSV missing required columns — ignoring.")
        risk_df = None

if risk_df is not None:
    risk_cols = [c for c in [
        "parsed_module", "quadrant", "risk_score_final",
        "impact_score", "detectability_score",
        "impact_score_ai", "detectability_score_ai",
        "probability_score_auto",
    ] if c in risk_df.columns]

    risk_df["parsed_module"] = risk_df["parsed_module"].apply(
        lambda x: normalise_module(x) if pd.notna(x) else x
    )
    risk_df_dedup = (
        risk_df[risk_cols]
        .sort_values("risk_score_final", ascending=False)
        .drop_duplicates(subset=["parsed_module"], keep="first")
    )
    df = df.merge(risk_df_dedup, on="parsed_module", how="left")
    df["quadrant"] = df.get("quadrant", pd.Series(["P4 - Low"] * len(df))).fillna("P4 - Low")
    df["risk_score_final"] = df.get("risk_score_final", pd.Series([0.0] * len(df))).fillna(0.0)
    risk_available = True

    n_unique    = risk_df_dedup["parsed_module"].nunique()
    ver_context = (
        f"version {sel_v[0]}" if len(sel_v) == 1
        else f"{len(sel_v)} versions selected" if sel_v
        else "all versions"
    )
    # Status is shown in the compact strip below the expanders, not as a
    # separate success box, to keep the sidebar tidy.
else:
    df["quadrant"]        = "Unknown"
    df["risk_score_final"] = 0.0
    risk_available         = False
if "Status" in df.columns:
    stats = sorted(df["Status"].dropna().unique().tolist())
    sel_s = st.sidebar.multiselect("Status", stats, default=stats)
    if sel_s:
        df = df[df["Status"].isin(sel_s)]

if "severity_num" in df.columns:
    sel_sev = st.sidebar.multiselect(
        "Severity", [1, 2, 3, 4], default=[1, 2, 3, 4],
        format_func=lambda x: {1: "S1-Critical", 2: "S2-Major", 3: "S3-Normal", 4: "S4-Minor"}[x],
    )
    if sel_sev:
        df = df[df["severity_num"].isin(sel_sev)]

if risk_available:
    all_q = sorted(df["quadrant"].dropna().unique().tolist())
    sel_q = st.sidebar.multiselect("Priority", all_q, default=all_q)
    if sel_q:
        df = df[df["quadrant"].isin(sel_q)]

active_label = "active " if not include_inactive else ""
st.sidebar.markdown(f"**Showing {len(df):,} {active_label}bugs**")


# =====================================================================
# TAB 1 – Module × Severity
# =====================================================================

if active_tab == "🗺️ Module × Severity":
    st.header("Module × Severity Heatmap")

    with st.expander("📖 How to read this chart", expanded=False):
        st.markdown("""
**One-line summary:** Where are bugs concentrated, and how serious are they? Rows = modules (or product categories), columns = severity. Darker cell = bigger problem.

**Why severity-weighted instead of raw counts:** A module with 2 crashes is more dangerous than a module with 20 cosmetic typos. We multiply each bug by a severity weight so the heatmap prioritises *impact* over *volume*:

| Severity | Weight | Meaning |
|----------|--------|---------|
| S1 — Critical | ×10 | App crash, data loss, showstopper |
| S2 — Major    | ×5  | Key feature broken, no workaround |
| S3 — Normal   | ×2  | Feature impaired, workaround exists |
| S4 — Minor    | ×1  | Cosmetic, low impact |

**Sort order:** Modules with the most S1 (Critical) bugs always appear at the top — ties broken by total weighted count. This makes sure crash-level bugs are never hidden behind a long tail of minor issues.

**View toggle:**
- **Category** (default) — 20 broad product areas, best for a quick executive overview ("which area is hurting?").
- **Module (top 30)** — drill into the 30 individual modules with the most weighted bugs, useful once you've spotted a problem area.

**`[P1]` / `[P2]` badges** next to module names appear when risk data is loaded (sidebar Step 2). They mark the module's FMEA test priority — see the Risk Heatmap tab for the full I×P×D scoring.

**Click any cell** to auto-fill the drill-down table below the chart with every matching bug — ECL link, severity, priority, Short Description — so you can investigate without leaving the dashboard.

**Reading tips for your manager:**
- Dark column on the left (S1) = release-blocking risk in that module.
- A row that's dark across all severity columns = chronic instability, not a one-off bug.
- Compare the same module against the [P1]/[P2] badge — a P1 module with a dark row is the highest priority on the dashboard.

---
**Where this data comes from:** Bugs are parsed from the n8n / ECL export by `parse_ecl_export.py`. Severity weights (S1=10, S2=5, S3=2, S4=1) are applied per bug at parse time, then aggregated here per module. **Closed/fixed bugs still appear in this view at full weight** — this tab shows the historical bug load. (The Defect Forecast tab decays closed-bug signal over time; that decay is *not* applied here.) The `[P1]`/`[P2]` badges come from `risk_register_scored.csv` produced by the I×P×D scoring pipeline (see Risk Heatmap tab).
""")

    vl = st.radio("View", ["Category", "Module (top 30)"], horizontal=True, key="t1v")
    gc = "module_category" if vl.startswith("C") else "parsed_module"

    df["severity_label"] = df["severity_num"].map(sl_map)

    pv = df.pivot_table(
        index=gc, columns="severity_label",
        values="severity_weight", aggfunc="sum", fill_value=0,
    )
    for lbl in ["1-Critical", "2-Major", "3-Normal", "4-Minor"]:
        if lbl not in pv.columns:
            pv[lbl] = 0
    pv = pv[["1-Critical", "2-Major", "3-Normal", "4-Minor"]]

    if vl.startswith("M"):
        pv = pv.loc[pv.sum(axis=1).nlargest(30).index]

    # Sort so highest-count modules appear at the top of the heatmap.
    # Primary sort: S1 (Critical) descending so the most critical modules
    # always float to the top; ties broken by total weighted count descending.
    sort_key = pd.DataFrame({
        "s1":    pv["1-Critical"],
        "total": pv.sum(axis=1),
    })
    pv = pv.loc[sort_key.sort_values(["s1", "total"], ascending=[False, False]).index]

    pv_display = pv.copy()
    if risk_available:
        q_map = df.groupby(gc)["quadrant"].first().to_dict()
        pv_display.index = [f"{y} [{q_map.get(y, '—')}]" for y in pv.index]

    # Use HEATMAP_COLOR_SCALE (blue gradient) so the continuous cell shading
    # does NOT conflict with the red/orange/yellow priority badge text on each row.
    fig = px.imshow(
        pv_display,
        labels=dict(x="Severity", y=gc, color="Weighted Count"),
        color_continuous_scale=HEATMAP_COLOR_SCALE,
        aspect="auto",
        text_auto=True,
    )
    fig.update_layout(
        height=max(400, len(pv_display) * 28),
        clickmode="event+select",
    )
    if risk_available:
        st.caption(
            "🔵 Cell shade = severity-weighted bug count (darker = more/heavier bugs). "
            "🔴 [P1] / 🟠 [P2] badges next to row labels = risk priority — kept separate "
            "from cell colour to avoid red-on-red confusion."
        )

    available_groups = sorted(df[gc].dropna().unique().tolist())
    if "dd_group" not in st.session_state or st.session_state["dd_group"] not in available_groups:
        st.session_state["dd_group"] = available_groups[0] if available_groups else None
    if "dd_sev" not in st.session_state or st.session_state["dd_sev"] not in SEV_OPTIONS:
        st.session_state["dd_sev"] = "All"

    event = st.plotly_chart(
        fig,
        width='stretch',
        on_select="rerun",
        selection_mode=["points"],
        key="heatmap_click",
    )

    if event and hasattr(event, "selection"):
        sel = event.selection
        pts = sel.points if hasattr(sel, "points") else sel.get("points", [])
        if pts:
            pt    = pts[0]
            raw_x = pt.get("x") if isinstance(pt, dict) else getattr(pt, "x", None)
            raw_y = pt.get("y") if isinstance(pt, dict) else getattr(pt, "y", None)
            clean_y = str(raw_y).split(" [")[0].strip() if raw_y is not None else None
            if clean_y and clean_y in available_groups:
                st.session_state["dd_group"] = clean_y
            if raw_x is not None and str(raw_x) in SEV_OPTIONS:
                st.session_state["dd_sev"] = str(raw_x)

    st.markdown("---")
    st.subheader("🔗 Bug Drill-Down")
    st.caption("Click a heatmap cell above to auto-fill, or select manually below.")

    col_a, col_b = st.columns(2)
    with col_a:
        drill_group = st.selectbox("Module / Category", options=available_groups, key="dd_group")
    with col_b:
        drill_sev = st.selectbox("Severity", options=SEV_OPTIONS, key="dd_sev")

    drill_df = df[df[gc] == drill_group].copy()
    if drill_sev != "All":
        drill_df = drill_df[drill_df["severity_num"] == SEV_NUM_MAP[drill_sev]]

    if len(drill_df) == 0:
        st.info("No bugs match this selection.")
    else:
        render_bug_table(drill_df)
        st.caption(f"Showing {len(drill_df)} bugs for **{drill_group}** / **{drill_sev}**")

    with st.expander("Raw pivot data"):
        show_dataframe(pv, width='stretch')


# =====================================================================
# TAB 2 – Version Timeline
# =====================================================================

elif active_tab == "📅 Version Timeline":
    st.header("Module × Version Timeline")

    with st.expander("📖 How to read this chart", expanded=False):
        st.markdown("""
**One-line summary:** A history of every module's bug load, version by version. Read it left to right like a timeline — oldest release on the left, newest on the right.

**The four patterns to look for** (and what each one means for your release):

| Pattern in the chart | What it means | Action to suggest |
|---|---|---|
| One cell suddenly darkens in a single column | **Regression** introduced in that specific release for that module | Check what changed in that build — recent merge or SDK update |
| A row stays warm across *every* version | **Chronic instability** — module is never stable regardless of release cadence | Needs an architectural fix, not another patch |
| One column is dark across *many* rows | **Bad release** — a large merge, SDK update, or new feature destabilised many areas at once | Post-mortem for that build; consider gating future merges of similar size |
| Dark → light → dark again | **Fix regressed** — the bug came back | Escalate root-cause review with RD |
| Gradually lightening over time | **Healthy trend** — bugs are being resolved without coming back | Confirm before sign-off |

**Version filter:** The sidebar version multiselect controls which columns appear.
- **Default = 3 most recent versions** — best for spotting fresh regressions in the current release cycle.
- **Select all versions** to see chronic problem rows that span the full history.

**View toggle:**
- **Category** — quick view of which product area had the worst release.
- **Module (top 25)** — pinpoint the exact sub-feature, useful once a category is flagged.

**Severity weighting** is the same as Tab 1 (S1×10, S2×5, S3×2, S4×1). A single S1 lights up the cell more than ten S4s — critical bugs stay visually loud.

**For your manager:** Hover any column header to see the version. The chart answers two questions at once: *"Which release was the worst?"* (compare columns) and *"Which module is the most consistent source of pain?"* (compare rows).

---
**Where this data comes from:** `parsed_version` is extracted from the n8n / ECL `Version` field by `parse_ecl_export.py`. Severity weighting is applied per bug at parse time. The pipeline also writes one `risk_register_<version>.csv` per version so per-release I×P×D scoring is available downstream.
""")

    if "parsed_version" in df.columns:
        vl2 = st.radio("View", ["Category", "Module (top 25)"], horizontal=True, key="t2v")
        gc2 = "module_category" if vl2.startswith("C") else "parsed_module"

        tl = df.pivot_table(
            index=gc2, columns="parsed_version",
            values="severity_weight", aggfunc="sum", fill_value=0,
        )
        tl = tl[sorted(tl.columns)]
        row_weights = tl.sum(axis=1)
        if vl2.startswith("M"):
            tl = tl.loc[row_weights.nlargest(25).sort_values(ascending=False).index]
        else:
            tl = tl.loc[row_weights.sort_values(ascending=False).index]

        fig2 = px.imshow(
            tl,
            labels=dict(x="Version", y=gc2, color="Weighted"),
            color_continuous_scale="YlOrRd",
            aspect="auto",
            text_auto=True,
        )
        fig2.update_layout(height=max(400, len(tl) * 28))
        st.plotly_chart(fig2, width='stretch')
    else:
        st.warning("No `parsed_version` column found.")


# =====================================================================
# TAB 3 – Tag Analysis
# =====================================================================

elif active_tab == "🏷️ Tag Analysis":
    st.header("Tag Distribution")

    with st.expander("📖 How to read this chart", expanded=False):
        st.markdown("""
**One-line summary:** What *kinds* of bugs is each module producing, and how well is automation catching them? Tags answer "what type of failure" — bug counts alone don't.

**Where tags come from:** Every bug's Short Description starts with `[TAGS]` set by the QA team. Example: `PDR-I 16.2.5 - [EDF][UX] AI Storytelling: subtitle misplaced` → tags `edf`, `ux`. The parser turns these into boolean columns per bug.

**The five tags that matter most:**

| Tag | Meaning | Why your manager should care |
|-----|---------|----------------------------|
| `Side Effect` | **Regression** — something that used to work broke after a code change | High side-effect rate = code base is fragile; need stronger regression suite |
| `AT Found` | Caught by **automated tests** | Good signal — AT is doing its job |
| `EDF` | **Engineering Design Flaw** — architectural / design root cause | Cannot be patched away; needs RD design review |
| `UX` | User experience problem | Polish gap — affects perception more than function |
| `MUI` | Multi-UI / platform consistency issue | Cross-device QA gap |

**Three views in this tab:**

1. **Main heatmap (top)** — raw tag counts per module/category. Darker = more bugs of that type. Use it to spot which kind of bug dominates each area (e.g. an `EDF`-heavy module needs design review, not just more testing).

2. **Regression Bug Rate (Side Effect %)** — bar chart ranking modules by *fraction* of bugs that are regressions, not raw count. Formula: `(bugs tagged [Side Effect]) ÷ (total bugs) × 100`.
   - **≥30%** = nearly 1 in 3 bugs is a regression. This module **must** be regression-tested every single version.

3. **Automation Coverage (AT Found Rate)** — bar chart ranking modules by *fraction* caught by automated tests:
   - 🟢 **≥30%** — strong AT safety net.
   - 🟡 **10–29%** — partial coverage; manual gaps remain.
   - 🔴 **<10%** — automation blind spot.

**Automation Blind Spots** (bottom of the tab) — explicit list of modules with **0% AT-found rate**. Every bug in these modules currently relies on manual discovery. These are the highest-ROI candidates for new automated tests.

**Manager pitch:** *"Tag analysis lets us tell the difference between a feature that needs more tests (high Side Effect rate) and one that needs a design fix (high EDF count). Same bug count, totally different action."*

---
**How this feeds the rest of the dashboard:**
- `regression_rate` and `automation_catch_rate` feed into **Detectability (D)** in `ai_risk_scorer.py`. A module with 0% AT coverage gets D+1, pushing its I×P×D risk score up — that's why the Risk Heatmap (Tab 7) doesn't only react to bug counts.
- The forecast (Tab 9) uses `tag_side_effect` history as one of its leading indicators of next-version regressions.
""")

    tc = [c for c in df.columns if c.startswith("tag_")]
    if tc:
        vl3 = st.radio("View", ["Category", "Module (top 25)"], horizontal=True, key="t3v")
        gc3 = "module_category" if vl3.startswith("C") else "parsed_module"

        tp = df.groupby(gc3)[tc].sum()
        tp.columns = [c.replace("tag_", "").replace("_", " ").title() for c in tp.columns]
        if vl3.startswith("M"):
            tp = tp.loc[tp.sum(axis=1).nlargest(25).index]

        fig3 = px.imshow(
            tp,
            labels=dict(x="Tag", y=gc3, color="Count"),
            color_continuous_scale="Viridis",
            aspect="auto",
            text_auto=True,
        )
        fig3.update_layout(height=max(400, len(tp) * 28))
        st.plotly_chart(fig3, width='stretch')

        st.subheader("Regression Bug Rate [Side Effect]")
        se = [c for c in tc if "side_effect" in c.lower()]
        if se:
            total_per_mod = df.groupby(gc3).size()
            se_count = df[df[se[0]] == True].groupby(gc3).size()
            se_rate = (
                (se_count / total_per_mod * 100)
                .fillna(0).sort_values(ascending=False).head(15)
            )
            fig_se = px.bar(
                se_rate.reset_index(), x=gc3, y=0,
                labels={gc3: "Module/Category", "0": "Regression Bug Rate (%)"},
                title="Top 15 by Regression Bug Rate (%)",
            )
            fig_se.update_layout(height=350, xaxis_tickangle=-30)
            st.plotly_chart(fig_se, width='stretch')

        st.subheader("Automation Coverage [AT Found Rate]")
        at = [c for c in tc if "at_found" in c.lower()]
        if at:
            total_per_mod = df.groupby(gc3).size()
            at_count = (
                df[df[at[0]] == True].groupby(gc3).size()
                .reindex(total_per_mod.index, fill_value=0)
            )
            at_rate = (at_count / total_per_mod * 100).fillna(0)
            at_df = pd.DataFrame({
                "Module": at_rate.index,
                "AT Found Rate (%)": at_rate.values,
                "Total Bugs": total_per_mod.values,
                "AT Found Bugs": at_count.values,
            }).sort_values("AT Found Rate (%)", ascending=False)

            fig_at = px.bar(
                at_df, x="Module", y="AT Found Rate (%)",
                color="AT Found Rate (%)",
                color_continuous_scale=["red", "orange", "green"],
                range_color=[0, 50],
                hover_data=["Total Bugs", "AT Found Bugs"],
                title="AT Found Rate by Module (Green ≥30%, Orange 10–29%, Red <10%)",
            )
            fig_at.update_layout(height=400, xaxis_tickangle=-30)
            st.plotly_chart(fig_at, width='stretch')

            blind_spots = at_df[at_df["AT Found Rate (%)"] == 0]
            st.subheader(f"🔴 Automation Blind Spots ({len(blind_spots)} modules with 0% AT coverage)")
            if len(blind_spots):
                show_dataframe(blind_spots, width='stretch', hide_index=True)
            else:
                st.success("All modules have AT coverage!")
    else:
        st.warning("No tag columns found — re-run parse_ecl_export.py")


# =====================================================================
# TAB 4 – Priority vs Severity Alignment
# =====================================================================

elif active_tab == "⚖️ P/S Alignment":
    st.header("⚖️ Priority vs Severity Alignment")

    with st.expander("📖 How to read this chart", expanded=True):
        st.markdown("""
**One-line summary:** Are QA and RD agreeing on what's important? QA picks **Severity** (how bad is the bug), RD picks **Priority** (how fast we'll fix it). When they disagree, this chart shows where.

**Why this matters:** A bug RD marks "No Matter" but QA marks Critical is either (a) a triage miss that could leak into release, or (b) a sign the team isn't aligned on user impact. Either way, you want to know.

**Chart layout:** RD Priority on the rows ↓, QA Severity on the columns →. Cell value = how many bugs sit in that combination.

| RD Priority (rows) ↓ \\ QA Severity (columns) → | **S1–S2 (Critical / Major)** | **S3–S4 (Normal / Minor)** |
|---|---|---|
| **Fix Now / Must Fix** | ✅ Aligned — both teams agree it's urgent | 🟡 **Inverse** — RD fast-tracked something QA rated minor; ask RD what they know that QA doesn't |
| **Better Fix / No Matter** | 🔴 **Mismatch** — QA flagged a critical bug, RD deprioritised it; **escalate** | ✅ Aligned — both teams agree it's low priority |
| **N/A (not yet triaged)** | 🔴 **Urgent Gap** — crash-level bug with no RD decision yet | ⚪ Expected — low-severity bugs often wait for triage |

The **diagonal cells** are healthy alignment. The **off-diagonal cells** are where the conversation needs to happen.

---
### 📊 The Three Mismatch Metrics (below the chart)

| Metric | Definition | What to do |
|--------|-----------|-----------|
| **🔴 Critical Mismatch** | S1 or S2 bugs with RD Priority 4 (No Matter) or N/A | Escalate to RD — a confirmed crash or major failure must not be deprioritised without written justification |
| **🟡 Inverse Mismatch** | S3 or S4 bugs RD marked Fix Now or Must Fix | Verify scope. RD may know about wider business impact that QA's rating missed. Either re-rate the severity up or recalibrate with RD |
| **⚪ Untriaged Critical** | S1 or S2 bugs where RD priority is still N/A | The most urgent gap — a critical bug with no triage decision is an uncontrolled release risk. Ping RD now |

**Critical Mismatch drill-down** auto-expands when mismatches exist — lists every offending bug with its ECL link so you can follow up immediately.

**Manager pitch:** *"This tab is the health check for QA↔RD alignment. Every Critical Mismatch is a triage decision that needs a meeting. Untriaged Critical is the worst signal — it means a crash-level bug is sitting in the queue with nobody owning the call."*

---
**Where this data comes from:**
- `priority_label` — mapped from the ECL `Priority` column by `parse_ecl_export.py` (1 = Fix Now → 5 = N/A).
- `severity_num` — mapped S1=1 (Critical) → S4=4 (Minor) at parse time.
- This tab uses the **current sidebar filter** (version, status, product) — change filters to see alignment for a specific release window.
""")

    if "priority_label" in df.columns and "severity_num" in df.columns:
        mm = df.pivot_table(
            index="priority_label", columns="severity_num",
            aggfunc="size", fill_value=0,
        )
        mm.columns = [
            f"S{int(c)}-{'Critical' if c==1 else 'Major' if c==2 else 'Normal' if c==3 else 'Minor'}"
            for c in mm.columns
        ]
        present_priorities = [p for p in PRIORITY_ORDER if p in mm.index]
        mm = mm.reindex(present_priorities)

        if mm.empty:
            st.warning("No bugs with matching priority/severity in current filters.")
        else:
            fig4 = px.imshow(
                mm, color_continuous_scale="Blues", aspect="auto", text_auto=True,
                labels={"x": "QA Severity", "y": "RD Priority", "color": "Bug Count"},
            )
            fig4.update_layout(height=350)
            st.plotly_chart(fig4, width='stretch')

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            high_sev = df["severity_num"].isin([1, 2])
            high_pri = df["priority_label"].isin(["1-Fix Now", "2-Must Fix"])
            low_pri  = df["priority_label"].isin(["4-No Matter", "5-N/A"])

            critical_mismatch  = df[high_sev & low_pri]
            inverse_mismatch   = df[~high_sev & high_pri]
            untriaged_critical = df[high_sev & (df["priority_label"] == "5-N/A")]

            col1.metric("🔴 Critical Mismatch",  f"{len(critical_mismatch):,}",
                        help="S1/S2 bugs with low priority or N/A — escalate to RD")
            col2.metric("🟡 Inverse Mismatch",   f"{len(inverse_mismatch):,}",
                        help="S3/S4 bugs with Fix Now/Must Fix — verify scope with RD")
            col3.metric("⚪ Untriaged Critical", f"{len(untriaged_critical):,}",
                        help="S1/S2 bugs where RD has not set priority (N/A)")

            if len(critical_mismatch) > 0:
                with st.expander(f"🔴 View Critical Mismatches ({len(critical_mismatch)} bugs)"):
                    render_bug_table(critical_mismatch)
    else:
        st.warning(
            "Need `priority_label` (or `Priority`) and `severity_num` columns. "
            "Re-run parse_ecl_export.py v2.3."
        )


# =====================================================================
# TAB 5 – Team Coverage
# =====================================================================

elif active_tab == "👥 Team Coverage":
    st.header("Tester × Module Coverage")

    with st.expander("📖 How to read this chart", expanded=False):
        st.markdown("""
**One-line summary:** Who on the QA team is exercising which product area? Rows = testers, columns = product category. Darker cell = more bugs that tester has filed there — a proxy for how much hands-on time they've spent in that area.

**Important caveat (read this before drawing conclusions):** This counts bugs *filed*, not bugs *tested*. A thorough tester who actively investigates will naturally file more bugs. A light cell can mean three different things:
1. The tester didn't cover that area (real coverage gap), **or**
2. The area was genuinely stable when they tested it, **or**
3. Another tester filed the duplicates first.

So treat this as a **rough experience signal**, not a precise coverage map. The most useful pattern is the *opposite*: rows that are dark across many columns = experienced generalists, columns that are dark across many rows = areas everyone touches.

---
### 🚨 Knowledge Silos (flagged below the chart)

A **silo** = a product category where only **one** tester has ever filed bugs. This is a **bus-factor risk**: if that person is unavailable (holiday, leave, reassignment), the entire area has *zero* team members with direct hands-on experience. In a release crunch, this is the kind of blind spot that lets a critical bug slip through.

**How to act on silos** (rough order of cost vs. value):
- **Pair the sole expert** with a second tester for 1–2 sprints of shadowed testing. Cheapest, fastest knowledge transfer.
- **Document test runbooks** for silo categories: what to test, known edge cases, recurring failure modes. The runbook preserves knowledge even if the expert leaves.
- **Cross-train during low-pressure sprints** — *not* mid-sprint before a release. Address bus-factor risk before it becomes urgent.
- **Prioritise silos that also appear in the P1 list (Risk Heatmap tab)** — high-risk *and* single-point-of-failure is the most dangerous combination on the dashboard.

**Manager pitch:** *"This isn't a performance scorecard for testers — it's a coverage map for the product. Light columns + a P1 badge = critical area without backup expertise."*

---
**Where this data comes from:** `Creator` and `module_category` are extracted by `parse_ecl_export.py` from the n8n / ECL export. The chart uses the current sidebar filter (version, status, product) — change filters to compare coverage across different release windows or products.
""")

    if "Creator" in df.columns and "module_category" in df.columns:
        cv = df.pivot_table(
            index="Creator", columns="module_category",
            aggfunc="size", fill_value=0,
        )
        fig5 = px.imshow(cv, color_continuous_scale="Greens", aspect="auto", text_auto=True)
        fig5.update_layout(height=max(400, len(cv) * 35))
        st.plotly_chart(fig5, width='stretch')

        st.subheader("Knowledge Silos")
        silos = [cat for cat in cv.columns if (cv[cat] > 0).sum() <= 1]
        if silos:
            for s in silos:
                st.warning(f"⚠️ **{s}**: only 1 tester")
        else:
            st.success("No knowledge silos detected.")
    else:
        st.warning("Need `Creator` and `module_category` columns.")


# =====================================================================
# TAB 6 – KPI Dashboard
# =====================================================================

elif active_tab == "📊 KPI Dashboard":
    st.header("Quality KPIs")

    with st.expander("📖 What these KPIs mean", expanded=False):
        st.markdown("""
**One-line summary:** The single page to glance at before a release. Each metric answers one yes/no question — "are we shipping this?".

| KPI | What it measures | Why it matters / target |
|-----|-----------------|----------------|
| **Total Bugs** | All bugs matching the current sidebar filters (version, status, product) | Baseline — sets the scale for everything else on this page |
| **Critical Bugs (S1)** | Count of Severity-1 (crash / data loss) bugs in the current filter | **Release gate.** Any S1 in the release candidate is a potential showstopper — zero S1s is the minimum bar |
| **Avg Days to Close** | Mean calendar days from `Create Date` → `Closed Date` (open bugs excluded) | Measures RD fix velocity. **Rising trend = backlog is growing faster than it's being burned down**. Compare across releases for trend |
| **Regression Bug Rate** | % of bugs tagged `[Side Effect]` | Regressions = features that previously worked and broke. **Above 20% = insufficient regression coverage** for the rate of change |
| **Active Bugs** | Open + In-Progress in the current filter | Live unresolved risk. The number you actually have to act on |
| **Inactive Bugs** | Closed + NAB + Won't Fix + duplicates | Resolved or dismissed. Counted separately because closed/fixed bugs **decay** in the prediction model (still informative, but not live risk) |
| **P1 Modules** | Modules with I×P×D risk score > 90 | **Test every version, no exceptions.** This is the must-cover list |
| **P2 Modules** | Modules with I×P×D risk score 70–90 | Test every version alongside P1 |
| **Avg Risk Score** | Mean I×P×D across all modules in the current filter | Health barometer. Rising over consecutive releases = cumulative risk growing |

---
### 📈 Weekly Bug Trend

Counts bugs **created per calendar week** (resampling `Create Date` into 7-day buckets).

**What healthy looks like vs. what to escalate:**

| Pattern | Stage of cycle | Read it as |
|---|---|---|
| **Declining** | Late cycle | ✅ Healthy — RD closing faster than QA is finding. Good sign before RC sign-off |
| **Flat** | Mid cycle | ✅ Steady state during active development |
| **Spike** | Mid cycle | ⚠️ Often a large merge, SDK update, or new feature drop. Cross-reference Tab 2 (Version Timeline) to find which module spiked |
| **Spike** | Late cycle | 🚨 Regression testing uncovered something deep. Consider delaying release until the spike resolves |

### 🥧 Severity Distribution

Pie chart of currently filtered bugs by severity. **Healthy end-of-cycle snapshot = mostly S3/S4** (the dangerous bugs have been resolved). **A large S1/S2 slice at release time = immediate RD escalation.**

**Manager pitch:** *"This tab is the four-second pre-release glance. If S1 isn't zero, if regression rate is above 20%, or if Avg Days to Close is climbing release-over-release, we have a release-readiness conversation. Everything else here is context."*

---
**Where these numbers come from:**
- Bug counts, severity, weekly trend: `ecl_parsed.csv` from `parse_ecl_export.py` / `fetch_from_n8n.py`.
- Active/Inactive split: `status_active` boolean from the parser. Inactive = `Close`, `NAB`, `Won't Fix`, `Need More Info`, `Not Reproducible`, `New Feature`, `External Issue`, `HQQA Close`, `FAE Close`.
- P1/P2 module counts and Avg Risk Score: `risk_register_scored.csv` from `ai_risk_scorer.py`.
- Regression Bug Rate: mean of `tag_side_effect` parsed from `[Side Effect]` tags.
""")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bugs (filtered)", f"{len(df):,}")
    c2.metric("Critical Bugs (S1)", f"{int((df['severity_num'] == 1).sum()):,}")

    if "days_to_close" in df.columns:
        ac = df["days_to_close"].dropna().mean()
        c3.metric("Avg Days to Close", f"{ac:.1f}" if not np.isnan(ac) else "N/A")
    else:
        c3.metric("Avg Days to Close", "N/A")

    se2 = [c for c in df.columns if "side_effect" in c.lower() and c.startswith("tag_")]
    c4.metric("Regression Bug Rate", f"{df[se2[0]].mean() * 100:.1f}%" if se2 else "N/A")

    if "status_active" in df.columns:
        st.markdown("---")
        sa1, sa2 = st.columns(2)
        sa1.metric("Active Bugs (in filter)", f"{df['status_active'].sum():,}")
        sa2.metric("Inactive Bugs (in filter)", f"{(~df['status_active']).sum():,}",
                   help="Close, NAB, Won't Fix, etc.")

    if risk_available:
        st.markdown("---")
        cr1, cr2, cr3 = st.columns(3)
        q_counts = df.groupby("quadrant").size()
        cr1.metric("P1 Modules (Critical)",  int(q_counts.get("P1 - Critical", 0)))
        cr2.metric("P2 Modules (High)", int(q_counts.get("P2 - High", 0)))
        cr3.metric("Avg Risk Score", f"{df['risk_score_final'].mean():.1f}")

    st.markdown("---")
    if "Create Date" in df.columns:
        st.subheader("Weekly Bug Trend")
        vd = df[df["Create Date"].notna()].copy()
        if len(vd):
            wk = vd.set_index("Create Date").resample("W").size().reset_index()
            wk.columns = ["Week", "Count"]
            st.plotly_chart(
                px.line(wk, x="Week", y="Count", markers=True).update_layout(height=300),
                width='stretch',
            )

    st.subheader("Severity Distribution")
    sd = df["severity_num"].value_counts().sort_index()
    sd.index = sd.index.map({1: "S1-Critical", 2: "S2-Major", 3: "S3-Normal", 4: "S4-Minor"})
    st.plotly_chart(
        px.pie(
            values=sd.values, names=sd.index,
            color_discrete_sequence=["#f44336", "#f57c00", "#fbc02d", "#4caf50"],
        ).update_layout(height=350),
        width='stretch',
    )


# =====================================================================
# TAB 7 – Risk Heatmap  (always left/right [6,4] split)
# =====================================================================

elif active_tab == "🔥 Risk Heatmap":

    if not risk_available:
        st.warning("""
**Risk data not loaded.** To unlock this tab:

1. Run: `python scripts/ai_risk_scorer.py data/risk_register.csv data/risk_register_scored.csv --provider heuristic`
2. In the **sidebar → Step 2**, upload `data/risk_register_scored.csv`
""")
        st.stop()

    # Build aggregation
    tm_agg = df.groupby(["module_category", "parsed_module"]).agg(
        bug_count=("parsed_module", "size"),
        sev_weight=("severity_weight", "sum"),
        critical_count=("severity_num", lambda x: (x == 1).sum()),
        risk_score=("risk_score_final", "mean"),
    ).reset_index()
    tm_agg = tm_agg[tm_agg["parsed_module"].notna() & (tm_agg["bug_count"] > 0)]
    tm_agg["quadrant"] = tm_agg["parsed_module"].map(
        df.groupby("parsed_module")["quadrant"].first().to_dict()
    ).fillna("P4 - Low")

    if risk_df is not None:
        for col, default in [("impact_score", 3), ("detectability_score", 3),
                             ("probability_score_auto", 3)]:
            if col in risk_df.columns:
                tm_agg[col] = tm_agg["parsed_module"].map(
                    risk_df.set_index("parsed_module")[col].to_dict()
                ).fillna(default).astype(int)

    all_modules    = sorted(tm_agg["parsed_module"].dropna().unique().tolist())
    all_categories = sorted(tm_agg["module_category"].dropna().unique().tolist())

    with st.expander("📖 How the Risk Score works — full pipeline explained", expanded=False):
        st.markdown("""
**One-line summary:** Every module on the treemap has a single number — the **Risk Score (0–125)** — that combines three things into one priority signal: *how much it hurts when it breaks*, *how often it breaks historically*, and *how easily we'd catch the bug before release*. This is a standard FMEA approach (Failure Mode and Effects Analysis) borrowed from manufacturing QA.

**The formula in one line:**
> **Risk Score = Impact × Probability × Detectability** (each scored 1–5, max = 125)

| Risk Score | Priority | What it means for testing |
|---|---|---|
| **> 90** | 🔴 **P1 — Critical** | Test every version, no exceptions |
| **70–90** | 🟠 **P2 — High** | Test every sprint / major version |
| **50–69** | 🟡 **P3 — Medium** | Test every release candidate |
| **< 50** | 🟢 **P4 — Low** | Full release cycle pass only |

**Manager pitch:** *"Bug counts alone are misleading — a module with 50 cosmetic bugs is less risky than a module with 2 hidden crashes that no automated test would catch. The Risk Score combines all three dimensions into one number so we can rank modules consistently."*

---

### How the score is computed (3-stage pipeline)

The risk scores in this tab are produced before the dashboard launches. Each stage is a script that writes a CSV consumed by the next.

### Step 1 — Parse bugs → `ecl_parsed.csv`
**Script:** `parse_ecl_export.py` (input from `fetch_from_n8n.py`)

Reads the raw n8n / ECL bug export and extracts structured fields from each bug's Short Description.
Example: `PDR-I 16.2.5 - [EDF][UX] AI Storytelling: subtitle misplaced`

Key outputs per bug row:
- `parsed_module` — normalised module name (typo aliases resolved, e.g. *HQ Auido Denoise* → *HQ Audio Denoise*)
- `module_category` — one of 20 top-level categories (with flat overrides + partial matching)
- `severity_num` / `severity_weight` — S1=10 pts, S2=5 pts, S3=2 pts, S4=1 pt
- `tag_side_effect`, `tag_at_found`, `tag_edf`, `tag_ux` etc. — boolean columns from `[TAG]` prefixes
- `days_to_close`, `builds_to_fix`, `repro_rate` — computed from ECL date and version fields
- `status_active` / `status_weight` — open=1.0, closed=0.5, invalid (NAB / Won't Fix / etc.)=0.0. Lets the prediction model treat fixed bugs as a faded signal instead of pretending they never happened

---

### Step 2 — Aggregate per-module metrics → `risk_register.csv`
**Script:** `compute_risk_scores.py`

Groups all bugs by module and computes:

| Column | How it is calculated |
|--------|---------------------|
| `total_bugs` | Count of all bugs for this module |
| `severity_weighted_total` | Sum of per-bug weights (S1x10 + S2x5 + S3x2 + S4x1) |
| `critical_count` / `major_count` | Count of S1 / S2 severity bugs |
| `regression_rate` | side_effect_count / total_bugs — fraction that are regressions |
| `automation_catch_rate` | at_found_count / total_bugs — fraction caught by automated tests |
| `avg_repro_rate` | Mean reproduce probability across all bugs in this module |
| `avg_days_to_close` | Mean calendar days from bug creation to Close status |
| `unique_reporters` | Number of distinct testers who have filed bugs here |
| **`probability_score_auto`** | **Quintile rank of total_bugs across all modules, scored 1–5.** Top 20% most bug-prone = 5, bottom 20% = 1. This is the P in I x P x D. |

Also produces one `risk_register_<version>.csv` per ECL version for per-version analysis.

---

### Step 3 — Score Impact and Detectability → `risk_register_scored.csv`
**Script:** `ai_risk_scorer.py`   Provider: heuristic / Ollama LLM / OpenAI

#### I — Impact (1–5): How severe is it when this module breaks?

Assigned from a domain knowledge table built by the QA team:

| Score | Example modules | Rationale |
|-------|----------------|-----------|
| 5 | Export, AI Storytelling, Auto Edit, Project, Produced Video | Core output — failure = complete feature loss for user |
| 4 | Image/Text to Video, AI Music Generator, Auto Captions, Camera, Trim | Key features that directly affect the deliverable |
| 3 | Most mid-tier features | Notable impact but a workaround exists |
| 2 | Trending, peripheral UI | Low user-visible impact |
| 1 | Settings, Preferences, Tutorials, Credit pages | Cosmetic or internal only |

**Auto-boost rules applied on top of the table:**
- `critical_count >= 10` → Impact +1 (capped at 5)
- `critical_count >= 5` and Impact < 4 → Impact raised to 4

With **Ollama or OpenAI**, the LLM receives module name, category, bug counts, and regression
rate, reasons about business impact, and returns a score 1–5 with written justification.

#### P — Probability (1–5): How likely is this module to have bugs in the next version?

This is `probability_score_auto` from Step 2 — the quintile rank of historical bug density.
It is **never overridden by AI**; it is always purely data-driven from ECL history.

#### D — Detectability (1–5): How hard is it to catch bugs before release?

Heuristic calculation (higher = harder to catch = more dangerous to miss):

| Condition | Effect on D |
|-----------|-------------|
| Base value | 3 |
| Automation catch rate > 10% | -1 (AT is finding these bugs) |
| Automation catch rate > 30% | -1 again (strong AT coverage) |
| Automation catch rate = 0% | +1 (no automated safety net at all) |
| Avg reproduce rate < 50% | +1 (intermittent bugs are hard to confirm) |
| Regression rate > 15% | +1 (side effects are unpredictable) |
| Final value | clamped to [1, 5] |

With Ollama or OpenAI, the LLM reasons about module complexity and test coverage instead.

---

### Final formula

```
Risk Score = I x P x D        (maximum = 5 x 5 x 5 = 125)
```

| Priority | Score threshold | Testing cadence |
|----------|----------------|-----------------|
| P1 — Critical  | > 90  | Every version, every sprint |
| P2 — High      | 70-90 | Every sprint / major version |
| P3 — Medium    | 50-69 | Every release candidate |
| P4 — Low       | < 50  | Full release cycle only |

---

### What each visual element shows

| Element | Data source | What it represents |
|---------|-------------|-------------------|
| **Block size** | `ecl_parsed.csv` | Bug count for this module |
| **Colour — Risk Score** | `risk_register_scored.csv` | Final I×P×D value |
| **Colour — Priority** | `risk_register_scored.csv` | P1–P4 assignment |
| **Colour — Critical Count** | `ecl_parsed.csv` | Number of S1 (crash-level) bugs |
| **Colour — Severity Weight** | `ecl_parsed.csv` | Weighted bug sum S1×10 + S2×5 + S3×2 + S4×1 |
| **Right detail panel** | Both files joined | Total bugs, critical count, risk score, and full bug list for the selected module. Before clicking a block, shows the top 5 modules by bug count as a quick reference. |
| **🌞 Sunburst View** (expander below treemap) | Both files | Same data in a nested pie chart. Use it to compare the proportion of bugs across categories — the outer ring is modules, the inner ring is categories. |
| **📋 Category Summary** (expander below treemap) | Both files | Aggregate table: one row per category showing total modules, total bugs, severity weight, average risk score, and count of P1 modules. Sort by Avg Risk Score to quickly identify the riskiest category. |
| **🔴 P1 + P2 Modules bar chart** | `risk_register_scored.csv` | All P1 and P2 modules ranked by risk score, highest first. These are the mandatory test-every-version targets. Each bar is coloured by priority. |
| **Risk Score vs Probability scatter** | `risk_register_scored.csv` | I×P×D score on the Y axis vs historical defect probability rank (1–5) on the X axis. X values are slightly jittered (±0.35) so overlapping dots separate visually — hover for the true integer value. Threshold lines at 50, 70, 90 define the four risk zones. Look for modules in the top-right corner: both high-risk AND historically bug-prone — these are the highest-priority targets. |

> **Tip:** Heuristic scores are a strong starting point but should be reviewed with the team.
> Open `risk_register_scored.csv` in a spreadsheet, adjust any `impact_score` or
> `detectability_score` values the team disagrees with, then re-run
> `ai_risk_scorer.py --provider heuristic` to recompute final scores and priorities.
""")

    # Version context banner
    if len(sel_v) == 1:
        st.info(
            f"Showing version **{sel_v[0]}** only. "
            "Risk scores loaded from per-version file where available. "
            "Clear the filter or select multiple versions to see all-version combined scores."
        )
    elif sel_v:
        st.info(
            f"Showing {len(sel_v)} versions: {chr(44).join(str(v) for v in sel_v)}. "
            "Risk scores are from the combined all-versions file."
        )
    else:
        st.info("Showing all versions combined. Risk scores are from the combined all-versions file.")

    color_opt = st.radio(
        "Color by",
        ["Risk Score (LLM)", "Priority", "Critical Count", "Severity Weight"],
        horizontal=True, key="tmc",
    )

    dynamic_height = min(1100, max(700, len(all_modules) * 12))

    col_map, col_detail = st.columns([6, 4], gap="medium")

    # ── LEFT: Treemap ────────────────────────────────────────────────
    with col_map:
        current_sel = st.session_state["tm_selected_module"]
        if current_sel:
            st.caption(f"🔍 **{current_sel}** selected · click another module to switch")
        else:
            st.caption("Click a module block to view its bugs →")

        _ipd_cols = {k: True for k in ["impact_score", "probability_score_auto", "detectability_score"]
                     if k in tm_agg.columns}
        _ipd_labels = {
            "impact_score": "I (Impact)",
            "probability_score_auto": "P (Probability)",
            "detectability_score": "D (Detectability)",
        }

        if color_opt == "Risk Score (LLM)":
            fig_tm = px.treemap(
                tm_agg, path=["module_category", "parsed_module"],
                values="bug_count", color="risk_score",
                color_continuous_scale="YlOrRd",
                range_color=[0, 125],
                labels={"bug_count": "Bugs", "risk_score": "Risk Score",
                        "module_category": "Category", "parsed_module": "Module", **_ipd_labels},
                hover_data={"bug_count": True, "risk_score": ":.1f",
                            "critical_count": True, "quadrant": True, **_ipd_cols},
            )
        elif color_opt == "Priority":
            fig_tm = px.treemap(
                tm_agg, path=["module_category", "parsed_module"],
                values="bug_count", color="quadrant",
                color_discrete_map=QUADRANT_COLORS,
                labels={"bug_count": "Bugs", "quadrant": "Priority",
                        "module_category": "Category", "parsed_module": "Module", **_ipd_labels},
                hover_data={"bug_count": True, "risk_score": ":.1f",
                            "critical_count": True, **_ipd_cols},
            )
        elif color_opt == "Critical Count":
            fig_tm = px.treemap(
                tm_agg, path=["module_category", "parsed_module"],
                values="bug_count", color="critical_count",
                color_continuous_scale="Reds",
                labels={"bug_count": "Bugs", "critical_count": "Critical Bugs",
                        "module_category": "Category", "parsed_module": "Module", **_ipd_labels},
                hover_data={**_ipd_cols},
            )
        else:
            fig_tm = px.treemap(
                tm_agg, path=["module_category", "parsed_module"],
                values="bug_count", color="sev_weight",
                color_continuous_scale="YlOrRd",
                labels={"bug_count": "Bugs", "sev_weight": "Severity Weight",
                        "module_category": "Category", "parsed_module": "Module", **_ipd_labels},
                hover_data={**_ipd_cols},
            )

        fig_tm.update_layout(
            height=dynamic_height,
            margin=dict(t=50, l=5, r=5, b=5),
            transition_duration=0,
        )
        fig_tm.update_traces(
            root_color="rgba(0,0,0,0)",
            textinfo="label+value",
        )

        tm_event = st.plotly_chart(
            fig_tm,
            width='stretch',
            on_select="rerun",
            selection_mode=["points"],
            key="treemap_click",
        )

        if tm_event and hasattr(tm_event, "selection"):
            sel = tm_event.selection
            pts = sel.points if hasattr(sel, "points") else sel.get("points", [])
            if pts:
                pt = pts[0]
                clicked_label = (
                    pt.get("label") if isinstance(pt, dict)
                    else getattr(pt, "label", None)
                )
                # Use the Plotly `parent` field to distinguish a module node from a
                # category node. Category nodes have an empty parent (they sit at the
                # root level), while module nodes have a non-empty parent that is the
                # category name. This correctly handles the edge case where a module
                # shares the same name as its category (e.g. "Shortcut" module inside
                # the "Shortcut" category) — the old `not in all_categories` check
                # would silently drop the click in that situation.
                clicked_parent = (
                    pt.get("parent") if isinstance(pt, dict)
                    else getattr(pt, "parent", None)
                )
                is_module_node = bool(clicked_parent)  # empty string / None = category root
                if (clicked_label
                        and clicked_label in all_modules
                        and is_module_node):
                    cat_row = tm_agg[tm_agg["parsed_module"] == clicked_label]
                    st.session_state["tm_selected_module"] = clicked_label
                    st.session_state["tm_selected_category"] = (
                        cat_row["module_category"].values[0] if len(cat_row) else ""
                    )

    # ── RIGHT: Bug detail panel ──────────────────────────────────────
    with col_detail:
        st.markdown(
            f"<div style='max-height:{dynamic_height}px; overflow-y:auto; padding-right:6px;'>",
            unsafe_allow_html=True,
        )

        selected     = st.session_state["tm_selected_module"]
        selected_cat = st.session_state.get("tm_selected_category", "")

        if not selected:
            st.markdown("### 📋 Bug Details")
            st.info("👈 Click a module in the treemap to see its bugs here.")
            st.markdown("**Top 5 modules by bug count:**")
            top5 = tm_agg.nlargest(5, "bug_count")[
                ["parsed_module", "bug_count", "critical_count", "risk_score", "quadrant"]
            ].copy()
            top5.columns = ["Module", "Bugs", "Critical", "Risk Score", "Priority"]
            show_dataframe(top5, width='stretch', hide_index=True)

        else:
            risk_row    = tm_agg[tm_agg["parsed_module"] == selected]
            module_bugs = df[df["parsed_module"] == selected].copy()

            st.markdown(f"### 📋 {selected}")
            if selected_cat:
                st.caption(f"Category: **{selected_cat}**")

            mm1, mm2, mm3 = st.columns(3)
            mm1.metric("Total",    len(module_bugs))
            mm2.metric("Critical", int((module_bugs["severity_num"] == 1).sum()))
            mm3.metric(
                "Risk Score",
                f"{risk_row['risk_score'].values[0]:.1f}" if len(risk_row) else "N/A"
            )

            quadrant_val = risk_row["quadrant"].values[0] if len(risk_row) else "—"
            qcolor = {"P1 - Critical": "🔴", "P2 - High": "🟠", "P3 - Medium": "🟡", "P4 - Low": "🟢"}.get(quadrant_val, "⚪")
            st.markdown(f"Priority: {qcolor} **{quadrant_val}**")

            st.markdown("---")

            detail_sev = st.radio(
                "Severity filter",
                options=SEV_OPTIONS,
                horizontal=True,
                key="detail_sev_filter",
            )
            if detail_sev != "All":
                module_bugs = module_bugs[
                    module_bugs["severity_num"] == SEV_NUM_MAP[detail_sev]
                ]

            if len(module_bugs) == 0:
                st.info("No bugs match this filter.")
            else:
                render_bug_table(module_bugs)
                st.caption(f"{len(module_bugs)} bug(s) · severity: **{detail_sev}**")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Supporting charts below ───────────────────────────────────────
    st.markdown("---")
    with st.expander("🌞 Sunburst View"):
        if color_opt == "Priority":
            fig_sb = px.sunburst(
                tm_agg, path=["module_category", "parsed_module"],
                values="bug_count", color="quadrant",
                color_discrete_map=QUADRANT_COLORS,
            )
        else:
            fig_sb = px.sunburst(
                tm_agg, path=["module_category", "parsed_module"],
                values="bug_count", color="risk_score",
                color_continuous_scale="YlOrRd",
            )
        fig_sb.update_layout(height=600, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig_sb, width='stretch')

    with st.expander("📋 Category Summary"):
        cats = (
            tm_agg.groupby("module_category").agg(
                Modules=("parsed_module", "nunique"),
                Total_Bugs=("bug_count", "sum"),
                Severity_Weight=("sev_weight", "sum"),
                Avg_Risk=("risk_score", "mean"),
                P1_Modules=("quadrant", lambda x: (x == "P1 - Critical").sum()),
            )
            .sort_values("Avg_Risk", ascending=False).reset_index()
        )
        cats.columns = ["Category", "Modules", "Total Bugs",
                        "Severity Weight", "Avg Risk Score", "P1 Modules"]
        show_dataframe(cats, width='stretch', hide_index=True)

    st.subheader("🔴 P1 + P2 Modules — Test Every Build")
    q4 = tm_agg[tm_agg["quadrant"].isin(["P1 - Critical", "P2 - High"])].sort_values("risk_score", ascending=False)
    if len(q4):
        fig_q4 = px.bar(
            q4.head(30), x="parsed_module", y="risk_score", color="quadrant",
            color_discrete_map=QUADRANT_COLORS,
            labels={"parsed_module": "Module", "risk_score": "Risk Score",
                    "quadrant": "Priority"},
            title="P1 + P2 Modules by Risk Score",
        )
        fig_q4.update_layout(height=400, xaxis_tickangle=-30)
        st.plotly_chart(fig_q4, width='stretch')
    else:
        st.info("No P1 or P2 modules in current filter.")

    if risk_df is not None and "probability_score_auto" in risk_df.columns:
        st.subheader("Risk Score vs Probability")
        scatter_df = tm_agg.copy()
        prob_map = risk_df.set_index("parsed_module")["probability_score_auto"].to_dict()
        scatter_df["probability"] = scatter_df["parsed_module"].map(prob_map).fillna(2.5)

        for col, default in [("impact_score", 3), ("detectability_score", 3)]:
            if col in risk_df.columns:
                scatter_df[col] = scatter_df["parsed_module"].map(
                    risk_df.set_index("parsed_module")[col].to_dict()
                ).fillna(default)

        def assign_zone(score: float) -> str:
            if score > 90: return "Critical Risk (>90)"
            if score >= 70: return "High Risk (70-90)"
            if score >= 50: return "Medium Risk (50-69)"
            return "Low Risk (<50)"

        scatter_df["risk_zone"] = scatter_df["risk_score"].apply(assign_zone)

        # Add small random jitter to x (probability) so dots at the same
        # integer score separate from each other and can be hovered individually.
        rng = np.random.default_rng(42)
        scatter_df["probability_jittered"] = (
            scatter_df["probability"]
            + rng.uniform(-0.35, 0.35, size=len(scatter_df))
        )

        zone_colors = {
            "Critical Risk (>90)": "#D62728",
            "High Risk (70-90)":   "#FF7F0E",
            "Medium Risk (50-69)": "#BCBD22",
            "Low Risk (<50)":      "#2CA02C",
        }
        zone_order = ["Critical Risk (>90)", "High Risk (70-90)", "Medium Risk (50-69)", "Low Risk (<50)"]
        scatter_df["risk_zone"] = pd.Categorical(scatter_df["risk_zone"], categories=zone_order, ordered=True)

        fig_scatter = px.scatter(
            scatter_df,
            x="probability_jittered",
            y="risk_score",
            color="risk_zone",
            color_discrete_map=zone_colors,
            category_orders={"risk_zone": zone_order},
            size="bug_count",
            hover_name="parsed_module",
            hover_data={
                "probability_jittered": False,        # hide the jittered value
                "probability": True,                  # show the real integer
                "impact_score": True,
                "detectability_score": True,
                "risk_score": ":.1f",
                "bug_count": True,
                "risk_zone": False,
            },
            labels={
                "probability_jittered": "Probability Score (1–5, jittered for readability)",
                "risk_score": "Risk Score (I×P×D)",
                "bug_count": "Bug Count",
                "impact_score": "Impact (I)",
                "detectability_score": "Detectability (D)",
                "probability": "Probability (P)",
            },
            size_max=32,
            opacity=0.80,
        )
        # Overlay true integer tick labels on x-axis so jitter doesn't confuse
        fig_scatter.update_xaxes(
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["1", "2", "3", "4", "5"],
            range=[0.3, 5.7],
        )
        # Pin threshold labels to the left to keep the right side clear for the legend
        for y_val, label in [
            (50, "Medium threshold"),
            (70, "High threshold"),
            (90, "Critical threshold"),
        ]:
            fig_scatter.add_hline(
                y=y_val, line_dash="dash", line_color="gray", line_width=1,
                annotation_text=label,
                annotation_position="top left",
            )
        fig_scatter.update_layout(height=520, legend=dict(x=1.01, y=1, xanchor="left"))
        st.caption(
            "ℹ️ X-axis values are slightly jittered (±0.35) so dots at the same probability "
            "level separate visually. Hover any dot for the true score."
        )
        st.plotly_chart(fig_scatter, width='stretch')

# =====================================================================
# TAB 8 – Bug Clusters
# =====================================================================

elif active_tab == "🔬 Bug Clusters":
    st.header("🔬 Bug Clusters — What Kinds of Problems Exist?")

    with st.expander("📖 How to read this tab — full guide", expanded=False):
        st.markdown("""
## 🔬 Bug Clusters — Complete Guide

**One-line summary:** Instead of reading hundreds of individual bug titles, this tab groups them into **issue groups** by their language similarity. Each issue group is a *recurring type of complaint* — a pattern, not a single bug. The tab answers: *"What categories of problems keep coming back, and how serious are they?"*

**Why this matters for your manager:** A long bug list looks scary but isn't actionable. *"We have 12 different subtitle bugs across 3 modules"* is — that's one root cause, one fix conversation, one regression test plan.

**How issue groups are named:**
- Default (TF-IDF mode): the 2–3 keywords that appear most often together (e.g. `ai storytelling | subtitle | timing`).
- With Ollama (recommended): a plain-English label generated by an LLM (e.g. *"subtitle rendering delay"*). Labels use the dedicated `--label-model` (default `qwen3:4b`) — separate from the embedding model — and have a 3-attempt retry to avoid empty labels from quantised models.

---

### 📊 Headline Metrics (top row)

| Metric | What it tells you |
|--------|------------------|
| **Total Bugs Analysed** | All bugs in the loaded clustered file |
| **Distinct Issue Groups Found** | Number of named clusters (the noise/unclustered bucket is excluded) |
| **Bugs Grouped into Issue Groups** | Count and % of bugs that belong to a named cluster — higher % means the algorithm found strong patterns |
| **Uncategorised Bugs** | Bugs whose descriptions were too short or too unique to match any cluster — normal to have some |

---

### 🎨 Colour Coding

Each issue group bar in the overview chart is coloured by the **average severity** of the bugs it contains:

| Colour | Average severity | Meaning |
|--------|-----------------|---------|
| 🔴 Red | ≤ 1.5 | Mostly Critical bugs (crashes, data loss) |
| 🟠 Orange | 1.5 – 2.5 | Mostly Major functional issues |
| 🟡 Yellow | 2.5 – 3.5 | Mostly Normal bugs |
| 🟢 Green | > 3.5 | Mostly Minor / cosmetic issues |

---

### 📏 How to Read the Overview Chart

- **Bar length** = number of bugs in that issue group. Longer = more bugs share this pattern.
- **Colour** = how severe those bugs are on average (see above).
- A **short red bar** (few critical bugs on a single issue group) may be more urgent than a **long yellow bar** (many minor bugs on a popular issue group). Use both dimensions together.

---

### 🃏 Issue Group Detail Cards

Expand any card to see:
- **Bug count** — how many bugs belong to this issue group
- **Avg severity** — 1 (Critical) to 4 (Minor)
- **Share of clustered bugs** — this issue group's proportion of all grouped bugs
- **Velocity (recent vs prior 2 versions)** — the acceleration ratio; >1.5 = growing, <0.67 = declining
- **Recurrence rate** — fraction of recent bugs from modules that also appeared in the prior 2-version window (high = same modules keep re-introducing this bug type)
- **Modules affected** — which modules contribute bugs to this issue group
- **Sample bug descriptions** — up to 6 real ECL examples so you can judge the pattern yourself
- **Action line** — a plain-English recommendation based on severity, amplified if the velocity or recurrence signals are also firing:
  - 🚨 **Immediate attention** (avg sev ≤ 1.5) — crash-level or data-loss bugs. Escalate to RD.
  - ⚠️ **High priority** (avg sev ≤ 2.5) — major functional issues. Add to next sprint regression.
  - 📋 **Standard priority** (avg sev ≤ 3.5) — normal issues. Cover in release-candidate testing.
  - ✅ **Low priority** (avg sev > 3.5) — cosmetic or minor issues. Full release cycle.

---

### 🚨 Alert Banners (shown before the chart when thresholds are hit)

**🔺 Growing issue group alerts** fire when `cluster_velocity_ratio` ≥ 1.5.
> **Velocity ratio** = (bugs in last 2 versions) ÷ (bugs in prior 2 versions). A ratio of 1.5 means 50% more bugs than the preceding window — a strong signal that something changed recently.

**🔁 Fix-not-holding alerts** fire when `recurrence_rate` ≥ 0.6.
> **Recurrence rate** = fraction of recent bugs (last 2 versions) from modules that also contributed to this issue group in the prior 2-version window. 60%+ means the same modules keep re-introducing the same type of bug — a root-cause fix, not another patch, is needed.

---

### 📈 Cluster Velocity Chart

Ranks all issue groups by velocity ratio. The **red dashed threshold line at 1.5×** marks the growing boundary.
- Bars to the right at 1.5 or beyond need immediate investigative attention regardless of total bug count.
- Green bars (declining) indicate issue groups that are resolving — the fix is holding.

---

### 🔀 Module Cluster Entropy Chart

**Shannon entropy** measures how spread out a module's bugs are across different issue groups.
The formula: H = −Σ(pᵢ × log₂(pᵢ)) where pᵢ = the fraction of bugs belonging to issue group i.

| Entropy | Meaning | What to do |
|---------|---------|-----------|
| **< 1 (✅ Focused)** | Bugs concentrated in one issue group — single well-defined failure mode | One root-cause fix likely resolves the majority of bugs |
| **1–2 (🔶 Spreading)** | Bugs spread across a few issue groups | Investigate whether the issue groups share a common component or API |
| **> 2 (⚠️ Broad instability)** | Bugs across many issue groups — systemic module instability | Comprehensive regression pass needed; no single fix will adequately cover this |

Orange dashed line at 1.0 and red dashed line at 2.0 mark the thresholds on the chart.

---

### 🔬 Severity-Stratified Tabs (S1/S2 and S3/S4)

When clustering was run with `--stratify-severity`, two additional sub-views appear:
- **🔴 S1/S2 tier** — clusters built only from Critical and Major bugs. These show the most dangerous patterns without minor-bug noise diluting the keyword weighting.
- **🟡 S3/S4 tier** — clusters built only from Normal and Minor bugs. Useful for UX and polish issues independently.

Each tier has its own overview bar chart and detail cards with velocity and recurrence metrics.

---

### 🔍 How to Investigate a Pattern

1. Check alert banners first — growing or fix-not-holding issue groups are highest priority.
2. Find the largest red/orange bars — combine severity and count.
3. Open the card and check which modules are affected.
   - **1 module** → isolated bug area. Add it to the current sprint focus.
   - **3+ modules** → shared root cause likely. Investigate common infrastructure (API, SDK version, shared component).
4. Read the sample descriptions — do they describe the same underlying failure? If yes, this is systematic.
5. Cross-check with **Tab 7 (Risk Heatmap)** — if the affected modules are P1/P2, the issue group is high-priority regardless of bug count.

---

### 📋 Full Issue Group Table

The "Full issue group table (raw data)" expander at the bottom provides a sortable table with all numeric columns: velocity ratio, trend, recurrence rate. Export or copy to a spreadsheet for offline analysis.

---

### 🛠️ Where Does This Data Come From?

Run the clustering script, then point the sidebar to the output files:

```bash
# Default (TF-IDF, fast, no Ollama required):
python scripts/cluster_bugs.py data/ecl_parsed.csv data/ecl_parsed_clustered.csv

# With severity stratification (adds S1/S2 and S3/S4 sub-clusters):
python scripts/cluster_bugs.py data/ecl_parsed.csv data/ecl_parsed_clustered.csv \\
  --stratify-severity

# Ollama mode with separate label & embed models (recommended):
python scripts/cluster_bugs.py data/ecl_parsed.csv data/ecl_parsed_clustered.csv \\
  --provider ollama --label-model qwen3:4b --embed-model mxbai-embed-large \\
  --stratify-severity

# Re-label only (skip embedding — fast iteration on label quality):
python scripts/cluster_bugs.py data/ecl_parsed.csv data/ecl_parsed_clustered.csv --relabel
```

**Pipeline shortcuts** (`refresh_pipeline.sh`):
- Default uses `CLUSTER_LABEL_MODEL=qwen3:4b` (the same model as `OLLAMA_MODEL`; still overridable separately).
- `./refresh_pipeline.sh --force-relabel` re-runs labelling and forces predictions to rebuild afterward (so Tab 9 picks up the new labels).

**Output files** (auto-loaded from `data/products/<slug>/clusters/`):
- `ecl_parsed_clustered.csv` — every bug tagged with its `cluster_id` and `cluster_label`.
- `ecl_parsed_cluster_summary.csv` — per-cluster: count, modules, avg severity, **velocity (last 2 vs prior 2 versions)**, trend, recurrence.
- `ecl_parsed_module_entropy.csv` — per-module Shannon entropy (focused vs spreading vs broad).
- `ecl_parsed_cluster_summary_s12.csv` / `_s34.csv` — critical/major and normal/minor tiers (requires `--stratify-severity`).

**Recommended cadence:** re-run **every Friday** or after each new batch of versions is parsed. The pipeline does this automatically via the weekday/weekend schedule.
""")

    if cluster_df is None:
        st.warning(
            "**Cluster data not loaded.** "
            "Run `python scripts/cluster_bugs.py data/ecl_parsed.csv` then set the path in **Sidebar → Step 3**."
        )
        st.stop()

    # ── Guard: need cluster columns ──────────────────────────────────────
    if "cluster_id" not in cluster_df.columns or "cluster_label" not in cluster_df.columns:
        st.error("Cluster file is missing `cluster_id` / `cluster_label` columns. Re-run cluster_bugs.py.")
        st.stop()

    # ── Headline metrics ─────────────────────────────────────────────────
    n_total    = len(cluster_df)
    n_clustered = (cluster_df["cluster_id"] != -1).sum()
    n_clusters  = cluster_df[cluster_df["cluster_id"] != -1]["cluster_id"].nunique()
    n_unclustered = n_total - n_clustered

    hm1, hm2, hm3, hm4 = st.columns(4)
    hm1.metric("Total Bugs Analysed", f"{n_total:,}")
    hm2.metric("Distinct Issue Groups Found", f"{n_clusters}")
    hm3.metric("Bugs Grouped into Issue Groups", f"{n_clustered:,}",
               help=f"{n_clustered/n_total*100:.0f}% of all bugs belong to a named issue group")
    hm4.metric("Uncategorised Bugs", f"{n_unclustered:,}",
               help="Bugs whose descriptions were too short or too unique to cluster")

    st.markdown("---")
    if cluster_sum_df is not None and not cluster_sum_df.empty:
        # Authoritative path: use the pre-computed summary CSV from cluster_bugs.py.
        summary = cluster_sum_df.copy().reset_index(drop=True)
        _velocity_source = "summary_csv"
    else:
        # Fallback: build basic summary from raw clustered bugs (count/modules/avg_sev only).
        # Velocity/trend/recurrence come from cluster_bugs.py's summary CSV — without it,
        # attempt inline computation only when Build# is present in the clustered data.
        clustered_only = cluster_df[cluster_df["cluster_id"] != -1]
        summary = (
            clustered_only
            .groupby(["cluster_id", "cluster_label"])
            .agg(
                count=("cluster_id", "size"),
                modules=("parsed_module",
                         lambda x: ", ".join(sorted(x.dropna().unique())[:5])),
                avg_sev=("severity_num", "mean"),
            )
            .sort_values("count", ascending=False)
            .reset_index()
        )
        if "Build#" in cluster_df.columns:
            # Build# available — compute velocity inline using the same logic as cluster_bugs.py.
            _build_num = pd.to_numeric(cluster_df["Build#"], errors="coerce")
            _max_build = _build_num.max()
            _vel_rows = []
            for _cid in summary["cluster_id"].unique():
                _cmask = (cluster_df["cluster_id"] == _cid)
                _cb = _build_num[_cmask]
                _recent = int((_cb >= _max_build - 2).sum())
                _prior  = int(((_cb >= _max_build - 5) & (_cb < _max_build - 2)).sum())
                _ratio  = round(_recent / max(_prior, 1), 2)
                _trend  = "growing" if _ratio >= 1.5 else "declining" if _ratio <= 0.67 else "stable"
                _r_mods = set(cluster_df.loc[_cmask & (_build_num >= _max_build - 2), "parsed_module"].dropna())
                _p_mods = set(cluster_df.loc[_cmask & (_build_num >= _max_build - 5) & (_build_num < _max_build - 2), "parsed_module"].dropna())
                _recur  = round(len(_r_mods & _p_mods) / max(len(_r_mods), 1), 3)
                _vel_rows.append({"cluster_id": _cid, "cluster_velocity_ratio": _ratio,
                                  "cluster_trend": _trend, "recurrence_rate": _recur})
            summary = summary.merge(pd.DataFrame(_vel_rows), on="cluster_id", how="left")
            _velocity_source = "inline_buildhash"
        else:
            # No Build# and no summary CSV — velocity data is genuinely unavailable.
            # Do NOT fill neutral defaults (1.0/stable/0.0); leave columns absent so the
            # display code can distinguish "unavailable" from "truly stable".
            _velocity_source = "unavailable"

    if summary.empty:
        st.info("No clusters found in the loaded data.")
        st.stop()

    # Ensure core display columns exist; do NOT add velocity defaults here.
    for col, default in [("count", 0), ("avg_sev", 3.0), ("modules", "")]:
        if col not in summary.columns:
            summary[col] = default

    # Determine whether velocity/recurrence data is usable.
    _velocity_available = "cluster_velocity_ratio" in summary.columns and _velocity_source != "unavailable"

    if _velocity_source == "unavailable":
        st.info(
            "ℹ️ **Velocity and recurrence metrics are unavailable.** "
            "The cluster summary CSV (`ecl_parsed_cluster_summary.csv`) from `cluster_bugs.py` "
            "was not loaded, and the clustered data has no `Build#` column for inline computation. "
            "Load the summary CSV in **Sidebar → Step 3** to enable growing issue-group alerts, "
            "velocity chart, and recurrence rates."
        )
    elif _velocity_source == "inline_buildhash":
        st.caption(
            "ℹ️ Velocity computed inline from `Build#` — load `ecl_parsed_cluster_summary.csv` "
            "in Sidebar → Step 3 for the authoritative values from `cluster_bugs.py`."
        )

    summary["avg_sev"] = pd.to_numeric(summary["avg_sev"], errors="coerce").fillna(3.0)
    summary["count"]   = pd.to_numeric(summary["count"],   errors="coerce").fillna(0).astype(int)

    # Human-readable severity label for the bar chart colour
    def _sev_band(v):
        if v <= 1.5: return "🔴 Mostly Critical"
        if v <= 2.5: return "🟠 Mostly Major"
        if v <= 3.5: return "🟡 Mostly Normal"
        return "🟢 Mostly Minor"

    summary["severity_band"] = summary["avg_sev"].apply(_sev_band)

    SEV_BAND_COLORS = {
        "🔴 Mostly Critical": "#ef4444",
        "🟠 Mostly Major":    "#f97316",
        "🟡 Mostly Normal":   "#eab308",
        "🟢 Mostly Minor":    "#22c55e",
    }

    # Truncate very long cluster labels for display
    summary["label_short"] = summary["cluster_label"].apply(
        lambda s: s[:45] + "…" if isinstance(s, str) and len(s) > 45 else s
    )

    # ── Growing-cluster alert banners ────────────────────────────────────
    # Only fire alerts when velocity data comes from an authoritative source
    # (cluster_bugs.py summary CSV or inline Build# computation).
    # Never fire on neutral defaults — that would produce false-calm signals.
    if _velocity_available:
        summary["cluster_velocity_ratio"] = pd.to_numeric(
            summary["cluster_velocity_ratio"], errors="coerce")
        _growing = summary[
            (summary["cluster_trend"] == "growing") &
            (summary["cluster_velocity_ratio"] >= 1.5)
        ].sort_values("cluster_velocity_ratio", ascending=False)
        _high_recur = summary[summary["recurrence_rate"] >= 0.6].sort_values(
            "recurrence_rate", ascending=False)

        if not _growing.empty:
            for _, _gr in _growing.head(3).iterrows():
                st.warning(
                    f"🔺 **Growing issue group:** _{_gr['label_short']}_ — "
                    f"{_gr['cluster_velocity_ratio']:.1f}× increase in recent versions "
                    f"({int(_gr['count'])} total bugs, avg sev {_gr['avg_sev']:.1f}). "
                    f"Modules: {_gr['modules']}"
                )
        if not _high_recur.empty:
            for _, _rr in _high_recur.head(2).iterrows():
                st.error(
                    f"🔁 **Fix not holding:** _{_rr['label_short']}_ — "
                    f"{_rr['recurrence_rate'] * 100:.0f}% recurrence rate. "
                    f"The same modules keep producing this type of bug. Escalate root-cause review."
                )

    # ── Overview bar chart ───────────────────────────────────────────────
    st.subheader("📊 Issue Group Overview — How Many Bugs per Issue Group?")
    st.caption(
        "Each bar is a recurring issue group. Colour shows average severity. "
        "Longer bar = more bugs share that issue group."
    )

    top_n_bar = st.slider("Show top N issue groups", min_value=5, max_value=min(30, len(summary)),
                          value=min(15, len(summary)), key="cluster_bar_n")
    bar_data = summary.head(top_n_bar).copy()

    fig_cl = px.bar(
        bar_data,
        x="count",
        y="label_short",
        orientation="h",
        color="severity_band",
        color_discrete_map=SEV_BAND_COLORS,
        hover_data={"count": True, "avg_sev": ":.2f", "modules": True,
                    "severity_band": False, "label_short": False},
        labels={"count": "Number of Bugs", "label_short": "Issue Group (keywords)",
                "avg_sev": "Avg Severity (1=Critical, 4=Minor)",
                "modules": "Modules affected"},
        text="count",
    )
    fig_cl.update_layout(
        height=max(380, top_n_bar * 30),
        yaxis={"categoryorder": "total ascending"},
        showlegend=True,
        legend_title_text="Severity",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_cl.update_traces(textposition="outside")
    st.plotly_chart(fig_cl, width='stretch')

    # ── Per-cluster detail cards ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("🃏 Issue Group Detail Cards")
    st.caption(
        "Each card shows one recurring issue group. "
        "Expand a card to see which modules are affected and example bugs."
    )

    for _, row in summary.head(top_n_bar).iterrows():
        cid    = row.get("cluster_id", -1)
        label  = str(row.get("cluster_label", "Unknown"))
        count  = int(row.get("count", 0))
        avg_sv = float(row.get("avg_sev", 3.0))
        mods   = str(row.get("modules", ""))
        sband  = row.get("severity_band", "🟡 Mostly Normal")
        rank   = summary.index[summary["cluster_label"] == label].tolist()
        rank_n = (rank[0] + 1) if rank else "?"

        # Pre-compute velocity/trend/recurrence so they can appear in the header.
        # Use None when the column is absent — never substitute neutral defaults.
        _raw_vel   = row.get("cluster_velocity_ratio")
        _raw_trend = row.get("cluster_trend")
        _raw_recur = row.get("recurrence_rate")
        vel_val    = float(_raw_vel)   if _raw_vel   is not None and not (isinstance(_raw_vel,   float) and np.isnan(_raw_vel))   else None
        trend_val  = str(_raw_trend)   if _raw_trend is not None else None
        recur_val  = float(_raw_recur) if _raw_recur is not None and not (isinstance(_raw_recur, float) and np.isnan(_raw_recur)) else None
        trend_icon = {"growing": "🔺", "declining": "✅", "stable": "➡️"}.get(trend_val or "", "➡️")
        sev_icon   = {"🔴 Mostly Critical": "🔴", "🟠 Mostly Major": "🟠",
                      "🟡 Mostly Normal": "🟡", "🟢 Mostly Minor": "🟢"}.get(sband, "⚪")

        if vel_val is not None:
            card_title = (
                f"{sev_icon} Issue Group #{rank_n} · **{count} bugs** "
                f"· {trend_icon} {trend_val} ({vel_val:.1f}×) · _{label}_"
            )
        else:
            card_title = f"{sev_icon} Issue Group #{rank_n} · **{count} bugs** · _{label}_"

        with st.expander(card_title, expanded=(rank_n == 1)):
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Bugs in issue group", count)
            cc2.metric("Avg severity", f"{avg_sv:.1f}",
                       help="1=Critical  2=Major  3=Normal  4=Minor")
            pct = f"{count / n_clustered * 100:.1f}%" if n_clustered else "—"
            cc3.metric("Share of clustered bugs", pct)

            # Velocity / trend / recurrence row — only shown when data is available.
            if vel_val is not None:
                vc1, vc2, vc3 = st.columns(3)
                vel_delta_color = "inverse" if trend_val == "declining" else ("off" if trend_val == "stable" else "normal")
                vc1.metric("Velocity (recent vs prior 2 versions)", f"{vel_val:.2f}×",
                           delta=f"{trend_icon} {trend_val.capitalize()}",
                           delta_color=vel_delta_color,
                           help="Ratio of bugs in the last 2 versions vs the 2 versions before that. Above 1.5 = growing.")
                _recur_display = f"{recur_val * 100:.0f}%" if recur_val is not None else "N/A"
                vc2.metric("Recurrence rate", _recur_display,
                           help="Fraction of recent bugs from modules that also contributed to this issue group in the prior window. High = fix not holding.")
                if recur_val is not None:
                    recur_label = "⚠️ Fix not holding" if recur_val > 0.5 else ("🔶 Moderate" if recur_val > 0.25 else "✅ Low")
                else:
                    recur_label = "N/A"
                vc3.metric("Recurrence risk", recur_label)

            if mods:
                st.markdown(f"**Modules affected:** {mods}")

            # Sample bugs from this cluster
            if cluster_df is not None and "cluster_id" in cluster_df.columns:
                cl_bugs = cluster_df[cluster_df["cluster_id"] == cid]
                if "parsed_description" in cl_bugs.columns:
                    samples = (
                        cl_bugs[cl_bugs["parsed_description"].notna()]
                        [["parsed_module", "parsed_description", "severity_label"]]
                        .head(6)
                        .rename(columns={
                            "parsed_module": "Module",
                            "parsed_description": "Bug description",
                            "severity_label": "Severity",
                        })
                    )
                    if not samples.empty:
                        st.markdown("**Sample bugs in this issue group:**")
                        show_dataframe(samples, hide_index=True, width='stretch')

            # Plain-English action — severity-based baseline
            if avg_sv <= 1.5:
                action = "🚨 **Immediate attention** — crash-level or data-loss bugs. Escalate to RD."
            elif avg_sv <= 2.5:
                action = "⚠️ **High priority** — major functionality issues. Add to next sprint regression."
            elif avg_sv <= 3.5:
                action = "📋 **Standard priority** — normal-severity issues. Cover in release-candidate testing."
            else:
                action = "✅ **Low priority** — mostly cosmetic or minor issues. Full release cycle."
            # Amplify when velocity or recurrence signals override the severity baseline.
            # Only apply when values come from an authoritative source (not absent).
            if vel_val is not None and trend_val == "growing" and vel_val >= 1.5:
                action += f"  🔺 **Growing fast ({vel_val:.1f}×)** — prioritise this issue group in the next version regardless of total count."
            if recur_val is not None and recur_val >= 0.5:
                action += "  🔁 **High recurrence** — same modules keep appearing; underlying root cause may not be resolved."
            st.info(action)

    # ── Raw summary table ────────────────────────────────────────────────
    with st.expander("📋 Full issue group table (raw data)"):
        display_cols_sum = [c for c in ["cluster_label", "count", "avg_sev", "modules",
                                         "severity_band", "cluster_velocity_ratio",
                                         "cluster_trend", "recurrence_rate"]
                            if c in summary.columns]
        disp_names = {
            "cluster_label": "Issue Group keywords", "count": "Bug count",
            "avg_sev": "Avg severity (1–4)", "modules": "Modules affected",
            "severity_band": "Severity band",
            "cluster_velocity_ratio": "Velocity ratio",
            "cluster_trend": "Trend",
            "recurrence_rate": "Recurrence rate",
        }
        display_sum = summary[display_cols_sum].rename(columns=disp_names)
        show_dataframe(display_sum, hide_index=True, width='stretch')

    # ── NEW v3.0 — Cluster velocity chart ───────────────────────────────
    if _velocity_available and "cluster_velocity_ratio" in summary.columns and "cluster_trend" in summary.columns:
        st.markdown("---")
        st.subheader("📈 Cluster Velocity — Which Issue Groups Are Growing?")
        st.caption(
            "Velocity ratio compares bug count in the most recent 2 versions vs the prior 2 versions. "
            "**Above 1.5** = issue group is growing (🔺 alert). **Below 0.67** = issue group is declining (✅ improving). "
            "Focus testing effort on growing issue groups."
        )

        vel_df = summary.copy()
        vel_df["cluster_velocity_ratio"] = pd.to_numeric(
            vel_df["cluster_velocity_ratio"], errors="coerce")
        # Drop clusters with undefined velocity (insufficient prior-window history)
        # so the chart doesn't mislead by showing them as "stable 1.0×".
        vel_undefined_ct = int(vel_df["cluster_velocity_ratio"].isna().sum())
        vel_df = vel_df.dropna(subset=["cluster_velocity_ratio"])
        if vel_df.empty:
            st.info("No cluster has enough prior-window history to compute a velocity ratio yet. "
                    "Re-run `cluster_bugs.py` after a few more versions of data are ingested.")
            _skip_velocity_chart = True
        else:
            _skip_velocity_chart = False
        # Show all clusters sorted by velocity descending — most urgent at top
        vel_df = vel_df.sort_values("cluster_velocity_ratio", ascending=False).head(20)
        if vel_undefined_ct and not _skip_velocity_chart:
            st.caption(f"_{vel_undefined_ct} cluster(s) hidden — not enough "
                       f"prior-window history to compute a ratio._")

        vel_df["trend_color"] = vel_df["cluster_trend"].map({
            "growing":              "🔺 Growing",
            "stable":               "➡️ Stable",
            "declining":            "✅ Declining",
            "insufficient_history": "⏳ New",
        }).fillna("➡️ Stable")

        TREND_COLORS = {
            "🔺 Growing":   "#ef4444",
            "➡️ Stable":    "#94a3b8",
            "✅ Declining": "#22c55e",
            "⏳ New":       "#a3a3a3",
        }
        if _skip_velocity_chart:
            fig_vel = None
        else:
            fig_vel = px.bar(
                vel_df,
                x="cluster_velocity_ratio",
                y="label_short",
                orientation="h",
                color="trend_color",
                color_discrete_map=TREND_COLORS,
                hover_data={"cluster_velocity_ratio": ":.2f",
                            "count": True, "modules": True,
                            "label_short": False, "trend_color": False},
                labels={"cluster_velocity_ratio": "Velocity ratio (recent / prior 2 versions)",
                        "label_short": "Issue Group",
                        "count": "Total bugs",
                        "modules": "Modules"},
                text="cluster_velocity_ratio",
            )
            fig_vel.update_traces(texttemplate="%{text:.2f}×", textposition="outside")
            fig_vel.add_vline(x=1.0, line_color="gray", line_width=1, line_dash="dash")
            fig_vel.add_vline(x=1.5, line_color="#ef4444", line_width=1, line_dash="dot",
                              annotation_text="Growing threshold", annotation_position="top right")
            fig_vel.update_layout(
                height=max(320, len(vel_df) * 28),
                yaxis={"categoryorder": "total ascending"},
                showlegend=True, legend_title_text="Trend",
                margin=dict(l=10, r=120, t=20, b=10),
            )
        if fig_vel is not None:
            st.plotly_chart(fig_vel, width='stretch')

    # ── NEW v3.0 — Module cluster entropy table ──────────────────────────
    # Use loaded entropy CSV if available; otherwise compute directly from cluster_df
    _ent_source_df = cluster_ent_df
    if _ent_source_df is None and cluster_df is not None and "cluster_id" in cluster_df.columns:
        _clustered_only = cluster_df[(cluster_df["cluster_id"] != -1)].dropna(
            subset=["parsed_module", "cluster_id"])
        _ent_rows = []
        for _mod, _grp in _clustered_only.groupby("parsed_module"):
            _counts = _grp["cluster_id"].value_counts().values.astype(float)
            _total  = _counts.sum()
            if _total > 0:
                _p   = _counts / _total
                _ent = float(-np.sum(_p * np.log2(_p + 1e-12)))
                _ent_rows.append({"module": _mod, "cluster_entropy": round(_ent, 3)})
        if _ent_rows:
            _ent_source_df = pd.DataFrame(_ent_rows)

    if _ent_source_df is not None and not _ent_source_df.empty:
        st.markdown("---")
        st.subheader("🔀 Module Cluster Entropy — Breadth of Bug Issue Groups")
        with st.expander("ℹ️ What is cluster entropy?", expanded=False):
            st.markdown("""
**Entropy** measures how spread out a module's bugs are across different issue groups.

| Entropy | Meaning | Action |
|---------|---------|--------|
| **Low (< 1)** | Bugs concentrated in one issue group — focused failure mode | Easy to fix: one root cause to address |
| **Medium (1–2)** | Bugs spread across a few issue groups | Review whether issue groups share a common component |
| **High (> 2)** | Bugs spread across many issue groups — broad instability | Module needs comprehensive regression testing |
""")
        ent_display = _ent_source_df.copy()
        ent_display["cluster_entropy"] = pd.to_numeric(
            ent_display["cluster_entropy"], errors="coerce").fillna(0.0)
        ent_display = ent_display.sort_values("cluster_entropy", ascending=False)
        ent_display["stability"] = ent_display["cluster_entropy"].apply(
            lambda v: "⚠️ Broad instability" if v > 2 else
                      "🔶 Spreading" if v > 1 else
                      "✅ Focused"
        )
        fig_ent = px.bar(
            ent_display.head(20),
            x="cluster_entropy", y="module", orientation="h",
            color="stability",
            color_discrete_map={
                "⚠️ Broad instability": "#ef4444",
                "🔶 Spreading":         "#f97316",
                "✅ Focused":           "#22c55e",
            },
            labels={"cluster_entropy": "Entropy (higher = more spread)",
                    "module": "Module", "stability": "Issue Group breadth"},
            text="cluster_entropy",
        )
        fig_ent.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_ent.add_vline(x=1.0, line_color="#f97316", line_width=1, line_dash="dash")
        fig_ent.add_vline(x=2.0, line_color="#ef4444", line_width=1, line_dash="dash")
        fig_ent.update_layout(
            height=max(300, min(20, len(ent_display)) * 30),
            yaxis={"categoryorder": "total ascending"},
            showlegend=True, legend_title_text="Issue Group breadth",
            margin=dict(l=10, r=80, t=10, b=10),
        )
        st.plotly_chart(fig_ent, width='stretch')

    # ── NEW v3.0 — Severity-stratified cluster views ─────────────────────
    if cluster_s12_df is not None or cluster_s34_df is not None:
        st.markdown("---")
        st.subheader("🔬 Severity-Stratified Issue Groups")
        st.caption(
            "Clusters run separately for S1/S2 (crash/major) and S3/S4 (normal/minor) bugs. "
            "The global view above mixes all severities; these tabs show each tier in isolation."
        )
        tier_tabs = st.tabs(["🔴 Critical/Major issue groups (S1–S2)", "🟡 Normal/Minor issue groups (S3–S4)"])

        for tab_widget, tier_df, tier_name in [
            (tier_tabs[0], cluster_s12_df, "S1/S2"),
            (tier_tabs[1], cluster_s34_df, "S3/S4"),
        ]:
            with tab_widget:
                if tier_df is None or tier_df.empty:
                    st.info(f"No stratified cluster data for {tier_name}. "
                            f"Re-run with `--stratify-severity` flag.")
                    continue

                for col, default in [("count", 0), ("avg_sev", 3.0), ("modules", "")]:
                    if col not in tier_df.columns:
                        tier_df[col] = default
                # Do NOT fill neutral defaults for velocity/recurrence — leave absent when unavailable.
                _tier_velocity_available = "cluster_velocity_ratio" in tier_df.columns

                tier_df = tier_df.copy()
                tier_df["label_short"] = tier_df["cluster_label"].apply(
                    lambda s: s[:45] + "…" if isinstance(s, str) and len(s) > 45 else s)
                tier_df["avg_sev"] = pd.to_numeric(tier_df["avg_sev"], errors="coerce").fillna(3.0)
                tier_df["count"]   = pd.to_numeric(tier_df["count"],   errors="coerce").fillna(0).astype(int)
                if _tier_velocity_available:
                    tier_df["cluster_velocity_ratio"] = pd.to_numeric(
                        tier_df["cluster_velocity_ratio"], errors="coerce")
                tier_df = tier_df.sort_values("count", ascending=False)

                # Overview bar — colour by trend when available, plain otherwise
                if _tier_velocity_available:
                    _tier_bar_color_col = "cluster_trend"
                    _tier_bar_color_map = {"growing": "#ef4444", "stable": "#94a3b8", "declining": "#22c55e"}
                    _tier_hover = {"count": True, "avg_sev": ":.2f", "modules": True,
                                   "cluster_velocity_ratio": ":.2f", "recurrence_rate": ":.0%"}
                else:
                    _tier_bar_color_col = "label_short"
                    _tier_bar_color_map = {}
                    _tier_hover = {"count": True, "avg_sev": ":.2f", "modules": True}

                fig_tier = px.bar(
                    tier_df.head(15),
                    x="count", y="label_short", orientation="h",
                    color=_tier_bar_color_col,
                    color_discrete_map=_tier_bar_color_map if _tier_bar_color_map else None,
                    hover_data=_tier_hover,
                    labels={"count": "Bugs", "label_short": "Issue Group",
                            "cluster_trend": "Trend",
                            "cluster_velocity_ratio": "Velocity",
                            "recurrence_rate": "Recurrence"},
                    text="count",
                )
                fig_tier.update_traces(textposition="outside")
                fig_tier.update_layout(
                    height=max(300, min(15, len(tier_df)) * 30),
                    yaxis={"categoryorder": "total ascending"},
                    showlegend=_tier_velocity_available,
                    legend_title_text="Trend",
                    margin=dict(l=10, r=40, t=10, b=10),
                )
                st.plotly_chart(fig_tier, width='stretch')

                if not _tier_velocity_available:
                    st.caption("ℹ️ Velocity/recurrence unavailable for this tier — run `cluster_bugs.py --stratify-severity` to generate stratified summary CSVs.")

                # Detail cards for top 10
                st.markdown(f"**Issue Group detail cards — {tier_name} tier**")
                tier_n_clustered = tier_df["count"].sum()
                for _ti, _tr in tier_df.head(10).iterrows():
                    _cid   = _tr.get("cluster_id", -1)
                    _label = str(_tr.get("cluster_label", "Unknown"))
                    _count = int(_tr.get("count", 0))
                    _avsev = float(_tr.get("avg_sev", 3.0))
                    _mods  = str(_tr.get("modules", ""))
                    _raw_vel   = _tr.get("cluster_velocity_ratio")
                    _raw_trend = _tr.get("cluster_trend")
                    _raw_recur = _tr.get("recurrence_rate")
                    _vel   = float(_raw_vel)   if _raw_vel   is not None and not (isinstance(_raw_vel,   float) and np.isnan(_raw_vel))   else None
                    _trend = str(_raw_trend)   if _raw_trend is not None else None
                    _recur = float(_raw_recur) if _raw_recur is not None and not (isinstance(_raw_recur, float) and np.isnan(_raw_recur)) else None
                    _sev_icon = "🔴" if _avsev <= 1.5 else "🟠" if _avsev <= 2.5 else "🟡" if _avsev <= 3.5 else "🟢"
                    _trend_icon = {"growing": "🔺", "declining": "✅", "stable": "➡️"}.get(_trend or "", "➡️")
                    if _vel is not None:
                        _title = f"{_sev_icon} **{_count} bugs** · {_trend_icon} {_trend} · _{_label}_"
                    else:
                        _title = f"{_sev_icon} **{_count} bugs** · _{_label}_"
                    with st.expander(_title, expanded=False):
                        _c1, _c2, _c3 = st.columns(3)
                        _c1.metric("Bugs", _count)
                        _c2.metric("Avg severity", f"{_avsev:.1f}")
                        _c3.metric("Share of tier", f"{_count / max(tier_n_clustered, 1) * 100:.1f}%")
                        if _vel is not None:
                            _v1, _v2 = st.columns(2)
                            _v1.metric("Velocity", f"{_vel:.2f}×",
                                       delta=f"{_trend_icon} {_trend.capitalize()}",
                                       delta_color="inverse" if _trend == "declining" else
                                                  ("off" if _trend == "stable" else "normal"))
                            _v2.metric("Recurrence", f"{_recur * 100:.0f}%" if _recur is not None else "N/A",
                                       help="Fraction of recent bugs from modules that contributed in prior window")
                        if _mods:
                            st.markdown(f"**Modules:** {_mods}")
                        # Sample bugs for this tier cluster
                        _id_col = "cluster_id_s12" if tier_name == "S1/S2" else "cluster_id_s34"
                        if cluster_df is not None and _id_col in cluster_df.columns:
                            _cl_bugs = cluster_df[cluster_df[_id_col] == _cid]
                        elif cluster_df is not None and "cluster_id" in cluster_df.columns:
                            _cl_bugs = cluster_df[cluster_df["cluster_id"] == _cid]
                        else:
                            _cl_bugs = pd.DataFrame()
                        if not _cl_bugs.empty and "parsed_description" in _cl_bugs.columns:
                            _samples = (
                                _cl_bugs[_cl_bugs["parsed_description"].notna()]
                                [["parsed_module", "parsed_description", "severity_label"]]
                                .head(5)
                                .rename(columns={"parsed_module": "Module",
                                                 "parsed_description": "Bug description",
                                                 "severity_label": "Severity"})
                            )
                            if not _samples.empty:
                                show_dataframe(_samples, hide_index=True, width='stretch')


# =====================================================================
# TAB 9 – Defect Forecast (v2.21 — scenario-based layout)
# =====================================================================

elif active_tab == "🔮 Defect Forecast":
    st.header("🔮 Defect Forecast — What's Likely to Break")

    with st.expander("📖 How to read this tab — full guide", expanded=False):
        st.markdown("""
## 🔮 Defect Forecast — Complete Guide

**One-line summary:** *"Where will the next version break, and what specifically should we test?"* A machine-learning model trained on every historical bug answers both questions in one tab.

**Manager pitch:** *"This is the only tab that looks forward instead of backward. Tabs 1–8 describe what already happened — this one ranks modules by their probability of producing a Critical bug in the next version, then converts that into a concrete test list."*

---

### 📊 Headline Metrics (Top Row)

| Metric | What it means |
|--------|--------------|
| **🔴 Critical modules** | >20% learned probability of an S1 critical bug next version — must be tested every version |
| **🟠 High-risk modules** | 10–20% probability of an S1 critical bug next version — test every sprint |
| **🎯 Predicted scenarios** | Total concrete, testable bug scenario predictions across all modules |
| **Total modules forecast** | Number of modules with enough history (≥5 versions) for the model to produce a prediction |

---

### 🎯 What to Test This Version (Primary Section)

Human-readable bug scenarios grouped by risk level, listed from most to least urgent (Critical → High → Medium). Each scenario answers: *"What specific thing should I test?"*

**Confidence levels** are based on how many similar bugs have recurred in history:
- **⬆️ High** — 3+ recurring bugs with similar descriptions (well-established pattern)
- **↔️ Medium** — 2 similar bugs (emerging pattern, include in testing)
- **⬇️ Low** — 1 occurrence (speculative; add to exploratory pass for Critical/High modules)

**Quadrant-aware scenario selection (new):** Scenarios are no longer ranked purely by ML score. Every **P1** module (and **P2** by default) is *always* included in the "What to Test" list as long as it has enough history — even if the model thinks risk is currently low. This keeps the test list aligned with the FMEA Risk Heatmap (Tab 7) so a P1 core module never silently disappears from coverage during a quiet stretch. Quadrant scoring uses these multipliers:

| Quadrant | Bonus weight |
|---|---|
| P1 — Critical | 1.00 (always included) |
| P2 — High | 0.60 |
| P3 — Medium | 0.20 |
| P4 — Low | 0.00 |

The list is hard-capped at 20 modules so it stays actionable.

**How scenarios are generated:**
1. Collect all bug Short Descriptions from the last 5 versions for each high-risk module.
2. Vectorise with TF-IDF and cluster similar descriptions with `AgglomerativeClustering` (cosine distance threshold).
3. Pick the most representative description from each cluster as the scenario text.
4. Assign confidence based on cluster size.
5. With `--provider ollama` / `--provider claude`, the AI receives the clusters and quadrant metadata, then produces more specific, actionable scenario text grounded in the real patterns.

---

### 📋 AI Risk Briefing — Next Version Focus Summary

A collapsible plain-English executive summary generated by `predict_defects.py`. Lists the top-risk modules with:
- Why they're at risk (which signals are currently elevated)
- What specifically to test based on historical patterns

Generated with Ollama or Claude when `--provider` is set; a heuristic text summary is produced even without AI.

---

### ⚠️ Severity Escalation Alerts

Shown automatically for modules where bugs are getting **more severe** over recent versions.

**Formula:**
```
severity_escalation = avg_severity(last version) − avg_severity(prior 3 versions)
```
Severity is numbered 1 (Critical) → 4 (Minor), so a **negative value means severity is worsening toward S1**. Modules with `severity_escalation < −0.3` appear here; `< −0.8` gets the 🚨 icon. Sorted by risk level so Critical modules surface first.

This signal can detect instability **before** raw bug counts spike — a pre-emptive warning that regression testing should be intensified immediately.

---

### 📋 Predicted Bug Categories — What Will Break?

A stacked bar chart (one bar per module, stacked by category) showing which **types** of bugs are expected based on the category distribution of recent bugs.

**The 6 QA categories** are assigned by keyword scoring on each bug's Short Description. Whichever category accumulates the most keyword hits wins:

| Category | Example trigger keywords |
|----------|------------------------|
| Crash / Stability | crash, exception, fatal, force close, hang, freeze |
| Feature not working as intended | not work, incorrect behavior, broken, regression |
| UI / Display problem | layout, alignment, render, icon, overlap, truncat |
| UX / Usability problem | confusing, navigation, slow, hard to find, unclear |
| Translation / Localization | translation, missing string, language, zh-tw, locale |
| Data / File / Sync issue | corrupt, import/export, sync, cannot save, wrong data |

`historical_pct` = this category's share of bugs in recent versions.
`expected_next_build` = `historical_pct × predicted_total` — how many bugs of this type are expected.

---

### 🔍 Predicted Bug Scenarios by Module

Expandable per-module section showing every predicted scenario with confidence badge, supporting QA categories, and the real source bug descriptions it was derived from. Critical-risk modules auto-expand; others are collapsed by default.

---

### 🃏 Module Forecast Cards

One card per Critical or High-risk module (top 10). Each card contains:

| Field | Explanation |
|-------|-------------|
| **Risk level** | Composite risk category (Critical/High/Medium/Low) |
| **Bug categories expected** | Count of distinct QA categories predicted for this module next version |
| **Actual last version** | Real observed bug count from the most recent version in training data — compare to the forecast for calibration |
| **Severity trend** | `severity_escalation` — see Alerts section above |
| **Versions since last critical** | `builds_since_last_crit` — number of versions elapsed since this module last produced an S1. Combined with `stable_mature` so the model can tell silent-but-quiet P1 cores apart from genuinely unknown new modules |
| **Issue Group breadth (entropy)** | Shannon entropy of issue-group distribution. Higher = bugs spread across many issue groups (broad instability). Lower = concentrated single failure mode. |
| **Leading signal** | The single feature with the highest Pearson correlation to future bug count for this module |
| **What types of bugs to expect** | Per-category breakdown from `_predictions_by_category.csv` showing historical % and expected count |
| **Predicted bug scenarios** | Top 3 scenarios from `_predictions_by_scenario.csv` |
| **AI risk briefing** | Narrative from Ollama/Claude when `--provider` is set |
| **What to test** | Concise test instruction based on risk level |

---

### 📡 Leading Indicators — What Predicts Future Bugs?

A horizontal bar chart ranking features by **Pearson correlation coefficient (r)** with future bug counts.

**How to read the chart:**
- **r close to +1 (red bar, right)** — when this signal goes up, more bugs follow in the next version
- **r close to −1 (green bar, left)** — when this signal goes up, fewer bugs follow (protective signal)
- **r near 0** — no consistent predictive relationship

The correlations are computed across all (module, version) rows in the training dataset by comparing each feature column against the `target` (actual bug count in the next version).

**Features ranked here include:**
- `crit_1/2/3` — critical bug momentum over last 1/2/3 versions
- `bugs_1/2/3` — total bug count momentum over last 1/2/3 versions
- `sev_1/2/3` — severity-weighted momentum (uses **status-weighted** counts so closed bugs decay over time)
- `trend` — last version minus 3 versions ago (upward slope)
- `severity_escalation` — worsening severity signal (avg severity tilting toward S1)
- `builds_since_last_crit` — versions elapsed since the last S1 in this module
- `crit_ratio` — proportion of S1 bugs in last 3 versions (high ratio = structurally dangerous module)
- `new_module` — flag for modules first appearing in recent versions (new features)
- `cross_module_spike` — count of other modules also spiking in the same version (correlated risk)
- `total_historical_bugs` — module maturity / overall activity level
- `stable_mature` (new) — flag for modules with ≥10 versions of history AND ≥8 versions since the last S1 AND no recent bug volume. Lets the classifier distinguish *genuinely calm core modules* from *unknown new ones*
- `impact_bug_ratio` (new) — bug count divided by FMEA Impact score; high values = the module is producing bugs at a rate disproportionate to how impactful it is
- `impact_weighted_spike` (new) — the cross-module spike signal weighted by FMEA impact
- `cluster_entropy_2/3` — issue-group diversity index (when cluster data is loaded)
- `top_cluster_velocity` — growth rate of the dominant issue group
- TF-IDF text features — keyword loadings from recent module descriptions (last 3 versions)
- Risk features (when loaded) — impact, detectability, probability scores from `ai_risk_scorer.py`

---

### 🔧 Advanced / Model Diagnostics (collapsed by default)

- **Predicted Bug Count bar chart** — raw predicted count per module coloured by risk level. Use the slider to control how many modules are shown.
- **Actual vs Predicted** — side-by-side bars comparing the model's last-version forecast against the real observed count. Useful for building trust in the model's accuracy before acting on it.
- **Module Signals Table** — sortable table of all numeric signals for every module, including both global and stratified forecasts, S1 ratio, and cross-module spike count.
- **Full Predictions Table** — complete output of `_predictions.csv` with all new features (crit_ratio, new_module, cross_module_spike).

---

### 🛠️ How the ML Model Works

**Bug count model:** `GradientBoostingRegressor` (scikit-learn, 200 trees, max_depth=4, learning_rate=0.1). Predicts bug count in the next version. Output includes both a **global model** (one GBR across all modules) and a **stratified model** (separate GBRs for high- and low-activity modules) — compare the "Forecast (global)" vs "Forecast (stratified)" columns in the Full Predictions Table to see which behaves better for your modules.

**Risk classifier:** `GradientBoostingClassifier` (150 trees, max_depth=3) calibrated with isotonic regression. Predicts the probability of at least one S1 (Critical) bug next version. The risk score IS that probability (0–100%). Replaces the old hand-tuned composite scoring. Falls back to weighted composite if the classifier has too few samples.

**Status weighting (important):** Bugs are not all weighted equally in training:
- **Open bugs → weight 1.0** (full live signal)
- **Invalid bugs (NAB / Won't Fix / Not a Bug / etc.) → weight 0.0** (excluded entirely; they were never real defects)
- **Closed/fixed bugs → weight 0.5, decaying linearly to 0.1 over 12 versions** (the fix lands → still a regression candidate next version → signal fades as more clean versions pass; never zero so a long-settled module still carries a faint echo of its bug history)

This means re-running the pipeline after a release closes out bugs gives the model a *softer* version of history, not a cliff drop — so quiet P1 cores stay flagged correctly.

**Training validation:** 3-fold `TimeSeriesSplit` cross-validation (respects time ordering — no data leakage). The CV MAE (Mean Absolute Error) is printed at run time: it tells you how many bugs off the forecast is on average.

**Feature matrix:** Built by `build_features()` in `predict_defects.py`. Each row is a (module, version) pair. Features include rolling window stats (1/2/3 versions), critical bug ratio, new module flag, cross-module spike count, rolling TF-IDF text features, and impact-aware features (`stable_mature`, `impact_bug_ratio`, `impact_weighted_spike`) when FMEA data is loaded. Requires at least **5 versions of history** per module; modules with less are excluded.

**Risk level thresholds:**
- **Critical > 20%** probability of S1 next version
- **High 10–20%**
- **Medium 5–10%**
- **Low < 5%**

S1 bugs are ~3.6% of all bugs in history, so a 10% prediction is already ~3× baseline elevation. Because these are *learned* probabilities, most modules will be "Low" when the product is healthy — only modules with genuine S1 risk patterns reach Critical.

---

### 🛠️ Where Does This Data Come From?

```bash
# Basic run (heuristic scenarios, no AI needed):
python scripts/predict_defects.py data/ecl_parsed.csv

# With cluster features (enables entropy signals and better accuracy):
python scripts/predict_defects.py data/ecl_parsed.csv \\
  --cluster-csv data/clusters/ecl_parsed_clustered.csv

# Full AI-powered scenarios + I×P×D risk features:
python scripts/predict_defects.py data/ecl_parsed.csv \\
  --cluster-csv data/clusters/ecl_parsed_clustered.csv \\
  --scored-csv data/risk_register_scored.csv \\
  --provider ollama --model qwen3:4b
```

**Pipeline shortcuts** (`refresh_pipeline.sh`):
- Default Friday refresh runs cluster + predict together so Tabs 8 and 9 stay in sync.
- `./refresh_pipeline.sh --force-relabel` re-labels clusters AND forces predictions to rebuild (so the new cluster labels propagate into the scenarios).
- `./refresh_pipeline.sh --skip-predict` skips this stage when you only need clustering.

**Output files** (auto-loaded from `data/products/<slug>/predictions/`):

| File | Contents |
|------|---------|
| `_predictions.csv` | Main forecast — predicted count, risk level, all signals, quadrant, priority_score |
| `_predictions_by_category.csv` | Per-module per-category historical % and expected count |
| `_predictions_by_scenario.csv` | Concrete bug scenario predictions with confidence (quadrant-aware selection) |
| `_predictions_by_cluster.csv` | Per-module per-cluster breakdown (requires `--cluster-csv`) |
| `_predictions_importance.csv` | Feature importance ranking from the trained model |
| `_predictions_leading_indicators.csv` | Pearson r correlation of each feature with future bugs |
| `_predictions_focus_summary.txt` | Plain-English risk briefing text |

**Recommended cadence:** re-run **every Friday** alongside `cluster_bugs.py` so Tabs 8 and 9 stay aligned.
""")

    if pred_df is None:
        st.warning(
            "**Prediction data not loaded.** "
            "Run `python scripts/predict_defects.py data/ecl_parsed.csv` then set the path in **Sidebar → Step 4**."
        )
        st.stop()

    required_pred_cols = {"module", "predicted", "risk_level"}
    if not required_pred_cols.issubset(pred_df.columns):
        st.error(f"Predictions file missing columns: {required_pred_cols - set(pred_df.columns)}")
        st.stop()

    # Build _ent_source_df for module entropy lookup (same fallback logic as Tab 8)
    _ent_source_df = cluster_ent_df
    if _ent_source_df is None and cluster_df is not None and "cluster_id" in cluster_df.columns:
        _ec_only = cluster_df[(cluster_df["cluster_id"] != -1)].dropna(
            subset=["parsed_module", "cluster_id"])
        _ent_rows_t9 = []
        for _mod, _grp in _ec_only.groupby("parsed_module"):
            _c = _grp["cluster_id"].value_counts().values.astype(float)
            if _c.sum() > 0:
                _p = _c / _c.sum()
                _ent_rows_t9.append({"module": _mod,
                                     "cluster_entropy": round(float(-np.sum(_p * np.log2(_p + 1e-12))), 3)})
        if _ent_rows_t9:
            _ent_source_df = pd.DataFrame(_ent_rows_t9)

    pred_df["predicted"] = pd.to_numeric(pred_df["predicted"], errors="coerce").fillna(0)
    # Sort by blended priority_score when present — keeps this tab's ordering
    # aligned with the Risk Heatmap (heatmap weights 65% of priority_score).
    if "priority_score" in pred_df.columns and pd.to_numeric(
            pred_df["priority_score"], errors="coerce").notna().any():
        pred_df["priority_score"] = pd.to_numeric(
            pred_df["priority_score"], errors="coerce").fillna(0)
        pred_df = pred_df.sort_values(
            ["priority_score", "predicted"], ascending=[False, False]
        ).reset_index(drop=True)
    else:
        pred_df = pred_df.sort_values("predicted", ascending=False).reset_index(drop=True)

    FORECAST_COLORS = {
        "Critical": "#ef4444",
        "High":     "#f97316",
        "Medium":   "#eab308",
        "Low":      "#22c55e",
    }
    RISK_ORDER = ["Critical", "High", "Medium", "Low"]
    RISK_ICONS   = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
    RISK_ADVICE  = {
        "Critical": "Test **every version**. Focus on crash scenarios, data loss, and any recently changed functionality.",
        "High":     "Test **every sprint**. Run full regression for this module and check for side effects in related areas.",
        "Medium":   "Include in **release-candidate** testing. Spot-check changed areas.",
        "Low":      "Cover in the full **release cycle** pass. No special urgency.",
    }
    # Confidence badges use arrows (not colored circles) to avoid conflicting
    # with RISK_ICONS, which already uses 🟢/🟡 for Low/Medium risk.
    _CONF_BADGES = {"high": "⬆️ High conf.", "medium": "↔️ Med. conf.", "low": "⬇️ Low conf."}

    # ── Headline metrics ─────────────────────────────────────────────────
    rl_counts = pred_df["risk_level"].value_counts()
    _scenario_count = len(pred_scenario_df) if pred_scenario_df is not None else 0
    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("🔴 Critical modules",  int(rl_counts.get("Critical", 0)),
               help=">20% probability of S1 critical bug next version (learned classifier)")
    pm2.metric("🟠 High-risk modules", int(rl_counts.get("High", 0)),
               help="10–20% probability of S1 critical bug next version (learned classifier)")
    pm3.metric("🎯 Predicted scenarios", _scenario_count,
               help="Total concrete bug scenario predictions across all modules")
    pm4.metric("Total modules forecast", len(pred_df))

    # ── Severity escalation alerts ── (shown before primary content so urgent signals are seen first)
    if "severity_escalation" in pred_df.columns:
        _esc = pred_df[["module", "risk_level", "severity_escalation"]].copy()
        _esc["severity_escalation"] = pd.to_numeric(_esc["severity_escalation"], errors="coerce")
        _RISK_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        _worsening = _esc[_esc["severity_escalation"] < -0.3].copy()
        _worsening["_risk_order"] = _worsening["risk_level"].map(_RISK_ORDER).fillna(99)
        _worsening = _worsening.sort_values(
            ["_risk_order", "severity_escalation"]).head(8)
        if not _worsening.empty:
            st.subheader("⚠️ Severity Escalation Alerts")
            st.caption(
                "These modules have bugs getting **more severe** in recent versions — "
                "the average severity is trending toward Critical (S1). "
                "Flag for immediate investigation even if raw bug counts look low."
            )
            for _, _er in _worsening.iterrows():
                _esc_val = float(_er["severity_escalation"])
                _esc_icon = "🚨" if _esc_val < -0.8 else "⚠️"
                st.warning(
                    f"{_esc_icon} ({_er['risk_level']} risk) **{_er['module']}** — "
                    f"severity worsening by {abs(_esc_val):.2f} points toward S1"
                )


    # ── Top Sub-Feature Risks — sortable bug-type × module ranking ─────────
    if (pred_category_df is not None and not pred_category_df.empty
            and "risk_score" in pred_category_df.columns):
        st.markdown("---")
        st.subheader("🎯 Top Sub-Feature Risks — Where Each Feature Will Break")
        st.caption("Per (module × bug type) ranking. risk_score combines historical "
                   "frequency × severity × recency. example_bug_codes link to source bugs.")
        _top_sub = pred_category_df.sort_values("risk_score", ascending=False).head(25).copy()
        _sub_cols = [c for c in [
            "module", "category", "risk_score", "expected_next_build",
            "expected_s1", "expected_s2", "expected_s3plus",
            "confidence", "example_bug_codes", "latest_bug_build",
        ] if c in _top_sub.columns]
        show_dataframe(_top_sub[_sub_cols], hide_index=True, width='stretch')

    # ── "What to Test This Version" — PRIMARY SECTION ──────────────────────
    st.markdown("---")
    st.subheader("🎯 What to Test This Version")

    if pred_scenario_df is not None and not pred_scenario_df.empty:
        _has_scenario_type = "scenario_type" in pred_scenario_df.columns
        _has_explanation   = "explanation"   in pred_scenario_df.columns
        _any_synth = bool(_has_scenario_type and
                          (pred_scenario_df["scenario_type"] == "ai_synthesized").any())
        _any_hist  = bool(_has_scenario_type and
                          (pred_scenario_df["scenario_type"] == "historical_pattern").any())
        _label_mix = []
        if _any_synth: _label_mix.append("🧠 AI-synthesized scenarios")
        if _any_hist:  _label_mix.append("📎 Historical patterns at risk of recurrence")
        _legend = " · ".join(_label_mix) if _label_mix else ""

        st.caption(
            "Concrete bug scenarios predicted for the next version, grouped by risk level. "
            "Each scenario is grounded in recurring historical patterns. "
            "Only modules with enough historical data to generate specific scenarios are shown here — "
            "see the **AI Risk Briefing** below for a full ranked list of all forecast modules."
            + (f"\n\n_Shown: {_legend}_" if _legend else "")
        )
        # Group scenarios by risk level
        _TYPE_BADGES = {
            "ai_synthesized":     "🧠 AI-synthesized",
            "historical_pattern": "📎 Historical pattern",
        }
        for _rl in ["Critical", "High", "Medium"]:
            _rl_scenarios = pred_scenario_df[pred_scenario_df["risk_level"] == _rl]
            if _rl_scenarios.empty:
                continue
            _rl_icon = RISK_ICONS.get(_rl, "⚪")
            st.markdown(f"#### {_rl_icon} {_rl} Risk")
            for _mod, _mod_sc in _rl_scenarios.groupby("module", sort=False):
                st.markdown(f"**{_mod}**")
                for _, _sc in _mod_sc.head(3).iterrows():
                    _conf = _CONF_BADGES.get(str(_sc.get("confidence", "medium")), "⬇️ Low conf.")
                    _text = str(_sc.get("scenario_text", ""))
                    # Per-row type badge only when there's a mix; otherwise the
                    # top-of-section legend already conveys the source.
                    _type_badge = (_TYPE_BADGES.get(str(_sc.get("scenario_type", "")), "")
                                   if (_any_synth and _any_hist) else "")
                    _badge_str = f" {_type_badge}" if _type_badge else ""
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{_conf}{_badge_str} — {_text}",
                                unsafe_allow_html=True)
            st.markdown("")
    else:
        # Fallback to AI Focus Summary if no scenarios
        if pred_summary_txt and pred_summary_txt.strip():
            st.caption(
                "Scenario predictions not available. Showing AI Focus Summary instead. "
                "Re-run predict_defects.py v5.0 to generate scenario predictions."
            )
            st.code(pred_summary_txt, language=None)
        else:
            st.info(
                "No scenario predictions available. Run `predict_defects.py` v5.0 "
                "to generate concrete bug scenario predictions."
            )

    # ── AI Focus Summary ─────────────────────────────────────────────────
    if pred_summary_txt and pred_summary_txt.strip():
        with st.expander("📋 AI Risk Briefing — Next Build Focus Summary", expanded=False):
            st.caption(
                "Plain-English summary generated by predict_defects.py. "
                "Lists the highest-risk modules with why they are at risk and what to test."
            )
            st.code(pred_summary_txt, language=None)

    st.markdown("---")

    # ── Bug Categories stacked bar chart ─────────────────────────────────
    # Compute top_n_pred for charts
    _pred_n = len(pred_df)
    if _pred_n <= 1:
        top_n_pred = _pred_n
    else:
        top_n_pred = min(20, _pred_n)

    if pred_category_df is not None and not pred_category_df.empty:
        st.subheader("📋 Predicted Bug Categories — What Will Break?")
        st.caption(
            "Each bar shows which types of bugs are expected per module, based on "
            "recent bug descriptions. This tells you WHAT to test for, not just how many."
        )
        _top_mods = pred_df.head(top_n_pred)["module"].tolist()
        _cat_chart = pred_category_df[pred_category_df["module"].isin(_top_mods)].copy()
        if not _cat_chart.empty:
            _CAT_COLORS = {
                "Crash / Stability":              "#ef4444",
                "Feature not working as intended": "#f97316",
                "UI / Display problem":            "#eab308",
                "UX / Usability problem":          "#a855f7",
                "Translation / Localization":      "#3b82f6",
                "Data / File / Sync issue":        "#06b6d4",
            }
            # Stack bottom→top; legend reversed so top of legend = top of bar stack.
            _CAT_STACK_ORDER = [
                "Data / File / Sync issue",
                "Translation / Localization",
                "UX / Usability problem",
                "UI / Display problem",
                "Feature not working as intended",
                "Crash / Stability",
            ]
            _mod_order = [m for m in _top_mods if m in _cat_chart["module"].values]
            _cat_chart["module"] = pd.Categorical(
                _cat_chart["module"], categories=_mod_order, ordered=True)
            _cat_chart["category"] = pd.Categorical(
                _cat_chart["category"], categories=_CAT_STACK_ORDER, ordered=True)
            _cat_chart = _cat_chart.sort_values(["module", "category"],
                                                ascending=[True, True])
            fig_cat = px.bar(
                _cat_chart,
                x="module", y="historical_pct", color="category",
                color_discrete_map=_CAT_COLORS,
                category_orders={"category": _CAT_STACK_ORDER},
                labels={"module": "Module", "historical_pct": "Proportion of recent bugs",
                        "category": "Bug Category"},
                hover_data={"historical_count": True, "expected_next_build": ":.1f"},
                barmode="stack",
            )
            fig_cat.update_traces(texttemplate="", textposition="inside")
            fig_cat.update_layout(
                height=460,
                xaxis_tickangle=-35,
                yaxis_tickformat=".0%",
                showlegend=True,
                legend_title_text="Bug Category",
                legend=dict(traceorder="reversed"),
                margin=dict(t=20, b=10),
            )
            st.plotly_chart(fig_cat, width='stretch')
        st.markdown("---")

    # Build description ↔ BugCode lookups once per session.
    # `_desc_lookup`     : description → BugCode (used to backfill missing codes)
    # `_bc_to_desc`      : BugCode → description (preferred source of truth when
    #                      rendering source bugs — protects against stale
    #                      prediction CSVs where source_bug_examples may not be
    #                      aligned with source_bug_codes)
    _DESC_LOOKUP_KEY = "__t9_desc_to_bugcode"
    _BC_LOOKUP_KEY   = "__t9_bugcode_to_desc"
    if _DESC_LOOKUP_KEY not in st.session_state or _BC_LOOKUP_KEY not in st.session_state:
        _raw_fp = st.session_state.get("fp_bugs", "")
        _desc_to_bc: dict[str, str] = {}
        _bc_to_desc: dict[str, str] = {}
        if _raw_fp and Path(_raw_fp).exists():
            _raw_df = load_csv(_raw_fp)
            _bc_col = next(
                (c for c in _raw_df.columns if "bugcode" in c.lower()), None
            )
            if _bc_col:
                for _d_col in ("Short Description", "parsed_description"):
                    if _d_col in _raw_df.columns:
                        for _d, _bc in zip(_raw_df[_d_col], _raw_df[_bc_col]):
                            if pd.notna(_d) and pd.notna(_bc) and str(_d).strip():
                                _desc_to_bc[str(_d)] = str(_bc)
                                _bc_to_desc.setdefault(str(_bc), str(_d))
        st.session_state[_DESC_LOOKUP_KEY] = _desc_to_bc
        st.session_state[_BC_LOOKUP_KEY]   = _bc_to_desc
    _desc_lookup: dict[str, str] = st.session_state[_DESC_LOOKUP_KEY]
    _bc_to_desc:  dict[str, str] = st.session_state[_BC_LOOKUP_KEY]

    # ── Predicted Bug Scenarios by Module (NEW detail section) ────────────
    if pred_scenario_df is not None and not pred_scenario_df.empty:
        st.subheader("🔍 Predicted Bug Scenarios by Module")
        st.caption(
            "Expand each module to see all predicted scenarios with confidence levels, "
            "supporting categories, and source bug examples."
        )

        _scenario_modules = pred_scenario_df["module"].unique()
        for _sc_mod in _scenario_modules:
            _sc_mod_data = pred_scenario_df[pred_scenario_df["module"] == _sc_mod]
            _sc_rl = str(_sc_mod_data.iloc[0].get("risk_level", "Medium"))
            _sc_icon = RISK_ICONS.get(_sc_rl, "⚪")
            _sc_count = len(_sc_mod_data)
            with st.expander(
                f"{_sc_icon} **{_sc_mod}** — {_sc_rl} risk · {_sc_count} scenario(s)",
                expanded=(_sc_rl == "Critical"),
            ):
                # Show the predicted version number once at the top of the module card
                _pred_build = _sc_mod_data.iloc[0].get("predicted_build")
                if _pred_build is not None and str(_pred_build) not in ("nan", ""):
                    st.caption(f"Predicted for version: **{_pred_build}**")

                for _, _sc_row in _sc_mod_data.iterrows():
                    _rank = int(_sc_row.get("scenario_rank", 0))
                    _text = str(_sc_row.get("scenario_text", ""))
                    _conf = str(_sc_row.get("confidence", "medium"))
                    _conf_badge = _CONF_BADGES.get(_conf, "⬇️ Low conf.")
                    _cats = str(_sc_row.get("supporting_categories", ""))
                    _examples = str(_sc_row.get("source_bug_examples", ""))
                    _sc_signal = str(_sc_row.get("leading_signal", ""))
                    _sc_type = str(_sc_row.get("scenario_type", ""))
                    _sc_expl = str(_sc_row.get("explanation", ""))
                    # Per-row type badge only when there's a mix; otherwise the
                    # top-of-section legend already conveys the source.
                    _type_badge = ({
                        "ai_synthesized":     "🧠 AI-synthesized",
                        "historical_pattern": "📎 Historical pattern",
                    }.get(_sc_type, "") if (_any_synth and _any_hist) else "")
                    _badge_str = f" {_type_badge}" if _type_badge else ""

                    st.markdown(f"**#{_rank}** {_conf_badge}{_badge_str} — {_text}")
                    _detail_parts = []
                    if _cats and _cats != "nan":
                        _detail_parts.append(f"Categories: {_cats}")
                    if _sc_signal and _sc_signal not in ("nan", ""):
                        _detail_parts.append(f"Signal: _{_sc_signal}_")
                    if _detail_parts:
                        st.caption(" · ".join(_detail_parts))
                    if _sc_expl and _sc_expl not in ("nan", ""):
                        with st.expander("❓ Why this scenario?", expanded=False):
                            st.markdown(_sc_expl.replace("\n", "  \n"))
                    # Source bugs — prefer direct BugCodes when present (no fuzzy match needed).
                    # Description for each BugCode is read from raw bugs via `_bc_to_desc`,
                    # NOT from `source_bug_examples` — that field can be misaligned on
                    # CSVs produced before the predict_defects.py alignment fix.
                    _codes_field = str(_sc_row.get("source_bug_codes", "")).strip()
                    if _codes_field and _codes_field not in ("nan", ""):
                        _codes_list = [c.strip() for c in _codes_field.split(" | ") if c.strip()]
                        _ex_descs = [d.strip() for d in _examples.split(" | ") if d.strip()] \
                                    if _examples and _examples != "nan" else []
                        _ex_rows = []
                        for _i, _bc in enumerate(_codes_list):
                            _ed = _bc_to_desc.get(_bc) or (
                                _ex_descs[_i] if _i < len(_ex_descs) else ""
                            )
                            _ex_rows.append((_bc, _ed, ECL_BUG_URL.format(bug_code=_bc)))
                        with st.expander(f"🐛 Source bugs — {len(_ex_rows)} linked", expanded=False):
                            for _bc, _ed, _url in _ex_rows:
                                if _ed:
                                    st.markdown(f"- [{_bc}]({_url}) — {_ed}")
                                else:
                                    st.markdown(f"- [{_bc}]({_url})")
                    elif _examples and _examples not in ("nan", ""):
                        _ex_descs = [d.strip() for d in _examples.split(" | ") if d.strip()]
                        if _ex_descs:
                            _ex_rows = []
                            for _ed in _ex_descs:
                                _bc = _desc_lookup.get(_ed)
                                if _bc and _bc not in ("nan", ""):
                                    _ex_rows.append((_bc, _ed,
                                        ECL_BUG_URL.format(bug_code=_bc)))
                                else:
                                    _ex_rows.append((None, _ed, None))
                            _link_count = sum(1 for r in _ex_rows if r[0])
                            _lbl = (f"🐛 Source bugs — {_link_count} linked"
                                    if _link_count else f"🐛 Source bugs ({len(_ex_rows)})")
                            with st.expander(_lbl, expanded=False):
                                for _bc, _ed, _url in _ex_rows:
                                    if _url:
                                        st.markdown(f"- [{_bc}]({_url}) — {_ed}")
                                    else:
                                        st.markdown(f"- _{_ed}_")
                    st.markdown("")

                # Bug Type Examples — per-category breakdown with example descriptions
                if pred_category_df is not None and not pred_category_df.empty:
                    _mod_cat_rows = pred_category_df[
                        (pred_category_df["module"] == _sc_mod)
                        & (pd.to_numeric(
                            pred_category_df["expected_next_build"], errors="coerce"
                        ).fillna(0) > 0)
                    ].sort_values("expected_next_build", ascending=False)
                    if not _mod_cat_rows.empty:
                        with st.expander("🏷️ Predicted Bug Types with Examples", expanded=False):
                            st.caption(
                                "What types of bugs are expected and what they typically look like, "
                                "based on historical patterns for this module."
                            )
                            for _, _cr in _mod_cat_rows.iterrows():
                                _cat_name = str(_cr.get("category", ""))
                                _exp_cnt  = float(_cr.get("expected_next_build", 0))
                                _exs      = str(_cr.get("example_descriptions", "")).strip()
                                _gen      = str(_cr.get("generated_description", "")).strip()
                                st.markdown(f"**{_cat_name}** — ~{_exp_cnt:.1f} expected")
                                if _exs and _exs not in ("nan", ""):
                                    for _ex in _exs.split(" | ")[:3]:
                                        _ex = _ex.strip()
                                        if not _ex:
                                            continue
                                        _bc = _desc_lookup.get(_ex)
                                        if _bc and _bc not in ("nan", ""):
                                            _url = ECL_BUG_URL.format(bug_code=_bc)
                                            st.markdown(f"- [{_bc}]({_url}) — {_ex}")
                                        else:
                                            st.markdown(f"- {_ex}")
                                elif _gen and _gen not in ("nan", ""):
                                    _gen_bc = str(_cr.get("generated_bug_code", "")).strip()
                                    if _gen_bc and _gen_bc not in ("nan", ""):
                                        _gen_url = ECL_BUG_URL.format(bug_code=_gen_bc)
                                        st.markdown(f"- [{_gen_bc}]({_gen_url}) — {_gen} _(similar existing bug)_")
                                    else:
                                        st.markdown(f"- {_gen} _(similar existing bug)_")
        st.markdown("---")

    # ── Module forecast cards ─────────────────────────────────────────────
    st.subheader("🃏 Module Forecast Cards")
    st.caption(
        "One card per high-risk module. Written for anyone on the team — "
        "no data science background needed."
    )

    cards_df = pred_df[pred_df["risk_level"].isin(["Critical", "High"])].head(10)
    if cards_df.empty:
        cards_df = pred_df.head(5)

    for _, row in cards_df.iterrows():
        mod       = str(row.get("module", "Unknown"))
        pred_val  = float(row.get("predicted", 0))
        rl        = str(row.get("risk_level", "Medium"))
        dom_type  = str(row.get("dominant_bug_type", "mixed"))
        lead_sig  = str(row.get("leading_signal", "recent bug momentum"))
        target_v  = row.get("target", None)
        icon      = RISK_ICONS.get(rl, "⚪")

        # v4.0 — build a short category summary for the card header
        _card_cat_hint = ""
        if pred_category_df is not None and not pred_category_df.empty:
            _mc = pred_category_df[pred_category_df["module"] == mod].head(2)
            if not _mc.empty:
                _card_cat_hint = " · ".join(_mc["category"].tolist())

        _quadrant = row.get("heatmap_quadrant")
        _priority = row.get("priority_score")
        _mod_vel  = row.get("module_cluster_velocity")
        _quadrant_tag = ""
        if _quadrant is not None and pd.notna(_quadrant):
            _quadrant_tag = f" · Heatmap {str(_quadrant).split('-')[0].strip()}"

        if _card_cat_hint:
            card_header = f"{icon} **{mod}** — {rl} risk{_quadrant_tag} · likely: {_card_cat_hint}"
        else:
            card_header = f"{icon} **{mod}** — {rl} risk{_quadrant_tag}"
        with st.expander(card_header, expanded=(rl == "Critical")):
            # Row 1 — risk + context metrics
            fc1, fc2, fc3 = st.columns(3)
            if _priority is not None and pd.notna(_priority):
                fc1.metric("Priority (blended)", f"{float(_priority):.0f}",
                           help="Blended rank: 65% heatmap risk_score_final + "
                                "20% ML P(S1) + 15% cluster velocity")
            else:
                fc1.metric("Risk level", rl)
            if pred_category_df is not None and not pred_category_df.empty:
                _mod_cats = pred_category_df[pred_category_df["module"] == mod]
                fc2.metric("Bug categories expected", f"{len(_mod_cats)}",
                           help="Number of distinct QA bug categories predicted for next version")
            else:
                fc2.metric("Predicted bugs", f"{pred_val:.0f}")
            if _mod_vel is not None and pd.notna(_mod_vel):
                _vel_f = float(_mod_vel)
                _vel_icon = "🔺" if _vel_f >= 1.5 else ("✅" if _vel_f <= 0.67 else "➡️")
                fc3.metric("Cluster velocity", f"{_vel_icon} {_vel_f:.2f}×",
                           help="Max growth rate across this module's bug clusters "
                                "(recent 2 versions / prior 2 versions). "
                                ">1.5× = growing.")
            elif target_v is not None:
                fc3.metric("Actual last version", f"{float(target_v):.0f}")

            # Row 2 — v3.0 signals
            sev_esc   = row.get("severity_escalation", None)
            bslc      = row.get("builds_since_last_crit", None)
            mod_ent   = None
            if _ent_source_df is not None and not _ent_source_df.empty:
                _ent_row = _ent_source_df[_ent_source_df["module"] == mod]
                if not _ent_row.empty:
                    mod_ent = float(_ent_row.iloc[0]["cluster_entropy"])

            if sev_esc is not None or bslc is not None or mod_ent is not None:
                sc1, sc2, sc3 = st.columns(3)
                if sev_esc is not None:
                    _sev_esc_f = float(sev_esc)
                    _esc_label = (
                        "🚨 Worsening toward S1" if _sev_esc_f < -0.5 else
                        "⚠️ Slight worsening"    if _sev_esc_f < -0.1 else
                        "✅ Stable / improving"
                    )
                    sc1.metric("Severity trend", _esc_label,
                               help="Negative = severity worsening toward S1 in recent versions")
                if bslc is not None:
                    sc2.metric("Versions since last critical", f"{int(float(bslc))}",
                               help="How many versions have passed since the last S1 bug in this module")
                if mod_ent is not None:
                    _ent_label = (
                        "⚠️ Broad instability" if mod_ent > 2.0 else
                        "🔶 Spreading issue groups"  if mod_ent > 1.0 else
                        "✅ Focused issue group"
                    )
                    sc3.metric("Issue Group breadth (entropy)", f"{mod_ent:.2f} — {_ent_label}",
                               help="High entropy = bugs spread across many issue groups; low = concentrated failure mode")

            st.markdown(f"**Leading signal:** _{lead_sig}_")

            # Per-category bug-type breakdown
            _showed_categories = False
            if pred_category_df is not None and not pred_category_df.empty:
                mod_cats = pred_category_df[pred_category_df["module"] == mod].head(6)
                if not mod_cats.empty:
                    _showed_categories = True
                    st.markdown("**What types of bugs to expect:**")
                    for _, cr in mod_cats.iterrows():
                        pct_str = f"{cr['historical_pct'] * 100:.0f}%"
                        cat_name = str(cr.get("category", "Unknown"))
                        exp_cnt  = cr.get("expected_next_build", 0)
                        _s1 = float(cr.get("expected_s1", 0) or 0)
                        _s2 = float(cr.get("expected_s2", 0) or 0)
                        _badge = ""
                        if _s1 > 0:
                            _badge += f" · 🚨 {_s1:.1f} S1"
                        if _s2 > 0:
                            _badge += f" · ⚠️ {_s2:.1f} S2"
                        st.markdown(
                            f"&nbsp;&nbsp;• **{cat_name}** — {pct_str} of recent bugs"
                            + (f" (~{exp_cnt:.0f} expected)" if exp_cnt >= 1 else "")
                            + _badge,
                            unsafe_allow_html=True,
                        )
                        _exs = str(cr.get("example_descriptions", "")).strip()
                        _gen = str(cr.get("generated_description", "")).strip()
                        if _exs and _exs not in ("nan", ""):
                            for _ex in _exs.split(" | ")[:2]:
                                _ex = _ex.strip()
                                if not _ex:
                                    continue
                                _bc = _desc_lookup.get(_ex)
                                if _bc and _bc not in ("nan", ""):
                                    _url = ECL_BUG_URL.format(bug_code=_bc)
                                    st.caption(
                                        f"&nbsp;&nbsp;&nbsp;&nbsp;↳ [{_bc}]({_url}) — {_ex}",
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.caption(
                                        f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {_ex}",
                                        unsafe_allow_html=True,
                                    )
                        elif _gen and _gen not in ("nan", ""):
                            _gen_bc = str(cr.get("generated_bug_code", "")).strip()
                            if _gen_bc and _gen_bc not in ("nan", ""):
                                _gen_url = ECL_BUG_URL.format(bug_code=_gen_bc)
                                st.caption(
                                    f"&nbsp;&nbsp;&nbsp;&nbsp;↳ [{_gen_bc}]({_gen_url}) — {_gen} _(similar existing bug)_",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.caption(
                                    f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {_gen} _(similar existing bug)_",
                                    unsafe_allow_html=True,
                                )

            # Fall back to cluster breakdown if no categories
            if not _showed_categories and pred_cluster_df is not None and not pred_cluster_df.empty:
                mod_clusters = pred_cluster_df[pred_cluster_df["module"] == mod].head(6)
                if not mod_clusters.empty:
                    st.markdown("**What to expect (by issue group):**")
                    for _, cr in mod_clusters.iterrows():
                        pct_str = f"{cr['historical_pct'] * 100:.0f}%"
                        label   = str(cr.get("cluster_label", f"Cluster {cr['cluster_id']}"))
                        if label not in ("No cluster data", "Unclustered"):
                            st.markdown(
                                f"&nbsp;&nbsp;• _{label}_ ({pct_str})",
                                unsafe_allow_html=True,
                            )

            # v5.0 — show top predicted scenarios in the card
            if pred_scenario_df is not None and not pred_scenario_df.empty:
                _card_scenarios = pred_scenario_df[pred_scenario_df["module"] == mod].head(3)
                if not _card_scenarios.empty:
                    st.markdown("**Predicted bug scenarios:**")
                    for _, _cs in _card_scenarios.iterrows():
                        _cs_conf = _CONF_BADGES.get(str(_cs.get("confidence", "medium")), "⬇️ Low conf.")
                        _cs_text = str(_cs.get("scenario_text", ""))
                        st.markdown(f"&nbsp;&nbsp;→ {_cs_conf} — {_cs_text}",
                                    unsafe_allow_html=True)

            # AI narrative
            _narrative = str(row.get("ai_narrative", "")).strip()
            if _narrative and _narrative not in ("", "nan"):
                st.markdown("**AI risk briefing:**")
                # The briefing is model prose and cites bug codes freely.
                st.info(linkify_bug_codes(_narrative))

            st.info(f"**What to test:** {RISK_ADVICE.get(rl, '')}")

    # ── Leading indicators ────────────────────────────────────────────────
    if pred_leading_df is not None and not pred_leading_df.empty:
        st.markdown("---")
        st.subheader("📡 Leading Indicators — What Predicts Future Bugs?")
        st.caption(
            "These are the current bug signals most strongly correlated with future bug counts. "
            "A high positive value means: 'when this signal is elevated now, more bugs tend to follow next version.'"
        )

        li = pred_leading_df.copy()
        li["pearson_r"] = pd.to_numeric(li["pearson_r"], errors="coerce").fillna(0)
        li["abs_r"]     = li["pearson_r"].abs()
        li = li.sort_values("abs_r", ascending=False).head(12)

        li["display_label"] = li.get("label", li.get("feature", li.index)).fillna(li.get("feature", ""))

        li["direction"] = li["pearson_r"].apply(
            lambda r: "📈 More now → more bugs later" if r > 0 else "📉 More now → fewer bugs later"
        )
        li["strength"]  = li["abs_r"].apply(
            lambda r: "Strong" if r >= 0.5 else "Moderate" if r >= 0.3 else "Weak"
        )

        fig_li = px.bar(
            li,
            x="pearson_r",
            y="display_label",
            orientation="h",
            color="pearson_r",
            color_continuous_scale=["#22c55e", "#f3f4f6", "#ef4444"],
            range_color=[-1, 1],
            hover_data={"pearson_r": ":.3f", "direction": True, "strength": True,
                        "display_label": False},
            labels={"pearson_r": "Correlation (−1 to +1)",
                    "display_label": "Signal",
                    "direction": "Direction",
                    "strength": "Strength"},
            text="pearson_r",
        )
        fig_li.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_li.update_layout(
            height=max(320, len(li) * 36),
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            margin=dict(l=10, r=60, t=10, b=10),
        )
        fig_li.add_vline(x=0, line_color="gray", line_width=1)
        st.plotly_chart(fig_li, width='stretch')

        with st.expander("📋 Full leading indicators table"):
            li_display = li[["display_label", "pearson_r", "direction", "strength"]].copy()
            li_display.columns = ["Signal", "Correlation", "Direction", "Strength"]
            show_dataframe(li_display, hide_index=True, width='stretch')

    # ── Cross-tab cluster callout ─────────────────────────────────────────
    # Placed after leading indicators — contextual link, not an urgent alert.
    if cluster_sum_df is not None and not cluster_sum_df.empty and "cluster_trend" in cluster_sum_df.columns:
        _growing_mods = set()
        for _, _cr in cluster_sum_df[cluster_sum_df["cluster_trend"] == "growing"].iterrows():
            _mod_str = str(_cr.get("modules", ""))
            for _m in _mod_str.split(","):
                _growing_mods.add(_m.strip())
        _high_risk_mods = set(cards_df["module"].tolist())
        _overlap = _high_risk_mods & _growing_mods
        if _overlap:
            st.info(
                f"📡 **Tab 8 cross-reference:** {len(_overlap)} high-risk module(s) also have "
                f"**growing issue groups** in the cluster analysis — "
                f"{', '.join(sorted(_overlap))}. "
                f"Open **Tab 8 → Bug Clusters** to see which specific issue groups are accelerating."
            )

    # ── Advanced / Model Diagnostics ─────────────────────────────────────
    st.markdown("---")
    with st.expander("🔧 Advanced / Model Diagnostics", expanded=False):
        st.caption(
            "Count-based model outputs and diagnostic tables. "
            "These are useful for model validation but not the primary view."
        )

        # Predicted Bug Count bar chart
        st.markdown("#### 📊 Predicted Bug Count — Next Build")
        _pred_n = len(pred_df)
        if _pred_n <= 1:
            _adv_top_n = _pred_n
        else:
            _slider_min = min(1, _pred_n - 1)
            _slider_max = min(40, _pred_n)
            _slider_val = min(20, _pred_n)
            _adv_top_n = st.slider(
                "Show top N modules",
                min_value=_slider_min,
                max_value=_slider_max,
                value=_slider_val,
                key="pred_bar_n_adv",
            )
        bar_pred = pred_df.head(_adv_top_n).copy()
        bar_pred["risk_level"] = pd.Categorical(
            bar_pred["risk_level"].astype(str), categories=RISK_ORDER, ordered=True
        )
        bar_pred = bar_pred.sort_values(["risk_level", "predicted"], ascending=[True, False])

        hover_extra = {}
        if "target" in bar_pred.columns:
            hover_extra["target"] = True
        if "dominant_bug_type" in bar_pred.columns:
            hover_extra["dominant_bug_type"] = True

        fig_pred = px.bar(
            bar_pred,
            x="module",
            y="predicted",
            color="risk_level",
            color_discrete_map=FORECAST_COLORS,
            category_orders={"risk_level": RISK_ORDER},
            hover_data={"predicted": ":.1f", "risk_level": True, **hover_extra},
            labels={"module": "Module", "predicted": "Predicted bugs (next version)",
                    "risk_level": "Risk level", "target": "Actual (last version)",
                    "dominant_bug_type": "Typical bug type"},
            text="predicted",
        )
        fig_pred.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_pred.update_layout(
            height=460,
            xaxis_tickangle=-35,
            showlegend=True,
            legend_title_text="Risk level",
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_pred, width='stretch')

        # Actual vs Predicted comparison
        if "target" in pred_df.columns:
            st.markdown("#### 📈 Actual vs Predicted (last known version)")
            avp = pred_df.head(_adv_top_n)[["module", "target", "predicted", "risk_level"]].copy()
            avp["target"]    = pd.to_numeric(avp["target"],    errors="coerce").fillna(0)
            avp["predicted"] = pd.to_numeric(avp["predicted"], errors="coerce").fillna(0)
            avp_melt = avp.melt(id_vars="module", value_vars=["target", "predicted"],
                                var_name="Type", value_name="Bugs")
            avp_melt["Type"] = avp_melt["Type"].map(
                {"target": "Actual (last version)", "predicted": "Forecast (next version)"}
            )
            fig_avp = px.bar(
                avp_melt, x="module", y="Bugs", color="Type", barmode="group",
                color_discrete_map={
                    "Actual (last version)":     "#6366f1",
                    "Forecast (next version)":   "#f97316",
                },
                labels={"module": "Module", "Bugs": "Bug count"},
            )
            fig_avp.update_layout(height=380, xaxis_tickangle=-35,
                                  legend_title_text="", margin=dict(t=10, b=10))
            st.plotly_chart(fig_avp, width='stretch')

        # Module Signals Table — all numeric signals from predict_defects.py
        _sig_cols = [c for c in [
            "module", "predicted", "predicted_stratified",
            "priority_score", "heatmap_quadrant", "module_cluster_velocity",
            "composite_risk", "risk_level",
            "severity_escalation", "builds_since_last_crit",
            "crit_ratio", "new_module", "cross_module_spike",
            "total_historical_bugs",
            "dominant_bug_type", "leading_signal",
        ] if c in pred_df.columns]
        _have_new_signals = any(c in pred_df.columns
                                for c in ["severity_escalation", "builds_since_last_crit"])
        if _have_new_signals:
            st.markdown("#### 📊 Module Signals Table")
            _sig_df = pred_df[_sig_cols].copy()
            if _ent_source_df is not None and not _ent_source_df.empty:
                _sig_df = _sig_df.merge(
                    _ent_source_df[["module", "cluster_entropy"]], on="module", how="left")
            _col_labels = {
                "module": "Module",
                "predicted": "Forecast (global)",
                "predicted_stratified": "Forecast (stratified)",
                "priority_score": "Priority (blended)",
                "heatmap_quadrant": "Heatmap quadrant",
                "module_cluster_velocity": "Cluster velocity",
                "composite_risk": "Risk score",
                "risk_level": "Risk",
                "severity_escalation": "Sev. escalation",
                "builds_since_last_crit": "Versions since S1",
                "crit_ratio": "S1 ratio",
                "new_module": "New module?",
                "cross_module_spike": "Cross-module spike",
                "total_historical_bugs": "Total historical bugs",
                "dominant_bug_type": "Typical bug type",
                "leading_signal": "Leading signal",
                "cluster_entropy": "Issue Group breadth",
            }
            _sig_df = _sig_df.rename(columns=_col_labels)
            show_dataframe(_sig_df, hide_index=True, width='stretch')

        # Full predictions table
        st.markdown("#### 📋 Full Predictions Table")
        _all_pred_cols = [c for c in [
            "module", "predicted", "predicted_stratified", "target",
            "priority_score", "heatmap_quadrant", "module_cluster_velocity",
            "composite_risk", "risk_level",
            "dominant_bug_type", "leading_signal",
            "severity_escalation", "builds_since_last_crit",
            "crit_ratio", "new_module", "cross_module_spike",
        ] if c in pred_df.columns]
        disp_pred = pred_df[_all_pred_cols].copy()
        _pred_col_labels = {
            "module": "Module",
            "predicted": "Forecast (global)",
            "predicted_stratified": "Forecast (stratified)",
            "target": "Actual (last version)",
            "priority_score": "Priority (blended)",
            "heatmap_quadrant": "Heatmap quadrant",
            "module_cluster_velocity": "Cluster velocity",
            "composite_risk": "Risk score",
            "risk_level": "Risk level",
            "dominant_bug_type": "Typical bug type",
            "leading_signal": "Leading signal",
            "severity_escalation": "Sev. escalation",
            "builds_since_last_crit": "Builds since S1",
            "crit_ratio": "S1 ratio (last 3 versions)",
            "new_module": "New module?",
            "cross_module_spike": "Cross-module spike",
        }
        disp_pred = disp_pred.rename(columns=_pred_col_labels)
        show_dataframe(disp_pred, hide_index=True, width='stretch')

        if pred_cluster_df is not None and not pred_cluster_df.empty:
            st.markdown("**Bug-type predictions by cluster:**")
            show_dataframe(
                pred_cluster_df[["module", "cluster_label", "historical_pct", "predicted_count"]]
                .rename(columns={
                    "module": "Module",
                    "cluster_label": "Issue group",
                    "historical_pct": "Historical %",
                    "predicted_count": "Predicted bugs",
                }),
                hide_index=True, width='stretch',
            )

        # ── Feature Importance — from _predictions_importance.csv ────────────
        # Complements leading_indicators: importance = what the MODEL weights most;
        # leading_indicators = what features are correlated with outcomes in the data.
        if pred_importance_df is not None and not pred_importance_df.empty:
            st.markdown("#### 🧠 Model Feature Importances")
            st.caption(
                "How much each feature influenced the model's predictions (GradientBoostingRegressor "
                "feature importances, summing to 1.0). High importance = the model leaned on this "
                "signal heavily. Compare with **Leading Indicators** above: a feature can be highly "
                "important to the model but have low Pearson r if its relationship with bugs is "
                "non-linear."
            )
            _imp = pred_importance_df.copy()
            # Legacy CSV (Series.to_csv with no name) loads as ["Unnamed: 0", "0"];
            # new CSV has an explicit "feature" column.
            if "feature" not in _imp.columns:
                unnamed = [c for c in _imp.columns if str(c).startswith("Unnamed")]
                if unnamed:
                    _imp = _imp.rename(columns={unnamed[0]: "feature"})
            if "feature" not in _imp.columns:
                non_numeric = [c for c in _imp.columns
                               if not pd.api.types.is_numeric_dtype(_imp[c])]
                if non_numeric:
                    _imp = _imp.rename(columns={non_numeric[0]: "feature"})
            imp_col = next((c for c in _imp.columns if c != "feature" and
                            pd.api.types.is_numeric_dtype(_imp[c])), None)
            if "feature" in _imp.columns and imp_col:
                _imp = _imp[["feature", imp_col]].copy()
                _imp[imp_col] = pd.to_numeric(_imp[imp_col], errors="coerce").fillna(0)
                _imp = _imp.sort_values(imp_col, ascending=False).head(20)
                # Map feature names to human-readable labels using predict_defects' label dict
                _FEAT_LABELS_T9 = {
                    "crit_1": "Critical-bug momentum (last version)",
                    "crit_2": "Critical-bug momentum (last 2 versions)",
                    "crit_3": "Critical-bug momentum (last 3 versions)",
                    "bugs_1": "Bug-count momentum (last version)",
                    "bugs_2": "Bug-count momentum (last 2 versions)",
                    "bugs_3": "Bug-count momentum (last 3 versions)",
                    "sev_1":  "Severity-weighted momentum (last version)",
                    "sev_2":  "Severity-weighted momentum (last 2 versions)",
                    "sev_3":  "Severity-weighted momentum (last 3 versions)",
                    "trend":  "Upward bug-count trend",
                    "severity_escalation":    "Severity escalation (→S1)",
                    "builds_since_last_crit": "Versions since last critical",
                    "cluster_entropy_2":      "Bug-issue group diversity (last 2 versions)",
                    "cluster_entropy_3":      "Bug-issue group diversity (last 3 versions)",
                    "top_cluster_velocity":   "Fastest-growing issue group velocity",
                    "crit_ratio":             "S1 bug proportion (last 3 versions)",
                    "new_module":             "New module flag",
                    "cross_module_spike":     "Correlated cross-module spike",
                    "total_historical_bugs":  "Total historical bug count",
                }
                _imp["label"] = _imp["feature"].map(_FEAT_LABELS_T9).fillna(_imp["feature"])
                fig_imp = px.bar(
                    _imp,
                    x=imp_col, y="label", orientation="h",
                    labels={imp_col: "Feature importance (0–1)", "label": "Feature"},
                    color=imp_col,
                    color_continuous_scale=["#e2e8f0", "#6366f1"],
                    text=_imp[imp_col].apply(lambda v: f"{v:.3f}"),
                )
                fig_imp.update_traces(textposition="outside")
                fig_imp.update_layout(
                    height=max(300, len(_imp) * 30),
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=60, t=10, b=10),
                )
                st.plotly_chart(fig_imp, width='stretch')


# =====================================================================
# TAB 10 – Release Pulse  (in-progress bugs → combined prediction view)
# =====================================================================

elif active_tab == "🎯 Release Pulse":
    import re as _re
    import requests as _requests

    _P_RISK_ICONS  = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
    _P_CONF_BADGES = {"high": "⬆️ High conf.", "medium": "↔️ Med. conf.", "low": "⬇️ Low conf."}
    _OLLAMA_BASE   = "http://localhost:11434"
    # Keep the dashboard's own generation on the same model as the pipeline.
    _PULSE_OLLAMA_MODEL = os.environ.get("PDRI_OLLAMA_MODEL", "qwen3:4b")
    try:
        from ollama_compat import (
            extract_json as _ollama_extract_json,
            response_text as _ollama_response_text,
        )
    except ModuleNotFoundError:
        from scripts.ollama_compat import (
            extract_json as _ollama_extract_json,
            response_text as _ollama_response_text,
        )

    st.header("🎯 Release Pulse — What to Test Right Now")

    with st.expander("📖 How to read this tab", expanded=False):
        st.markdown("""
**One-line summary:** *"Which modules have in-flight bugs this version, and what should QA verify — combining live signals with ML predictions?"*

**RDResolved** — RD has applied a fix; awaiting code review, not yet in a build.
**TRCreated** — a **T**esting **R**equest now covers the fix: it is built and
waiting for QA to verify it. This is the last stage before Close.

The observed order is `Assigned → RDResolved → NeedCodeReview → CodeReviewed →
BuildResolved → TRCreated → Close`.

These statuses mark work **not yet QA-verified** for the current version.
The tab always reads the full unfiltered dataset — sidebar Status/Severity filters do not affect it.

**Why this is a better signal than historical data alone:**
TRCreated/RDResolved bugs tell you exactly *what is being changed right now*, so regression risk is immediate and concrete. This tab blends:
1. **In-flight signal** — modules actively being touched (TRCreated/RDResolved counts)
2. **ML predictions** — historical scenarios from the Defect Forecast (Tab 9), if loaded
3. **FMEA risk** — long-term risk quadrant from the Risk Heatmap (Tab 7)
4. **Ollama synthesis** — combines all three into targeted, version-specific test recommendations

**Risk levels** — derived from ML classifier probability (same as Tab 9) when forecast data is loaded; otherwise estimated from FMEA risk score:

| Risk | ML P(S1) | FMEA fallback |
|------|----------|---------------|
| Critical | >20% | risk_score > 0.7 |
| High | 10–20% | risk_score > 0.4 |
| Medium | 5–10% | risk_score > 0.2 |
| Low | <5% | otherwise |

Modules with TRCreated bugs are boosted one tier — active implementation increases regression risk.
**AI recommendations** are cached per version — click once, re-renders are instant.
""")

    # ── Load unfiltered bug data ───────────────────────────────────────────
    _pulse_fp = st.session_state.get("fp_bugs", "")
    if not _pulse_fp or not Path(_pulse_fp).exists():
        st.warning("Bug data not loaded. Configure Step 1 in the sidebar first.")
        st.stop()

    df_raw_pulse = load_csv(_pulse_fp)
    if "Create Date" in df_raw_pulse.columns:
        df_raw_pulse["Create Date"] = pd.to_datetime(df_raw_pulse["Create Date"], errors="coerce")
    if "Closed Date" in df_raw_pulse.columns:
        df_raw_pulse["Closed Date"] = pd.to_datetime(df_raw_pulse["Closed Date"], errors="coerce")
    if "parsed_module" in df_raw_pulse.columns:
        df_raw_pulse["parsed_module"] = df_raw_pulse["parsed_module"].apply(
            lambda x: normalise_module(x) if pd.notna(x) else x
        )

    # BugCode → Short Description lookup, used everywhere on this tab to display
    # the real bug description next to a BugCode (avoids relying on potentially
    # misaligned `source_bug_examples` from older prediction CSVs).
    _bc_col_pulse = next(
        (c for c in df_raw_pulse.columns if "bugcode" in c.lower()), None
    )
    _desc_col_lookup = (
        "Short Description" if "Short Description" in df_raw_pulse.columns
        else ("parsed_description" if "parsed_description" in df_raw_pulse.columns else None)
    )
    _bc_to_desc_pulse: dict[str, str] = {}
    if _bc_col_pulse and _desc_col_lookup:
        for _bc, _d in zip(df_raw_pulse[_bc_col_pulse], df_raw_pulse[_desc_col_lookup]):
            if pd.notna(_bc) and pd.notna(_d) and str(_bc).strip():
                _bc_to_desc_pulse.setdefault(str(_bc).strip(), str(_d).strip())

    PULSE_STATUSES = {"TRCreated", "RDResolved"}

    # ── Find newest version ────────────────────────────────────────────────
    if "parsed_version" not in df_raw_pulse.columns or "Create Date" not in df_raw_pulse.columns:
        st.error("Bug data missing `parsed_version` or `Create Date` columns.")
        st.stop()

    _ver_dates = (
        df_raw_pulse.dropna(subset=["parsed_version", "Create Date"])
        .groupby("parsed_version")["Create Date"].max()
    )
    if _ver_dates.empty:
        st.warning("No version/date data found.")
        st.stop()
    newest_version = str(_ver_dates.idxmax())

    # ── Close-date filter (default: today) ─────────────────────────────────
    # Bugs that are created and immediately closed never appear in the
    # TRCreated/RDResolved-only view. Adding "Closed Date == <date>" as a
    # second inclusion branch surfaces those — usually the newest verified
    # fixes.
    from datetime import date as _date, timedelta as _timedelta
    _default_closed_on = _date.today()
    _has_closed_col = "Closed Date" in df_raw_pulse.columns
    if _has_closed_col:
        _closed_on = st.date_input(
            "Include bugs closed on:",
            value=_default_closed_on,
            help="Bugs with this Closed Date are added to the TRCreated/RDResolved cohort. "
                 "Defaults to today.",
            key="pulse_closed_on",
        )
    else:
        _closed_on = None

    # ── Time-travel: load the snapshot for the picked date, if one exists ──
    # When `<picked_date>.json` exists in release_pulse_history/, the page
    # renders the world as recorded that morning instead of live state:
    #   • In-flight cohort  ← snapshot.inflight_bugcodes_by_module
    #   • AI scenarios      ← snapshot.scenarios
    # Snapshots written before inflight_bugcodes_by_module was added still
    # supply scenarios but cannot override the in-flight cohort.
    _pulse_snap: dict | None = None
    _snap_inflight_bcs: set[str] = set()
    if _has_closed_col and _closed_on is not None:
        _pulse_snap_fp = (
            Path(_product_dir) / "predictions" / "release_pulse_history"
            / f"{_closed_on.isoformat()}.json"
        )
        if _pulse_snap_fp.exists():
            try:
                _pulse_snap = json.loads(_pulse_snap_fp.read_text())
            except Exception as _snap_err:
                st.warning(f"Could not read snapshot for {_closed_on.isoformat()}: {_snap_err}")
                _pulse_snap = None
            if _pulse_snap is not None:
                for _bcs in (_pulse_snap.get("inflight_bugcodes_by_module") or {}).values():
                    for _b in (_bcs or []):
                        _b = str(_b).strip()
                        if _b:
                            _snap_inflight_bcs.add(_b)

    # Pin to the snapshot's version when time-traveling, so the page reflects
    # the release that was current on the picked date.
    if _pulse_snap is not None:
        _snap_v = str(_pulse_snap.get("version", "")).strip()
        if _snap_v:
            newest_version = _snap_v

    if _pulse_snap is not None:
        _snap_ver = str(_pulse_snap.get("version", "")) or "?"
        _snap_gen = str(_pulse_snap.get("generated_at", ""))[:10] or _closed_on.isoformat()
        _has_full_travel = bool(
            _snap_inflight_bcs
            and (_pulse_snap.get("risk_classifications_by_module") or {})
        )
        if _has_full_travel:
            st.success(
                f"📸 Showing snapshot from **{_snap_gen}** (version {_snap_ver}) — "
                "cohort, metrics, risk levels, FMEA quadrants, and scenarios all reflect that day."
            )
        elif _snap_inflight_bcs:
            st.info(
                f"📸 Snapshot from **{_snap_gen}** (version {_snap_ver}) — cohort and scenarios "
                "reflect that day, but **risk classifications** still come from live data "
                "(snapshot pre-dates the risk-classification field)."
            )
        else:
            st.info(
                f"📸 Snapshot from **{_snap_gen}** (version {_snap_ver}) found but pre-dates the "
                "in-flight-cohort field — showing snapshot scenarios with **live** in-flight metrics."
            )
    elif _closed_on is not None and _closed_on != _date.today():
        st.caption(
            f"_No snapshot saved for {_closed_on.isoformat()} — showing live state. "
            "Snapshots are written daily by the 3am pipeline._"
        )

    # ── Filter pulse bugs ──────────────────────────────────────────────────
    _desc_col = "Short Description" if "Short Description" in df_raw_pulse.columns else "parsed_description"
    if _snap_inflight_bcs and "BugCode" in df_raw_pulse.columns:
        # Time-travel: in-flight cohort comes from the snapshot.
        _status_mask = (
            df_raw_pulse["BugCode"].astype(str).str.strip().isin(_snap_inflight_bcs)
        )
    else:
        _status_mask = df_raw_pulse["Status"].isin(PULSE_STATUSES)
    if _has_closed_col and _closed_on is not None:
        _closed_mask = df_raw_pulse["Closed Date"].dt.date == _closed_on
        _include_mask = _status_mask | _closed_mask
    else:
        _closed_mask = pd.Series(False, index=df_raw_pulse.index)
        _include_mask = _status_mask
    df_pulse = df_raw_pulse[
        _include_mask &
        (df_raw_pulse["parsed_version"] == newest_version)
    ].copy()

    # Time-travel: restore each in-flight bug's Status as it was when the
    # snapshot was taken (the live CSV has since updated it). Without this,
    # bugs that have moved to Close/RDResolved since snapshot time would
    # under-count the TRCreated metric.
    if _pulse_snap is not None and "BugCode" in df_pulse.columns:
        _status_map = _pulse_snap.get("inflight_status_by_bugcode") or {}
        if _status_map:
            _bc_series = df_pulse["BugCode"].astype(str).str.strip()
            _hits = _bc_series.isin(_status_map)
            df_pulse.loc[_hits, "Status"] = _bc_series[_hits].map(_status_map)

    # Tag each row with its inclusion reason so per-module aggregation can
    # report "closed on the chosen date" separately from TRCreated/RDResolved.
    if _has_closed_col and _closed_on is not None:
        df_pulse["_in_closed_cohort"] = (
            df_pulse["Closed Date"].dt.date == _closed_on
        ).fillna(False)
    else:
        df_pulse["_in_closed_cohort"] = False

    # ── Merge risk scores ──────────────────────────────────────────────────
    # Time-travel: when the snapshot recorded per-module risk classifications,
    # merge from those instead of the live risk register so quadrant + score
    # reflect the picked date.
    _snap_risk_cls = (_pulse_snap or {}).get("risk_classifications_by_module") if _pulse_snap else None
    if _snap_risk_cls and "parsed_module" in df_pulse.columns:
        _snap_risk_df = pd.DataFrame([
            {"parsed_module": m, "quadrant": c.get("quadrant", "Unknown"),
             "risk_score_final": c.get("risk_score", 0.0)}
            for m, c in _snap_risk_cls.items()
        ])
        df_pulse = df_pulse.merge(_snap_risk_df, on="parsed_module", how="left")
    elif risk_available and "parsed_module" in df_pulse.columns:
        _rc = [c for c in ["parsed_module", "quadrant", "risk_score_final",
                            "impact_score", "detectability_score"]
               if c in risk_df_dedup.columns]
        df_pulse = df_pulse.merge(risk_df_dedup[_rc], on="parsed_module", how="left")
    df_pulse["quadrant"]        = df_pulse.get("quadrant",        pd.Series()).fillna("Unknown")
    df_pulse["risk_score_final"] = pd.to_numeric(
        df_pulse.get("risk_score_final", pd.Series()), errors="coerce"
    ).fillna(0.0)

    # ── Build per-module summary ───────────────────────────────────────────
    _pulse_grp = (
        df_pulse.groupby("parsed_module")
        .agg(
            quadrant       =("quadrant",        "first"),
            risk_score     =("risk_score_final", "first"),
            trc_count      =("Status", lambda s: (s == "TRCreated").sum()),
            rdr_count      =("Status", lambda s: (s == "RDResolved").sum()),
            closed_count   =("_in_closed_cohort", "sum"),
            total          =("Status",           "count"),
            sample_descs   =(_desc_col, lambda x: list(x.dropna().head(10))),
        )
        .reset_index()
        .sort_values(["risk_score", "total"], ascending=[False, False])
    )
    _pulse_grp["closed_count"] = _pulse_grp["closed_count"].fillna(0).astype(int)

    # ── Derive risk level ──────────────────────────────────────────────────
    # Prefer ML predictions from pred_df (Tab 9 model); fall back to FMEA score.
    # Boost one tier for modules with TRCreated bugs (active implementation risk).
    _RISK_ORDER = ["Low", "Medium", "High", "Critical"]

    def _boost_risk(level: str, trc: int) -> str:
        if trc == 0:
            return level
        idx = _RISK_ORDER.index(level) if level in _RISK_ORDER else 0
        return _RISK_ORDER[min(idx + 1, len(_RISK_ORDER) - 1)]

    def _score_to_risk(score: float) -> str:
        if score > 0.7:  return "Critical"
        if score > 0.4:  return "High"
        if score > 0.2:  return "Medium"
        return "Low"

    _pred_rl_map: dict[str, str] = {}
    _pred_leading_map: dict[str, str] = {}
    # Time-travel: prefer the snapshot's recorded risk classifications.
    if _snap_risk_cls:
        for _m, _c in _snap_risk_cls.items():
            _pred_rl_map[str(_m)] = str(_c.get("risk_level", "Low"))
            _pred_leading_map[str(_m)] = str(_c.get("leading_signal", ""))
    elif pred_df is not None and "module" in pred_df.columns and "risk_level" in pred_df.columns:
        for _, _pr in pred_df.iterrows():
            _pred_rl_map[str(_pr["module"])] = str(_pr["risk_level"])
            if "leading_signal" in pred_df.columns:
                _pred_leading_map[str(_pr["module"])] = str(_pr.get("leading_signal", ""))

    def _module_risk(mod: str, score: float, trc: int) -> str:
        base = _pred_rl_map.get(mod, _score_to_risk(score))
        return _boost_risk(base, trc)

    # Guard against empty DataFrame: df.apply(axis=1) on empty df returns the
    # empty DataFrame itself, which can't be assigned to a single column.
    if _pulse_grp.empty:
        _pulse_grp["risk_level"] = pd.Series(dtype=str)
    else:
        _pulse_grp["risk_level"] = _pulse_grp.apply(
            lambda r: _module_risk(r["parsed_module"], r["risk_score"], int(r["trc_count"])), axis=1
        )
    _pulse_grp = _pulse_grp.sort_values(
        ["risk_level", "risk_score", "total"],
        key=lambda c: c.map(_RISK_ORDER.index) if c.name == "risk_level" else c,
        ascending=[False, False, False],
    )

    # ── Headline metrics ───────────────────────────────────────────────────
    _n_trc      = int(df_pulse["Status"].eq("TRCreated").sum())
    _n_rdr      = int(df_pulse["Status"].eq("RDResolved").sum())
    _n_closed   = int(df_pulse["_in_closed_cohort"].sum())
    _n_mods     = int(_pulse_grp["parsed_module"].nunique())
    _n_critical = int((_pulse_grp["risk_level"] == "Critical").sum())

    st.markdown(f"#### Version **{newest_version}** — in-progress bugs")
    _m1, _m2, _m3, _m4, _m5 = st.columns(5)
    _m1.metric("TRCreated",        _n_trc,
               help="Bugs in Technical Review — actively being implemented")
    _m2.metric("RDResolved",       _n_rdr,
               help="Bugs resolved at RD level — fix applied, not yet QA-verified")
    _m3.metric(
        f"Closed {_closed_on.isoformat() if _closed_on else '—'}",
        _n_closed,
        help="Bugs whose Closed Date matches the date picker above. "
             "Includes bugs that were created and closed in the same window.",
    )
    _m4.metric("Modules affected", _n_mods)
    _m5.metric("🔴 Critical modules", _n_critical,
               help="Modules where risk_level=Critical (ML or FMEA) and/or active TRCreated bugs")

    if df_pulse.empty:
        st.info(
            f"No bugs matched the filter for version **{newest_version}** "
            f"(TRCreated/RDResolved or Closed on {_closed_on.isoformat() if _closed_on else '—'})."
        )
        st.stop()

    # ── "What to Test Right Now" — primary section ─────────────────────────
    st.markdown("---")
    st.subheader("🎯 What to Test Right Now")

    # Gather historical scenarios. In time-travel mode, read from the
    # snapshot's recorded history; otherwise from the live predictions CSV.
    _pulse_mods_set = set(_pulse_grp["parsed_module"])
    _hist_sc: dict[str, list[dict]] = {}   # module → list of scenario dicts
    _snap_hist_sc = (_pulse_snap or {}).get("historical_scenarios_by_module") if _pulse_snap else None
    if _snap_hist_sc:
        for _mod, _recs in _snap_hist_sc.items():
            if _mod in _pulse_mods_set:
                _hist_sc[str(_mod)] = list(_recs or [])
    elif pred_scenario_df is not None and not pred_scenario_df.empty:
        _sc_mod_col = "module" if "module" in pred_scenario_df.columns else None
        if _sc_mod_col:
            for _mod, _msc in pred_scenario_df.groupby(_sc_mod_col):
                if _mod in _pulse_mods_set:
                    _hist_sc[_mod] = _msc.to_dict("records")

    _has_hist = bool(_hist_sc)
    _legend_parts = []
    if _has_hist:         _legend_parts.append("📎 Historical pattern (from Defect Forecast)")
    _legend_parts.append("🔄 In-flight context (live TRCreated/RDResolved bugs)")
    st.caption(
        "Scenarios grouped by risk level — highest urgency first. "
        "Historical patterns are pre-computed by the ML model (Tab 9); "
        "in-flight scenarios are generated on-demand from active bugs."
        + f"\n\n_{' · '.join(_legend_parts)}_"
    )

    # Display historical scenarios grouped by risk level (same layout as Tab 9)
    for _rl in ["Critical", "High", "Medium", "Low"]:
        _rl_mods = _pulse_grp[_pulse_grp["risk_level"] == _rl]
        if _rl_mods.empty:
            continue
        _rl_icon = _P_RISK_ICONS.get(_rl, "⚪")
        st.markdown(f"#### {_rl_icon} {_rl} Risk")
        for _, _mr in _rl_mods.iterrows():
            _mod = _mr["parsed_module"]
            _trc = int(_mr["trc_count"])
            _rdr = int(_mr["rdr_count"])
            _cls = int(_mr.get("closed_count", 0))
            _badge_parts = [f"TRCreated: {_trc}", f"RDResolved: {_rdr}"]
            if _cls > 0:
                _badge_parts.append(f"Closed {_closed_on.isoformat()}: {_cls}")
            _badge = " · ".join(_badge_parts)
            st.markdown(f"**{_mod}** — _{_badge}_")
            # Historical scenarios from ML model
            for _hs in _hist_sc.get(_mod, [])[:3]:
                _hconf  = _P_CONF_BADGES.get(str(_hs.get("confidence", "medium")), "⬇️ Low conf.")
                _htext  = str(_hs.get("scenario_text", ""))
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{_hconf} 📎 Historical — {_htext}",
                            unsafe_allow_html=True)
                # Source BugCodes — clickable ECL links for traceability
                _hs_codes = str(_hs.get("source_bug_codes", "")).strip()
                if _hs_codes and _hs_codes not in ("nan", ""):
                    _bc_links = []
                    for _bc in _hs_codes.split(" | "):
                        _bc = _bc.strip()
                        if not _bc:
                            continue
                        _bc_links.append(f"[{_bc}]({ECL_BUG_URL.format(bug_code=_bc)})")
                    if _bc_links:
                        st.markdown(
                            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                            f"_Source bugs:_ {' · '.join(_bc_links)}",
                            unsafe_allow_html=True,
                        )
            # In-flight preview: top bug description as a stub until AI runs
            _samples = _mr["sample_descs"][:2]
            for _sd in _samples:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🔄 In-flight — {_sd}",
                            unsafe_allow_html=True)
        st.markdown("")

    # ── Module breakdown table ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Module Breakdown")
    _closed_label = (
        f"Closed {_closed_on.isoformat()}" if _closed_on else "Closed"
    )
    _display_grp = _pulse_grp[["parsed_module", "risk_level", "quadrant", "risk_score",
                                "trc_count", "rdr_count", "closed_count", "total"]].rename(columns={
        "parsed_module": "Module", "risk_level": "Risk Level",
        "quadrant": "FMEA Quadrant", "risk_score": "Risk Score",
        "trc_count": "TRCreated", "rdr_count": "RDResolved",
        "closed_count": _closed_label, "total": "Total",
    })
    show_dataframe(
        _display_grp,
        hide_index=True,
        column_config={"Risk Score": st.column_config.NumberColumn(format="%.3f")},
        width="stretch",
    )

    # ── Raw bug list ───────────────────────────────────────────────────────
    _show_cols = [c for c in ["BugCode", "Status", _desc_col, "severity_label",
                               "Handler", "parsed_module", "quadrant", "Closed Date"]
                  if c in df_pulse.columns]
    with st.expander(
        f"📋 All {len(df_pulse)} bugs in cohort "
        f"(TRCreated/RDResolved or Closed on {_closed_on.isoformat() if _closed_on else '—'})",
        expanded=False,
    ):
        show_dataframe(df_pulse[_show_cols], hide_index=True, width="stretch")

    # ── Ollama synthesis ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 AI-Synthesized Test Scenarios")

    # Scope cache by date when time-traveling so snapshot scenarios don't
    # leak into the live cache when the user navigates back to today.
    if _pulse_snap is not None and _closed_on is not None:
        _cache_key = f"pulse_ai_scenarios_{newest_version}_{_closed_on.isoformat()}"
    else:
        _cache_key = f"pulse_ai_scenarios_{newest_version}"

    def _ollama_reachable() -> bool:
        try:
            r = _requests.get(f"{_OLLAMA_BASE}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def _call_ollama_pulse_scenarios(
        module: str, quadrant: str, risk_level: str, risk_score: float,
        leading_signal: str, inflight_descs: list[str],
        hist_scenarios: list[str], version: str,
    ) -> list[dict]:
        _inflight_bullets = "\n".join(f"- {d}" for d in inflight_descs[:12])
        _hist_bullets = (
            "\n".join(f"- {s}" for s in hist_scenarios[:5])
            if hist_scenarios else "  (no historical predictions loaded)"
        )
        _prompt = (
            f"You are a QA lead reviewing bugs currently in active development for a video editing iOS app.\n\n"
            f"Version: {version}\n"
            f"Module: {module}\n"
            f"ML-predicted risk level: {risk_level}\n"
            f"FMEA quadrant: {quadrant} (risk score: {risk_score:.3f})\n"
            f"Strongest historical signal: {leading_signal or 'N/A'}\n\n"
            f"Active in-progress bugs (TRCreated / RDResolved — not yet QA-verified):\n"
            f"{_inflight_bullets}\n\n"
            f"ML-predicted historical risk patterns for this module:\n"
            f"{_hist_bullets}\n\n"
            f"Given the current in-flight changes AND the historical risk patterns above, "
            f"predict 3-5 specific bug scenarios QA should test. Focus on:\n"
            f"1. Regressions introduced by the in-flight changes\n"
            f"2. Edge cases around the active bug fixes\n"
            f"3. Interaction effects with the known historical risk areas\n\n"
            f"Each scenario must be a concrete, testable statement (like a real bug title). "
            f"Do NOT write vague summaries. Do NOT mention counts.\n"
            f"Return exactly this JSON object and nothing else. Do not echo "
            f"the context above:\n"
            f'{{"scenarios": [{{"scenario": "...", "confidence": "high|medium|low", '
            f'"based_on": "in_flight|historical|both", '
            f'"explanation": "**Why likely:** [reason]. **Steps to reproduce:** [steps]. '
            f'**What to verify:** [expected vs actual]."}}]}}'
        )
        try:
            resp = _requests.post(
                f"{_OLLAMA_BASE}/api/generate",
                json={
                    "model": _PULSE_OLLAMA_MODEL,
                    "prompt": _prompt,
                    "stream": False,
                    "format": "json",
                    "think": False,
                    "options": {"temperature": 0.3, "num_predict": 2000},
                },
                timeout=120,
            )
            resp.raise_for_status()
            raw = _ollama_response_text(resp.json())
            parsed = _ollama_extract_json(raw)
            if isinstance(parsed, list):
                return parsed
            # Sometimes model wraps in a dict
            for key in ("scenarios", "results", "predictions"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            return []
        except Exception:
            try:
                salvaged = _ollama_extract_json(raw, expect="array")
                if isinstance(salvaged, list):
                    return salvaged
            except Exception:
                pass
            return []

    # In time-travel mode the snapshot's scenarios are the source of truth —
    # they cannot be regenerated and they pin the dashboard to that day.
    _snap_scenarios_dict = (
        (_pulse_snap or {}).get("scenarios") if _pulse_snap else None
    )
    _in_snapshot_mode = bool(_snap_scenarios_dict)

    if _in_snapshot_mode:
        # Bypass the session cache entirely so navigating between snapshot
        # dates always re-reads the file (cheap; the JSON is small).
        st.session_state[_cache_key] = _snap_scenarios_dict
    else:
        # Hydrate the in-memory cache from the pre-generated pipeline output
        # if one exists for this version. This is what makes the section
        # render immediately after a fresh `refresh_pipeline.sh` run.
        _pulse_pregen_fp = Path(_product_dir) / "predictions" / "release_pulse_scenarios.json"
        if _pulse_pregen_fp.exists() and not st.session_state.get(_cache_key):
            try:
                _pregen = json.loads(_pulse_pregen_fp.read_text())
                if str(_pregen.get("version", "")) == str(newest_version):
                    _pregen_scenarios = _pregen.get("scenarios", {}) or {}
                    if _pregen_scenarios:
                        st.session_state[_cache_key] = _pregen_scenarios
            except Exception:
                pass

    _ollama_up = _ollama_reachable()

    # Generate live only when we have no cache AND Ollama is reachable AND
    # we are not viewing a historical snapshot (regenerating in snapshot mode
    # would defeat the point of time-travel).
    if not _in_snapshot_mode and not st.session_state.get(_cache_key) and _ollama_up:
        _top_mods = _pulse_grp.head(10)
        _ai_sc: dict[str, list[dict]] = {}
        _prog = st.progress(0, text="Generating scenarios…")
        for _i, (_idx, _mr) in enumerate((_top_mods.iterrows()), 1):
            _mod = _mr["parsed_module"]
            _hist_txts = [str(s.get("scenario_text", ""))
                          for s in _hist_sc.get(_mod, [])]
            _ai_sc[_mod] = _call_ollama_pulse_scenarios(
                module=_mod,
                quadrant=str(_mr["quadrant"]),
                risk_level=str(_mr["risk_level"]),
                risk_score=float(_mr["risk_score"]),
                leading_signal=_pred_leading_map.get(_mod, ""),
                inflight_descs=list(_mr["sample_descs"]),
                hist_scenarios=_hist_txts,
                version=newest_version,
            )
            _prog.progress(_i / len(_top_mods),
                           text=f"Processing {_mod} ({_i}/{len(_top_mods)})…")
        _prog.empty()
        st.session_state[_cache_key] = _ai_sc

    _ai_cached = st.session_state.get(_cache_key, {})

    if _ai_cached:
        if _in_snapshot_mode:
            st.caption(
                "AI scenarios loaded from the snapshot — pick today's date in the filter "
                "above to switch back to the live cache."
            )
        else:
            st.caption("AI scenarios cached for this version — click **Clear cache** to regenerate.")
            if st.button("🗑️ Clear cache", key="pulse_clear_cache"):
                st.session_state.pop(_cache_key, None)
                st.rerun()
        st.markdown("---")
        # Display per-module cards (matching Tab 9 expanded card style)
        for _rl in ["Critical", "High", "Medium", "Low"]:
            _rl_mods = _pulse_grp[_pulse_grp["risk_level"] == _rl]
            for _, _mr in _rl_mods.iterrows():
                _mod = _mr["parsed_module"]
                if _mod not in _ai_cached:
                    continue
                _rl_icon   = _P_RISK_ICONS.get(_rl, "⚪")
                _scenarios = _ai_cached[_mod]
                _trc       = int(_mr["trc_count"])
                _rdr       = int(_mr["rdr_count"])
                _leading   = _pred_leading_map.get(_mod, "")
                with st.expander(
                    f"{_rl_icon} **{_mod}** — {_rl} risk · "
                    f"TRCreated: {_trc} · RDResolved: {_rdr} · "
                    f"{len(_scenarios)} scenario(s)",
                    expanded=False,
                ):
                    _c1, _c2, _c3 = st.columns(3)
                    _c1.metric("FMEA Quadrant",  str(_mr["quadrant"]))
                    _c2.metric("Risk Score",     f"{_mr['risk_score']:.3f}")
                    _c3.metric("In-flight bugs", _trc + _rdr)
                    if _leading:
                        st.caption(f"**Strongest signal:** {_leading}")

                    if _scenarios:
                        st.markdown("**Predicted scenarios:**")
                        for _sc_i, _sc in enumerate(_scenarios, 1):
                            _conf_badge = _P_CONF_BADGES.get(
                                str(_sc.get("confidence", "medium")), "⬇️ Low conf."
                            )
                            _based_on   = str(_sc.get("based_on", ""))
                            _src_badge  = {
                                "in_flight":  "🔄 In-flight",
                                "historical": "📎 Historical",
                                "both":       "🔄📎 Combined",
                            }.get(_based_on, "🔄 In-flight")
                            _stext      = str(_sc.get("scenario", ""))
                            _expl       = str(_sc.get("explanation", ""))
                            st.markdown(
                                f"&nbsp;&nbsp;**#{_sc_i}** {_conf_badge} {_src_badge} — {_stext}",
                                unsafe_allow_html=True,
                            )
                            if _expl:
                                with st.expander("❓ Why this scenario?", expanded=False):
                                    st.markdown(_expl)
                    else:
                        st.caption("No structured scenarios returned for this module.")

                    # Show historical ML scenarios alongside for comparison
                    _hs_list = _hist_sc.get(_mod, [])
                    if _hs_list:
                        st.markdown("**Historical risk patterns (from Defect Forecast ML):**")
                        for _hs in _hs_list[:3]:
                            _hc = _P_CONF_BADGES.get(
                                str(_hs.get("confidence", "medium")), "⬇️ Low conf."
                            )
                            st.markdown(
                                f"&nbsp;&nbsp;{_hc} 📎 — {_hs.get('scenario_text', '')}",
                                unsafe_allow_html=True,
                            )
                            _hs_codes = str(_hs.get("source_bug_codes", "")).strip()
                            if _hs_codes and _hs_codes not in ("nan", ""):
                                _bc_links = [
                                    f"[{_bc.strip()}]({ECL_BUG_URL.format(bug_code=_bc.strip())})"
                                    for _bc in _hs_codes.split(" | ") if _bc.strip()
                                ]
                                if _bc_links:
                                    st.markdown(
                                        "&nbsp;&nbsp;&nbsp;&nbsp;"
                                        f"_Source bugs:_ {' · '.join(_bc_links)}",
                                        unsafe_allow_html=True,
                                    )
    else:
        # ── Heuristic fallback ─────────────────────────────────────────────
        if _ollama_up:
            st.caption(
                "No AI scenarios cached yet — re-run `./refresh_pipeline.sh` "
                "to populate the cache, or refresh this page to generate live."
            )
        else:
            st.caption(
                "Ollama is not reachable and no pre-generated cache was found. "
                "Start Ollama (`ollama serve`) and refresh, or run "
                "`./refresh_pipeline.sh` to pre-generate scenarios."
            )
        for _, _mr in _pulse_grp.head(10).iterrows():
            _mod   = _mr["parsed_module"]
            _rl    = str(_mr["risk_level"])
            _q     = str(_mr["quadrant"])
            _rs    = float(_mr["risk_score"])
            _trc   = int(_mr["trc_count"])
            _rdr   = int(_mr["rdr_count"])
            _icon  = _P_RISK_ICONS.get(_rl, "⚪")
            with st.expander(
                f"{_icon} **{_mod}** — {_rl} risk · {_q} · TRCreated: {_trc} · RDResolved: {_rdr}",
                expanded=_rl in ("Critical", "High"),
            ):
                st.markdown("**In-flight bugs (up to 5):**")
                for _b in _mr["sample_descs"][:5]:
                    st.markdown(f"- {_b}")
                _hs_list = _hist_sc.get(_mod, [])
                if _hs_list:
                    st.markdown("**Historical ML scenarios:**")
                    for _hs in _hs_list[:3]:
                        _hc = _P_CONF_BADGES.get(str(_hs.get("confidence", "medium")), "⬇️ Low conf.")
                        st.markdown(f"- {_hc} — {_hs.get('scenario_text', '')}")
                        _hs_codes = str(_hs.get("source_bug_codes", "")).strip()
                        if _hs_codes and _hs_codes not in ("nan", ""):
                            _bc_links = [
                                f"[{_bc.strip()}]({ECL_BUG_URL.format(bug_code=_bc.strip())})"
                                for _bc in _hs_codes.split(" | ") if _bc.strip()
                            ]
                            if _bc_links:
                                st.markdown(f"  _Source bugs:_ {' · '.join(_bc_links)}")
                st.caption(
                    f"Risk: **{_rl}** ({_q}, score {_rs:.3f}) — "
                    f"{'prioritise every sprint' if _rl in ('Critical', 'High') else 'include in regression pass'}."
                )

    # ── Prediction effectiveness panel ─────────────────────────────────────
    # Browses any saved snapshot in release_pulse_history/ and evaluates it
    # against bugs whose Closed Date matches the snapshot date. A scenario
    # "matches" a bug if the bug's BugCode appears in the scenario's source
    # set OR if the scenario_text shares ≥ 3 meaningful tokens with the
    # bug's Short Description.
    st.markdown("---")
    _hist_dir = Path(_product_dir) / "predictions" / "release_pulse_history"
    _yesterday = _date.today() - _timedelta(days=1)

    _available_dates: list = []
    if _hist_dir.is_dir():
        for _f in _hist_dir.glob("*.json"):
            try:
                _available_dates.append(_date.fromisoformat(_f.stem))
            except ValueError:
                continue
        _available_dates.sort(reverse=True)

    # Mirror the top "Include bugs closed on:" date into the snapshot selectbox
    # whenever it changes, so the two controls stay in sync. Session state must
    # be written before the selectbox is instantiated below.
    if _closed_on is not None and _available_dates:
        _prev_top_date = st.session_state.get("_pulse_prev_top_date")
        if _prev_top_date != _closed_on and _closed_on in _available_dates:
            st.session_state["pulse_snapshot_date"] = _closed_on
        st.session_state["_pulse_prev_top_date"] = _closed_on

    with st.expander("📊 Did predictions catch any bugs? (browse snapshots)", expanded=False):
        if not _available_dates:
            st.info(
                f"No saved snapshots yet — effectiveness becomes available after the next "
                f"pipeline run. Snapshots will be written to `{_hist_dir}`."
            )
        else:
            _sel_date = st.selectbox(
                "Evaluate snapshot from:",
                options=_available_dates,
                index=0,
                format_func=lambda d: (
                    d.isoformat() + (" (yesterday)" if d == _yesterday else "")
                ),
                key="pulse_snapshot_date",
                help=f"{len(_available_dates)} snapshot(s) available in release_pulse_history/.",
            )
            _hist_fp = _hist_dir / f"{_sel_date.isoformat()}.json"

            st.caption(
                f"Evaluates the snapshot from **{_sel_date.isoformat()}** against bugs whose "
                f"`Closed Date` is **{_sel_date.isoformat()}**. A scenario is marked ✅ matched when "
                "a closed bug's BugCode appears in the historical source set, OR when the scenario "
                "text shares **5+** meaningful tokens with the bug's Short Description **and** the "
                "bug's `parsed_module` matches the scenario's module. Bugs whose BugCode was already "
                "in the snapshot's input window (closed-on cohort or in-flight) are excluded so the "
                "snapshot can't be graded on bugs it was shown."
            )

            try:
                _yday_snapshot = json.loads(_hist_fp.read_text())
            except Exception as _e:
                st.warning(f"Could not read snapshot {_hist_fp.name}: {_e}")
                _yday_snapshot = None

            # Bugs closed on the evaluation date (independent of cohort filter above).
            if "Closed Date" in df_raw_pulse.columns:
                _closed_yday_df = df_raw_pulse[
                    df_raw_pulse["Closed Date"].dt.date == _sel_date
                ].copy()
            else:
                _closed_yday_df = df_raw_pulse.iloc[0:0].copy()

            if _yday_snapshot is None:
                pass
            elif not _yday_snapshot.get("scenarios"):
                st.info(f"Snapshot for {_sel_date.isoformat()} contains no predicted scenarios.")
            else:
                if _closed_yday_df.empty:
                    st.info(
                        f"No bugs closed on {_sel_date.isoformat()} — showing snapshot scenarios "
                        "with 0 matches."
                    )
                _STOPWORDS = {
                    "the","a","an","and","or","of","to","in","on","at","is","it","for",
                    "with","by","be","not","this","that","when","while","from","as","but",
                    "if","then","else","do","does","did","has","have","had","can","will",
                    "would","should","could","may","might","must","shall","i","you","we",
                    "they","he","she","its","their","our","my","your","cannot","cant",
                    "wont","arent","isnt","wasnt","werent",
                }
                def _tokens(s: str) -> set[str]:
                    return {
                        t for t in _re.findall(r"[a-z0-9]+", s.lower())
                        if len(t) >= 3 and t not in _STOPWORDS
                    }

                # BugCodes the snapshot already saw — excluded from evaluation so
                # a bug can't both inform a scenario and be counted as a match.
                # Falls back to closed_bugcodes only on snapshots written before
                # inflight_bugcodes_by_module was added.
                _seen_bcs: set[str] = {
                    str(b).strip() for b in (_yday_snapshot.get("closed_bugcodes") or [])
                    if str(b).strip()
                }
                _inflight_by_mod_snap: dict = (
                    _yday_snapshot.get("inflight_bugcodes_by_module") or {}
                )
                for _bcs in _inflight_by_mod_snap.values():
                    for _b in (_bcs or []):
                        _b = str(_b).strip()
                        if _b:
                            _seen_bcs.add(_b)

                # Closed bug rows — ensure BugCode/Short Description columns exist.
                _bc_col_eval = next(
                    (c for c in _closed_yday_df.columns if "bugcode" in c.lower()), None
                )
                _desc_col_eval = (
                    "Short Description" if "Short Description" in _closed_yday_df.columns
                    else "parsed_description"
                )
                _has_module_col = "parsed_module" in _closed_yday_df.columns
                _closed_records = []
                _excluded_seen = 0
                for _, _r in _closed_yday_df.iterrows():
                    _bc_val = str(_r.get(_bc_col_eval, "")).strip() if _bc_col_eval else ""
                    _desc_val = str(_r.get(_desc_col_eval, "")).strip()
                    if _bc_val and _bc_val in _seen_bcs:
                        _excluded_seen += 1
                        continue
                    _mod_val = str(_r.get("parsed_module", "")).strip() if _has_module_col else ""
                    if _bc_val or _desc_val:
                        _closed_records.append({
                            "bugcode": _bc_val,
                            "desc":    _desc_val,
                            "tokens":  _tokens(_desc_val),
                            "module":  _mod_val,
                        })

                _scenarios_by_mod = _yday_snapshot.get("scenarios", {}) or {}
                _src_codes_by_mod = _yday_snapshot.get("source_bug_codes_by_module", {}) or {}

                _total_predicted = 0
                _total_matched   = 0
                _matched_rows: list[dict] = []
                _miss_rows:    list[dict] = []
                _KEYWORD_MIN = 5

                for _mod, _scn_list in _scenarios_by_mod.items():
                    _src_codes = set(_src_codes_by_mod.get(_mod, []))
                    for _scn in (_scn_list or []):
                        _scn_text = str(_scn.get("scenario", "")).strip()
                        if not _scn_text:
                            continue
                        _total_predicted += 1
                        _scn_tokens = _tokens(_scn_text)
                        _hits = []
                        for _cr in _closed_records:
                            _direct = bool(_cr["bugcode"]) and _cr["bugcode"] in _src_codes
                            _same_module = bool(_cr["module"]) and _cr["module"] == _mod
                            _overlap = len(_scn_tokens & _cr["tokens"])
                            _keyword_hit = _same_module and _overlap >= _KEYWORD_MIN
                            if _direct or _keyword_hit:
                                _hits.append({
                                    **_cr,
                                    "method": "BugCode" if _direct else f"keyword ×{_overlap} (same module)",
                                })
                        row = {
                            "module":    _mod,
                            "scenario":  _scn_text,
                            "confidence": str(_scn.get("confidence", "")),
                            "based_on":  str(_scn.get("based_on", "")).strip().lower(),
                            "hits":      _hits,
                        }
                        if _hits:
                            _total_matched += 1
                            _matched_rows.append(row)
                        else:
                            _miss_rows.append(row)

                _excl_note = (
                    f" · excluded {_excluded_seen} bug(s) the snapshot already saw"
                    if _excluded_seen else ""
                )
                st.markdown(
                    f"**{_sel_date.isoformat()}: {_total_matched} / {_total_predicted} predicted "
                    f"scenarios matched a closed bug · {len(_closed_records)} bugs closed that day"
                    f"{_excl_note}.**"
                )

                # ── Per-scenario manual annotations ────────────────────────
                # Two flags per scenario, persisted to <date>.annotations.json
                # alongside the snapshot. The snapshot file itself stays
                # immutable so scenarios can't drift between annotation runs.
                import hashlib as _hashlib
                from datetime import datetime as _datetime
                _anno_fp = _hist_dir / f"{_sel_date.isoformat()}.annotations.json"

                def _scn_key(module: str, scenario_text: str) -> str:
                    _h = _hashlib.sha1(
                        scenario_text.encode("utf-8", errors="replace")
                    ).hexdigest()[:10]
                    return f"{module}::{_h}"

                def _load_annotations() -> dict:
                    if not _anno_fp.exists():
                        return {}
                    try:
                        return dict(json.loads(_anno_fp.read_text()).get("annotations", {}))
                    except Exception:
                        return {}

                def _save_annotations(annos: dict) -> None:
                    _anno_fp.parent.mkdir(parents=True, exist_ok=True)
                    _anno_fp.write_text(json.dumps({
                        "last_updated": _datetime.now().isoformat(timespec="seconds"),
                        "snapshot_date": _sel_date.isoformat(),
                        "annotations": annos,
                    }, indent=2))

                _annotations = _load_annotations()

                def _on_anno_toggle(sk, module, scenario, field):
                    widget_key = (
                        f"pulse_anno_{_sel_date.isoformat()}_{sk}_{field}"
                    )
                    val = bool(st.session_state.get(widget_key, False))
                    cur = _annotations.setdefault(sk, {})
                    cur["module"] = module
                    cur["scenario"] = scenario
                    cur[field] = val
                    cur["updated_at"] = _datetime.now().isoformat(timespec="seconds")
                    _save_annotations(_annotations)

                # Scenarios derived from in-flight bugs (or "both") describe
                # already-known active problems — annotation adds no signal.
                _ANNO_SKIP_SOURCES = {"in_flight", "both"}

                def _render_anno_controls(module: str, scenario: str, based_on: str = "") -> None:
                    if based_on in _ANNO_SKIP_SOURCES:
                        st.caption(
                            "&nbsp;&nbsp;&nbsp;&nbsp;_🔄 In-flight scenario — already a "
                            "known problem; no annotation needed._",
                            unsafe_allow_html=True,
                        )
                        return
                    sk = _scn_key(module, scenario)
                    cur = _annotations.get(sk, {})
                    _ac1, _ac2 = st.columns(2)
                    _ac1.toggle(
                        "✋ Findable manually",
                        value=bool(cur.get("findable_manually", False)),
                        key=f"pulse_anno_{_sel_date.isoformat()}_{sk}_findable_manually",
                        on_change=_on_anno_toggle,
                        args=(sk, module, scenario, "findable_manually"),
                        help="Tick if the feature/scenario can be exercised through the live app UI.",
                    )
                    _ac2.toggle(
                        "🐛 Actual problem",
                        value=bool(cur.get("has_actual_problems", False)),
                        key=f"pulse_anno_{_sel_date.isoformat()}_{sk}_has_actual_problems",
                        on_change=_on_anno_toggle,
                        args=(sk, module, scenario, "has_actual_problems"),
                        help="Tick if manual testing actually surfaced a bug.",
                    )

                _n_annotatable = sum(
                    1 for r in (_matched_rows + _miss_rows)
                    if r["based_on"] not in _ANNO_SKIP_SOURCES
                )
                _n_anno     = len(_annotations)
                _n_findable = sum(1 for a in _annotations.values() if a.get("findable_manually"))
                _n_problem  = sum(1 for a in _annotations.values() if a.get("has_actual_problems"))
                st.caption(
                    f"📝 Annotated: **{_n_anno} / {_n_annotatable}** annotatable · "
                    f"✋ Findable: **{_n_findable}** · 🐛 Actual problems: **{_n_problem}** · "
                    f"_in-flight scenarios skipped (always real); saved to `{_anno_fp.name}`_"
                )

                if _matched_rows:
                    st.markdown(f"#### ✅ Matched ({len(_matched_rows)})")
                    for _row in _matched_rows:
                        st.markdown(f"**{_row['module']}** — {_row['scenario']}")
                        for _h in _row["hits"]:
                            _bc = _h["bugcode"]
                            _desc = _bc_to_desc_pulse.get(_bc, _h["desc"]) if _bc else _h["desc"]
                            if _bc:
                                _url = ECL_BUG_URL.format(bug_code=_bc)
                                st.markdown(
                                    f"&nbsp;&nbsp;&nbsp;&nbsp;✅ [{_bc}]({_url}) — {_desc} "
                                    f"_(matched by {_h['method']})_",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"&nbsp;&nbsp;&nbsp;&nbsp;✅ {_desc} _(matched by {_h['method']})_",
                                    unsafe_allow_html=True,
                                )
                        _render_anno_controls(_row["module"], _row["scenario"], _row.get("based_on", ""))

                if _miss_rows:
                    # Bucket by Ollama's `based_on` field so the user can see
                    # whether unmatched scenarios came from in-flight bugs vs
                    # historical risk patterns vs both.
                    _BUCKETS = [
                        ("in_flight",  "🔄 In-flight"),
                        ("both",       "🔀 Both (in-flight + historical)"),
                        ("historical", "📎 Historical pattern"),
                    ]
                    _known = {k for k, _ in _BUCKETS}
                    _grouped: dict[str, list[dict]] = {k: [] for k, _ in _BUCKETS}
                    _grouped["other"] = []
                    for _row in _miss_rows:
                        _key = _row["based_on"] if _row["based_on"] in _known else "other"
                        _grouped[_key].append(_row)

                    with st.expander(f"❌ Not matched ({len(_miss_rows)})", expanded=False):
                        for _k, _label in _BUCKETS:
                            _bucket = _grouped[_k]
                            if not _bucket:
                                continue
                            st.markdown(f"##### {_label} ({len(_bucket)})")
                            for _row in _bucket:
                                st.markdown(f"- **{_row['module']}** — {_row['scenario']}")
                                _render_anno_controls(_row["module"], _row["scenario"], _row.get("based_on", ""))
                        if _grouped["other"]:
                            st.markdown(f"##### 🧪 Other / unspecified ({len(_grouped['other'])})")
                            for _row in _grouped["other"]:
                                _src = _row["based_on"] or "—"
                                st.markdown(f"- **{_row['module']}** — {_row['scenario']}  _(based_on: {_src})_")
                                _render_anno_controls(_row["module"], _row["scenario"], _row.get("based_on", ""))


# =====================================================================
# TAB 11 – TR Test Focus  (newest TR fixed bugs → actionable QA report)
# =====================================================================

elif active_tab == "🧪 TR Test Focus":
    import re as _tr_re

    st.markdown("""
<style>
.tr-focus-hero {
    padding: 1.25rem 1.4rem;
    margin: 0 0 1rem;
    border: 1px solid rgba(99, 102, 241, .24);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(99, 102, 241, .13), rgba(14, 165, 233, .07));
}
.tr-focus-hero .eyebrow {
    color: #6366f1;
    font-size: .76rem;
    font-weight: 750;
    letter-spacing: .09em;
    text-transform: uppercase;
}
.tr-focus-hero h2 { margin: .3rem 0 .35rem; }
.tr-focus-hero p { margin: 0; opacity: .78; }
.tr-metric-grid {
    display: grid;
    grid-template-columns: 1fr .7fr 1.65fr;
    gap: 1rem;
    margin: .25rem 0 .8rem;
}
.tr-metric-card {
    min-width: 0;
    padding: .9rem 1rem 1rem;
    border: 1px solid rgba(99, 102, 241, .18);
    border-radius: 14px;
    background: rgba(99, 102, 241, .045);
}
.tr-metric-label {
    margin-bottom: .5rem;
    font-size: .82rem;
    font-weight: 650;
    opacity: .78;
}
.tr-metric-note {
    margin: -.25rem 0 .55rem;
    font-size: .72rem;
    line-height: 1.35;
    opacity: .62;
}
.tr-metric-value {
    font-size: 1.85rem;
    font-weight: 650;
    line-height: 1.2;
    overflow-wrap: anywhere;
}
.tr-code-list {
    display: flex;
    flex-wrap: wrap;
    gap: .4rem;
}
a.tr-code-chip { color: inherit; text-decoration: none; }
/* The focus list links the build number in its heading to the build's TR;
   it should read as part of the title, not as a blue link. */
div[data-testid="stMarkdownContainer"] :is(h1, h2, h3) a {
    color: inherit;
    text-decoration: none;
}
div[data-testid="stMarkdownContainer"] :is(h1, h2, h3) a:hover {
    text-decoration: underline;
}
a.tr-code-chip:hover { text-decoration: underline; }
.tr-code-chip {
    padding: .25rem .55rem;
    border: 1px solid rgba(99, 102, 241, .28);
    border-radius: 999px;
    background: rgba(99, 102, 241, .09);
    font-size: .9rem;
    font-weight: 600;
    white-space: nowrap;
}
@media (max-width: 760px) {
    .tr-metric-grid { grid-template-columns: 1fr; }
}
div[data-testid="stExpander"] { border-radius: 12px; }
</style>
<div class="tr-focus-hero">
  <div class="eyebrow">Regression test guidance</div>
  <h2>🧪 Newest TR Test Focus</h2>
  <p>Start with <strong>Waiting on you</strong> — the bugs this build's TR was created to verify — then the rest of <strong>What to test</strong>.</p>
</div>
""", unsafe_allow_html=True)

    st.caption(
        "A focus list assembled from what eCL and RD recorded for this Close "
        "Build, plus a verification checklist per fixed bug, from the included-bug "
        "status logs and RD fix details."
    )

    # One orientation block: what the page is, how to read it, then how it is
    # built. The deterministic rule notes live in their own expander below.
    with st.expander("📖 How to read this report", expanded=False):
        st.markdown("""
**In short.** This page covers the newest **Close Build**, the build RD closed the
newest fixes into, and the **Testing Request (TR)** eCL opened to test it. Start
with the focus list at the top: it lists what the TR asks you to verify, and what
changed in the build. Everything below it is supporting detail.

#### The three bug lists on an eCL TR

| eCL calls it | What it is | On this page |
|---|---|---|
| **Bugs from which this Testing Request is created** | The bugs this TR exists to verify: the main scope of the round | **1. Waiting on you** (not verified yet) and **Already verified** |
| **This TR Generated the following eBug(s)** | New bugs opened while testing this TR | **Found on this TR so far** |
| **Known bugs** | Open bugs that already existed before this TR | Not shown: eCL's API does not expose this list |

A TR's scope and the Close Build's fixes overlap without being the same set. A
TR is often created from bugs closed into an earlier build whose own TR was
cancelled, so a bug in "Waiting on you" can come from outside this build's
fixes. Such bugs are marked with the build they were fixed into.

#### The focus list (📋 What to test)

Built from what eCL and RD recorded, with no model involved, so it is never empty:

1. **Waiting on you**: bugs this TR was created from that nobody has verified.
   It reads the TR live on every refresh, so it moves as colleagues verify. It
   says *unknown*, never "nothing", when the TR could not be read.
2. **Where the changes are**: this build's fixes ranked by module.
3. **RD asked for this specifically**: RD's own "Risk" notes and languages.
4. **Shared paths named in the fix notes**: RD's words for what else the
   changed code touches.
5. **Thin evidence**: fixes whose eCL record came back incomplete.
6. **Seen before**: modules where these fixes have similar past defects.
7. **Fixed earlier in this version**: fixes from earlier builds of the same
   version, which also ship in this build.
- **Found on this TR so far** and **Already verified** close the list. A QA
  *Fail* counts as finished work that went back to RD, not as your to-do.

The **🤖 Automated tests** expander under the list matches AT_PDRi test files
to the changed modules (pdri only) and shows the commands. It lists them without
running them.

#### The rest of the page

- **Close Build card**: the build number links to its TR (*Tested by*). **Bug
  Creation TRs** are where this build's bugs were first *reported*, which is
  provenance only. Those TRs tested older builds.
- **✅ Per-Bug Verification**: steps to confirm each of this build's fixes,
  built from that bug's recorded repro steps and symptom.
- **🎯 Recommended Regression Areas**: AI *predictions* of what a fix may have
  broken. The section appears only when something clears the grounding gates.
  Otherwise it is hidden, and the withheld drafts stay under diagnostics.
- **🗓️ Recent runs**: the last 10 days of published runs. A day with no entry
  is a day when nothing changed, not a failed run.
- **📈 Previous TR Prediction Check**: a rough look back at whether an earlier
  build's predictions produced new bugs. It is a signal for tuning, not a score.

#### The labels

- **Risk** = how bad the failure would be *if it happened*, read from the described
  failure — not from the source ticket's severity field.
- **Confidence** = how much independent evidence from *these* fixes backs the claim —
  not how sure the AI sounded.
- **Evidence quality** = how complete the source status log and fix note were.

#### Where the data comes from

The refresh pipeline picks the newest **Close Build#** among the **TRCreated**
bugs, then takes *every* bug closed into that build as its cohort, whatever
their status is now. Filtering on status would make the cohort depend on when
the pipeline ran. Bug details come from the n8n workflow, with eCL's API as a
backup. The TR's own bug lists come from eCL's API directly. Ollama writes the
per-bug checklists and the regression predictions. The latest validated report
stays on the page if a later run fails.

#### Why all the fine print

The AI only proposes candidates. Every published claim must point at exact text from
these fixes; fixed rules then set risk, confidence, and order, and can only lower
them. Anything unproven is withheld and kept in the audit trail instead of being
shown to you. *Rules that constrain these predictions* lists those rules, and
*What happened when this report was built* shows how many claims each one removed.

#### Glossary for the notes on this page

| Term | What it means here |
|---|---|
| **Draft / candidate** | One proposed regression claim, before any checking. |
| **Target** | The feature or module a claim tells you to re-test. |
| **Signal / current-fix evidence** | An exact quoted line from an RD fix note or status log in *this* Close Build. |
| **Shared path** | The concrete mechanism (component, code path) a claim says connects the fixed bug to its target. |
| **Trigger / failure mode** | What you would do to provoke the problem, and what you would observe if it occurred. |
| **Grounding audit** | The check that every part of a claim traces back to quoted evidence. |
| **Gate** | A fixed pass/fail rule that runs *before* any AI review and cannot be overridden by it. |
| **Withheld** | Not published — kept in the stored record with the reason. |
| **Provenance** | The immutable record of everything considered, including what was dropped. |
| **Coverage-equivalent** | A claim that says the same thing in different words; only one wording is shown. |
| **Historical analog** | An older resolved bug that looks similar. Context only — it can never raise risk or confidence. |
| **Alias** | A human-confirmed synonym pair, so a target term can match differently-worded evidence. |
| **Schema version** | Which generation of these rules produced the stored report; each version adds guardrails. |
""")

    _tr_latest_fp = Path(_product_dir) / "tr_reports" / "latest.json"
    _tr_status_fp = Path(_product_dir) / "tr_reports" / "status.json"
    _tr_status = {}
    if _tr_status_fp.exists():
        try:
            _tr_status = json.loads(_tr_status_fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _tr_status = {"outcome": "status_unreadable"}
    if not _tr_latest_fp.exists():
        st.info(
            "No fixed-bug TR report has been generated for this product yet. "
            "Run the refresh pipeline when the newest version has a `TRCreated` cohort."
        )
        if _tr_status.get("outcome") == "no_cohort":
            st.caption("Latest check: no TRCreated cohort exists in the current parsed data.")
        elif _tr_status.get("outcome") in {
            "fetch_failed", "analysis_failed", "validation_failed", "status_unreadable"
        }:
            st.warning(
                "Latest TR Test Focus attempt: "
                + str(_tr_status.get("message") or _tr_status.get("outcome"))
            )
    else:
        try:
            try:
                from process_newest_tr import (
                    audit_report_manifest, extract_manifest_checklists,
                    resolve_product_artifact, select_newest_tr_cohort,
                )
            except ModuleNotFoundError:
                from scripts.process_newest_tr import (
                    audit_report_manifest, extract_manifest_checklists,
                    resolve_product_artifact, select_newest_tr_cohort,
                )
            try:
                from tr_prediction_reviews import (
                    REJECTION_REASON_LABELS, ReviewLedgerError,
                    build_analysis_guardrail_policy, build_target_alias_policy,
                    load_review_ledger,
                    grounding_review_summary, load_grounding_review_ledger,
                    guardrail_effectiveness_comparison, overlay_reviews,
                    overlay_grounding_reviews,
                    repair_origin_effectiveness_comparison,
                    upsert_grounding_review, upsert_review,
                )
            except ModuleNotFoundError:
                from scripts.tr_prediction_reviews import (
                    REJECTION_REASON_LABELS, ReviewLedgerError,
                    build_analysis_guardrail_policy, build_target_alias_policy,
                    load_review_ledger,
                    grounding_review_summary, load_grounding_review_ledger,
                    guardrail_effectiveness_comparison, overlay_reviews,
                    overlay_grounding_reviews,
                    repair_origin_effectiveness_comparison,
                    upsert_grounding_review, upsert_review,
                )

            _tr_manifest = json.loads(_tr_latest_fp.read_text(encoding="utf-8"))
            _tr_build = str(_tr_manifest.get("close_build") or "Unknown")
            _tr_codes = [str(v) for v in (_tr_manifest.get("tr_codes") or [])]
            _tr_bug_codes = [str(v) for v in (_tr_manifest.get("bug_codes") or [])]
            _tr_bug_total = int(
                _tr_manifest.get("bug_count_total")
                or _tr_manifest.get("bug_count")
                or len(_tr_bug_codes)
            )
            _tr_bug_analyzed = int(
                _tr_manifest.get("bug_count_analyzed")
                or _tr_manifest.get("bug_count")
                or len(_tr_bug_codes)
            )
            _tr_missing_details = [
                str(v) for v in (_tr_manifest.get("missing_detail_bug_codes") or [])
            ]
            _tr_generated = str(_tr_manifest.get("generated_at") or "")
            _tr_audit = audit_report_manifest(Path(_product_dir), _tr_manifest)
            _tr_report_fp = _tr_audit["paths"].get("report_path")

            if _tr_status.get("outcome") == "status_unreadable":
                _tr_audit["warnings"].append("The latest-attempt status file is unreadable.")

            _tr_current_csv = Path(_product_dir) / "ecl_parsed.csv"
            if _tr_current_csv.exists():
                try:
                    _tr_current_build, _tr_current_codes = select_newest_tr_cohort(_tr_current_csv)
                    if not _tr_current_codes:
                        _tr_audit["warnings"].append(
                            "The current parsed data has no TRCreated cohort; this is the last "
                            "successful historical report."
                        )
                    elif (
                        _tr_current_build != _tr_build
                        or set(_tr_current_codes) != set(_tr_bug_codes)
                    ):
                        _tr_audit["warnings"].append(
                            f"A newer or different current cohort is present ({_tr_current_build}); "
                            "this report remains the last successful result."
                        )
                except (OSError, ValueError, TypeError):
                    _tr_audit["warnings"].append(
                        "The current parsed cohort could not be checked for freshness."
                    )

            _tr_outcome = str(_tr_status.get("outcome") or "")
            if _tr_outcome in {"fetch_failed", "analysis_failed", "validation_failed"}:
                _tr_audit["warnings"].append(
                    "The latest TR Test Focus attempt failed: "
                    + str(_tr_status.get("message") or _tr_outcome)
                )

            for _tr_warning in list(dict.fromkeys(_tr_audit["warnings"])):
                st.warning(_tr_warning)

            import html as _tr_html

            # Loaded before the metric cards so the Close Build card can name
            # the TR that tests it. The details file is the same artifact
            # audit_report_manifest already hash-verified above.
            _tr_details_fp = _tr_audit["paths"].get("details_path")
            _tr_details, _tr_details_meta = {}, {}
            _tr_details_unreadable = False
            if _tr_details_fp and _tr_details_fp.exists():
                try:
                    _tr_details = json.loads(
                        _tr_details_fp.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    _tr_details = {}
                    _tr_details_unreadable = True
                _tr_details_meta = _tr_details.get("meta") or {}

            # status.json is rewritten on every attempt, an unchanged run is
            # never republished, and the build TR's queue moves as QA verifies
            # while the cohort stays put. So the newest lookup for this same
            # build wins; the run's own copy is the fallback.
            _tr_build_trs = list(_tr_details_meta.get("build_trs") or [])
            _tr_live_build_trs = False
            if str(_tr_status.get("selected_close_build") or "") == _tr_build:
                _tr_status_trs = list(_tr_status.get("build_trs") or [])
                if _tr_status_trs and (
                    not _tr_build_trs
                    or any(tr.get("bugs_loaded") for tr in _tr_status_trs)
                ):
                    _tr_live_build_trs = _tr_status_trs != _tr_build_trs
                    _tr_build_trs = _tr_status_trs
            _tr_build_tr_html = ""
            _tr_build_html = _tr_html.escape(_tr_build)
            if len(_tr_build_trs) == 1 and _tr_build_trs[0].get("tr_code"):
                _tr_build_html = (
                    f'<a href="{_tr_html.escape(tr_url(_tr_build_trs[0]["tr_code"]))}" '
                    f'target="_blank" style="color:inherit">{_tr_build_html}</a>'
                )
            if _tr_build_trs:
                _tr_build_tr_html = (
                    '<div class="tr-metric-note">Tested by '
                    + ", ".join(
                        f'<a href="{_tr_html.escape(tr_url(tr["tr_code"]))}" '
                        f'target="_blank">{_tr_html.escape(tr["tr_code"])}</a>'
                        + (" (cancelled)" if tr.get("form_state") == "Cancel" else "")
                        for tr in _tr_build_trs if tr.get("tr_code")
                    )
                    + "</div>"
                )

            _tr_code_chips = "".join(
                f'<a class="tr-code-chip" href="{_tr_html.escape(tr_url(code))}" '
                f'target="_blank">{_tr_html.escape(code)}</a>'
                for code in _tr_codes
            ) or '<span class="tr-code-chip">Not supplied</span>'
            st.markdown(f"""
<div class="tr-metric-grid">
  <div class="tr-metric-card">
    <div class="tr-metric-label">Close Build</div>
    <div class="tr-metric-value">{_tr_build_html}</div>{_tr_build_tr_html}
  </div>
  <div class="tr-metric-card">
    <div class="tr-metric-label">Fixed bugs analyzed</div>
    <div class="tr-metric-value">{_tr_bug_analyzed} of {_tr_bug_total}</div>
  </div>
  <div class="tr-metric-card">
    <div class="tr-metric-label">Bug Creation TRs</div>
    <div class="tr-metric-note">Parent TRs where these bugs were originally created—not the TR for this Close Build.</div>
    <div class="tr-code-list">{_tr_code_chips}</div>
  </div>
</div>
""", unsafe_allow_html=True)
            if _tr_generated:
                st.caption(f"Generated {_tr_generated.replace('T', ' ')[:19]}")

            # Cohort provenance qualifies every recommendation below it, so it
            # is rendered before them.
            _tr_changesets = {}
            # First thing on the page: what to test, as recorded fact. The
            # slot is reserved here and filled once the analysis is loaded
            # below, so it renders above everything without reordering the
            # tab's existing data loading.
            _tr_focus_slot = st.container()
            if _tr_details_fp and _tr_details_fp.exists():
                if _tr_details_unreadable:
                    st.caption("Cohort provenance could not be read.")
                _tr_changesets = collect_changesets(_tr_details)

                if _tr_details_meta:
                    render_cohort_provenance(
                        st, {**_tr_details_meta, "build_trs": _tr_build_trs},
                        ECL_BUG_URL)

            _tr_analysis = {}
            _tr_analysis_fp = _tr_audit["paths"].get("analysis_path")
            if _tr_analysis_fp and _tr_analysis_fp.exists():
                try:
                    _tr_analysis = json.loads(
                        _tr_analysis_fp.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    st.warning(
                        "Structured prediction evidence could not be displayed."
                    )
            # Fill the reserved focus-list slot. Prefer the artifact the
            # overnight run wrote, so the file QA reads and the page they open
            # cannot disagree; rebuild only for runs made before it existed.
            _tr_focus_md = ""
            _tr_focus_note = ""
            _tr_focus_fp = resolve_product_artifact(
                Path(_product_dir), _tr_manifest.get("focus_list_path")
            )
            if _tr_focus_fp and _tr_focus_fp.is_file():
                try:
                    _tr_focus_md = _tr_focus_fp.read_text(encoding="utf-8")
                except OSError:
                    _tr_focus_md = ""
            # A file written by an older builder is rebuilt from the same run's
            # stored evidence (deterministic), so a content fix reaches the page
            # without waiting for the evidence to change and republish.
            if _tr_focus_md and not focus_list_is_current(_tr_focus_md) and _tr_details:
                _tr_focus_md = ""
                _tr_focus_note = (
                    "Rebuilt in the current focus-list format from this run's "
                    "stored evidence; the run's own file predates it."
                )
            elif _tr_focus_md and _tr_live_build_trs and _tr_details:
                _tr_focus_md = ""
                _tr_focus_note = (
                    "“Waiting on you” reflects this build's TR as eCL reported it "
                    f"at {str(_tr_status.get('checked_at') or '')[:16].replace('T', ' ')} "
                    "UTC, newer than this run's stored file."
                )
            _tr_focus = {}
            if _tr_details:
                _tr_focus_details = _tr_details
                if _tr_build_trs and _tr_build_trs != _tr_details_meta.get("build_trs"):
                    _tr_focus_details = {
                        **_tr_details,
                        "meta": {**_tr_details_meta, "build_trs": _tr_build_trs},
                    }
                if not (_tr_details_meta.get("version_fixes")) and Path(_bugs_default).is_file():
                    # Runs made before Stage 8 stored earlier fixes of this
                    # version: read them from the current bug list instead.
                    import csv as _tr_csv
                    try:
                        with open(_bugs_default, newline="", encoding="utf-8-sig") as _fh:
                            _tr_vf = collect_version_fixes(
                                list(_tr_csv.DictReader(_fh)), _tr_build,
                                [str(b.get("BugCode")) for _v in
                                 (_tr_details.get("products") or {}).values() for b in _v],
                            )
                    except (OSError, _tr_csv.Error):
                        _tr_vf = {}
                    if _tr_vf:
                        _tr_focus_details = {
                            **_tr_focus_details,
                            "meta": {**(_tr_focus_details.get("meta") or {}),
                                     "version_fixes": _tr_vf},
                        }
                        if not _tr_focus_md:
                            _tr_focus_note += (
                                " Earlier fixes in this version were read from the "
                                "current bug list — this run predates storing them."
                            )
                try:
                    _tr_focus = build_focus_list(_tr_focus_details, _tr_analysis)
                except (KeyError, TypeError, ValueError):
                    _tr_focus = {}
            if not _tr_focus_md and _tr_focus:
                _tr_focus_md = focus_list_markdown(_tr_focus, _tr_generated)
            # ``focus_list_markdown`` links codes itself, so this is a no-op for
            # anything written since. A file promoted before it did is served
            # from the run directory verbatim and would otherwise be the one
            # place on the page where a bug code is not clickable. Applied
            # before the download too: QA reads the file more than the page.
            _tr_focus_md = linkify_bug_codes(_tr_focus_md)
            if _tr_focus_md:
                with _tr_focus_slot:
                    with st.container(border=True):
                        st.markdown(_tr_focus_md)
                    if _tr_focus_note:
                        st.caption(_tr_focus_note.strip())
                    st.download_button(
                        "Download this list", _tr_focus_md,
                        file_name="what_to_test_"
                        + _tr_re.sub(r"[^A-Za-z0-9._-]+", "_", _tr_build)
                        + ".md",
                        mime="text/markdown",
                        key=f"tr_focus_dl_{selected_slug}",
                    )
                    # AT_PDRi cases for the changed modules, read from that repo
                    # at render time — so kept out of the run's focus_list.md.
                    if _tr_focus and selected_slug in AT_REPOS:
                        _tr_at_repo = at_repo_dir(selected_slug)
                        _tr_at_set = configured_at_dir(selected_slug)
                        _tr_at_files, _tr_at, _tr_at_sel = [], {}, []
                        if _tr_at_repo is not None:
                            _tr_at_files = _load_at_index(str(_tr_at_repo))
                            _tr_at = at_cases_for_focus(_tr_focus, _tr_at_files)
                            _tr_at_sel = _tr_at["selected"]
                        _tr_at_label = (
                            f"🤖 Automated tests for this build — "
                            f"{len(_tr_at_sel)} AT file(s), "
                            f"{sum(e['case_count'] for e in _tr_at_sel)} case(s), "
                            "not run yet"
                            if _tr_at_repo is not None else
                            "🤖 Automated tests for this build — no AT folder set"
                        )
                        with st.expander(_tr_at_label, expanded=_tr_at_repo is None):
                            # Saved to .at_repos.json, so it is set once per
                            # machine and every later render reuses it.
                            with st.form(f"tr_at_dir_{selected_slug}", border=False):
                                _tr_at_input = st.text_input(
                                    "AT repo folder",
                                    value=_tr_at_set,
                                    placeholder="/Users/<you>/Desktop/AT_PDRi",
                                    help="The AT repo root — the folder holding "
                                         "tests/new_bft_ios and tests/new_sft_ios. "
                                         "Remembered on this machine.",
                                )
                                if st.form_submit_button("Save folder"):
                                    try:
                                        save_at_repo_dir(selected_slug, _tr_at_input)
                                    except (ValueError, OSError) as _exc:
                                        st.error(f"Not saved: {_exc}")
                                    else:
                                        _load_at_index.clear()
                                        st.rerun()
                            if os.environ.get(AT_REPOS[selected_slug]):
                                st.caption(
                                    f"`${AT_REPOS[selected_slug]}` is set and "
                                    "overrides the saved folder."
                                )
                            if _tr_at_repo is None:
                                st.info(
                                    "No AT folder set, so no automated cases are "
                                    "listed. Save the AT repo folder above."
                                    if not _tr_at_set else
                                    f"The saved AT folder `{_tr_at_set}` no longer "
                                    "exists. Save its new location above."
                                )
                            else:
                                _tr_at_cmds = pytest_commands(
                                    _tr_at_sel, _tr_at_files, _tr_build)
                                st.caption(
                                    f"From `{_tr_at_repo}`. Listed only; nothing "
                                    "is triggered from this page. Selected by "
                                    "file: every case is a dependency chain, so "
                                    "a case cannot run without the ones before it."
                                )
                                if _tr_at_sel:
                                    show_dataframe(
                                        pd.DataFrame([
                                            {"Suite": e["suite"],
                                             "Module": e["module"],
                                             "Why": e["reason"],
                                             "Entry point": c["description"] or c["name"],
                                             "Cases": len(c["cases"]),
                                             "File": e["file"].rsplit("/", 1)[-1]}
                                            for e in _tr_at_sel for c in e["classes"]
                                        ]),
                                        hide_index=True, use_container_width=True,
                                    )
                                    st.download_button(
                                        "Download the full case list (CSV)",
                                        pd.DataFrame([
                                            {"suite": e["suite"], "module": e["module"],
                                             "reason": e["reason"],
                                             "node_id": case["node_id"],
                                             "step": case["title"]}
                                            for e in _tr_at_sel for c in e["classes"]
                                            for case in c["cases"]
                                        ]).to_csv(index=False),
                                        file_name="at_cases_"
                                        + _tr_re.sub(r"[^A-Za-z0-9._-]+", "_", _tr_build)
                                        + ".csv",
                                        mime="text/csv",
                                        key=f"tr_at_dl_{selected_slug}",
                                    )
                                    for _suite, _cmd in _tr_at_cmds.items():
                                        st.markdown(
                                            "**BFT** — quick pass; installs this "
                                            "Close Build first:"
                                            if _suite == "BFT" else
                                            "**SFT** — full checklist, changed and "
                                            "regression modules; installs this Close "
                                            "Build first and ends with the My "
                                            "Artwork sweep. Hours of rendering, "
                                            "and it spends credits:"
                                        )
                                        st.code(_cmd, language="bash")
                                    _tr_costly = [
                                        e["file"].rsplit("/", 1)[-1] for e in _tr_at_sel
                                        if e["cost_warning"] and e["suite"] == "SFT"
                                    ]
                                    if _tr_costly:
                                        st.warning(
                                            "Carries a COST WARNING in AT_PDRi: "
                                            + ", ".join(f"`{f}`" for f in _tr_costly)
                                            + ". Run with `TTV_SUBMIT_GENERATIONS=0` "
                                            "to check locators without submitting "
                                            "generations."
                                        )
                                if _tr_at["unmatched"]:
                                    st.caption(
                                        "No AT file for: " + ", ".join(
                                            f"{u['module']} ({u['reason']})"
                                            for u in _tr_at["unmatched"]
                                        )
                                    )

            _tr_quality = _tr_analysis.get("evidence_quality_counts") or {}
            if _tr_quality:
                st.caption(
                    "Source evidence quality — "
                    f"High: {int(_tr_quality.get('high', 0))} · "
                    f"Medium: {int(_tr_quality.get('medium', 0))} · "
                    f"Low: {int(_tr_quality.get('low', 0))}"
                )
            st.caption(
                "Source quality describes the available bug details. Recommendation "
                "confidence describes support for the proposed regression check; "
                "neither is a release-readiness score."
            )
            _tr_checklist_gen = _tr_analysis.get("checklist_generation") or {}
            _tr_fallback_count = int(
                _tr_checklist_gen.get("deterministic_fallback_count") or 0
            )
            if _tr_fallback_count:
                _tr_fallback_codes = ", ".join(
                    bug_link(item.get("bug_code"), code_style=False)
                    or "?"
                    for item in _tr_checklist_gen.get("deterministic_fallbacks") or []
                )
                st.caption(
                    f"⚠️ {_tr_fallback_count} checklist(s) are evidence-only "
                    "fallbacks — the model could not produce a valid checklist, "
                    "so the steps restate stored evidence and need review: "
                    f"{_tr_fallback_codes}"
                )
            # Reserve the main workflow ahead of methodology and audit output.
            _tr_work_slot = st.container()

            # ── Recent history ───────────────────────────────────────────
            # Everything above describes one run. Stage 8 never rewrites a
            # published run, so the run tree is already a record of the past
            # builds and of how each cohort grew during a day; only
            # latest.json is overwritten. This reads that tree — no fetching,
            # no re-analysis — and renders after the current work so it stays
            # reference material rather than competing with "what to test".
            st.markdown("### 🗓️ Recent runs")
            _tr_hist_days = st.radio(
                "Window", [7, DEFAULT_HISTORY_DAYS, 30], index=1, horizontal=True,
                format_func=lambda value: f"Last {value} days",
                key=f"tr_hist_days_{selected_slug}", label_visibility="collapsed",
            )
            _tr_hist_current = current_run_dir(Path(_product_dir), _tr_manifest)
            _tr_hist = run_history(
                Path(_product_dir), days=int(_tr_hist_days),
                current_dir=_tr_hist_current,
            )
            if not _tr_hist:
                st.caption(
                    "No TR Test Focus run was published for this product in the "
                    f"last {int(_tr_hist_days)} days."
                )
            else:
                _tr_hist_digest = history_digest(_tr_hist, int(_tr_hist_days))
                st.caption(
                    f"{_tr_hist_digest['run_count']} published run(s) on "
                    f"{_tr_hist_digest['days_with_runs']} day(s), covering "
                    f"{len(_tr_hist_digest['builds'])} Close Build(s). A run is "
                    "published only when the evidence, the model, or the analysis "
                    "schema changed, so a day with no entry is a day nothing "
                    "changed — not a failed run. Only the newest attempt keeps a "
                    "status, so failures leave no trace here."
                )
                show_dataframe(
                    pd.DataFrame([
                        {
                            "Run (UTC)": row["run_at"].strftime("%Y-%m-%d %H:%M"),
                            "Close Build": row["close_build"],
                            # Kept as text: a run with no recorded count would
                            # otherwise mix int and str in one Arrow column.
                            "Bugs": (
                                "—" if row["bug_count"] is None
                                else str(row["bug_count"])
                            ),
                            # "0" and "—" are different answers: the cohort
                            # did not change, versus there is nothing to
                            # compare it with.
                            "Δ same build": (
                                "—" if row["bug_delta"] is None
                                else f"{row['bug_delta']:+d}" if row["bug_delta"]
                                else "0"
                            ),
                            "Model": row["model"] or "—",
                            "Showing now": "✅" if row["is_current"] else "",
                            "Run ID": row["run_id"],
                        }
                        for row in _tr_hist
                    ]),
                    hide_index=True, width="stretch",
                )
                st.caption(
                    "**Δ same build** compares a run with the previous run of the "
                    "*same* Close Build — a cohort that grew during the day as more "
                    "bugs closed into the build. It is blank across a build change, "
                    "where the two numbers are different cohorts rather than a change "
                    "in one."
                )
                if not any(row["is_current"] for row in _tr_hist):
                    st.caption(
                        "⚠️ The report shown above was published before this window "
                        "and so has no row here."
                    )

                _tr_hist_labels = {
                    (
                        f"{row['run_at']:%Y-%m-%d %H:%M} · {row['close_build']}"
                        + (
                            f" · {row['bug_count']} bugs"
                            if row["bug_count"] is not None else ""
                        )
                        + (" · showing now" if row["is_current"] else "")
                    ): row
                    for row in _tr_hist
                }
                _tr_hist_pick = st.selectbox(
                    "Open the focus list from a past run",
                    list(_tr_hist_labels),
                    key=f"tr_hist_pick_{selected_slug}",
                )
                _tr_hist_row = _tr_hist_labels[_tr_hist_pick]
                _tr_hist_md, _tr_hist_note = run_focus_list(_tr_hist_row)
                _tr_hist_md = linkify_bug_codes(_tr_hist_md)
                st.caption(_tr_hist_note)
                if _tr_hist_md:
                    with st.container(border=True):
                        st.markdown(_tr_hist_md)
                    st.download_button(
                        "Download this past list", _tr_hist_md,
                        file_name="what_to_test_"
                        + _tr_re.sub(
                            r"[^A-Za-z0-9._-]+", "_", _tr_hist_row["close_build"]
                        )
                        + "_" + _tr_hist_row["run_id"] + ".md",
                        mime="text/markdown",
                        key=f"tr_hist_dl_{selected_slug}",
                    )

            st.markdown("### Report evidence & diagnostics")
            _tr_schema = int(_tr_analysis.get("schema_version") or 0)
            _tr_guardrail_labels = {
                "require_stronger_shared_path_grounding": (
                    "stronger shared-path grounding"
                ),
                "narrow_scope_and_disconfirmation": (
                    "narrower scope and earlier disconfirmation"
                ),
            }

            # The methodology notes below describe fixed rules, not this run's
            # results, so they are grouped and collapsed instead of stacked as
            # ~20 separate captions above the recommendations.
            _tr_rule_groups = [
                (
                    "published",
                    "What actually gets published",
                    "A claim reaches the list only if it is specific, distinct, "
                    "and still standing after the final evidence audit. "
                    "Everything dropped is kept in the stored record.",
                    [],
                ),
                (
                    "labels",
                    "How Risk, Confidence and order are decided",
                    "The model's own labels are discarded. These rules recompute "
                    "them from the cited evidence and can only lower them.",
                    [],
                ),
                (
                    "gates",
                    "What must be proven before a claim survives",
                    "Hard pass/fail checks that run before any AI critic. A claim "
                    "that cannot point at exact quoted text from these fixes is "
                    "withheld, and no critic, alias, or historical analog can "
                    "overrule that.",
                    [],
                ),
            ]
            _tr_rules = {key: items for key, _, _, items in _tr_rule_groups}

            def _tr_rule(group, title, body):
                _tr_rules[group].append(f"**{title}** — {body}")

            if _tr_schema >= 12:
                _tr_rule(
                    "published", "Prediction specificity",
                    "every published adjacent claim has a distinctive target plus "
                    "a trigger or observable failure term, using the same "
                    "vocabulary as future-cohort scenario matching.",
                )
            if _tr_schema >= 13:
                _tr_rule(
                    "published", "Coverage diversity",
                    "recommendations with identical normalized category, target, "
                    "trigger, failure, boundary, disconfirmation, and QA-action "
                    "terms are shown once; every suppressed wording variant and "
                    "its evidence remain in immutable audit provenance.",
                )
            if _tr_schema >= 14:
                _tr_rule(
                    "published", "Grounding-aware selection",
                    "every exact synthesized wording variant receives the final "
                    "adversarial audit before supported coverage-equivalents are "
                    "collapsed, so a withheld preferred wording cannot hide a "
                    "supported fallback.",
                )
            if _tr_schema >= 15:
                _tr_rule(
                    "labels", "Deterministic evidence confidence",
                    "model-selected confidence is discarded. High requires "
                    "multiple current BugCodes with medium-or-better evidence "
                    "including at least one high-quality source; historical "
                    "analogs never raise the label.",
                )
            if _tr_schema >= 16:
                _tr_rule(
                    "labels", "Deterministic impact risk",
                    "model-selected risk and impact basis are discarded. "
                    "Observable failure wording determines presentation, "
                    "functional, crash, launch-blocking, billing, data-loss, or "
                    "security/privacy impact; evidence confidence caps "
                    "unsupported severe labels.",
                )
            if _tr_schema >= 17:
                _tr_rule(
                    "labels", "Deterministic target category",
                    "an unambiguous known target module wins; otherwise a "
                    "category is retained only when it matches the unique "
                    "category of the cited current BugCodes. Unsupported or "
                    "ambiguous labels abstain as cross-cutting.",
                )
            if _tr_schema >= 18:
                _tr_rule(
                    "labels", "Correlation-aware evidence confidence",
                    "cited BugCodes with identical signal terms, or near-duplicate "
                    "signals from the same module, count as one evidence group. "
                    "High confidence requires at least two deterministic groups.",
                )
            if _tr_schema >= 19:
                _tr_rule(
                    "labels", "Target-anchored confidence",
                    "High is capped to Medium unless a distinctive term from the "
                    "proposed adjacent target appears in at least one exact cited "
                    "current-fix signal. Generic shared component terms do not "
                    "count as a target anchor.",
                )
            if _tr_schema >= 20:
                _tr_rule(
                    "gates", "Evidence-anchored verification",
                    "each per-bug checklist is deterministically tied to the "
                    "stored reproduction context and reported symptom. Invented "
                    "versions, quoted strings, platforms, locales, and internal "
                    "components are rejected before publication.",
                )
            if _tr_schema >= 21:
                _tr_rule(
                    "labels", "Harm-specific, negation-aware risk",
                    "bare billing, payment, subscription, entitlement, security, "
                    "or privacy words do not raise severity. Critical requires "
                    "explicit observable harm, and negated passing signals are "
                    "recorded and ignored.",
                )
            if _tr_schema >= 22:
                _tr_rule(
                    "labels", "Deterministic recommendation priority",
                    "impact risk and evidence confidence come first, followed by "
                    "target-anchored independent groups and their evidence "
                    "quality. Model response order, historical analogs, and "
                    "source-ticket severity cannot move a recommendation upward.",
                )
            if _tr_schema >= 23:
                _tr_rule(
                    "published", "Synthesis-omission safeguard",
                    "the strongest grounded drafts omitted by AI consolidation are "
                    "deterministically backfilled into the final adversarial audit "
                    "with coverage diversity first. Only the top eight supported "
                    "claims are shown; supported overflow remains in immutable "
                    "provenance.",
                )
            if _tr_schema >= 24:
                _tr_rule(
                    "published", "Complementary candidate discovery",
                    "a second independent pass uses the same frozen evidence to "
                    "seek only materially distinct missed adjacent coverage. "
                    "Equivalent variants are filtered before grounding; distinct "
                    "additions still face the complete critic, synthesis "
                    "safeguard, and final audit.",
                )
            if _tr_schema >= 25:
                _tr_rule(
                    "gates", "Deterministic adjacent-target gate",
                    "before any model critic, a distinctive term from the proposed "
                    "target must occur in an exact cited current-fix signal. "
                    "Unanchored targets are preserved as withheld provenance; "
                    "neither the critic nor historical analogs can override the "
                    "decision.",
                )
            if _tr_schema >= 26:
                _tr_rule(
                    "gates", "Bounded exact-reference completion",
                    "when the target is absent from the model-selected signal, the "
                    "analyzer checks only other current-fix signals from the "
                    "candidate's existing BugCodes and may add an exact matching "
                    "reference within the four-reference cap before applying the "
                    "unchanged target gate.",
                )
            if _tr_schema >= 27:
                _tr_active_target_aliases = (
                    (_tr_analysis.get("target_alias_policy") or {}).get(
                        "active_aliases"
                    ) or {}
                )
                _tr_rule(
                    "gates", "Reviewed directional target aliases",
                    "only explicit target-term → cited-signal-term pairs confirmed "
                    "across at least four reviews, three Close Builds, and four "
                    "immutable source runs may satisfy the target gate. Aliases "
                    "never raise confidence, risk, or priority. Active for this "
                    "run: "
                    + (
                        " · ".join(
                            f"{target} → {evidence_term}"
                            for target, evidence_terms in sorted(
                                _tr_active_target_aliases.items()
                            )
                            for evidence_term in evidence_terms
                        )
                        if _tr_active_target_aliases else "none"
                    )
                    + "."
                    + (
                        " Schema 28 preserves exact alias usage into future-cohort "
                        "reviews and can retire a repeatedly misleading pair only "
                        "from diverse alias-specific outcomes; matcher errors do "
                        "not count."
                        if _tr_schema >= 28 else ""
                    ),
                )
            if _tr_schema >= 29:
                _tr_rule(
                    "gates", "Deterministic shared-path gate",
                    "the rationale must name a distinctive non-target mechanism "
                    "present in an exact cited current-fix signal. Generic path "
                    "language alone is withheld before the model critic; aliases, "
                    "analogs, confidence, risk, and priority cannot override the "
                    "result.",
                )
            if _tr_schema >= 30:
                _tr_rule(
                    "labels", "Current-evidence-bounded harm",
                    "crash, launch blocking, data loss, persistent corruption, "
                    "billing/entitlement, and security/privacy predictions can "
                    "rank above Medium only when the same non-negated harm class "
                    "occurs in the cited BugCodes' summary, reproduction evidence, "
                    "or exact current-fix signal.",
                )
            if _tr_schema >= 31:
                _tr_portfolio_policy = (
                    _tr_analysis.get("recommendation_portfolio_policy") or {}
                )
                _tr_cutoff_tier = _tr_portfolio_policy.get("cutoff_tier") or {}
                _tr_rule(
                    "published", "Severity-safe portfolio diversity",
                    "stronger risk and confidence tiers are exhausted first. When "
                    "one equal tier crosses the top-eight boundary, its strongest "
                    "claim remains the anchor and remaining slots favor new exact "
                    "path mechanisms, target categories, and source BugCodes. "
                    "Cutoff tier: "
                    + (
                        str(_tr_cutoff_tier.get("risk") or "").title()
                        + " / "
                        + str(_tr_cutoff_tier.get("confidence") or "").title()
                        if _tr_cutoff_tier.get("risk") else "not reached"
                    )
                    + ".",
                )
            if _tr_schema >= 35:
                _tr_rule(
                    "gates", "Deterministic direct-fix check",
                    "a claim whose target is the source bug's own module and "
                    "whose wording adds nothing that bug's evidence does not "
                    "already say is the fix restated, not an adjacent area — "
                    "the per-bug checklist already covers it — so it is "
                    "withheld before the model critic. A different module, or "
                    "a distinctive new surface inside the same module, still "
                    "qualifies.",
                )
            if _tr_schema >= 34:
                _tr_rule(
                    "gates", "Stated-generalization target support",
                    "when a cited fix note states the change matches a family "
                    "of call sites — \"matching every other text-mutation path\" "
                    "— a target that is another member of that family is "
                    "supported even though the note never names it. The claim "
                    "must invoke that family in its rationale and touch a "
                    "mechanism the same quote names. It overrides the model "
                    "critic on the target check only; any other failed check "
                    "still withholds the draft, and it never changes "
                    "confidence, risk, or priority.",
                )
            if _tr_schema >= 33:
                _tr_rule(
                    "gates", "Mechanism-aware adjacency evidence",
                    "a fix note supplies adjacency evidence when it names shared "
                    "architecture, states that the fix generalizes to other call "
                    "sites, or names a concrete reusable mechanism such as a "
                    "symbol, design token, or localized string key. Fix notes "
                    "with none of the three still yield nothing; the target and "
                    "shared-path gates are unchanged, so breadth here only "
                    "decides what may be proposed, never what is published.",
                )
            if _tr_schema >= 32:
                _tr_rule(
                    "published", "Evidence-bounded scenario specificity",
                    "exact versions, numbers, quoted UI text, platforms/devices, "
                    "locales, lifecycle or network states, media formats, and "
                    "internal components in a trigger or failure must occur in "
                    "that prediction's frozen current BugCode evidence. Invalid "
                    "precision is corrected before the model critic.",
                )
            if _tr_schema >= 37:
                _tr_rule(
                    "gates", "Instruction consistency",
                    "exact action and passing-observation details must occur in "
                    "current source evidence. Explicit conflicting credit amounts "
                    "or periods are rejected; the evidence critic also checks "
                    "scenario consistency.",
                )
            _tr_applied_guardrails = (
                (_tr_analysis.get("analysis_guardrail_policy") or {}).get(
                    "active_guardrails"
                ) or {}
            )
            if _tr_applied_guardrails:
                _tr_rule(
                    "gates", "Reviewed category guardrails applied to this run",
                    " · ".join(
                        f"{category}: "
                        + _tr_guardrail_labels.get(
                            str(item.get("guardrail") or ""),
                            str(item.get("guardrail") or "").replace("_", " "),
                        )
                        for category, item in sorted(
                            _tr_applied_guardrails.items(),
                            key=lambda pair: str(pair[0]).casefold(),
                        )
                        if isinstance(item, dict)
                    ),
                )

            if any(_tr_rules.values()):
                with st.expander(
                    "🔒 Rules that constrain these predictions"
                    + (f" (schema {_tr_schema})" if _tr_schema else ""),
                    expanded=False,
                ):
                    st.caption(
                        "The AI proposes; these fixed rules decide what survives "
                        "and how it is labeled. They can only remove claims or "
                        "lower their labels, never promote them."
                    )
                    for _tr_key, _tr_heading, _tr_lead, _tr_items in _tr_rule_groups:
                        if not _tr_items:
                            continue
                        st.markdown(f"##### {_tr_heading}")
                        st.caption(_tr_lead)
                        st.markdown(
                            "\n".join(f"- {item}" for item in _tr_items)
                        )

            # Per-run counts are buffered and rendered together in this slot so
            # they read as one block instead of interleaving with the drill-down
            # tables they describe.
            _tr_run_notes = []
            _tr_run_notes_slot = st.container()
            _tr_grounding_reason_labels = {
                "adjacent_target_absent_from_current_signal": (
                    "Adjacent target absent from cited fix signal"
                ),
                "target_not_supported": "Adjacent target not supported",
                "trigger_not_supported": "Trigger not supported",
                "failure_mode_not_supported": "Failure mode not supported",
                "shared_path_not_concrete": "Shared path not concrete",
                "boundary_not_supported": "Scope boundary not supported",
                "direct_fix_restatement": "Direct-fix restatement",
            }
            _tr_grounding_audit = (
                _tr_analysis.get("candidate_grounding_audit") or {}
            )
            _tr_grounding_review_fp = (
                Path(_product_dir) / "tr_reports" / "grounding_reviews.json"
            )
            _tr_grounding_review_ledger = {
                "schema_version": 1,
                "product": selected_slug or "legacy",
                "reviews": {},
            }
            _tr_alias_prediction_review_ledger = {"reviews": {}}
            try:
                _tr_alias_prediction_review_ledger = load_review_ledger(
                    Path(_product_dir) / "tr_reports" / "prediction_reviews.json",
                    selected_slug or "legacy",
                )
            except ReviewLedgerError:
                # The prediction-review section below reports the detailed error.
                pass
            _tr_grounding_review_summary = grounding_review_summary(
                _tr_grounding_review_ledger
            )
            _tr_current_grounding_review_summary = grounding_review_summary(
                _tr_grounding_review_ledger, set()
            )
            _tr_target_alias_review_policy = build_target_alias_policy(
                _tr_grounding_review_ledger,
                _tr_alias_prediction_review_ledger,
            )
            try:
                _tr_grounding_review_ledger = load_grounding_review_ledger(
                    _tr_grounding_review_fp, selected_slug or "legacy"
                )
                _tr_grounding_review_summary = grounding_review_summary(
                    _tr_grounding_review_ledger
                )
                _tr_target_alias_review_policy = build_target_alias_policy(
                    _tr_grounding_review_ledger,
                    _tr_alias_prediction_review_ledger,
                )
                if isinstance(_tr_grounding_audit, dict):
                    (
                        _tr_grounding_audit,
                        _tr_current_grounding_review_summary,
                    ) = overlay_grounding_reviews(
                        _tr_grounding_audit,
                        _tr_grounding_review_ledger,
                        str(_tr_analysis.get("source_sha256") or ""),
                        str(_tr_analysis.get("engine_sha256") or ""),
                        str(_tr_analysis.get("model") or ""),
                        _tr_build,
                    )
            except ReviewLedgerError as _tr_grounding_review_error:
                st.warning(
                    "Grounding recall reviews could not be loaded; the separate "
                    "ledger was left unchanged: "
                    + str(_tr_grounding_review_error)
                )
            if isinstance(_tr_grounding_audit, dict) and (
                _tr_grounding_audit.get("schema_version") == 1
            ):
                _tr_grounding_audited = int(
                    _tr_grounding_audit.get("audited_candidate_count") or 0
                )
                _tr_grounding_supported = int(
                    _tr_grounding_audit.get("supported_candidate_count") or 0
                )
                _tr_grounding_withheld = int(
                    _tr_grounding_audit.get("withheld_candidate_count") or 0
                )
                _tr_grounding_duplicates = int(
                    _tr_grounding_audit.get("duplicate_candidate_count") or 0
                )
                _tr_final_grounding_available = (
                    "final_recommendation_audited_count" in _tr_grounding_audit
                )
                _tr_final_grounding_audited = int(
                    _tr_grounding_audit.get(
                        "final_recommendation_audited_count"
                    ) or 0
                )
                _tr_final_grounding_supported = int(
                    _tr_grounding_audit.get(
                        "final_recommendation_supported_count"
                    ) or 0
                )
                _tr_final_grounding_grounded = int(
                    _tr_grounding_audit.get(
                        "final_recommendation_grounded_count",
                        _tr_final_grounding_supported
                        + int(_tr_grounding_audit.get(
                            "coverage_equivalent_recommendation_count"
                        ) or 0),
                    ) or 0
                )
                _tr_final_grounding_withheld = int(
                    _tr_grounding_audit.get(
                        "final_recommendation_withheld_count"
                    ) or 0
                )
                _tr_final_grounding_duplicates = int(
                    _tr_grounding_audit.get(
                        "duplicate_final_recommendation_count"
                    ) or 0
                )
                _tr_coverage_equivalent_count = int(
                    _tr_grounding_audit.get(
                        "coverage_equivalent_recommendation_count"
                    ) or 0
                )
                _tr_synthesis_safeguard = (
                    _tr_grounding_audit.get(
                        "synthesis_selection_safeguard"
                    ) or {}
                )
                _tr_synthesis_omitted = int(
                    _tr_synthesis_safeguard.get(
                        "synthesis_omitted_candidate_count"
                    ) or 0
                )
                _tr_synthesis_backfilled = int(
                    _tr_synthesis_safeguard.get(
                        "backfill_candidate_count"
                    ) or 0
                )
                _tr_synthesis_unbackfilled = int(
                    _tr_synthesis_safeguard.get(
                        "unbackfilled_omitted_candidate_count"
                    ) or 0
                )
                _tr_published_recommendations = int(
                    _tr_grounding_audit.get(
                        "published_recommendation_count",
                        _tr_final_grounding_supported,
                    ) or 0
                )
                _tr_portfolio_excluded_count = int(
                    _tr_grounding_audit.get(
                        "portfolio_excluded_recommendation_count"
                    ) or 0
                )
                _tr_grounding_repair_available = (
                    "initial_supported_candidate_count" in _tr_grounding_audit
                )
                _tr_grounding_initial_supported = int(
                    _tr_grounding_audit.get(
                        "initial_supported_candidate_count"
                    ) or 0
                )
                _tr_grounding_initial_withheld = int(
                    _tr_grounding_audit.get(
                        "initial_withheld_candidate_count"
                    ) or 0
                )
                _tr_grounding_recovered = int(
                    _tr_grounding_audit.get("recovered_candidate_count") or 0
                )
                _tr_discovery_audit = (
                    _tr_grounding_audit.get("candidate_discovery_audit") or {}
                )
                _tr_adjacency_gate_withheld_records = []
                if (
                    int(_tr_analysis.get("schema_version") or 0) >= 24
                    and isinstance(_tr_discovery_audit, dict)
                ):
                    _tr_discovery_primary = int(
                        _tr_discovery_audit.get("primary_candidate_count") or 0
                    )
                    _tr_discovery_challenge = int(
                        _tr_discovery_audit.get(
                            "coverage_challenge_candidate_count"
                        ) or 0
                    )
                    _tr_discovery_distinct = int(
                        _tr_discovery_audit.get(
                            "coverage_challenge_distinct_count"
                        ) or 0
                    )
                    _tr_discovery_overlap = int(
                        _tr_discovery_audit.get(
                            "coverage_overlap_filtered_count"
                        ) or 0
                    )
                    _tr_run_notes.append(
                        "**Discovery completeness** — primary pass proposed "
                        f"{_tr_discovery_primary} draft"
                        + ("s" if _tr_discovery_primary != 1 else "")
                        + "; the independent coverage challenge proposed "
                        f"{_tr_discovery_challenge}, adding "
                        f"{_tr_discovery_distinct} materially distinct draft"
                        + ("s" if _tr_discovery_distinct != 1 else "")
                        + f" and filtering {_tr_discovery_overlap} coverage-equivalent "
                        "variant"
                        + ("s" if _tr_discovery_overlap != 1 else "")
                        + " before grounding. Discovery path never changes confidence, "
                        "risk, or priority."
                    )
                    _tr_malformed_dropped = int(
                        _tr_discovery_audit.get(
                            "malformed_candidate_dropped_count"
                        ) or 0
                    )
                    _tr_malformed_records = (
                        _tr_discovery_audit.get(
                            "malformed_candidate_drop_records"
                        ) or []
                    )
                    if _tr_malformed_dropped:
                        _tr_run_notes.append(
                            "**Discarded model output** — "
                            f"{_tr_malformed_dropped} proposed draft"
                            + ("s" if _tr_malformed_dropped != 1 else "")
                            + (" were" if _tr_malformed_dropped != 1 else " was")
                            + " rejected as malformed before any evidence gate "
                            "and never became a candidate. A high share here "
                            "means the model is failing the response contract, "
                            "not that the evidence was weak."
                            + (
                                " Open *Drafts rejected as malformed* for the "
                                "reason each one failed."
                                if _tr_malformed_records else
                                " This run predates per-draft reason capture."
                            )
                        )
                    if _tr_malformed_records:
                        with st.expander(
                            "Drafts rejected as malformed",
                            expanded=False,
                        ):
                            st.caption(
                                "These proposals never reached the adjacency or "
                                "grounding gates: they failed the response "
                                "contract itself — a missing or wrong-typed "
                                "field, an uncitable BugCode, or an unparseable "
                                "response. They are a model-quality signal, not "
                                "a judgement about the build."
                            )
                            show_dataframe(
                                pd.DataFrame([
                                    {
                                        "Discovery path": str(
                                            _tr_dropped.get("discovery_origin")
                                            or ""
                                        ).replace("_", " "),
                                        "Batch": _tr_dropped.get("batch_number"),
                                        "Draft #": (
                                            _tr_dropped.get("candidate_index")
                                            if _tr_dropped.get("candidate_index")
                                            is not None else "whole response"
                                        ),
                                        "Proposed area": str(
                                            _tr_dropped.get("area") or "—"
                                        ),
                                        "Cited bugs": " · ".join(
                                            str(code) for code in (
                                                _tr_dropped.get(
                                                    "source_bug_codes"
                                                ) or []
                                            )
                                        ) or "—",
                                        "Why it was rejected": " · ".join(
                                            str(reason) for reason in (
                                                _tr_dropped.get(
                                                    "rejection_reasons"
                                                ) or []
                                            )
                                        ),
                                    }
                                    for _tr_dropped in _tr_malformed_records
                                    if isinstance(_tr_dropped, dict)
                                ]),
                                width="stretch",
                                hide_index=True,
                            )
                    if int(_tr_analysis.get("schema_version") or 0) >= 25:
                        _tr_adjacency_gate_evaluated = int(
                            _tr_discovery_audit.get(
                                "adjacency_gate_evaluated_count"
                            ) or 0
                        )
                        _tr_adjacency_gate_supported = int(
                            _tr_discovery_audit.get(
                                "adjacency_gate_supported_count"
                            ) or 0
                        )
                        _tr_adjacency_gate_withheld = int(
                            _tr_discovery_audit.get(
                                "adjacency_gate_withheld_count"
                            ) or 0
                        )
                        _tr_shared_path_gate_withheld = int(
                            _tr_discovery_audit.get(
                                "shared_path_gate_withheld_count"
                            ) or 0
                        )
                        _tr_run_notes.append(
                            "**Deterministic adjacency support** — "
                            f"{_tr_adjacency_gate_supported} of "
                            f"{_tr_adjacency_gate_evaluated} discovered draft"
                            + (
                                "s" if _tr_adjacency_gate_evaluated != 1 else ""
                            )
                            + (
                                " named both their target and a distinctive "
                                "non-target shared-path term in exact cited "
                                "current-fix evidence and could proceed; "
                                if int(_tr_analysis.get("schema_version") or 0) >= 29
                                else " named their target in exact cited current-fix "
                                "evidence and could proceed; "
                            )
                            + f"{_tr_adjacency_gate_withheld} unsupported draft"
                            + (
                                "s" if _tr_adjacency_gate_withheld != 1 else ""
                            )
                            + " were withheld before the model critic."
                            + (
                                f" {_tr_shared_path_gate_withheld} failed only the "
                                "deterministic shared-path rationale check."
                                if int(_tr_analysis.get("schema_version") or 0) >= 29
                                else ""
                            )
                            + (
                                f" {int(_tr_discovery_audit.get('direct_fix_gate_withheld_count') or 0)}"
                                " restated the cited fix instead of naming an "
                                "adjacent area."
                                if int(_tr_discovery_audit.get(
                                    "direct_fix_gate_withheld_count"
                                ) or 0) else ""
                            )
                        )
                        _tr_adjacency_gate_withheld_records = (
                            _tr_discovery_audit.get(
                                "adjacency_gate_withheld_candidates"
                            ) or []
                        )
                        if _tr_adjacency_gate_withheld_records:
                            with st.expander(
                                "Drafts withheld by deterministic evidence gate",
                                expanded=False,
                            ):
                                show_dataframe([
                                    {
                                        "Candidate ID": str(
                                            record.get("candidate_id") or ""
                                        ),
                                        "Discovery path": str(
                                            record.get("discovery_origin") or ""
                                        ).replace("_", " "),
                                        "Area": str(
                                            (record.get("candidate") or {}).get(
                                                "area"
                                            ) or ""
                                        ),
                                        "Reason": _tr_grounding_reason_labels.get(
                                            str(record.get("reason") or ""),
                                            str(record.get("reason") or "").replace(
                                                "_", " "
                                            ),
                                        ),
                                        "Required target terms": ", ".join(
                                            str(term) for term in (
                                                record.get(
                                                    "adjacency_support_basis"
                                                ) or {}
                                            ).get("target_terms") or []
                                        ),
                                        "Matched target terms": ", ".join(
                                            str(term) for term in (
                                                record.get(
                                                    "adjacency_support_basis"
                                                ) or {}
                                            ).get("target_anchor_terms") or []
                                        ),
                                        "Rationale path terms": ", ".join(
                                            str(term) for term in (
                                                (record.get("candidate") or {}).get(
                                                    "shared_path_support_basis"
                                                ) or {}
                                            ).get("matched_path_terms") or []
                                        ),
                                    }
                                    for record in (
                                        _tr_adjacency_gate_withheld_records
                                    )
                                    if isinstance(record, dict)
                                ], hide_index=True, width="stretch")
                        if int(_tr_analysis.get("schema_version") or 0) >= 26:
                            _tr_ref_completion_records = (
                                _tr_discovery_audit.get(
                                    "evidence_ref_completion_candidates"
                                ) or []
                            )
                            _tr_ref_completion_count = int(
                                _tr_discovery_audit.get(
                                    "evidence_ref_completion_applied_count"
                                ) or 0
                            )
                            _tr_ref_completion_recovers_families = int(
                                _tr_analysis.get("schema_version") or 0
                            ) >= 40
                            _tr_run_notes.append(
                                "**Deterministic reference recovery** — "
                                f"{_tr_ref_completion_count} discovered draft"
                                + (
                                    "s" if _tr_ref_completion_count != 1 else ""
                                )
                                + (
                                    " named a target, or invoked a stated fix "
                                    "family, that was present in another exact "
                                    "signal from an already-cited BugCode."
                                    if _tr_ref_completion_recovers_families
                                    else " named a target present in another "
                                    "exact signal from an already-cited BugCode."
                                )
                                + " Those references were "
                                "added within the four-reference cap before the target "
                                "gate; no BugCode or synonym scope was added."
                            )
                            if _tr_ref_completion_records:
                                with st.expander(
                                    "Exact references recovered before evidence gate"
                                    if _tr_ref_completion_recovers_families
                                    else "Target references recovered before evidence gate",
                                    expanded=False,
                                ):
                                    show_dataframe([
                                        {
                                            "Candidate ID": str(
                                                record.get("candidate_id") or ""
                                            ),
                                            "Discovery path": str(
                                                record.get("discovery_origin") or ""
                                            ).replace("_", " "),
                                            "Area": str(record.get("area") or ""),
                                            "Original references": " · ".join(
                                                f"{ref.get('bug_code')}"
                                                f"[{ref.get('signal_index')}]"
                                                for ref in (
                                                    record.get(
                                                        "original_evidence_refs"
                                                    ) or []
                                                ) if isinstance(ref, dict)
                                            ),
                                            "Added exact evidence": " · ".join(
                                                f"{ref.get('bug_code')}"
                                                f"[{ref.get('signal_index')}]: "
                                                f"{ref.get('quote')}"
                                                for ref in (
                                                    record.get(
                                                        "added_evidence_refs"
                                                    ) or []
                                                ) if isinstance(ref, dict)
                                            ),
                                        }
                                        for record in _tr_ref_completion_records
                                        if isinstance(record, dict)
                                    ], hide_index=True, width="stretch")
                        if int(_tr_analysis.get("schema_version") or 0) >= 27:
                            _tr_alias_supported_records = (
                                _tr_discovery_audit.get(
                                    "reviewed_alias_supported_candidates"
                                ) or []
                            )
                            _tr_alias_supported_count = int(
                                _tr_discovery_audit.get(
                                    "reviewed_alias_supported_count"
                                ) or 0
                            )
                            _tr_run_notes.append(
                                "**Reviewed vocabulary support** — "
                                f"{_tr_alias_supported_count} discovered draft"
                                + (
                                    "s" if _tr_alias_supported_count != 1 else ""
                                )
                                + " passed the target gate through an active human-"
                                "confirmed directional alias. These aliases do not "
                                "change confidence, risk, or priority."
                            )
                            if _tr_alias_supported_records:
                                with st.expander(
                                    "Predictions supported by reviewed vocabulary",
                                    expanded=False,
                                ):
                                    show_dataframe([
                                        {
                                            "Candidate ID": str(
                                                record.get("candidate_id") or ""
                                            ),
                                            "Discovery path": str(
                                                record.get("discovery_origin") or ""
                                            ).replace("_", " "),
                                            "Area": str(record.get("area") or ""),
                                            "Reviewed matches": " · ".join(
                                                f"{match.get('target_term')} → "
                                                f"{match.get('evidence_alias_term')} "
                                                f"at {match.get('bug_code')}"
                                                f"[{match.get('signal_index')}]"
                                                for match in (
                                                    record.get(
                                                        "reviewed_alias_matches"
                                                    ) or []
                                                ) if isinstance(match, dict)
                                            ),
                                        }
                                        for record in _tr_alias_supported_records
                                        if isinstance(record, dict)
                                    ], hide_index=True, width="stretch")
                    _tr_discovery_overlap_records = (
                        _tr_discovery_audit.get(
                            "coverage_overlap_filtered_candidates"
                        ) or []
                    )
                    if _tr_discovery_overlap_records:
                        with st.expander(
                            "Coverage-challenge variants filtered before grounding",
                            expanded=False,
                        ):
                            show_dataframe([
                                {
                                    "Candidate ID": str(
                                        record.get("candidate_id") or ""
                                    ),
                                    "Covered by": str(
                                        record.get(
                                            "equivalent_to_candidate_id"
                                        ) or ""
                                    ),
                                    "Coverage signature": str(
                                        record.get("coverage_signature") or ""
                                    ),
                                    "Area": str(
                                        (record.get("candidate") or {}).get(
                                            "area"
                                        ) or ""
                                    ),
                                }
                                for record in _tr_discovery_overlap_records
                                if isinstance(record, dict)
                            ], hide_index=True, width="stretch")
                _tr_run_notes.append(
                    "**Adversarial grounding audit** — "
                    + (
                        f"{_tr_grounding_initial_supported} of "
                        f"{_tr_grounding_audited} unique drafts passed initially; "
                        f"{_tr_grounding_recovered} of "
                        f"{_tr_grounding_initial_withheld} initially withheld drafts "
                        "were recovered by one bounded repair and fresh six-check "
                        f"audit; {_tr_grounding_withheld} remained withheld"
                        if _tr_grounding_repair_available else
                        f"{_tr_grounding_supported} of {_tr_grounding_audited} unique "
                        "drafts passed all six evidence checks"
                    )
                    + (
                        f"; {_tr_grounding_withheld} withheld before synthesis"
                        if _tr_grounding_withheld
                        and not _tr_grounding_repair_available else ""
                    )
                    + (
                        f"; {_tr_grounding_duplicates} exact duplicates removed"
                        if _tr_grounding_duplicates else ""
                    )
                    + (
                        f"; {_tr_final_grounding_grounded} of "
                        f"{_tr_final_grounding_audited} "
                        + (
                            "synthesized or backfilled recommendations "
                            if int(_tr_analysis.get("schema_version") or 0) >= 23
                            else "synthesized recommendations "
                        )
                        + "passed the final audit"
                        if _tr_final_grounding_available else ""
                    )
                    + (
                        f"; {_tr_final_grounding_withheld} withheld after synthesis"
                        if _tr_final_grounding_withheld else ""
                    )
                    + (
                        f"; {_tr_final_grounding_duplicates} exact synthesized "
                        "duplicates removed"
                        if _tr_final_grounding_duplicates else ""
                    )
                    + (
                        f"; {_tr_coverage_equivalent_count} wording variant"
                        + ("s" if _tr_coverage_equivalent_count != 1 else "")
                        + " suppressed after passing as coverage-equivalent"
                        if _tr_coverage_equivalent_count else ""
                    )
                    + (
                        f"; {_tr_final_grounding_supported} distinct supported "
                        "recommendation"
                        + ("s" if _tr_final_grounding_supported != 1 else "")
                        + (
                            f", {_tr_published_recommendations} published"
                            f" and {_tr_portfolio_excluded_count} preserved outside "
                            "the visible portfolio"
                            if int(_tr_analysis.get("schema_version") or 0) >= 23
                            and _tr_portfolio_excluded_count
                            else " published"
                        )
                        if int(_tr_analysis.get("schema_version") or 0) >= 14
                        else ""
                    )
                    + "."
                )
                _tr_generalization_overrides = int(
                    _tr_grounding_audit.get(
                        "generalization_target_override_count"
                    ) or 0
                )
                if _tr_generalization_overrides:
                    _tr_run_notes.append(
                        "**Stated-generalization support** — "
                        f"{_tr_generalization_overrides} draft"
                        + ("s" if _tr_generalization_overrides != 1 else "")
                        + " kept a target the cited fix note generalizes to "
                        "without naming, over the model critic's target "
                        "verdict. No other check was overridden and no label "
                        "was raised."
                    )
                if int(_tr_analysis.get("schema_version") or 0) >= 23:
                    _tr_run_notes.append(
                        "**Selection completeness** — AI synthesis omitted "
                        f"{_tr_synthesis_omitted} grounded draft"
                        + ("s" if _tr_synthesis_omitted != 1 else "")
                        + "; deterministic coverage-diverse backfill promoted "
                        f"{_tr_synthesis_backfilled} into the final audit"
                        + (
                            f"; {_tr_synthesis_unbackfilled} remained in accepted-"
                            "draft provenance outside the bounded backfill"
                            if _tr_synthesis_unbackfilled else ""
                        )
                        + "."
                    )
                _tr_coverage_equivalents = (
                    _tr_grounding_audit.get(
                        "coverage_equivalent_recommendations"
                    ) or []
                )
                if _tr_coverage_equivalents:
                    _tr_coverage_rows = []
                    for _tr_record in _tr_coverage_equivalents:
                        if not isinstance(_tr_record, dict):
                            continue
                        _tr_suppressed = _tr_record.get(
                            "suppressed_recommendation"
                        ) or {}
                        _tr_coverage_rows.append({
                            "Coverage signature": str(
                                _tr_record.get("coverage_signature") or ""
                            ),
                            "Retained claim": str(
                                _tr_record.get("kept_recommendation_id") or ""
                            ),
                            "Suppressed claim": str(
                                _tr_record.get(
                                    "suppressed_recommendation_id"
                                ) or ""
                            ),
                            "Final grounding": str(
                                _tr_record.get("suppressed_grounding_status")
                                or "legacy"
                            ),
                            "Target category": str(
                                _tr_suppressed.get("target_category") or ""
                            ),
                            "Suppressed wording": " | ".join(filter(None, [
                                str(_tr_suppressed.get("area") or ""),
                                str(_tr_suppressed.get("trigger") or ""),
                                str(_tr_suppressed.get(
                                    "expected_failure_mode"
                                ) or ""),
                            ])),
                            "Source BugCodes": " · ".join(
                                str(code) for code in (
                                    _tr_suppressed.get("source_bug_codes") or []
                                )
                            ),
                            "Evidence preserved": " · ".join(
                                f"{ref.get('bug_code')}[{ref.get('signal_index')}]"
                                for ref in (
                                    _tr_suppressed.get("evidence_refs") or []
                                ) if isinstance(ref, dict)
                            ),
                        })
                    with st.expander(
                        "Coverage-equivalent wording variants",
                        expanded=False,
                    ):
                        st.caption(
                            "Only exact normalized facet sets are collapsed. Any "
                            "different category, target, trigger, failure, boundary, "
                            "disconfirmation, or QA-action term remains a separate "
                            "recommendation. In schema 14, every displayed row first "
                            "passed the final adversarial grounding audit."
                        )
                        show_dataframe(
                            pd.DataFrame(_tr_coverage_rows),
                            width="stretch",
                            hide_index=True,
                        )
                _tr_portfolio_excluded = (
                    _tr_grounding_audit.get(
                        "portfolio_excluded_recommendations"
                    ) or []
                )
                _tr_portfolio_policy = (
                    _tr_analysis.get("recommendation_portfolio_policy") or {}
                )
                _tr_cutoff_selection_records = (
                    _tr_portfolio_policy.get("cutoff_selection_records") or []
                )
                if _tr_cutoff_selection_records:
                    with st.expander(
                        "Equal-tier visible portfolio selection evidence",
                        expanded=False,
                    ):
                        st.caption(
                            "Original priority is preserved as the visible order. "
                            "Coverage novelty affects membership only inside the "
                            "single risk/confidence tier crossing the cutoff."
                        )
                        show_dataframe(
                            pd.DataFrame([{
                                "Selection round": int(
                                    record.get("selection_round") or 0
                                ),
                                "Original priority": int(
                                    record.get("original_priority_position") or 0
                                ),
                                "Hypothesis": str(
                                    record.get("hypothesis_id") or ""
                                ),
                                "Reason": str(
                                    record.get("selection_reason") or ""
                                ).replace("_", " "),
                                "New category": str(
                                    record.get("novel_target_category") or ""
                                ),
                                "New path terms": " · ".join(
                                    str(term) for term in (
                                        record.get("novel_path_terms") or []
                                    )
                                ),
                                "New source BugCodes": " · ".join(
                                    str(code) for code in (
                                        record.get("novel_source_bug_codes") or []
                                    )
                                ),
                            } for record in _tr_cutoff_selection_records]),
                            width="stretch",
                            hide_index=True,
                        )
                if _tr_portfolio_excluded:
                    _tr_portfolio_rows = []
                    for _tr_record in _tr_portfolio_excluded:
                        if not isinstance(_tr_record, dict):
                            continue
                        _tr_item = _tr_record.get("recommendation") or {}
                        _tr_portfolio_rows.append({
                            "Original priority position": int(
                                _tr_record.get("portfolio_position") or 0
                            ),
                            "Area": str(_tr_item.get("area") or ""),
                            "Risk": str(_tr_item.get("risk") or ""),
                            "Confidence": str(
                                _tr_item.get("confidence") or ""
                            ).title(),
                            "Exclusion reason": str(
                                _tr_record.get("reason") or ""
                            ).replace("_", " "),
                            "Selection path": str(
                                _tr_item.get("selection_origin") or ""
                            ).replace("_", " "),
                            "Source BugCodes": " · ".join(
                                str(code) for code in (
                                    _tr_item.get("source_bug_codes") or []
                                )
                            ),
                        })
                    with st.expander(
                        "Supported claims outside the visible top-eight portfolio",
                        expanded=False,
                    ):
                        st.caption(
                            "These claims passed the final grounding audit. They "
                            "either fell below the risk/confidence cutoff or were "
                            "outside the diversified slots within the equal cutoff "
                            "tier; their original priority and reason remain frozen."
                        )
                        show_dataframe(
                            pd.DataFrame(_tr_portfolio_rows),
                            width="stretch",
                            hide_index=True,
                        )
                _tr_withheld_candidates = (
                    _tr_grounding_audit.get("withheld_candidates") or []
                )
                _tr_final_withheld_recommendations = (
                    _tr_grounding_audit.get(
                        "final_withheld_recommendations"
                    ) or []
                )
                _tr_recovered_candidates = (
                    _tr_grounding_audit.get("recovered_candidates") or []
                )
                if _tr_recovered_candidates:
                    _tr_recovered_rows = []
                    for _tr_recovered in _tr_recovered_candidates:
                        if not isinstance(_tr_recovered, dict):
                            continue
                        _tr_recovered_rows.append({
                            "Original candidate": str(
                                _tr_recovered.get("original_candidate_id") or ""
                            ),
                            "Repaired candidate": str(
                                _tr_recovered.get("repaired_candidate_id") or ""
                            ),
                            "Target category": str(
                                _tr_recovered.get("target_category") or ""
                            ),
                            "Area": str(_tr_recovered.get("area") or ""),
                            "Initial failure": _tr_grounding_reason_labels.get(
                                str(_tr_recovered.get("initial_reason") or ""),
                                str(_tr_recovered.get("initial_reason") or "")
                                .replace("_", " "),
                            ),
                            "Changed fields": " · ".join(
                                str(field).replace("_", " ") for field in (
                                    _tr_recovered.get("changed_fields") or []
                                )
                            ),
                            "Evidence preserved": " · ".join(
                                f"{ref.get('bug_code')}[{ref.get('signal_index')}]"
                                for ref in (
                                    _tr_recovered.get("evidence_refs") or []
                                ) if isinstance(ref, dict)
                            ),
                        })
                    with st.expander(
                        "Drafts recovered by bounded grounding repair",
                        expanded=False,
                    ):
                        st.caption(
                            "Each draft kept its category, source BugCodes, exact "
                            "evidence and historical references, and did not increase "
                            "confidence. It reached synthesis only after a fresh "
                            "six-check adversarial pass."
                        )
                        show_dataframe(
                            pd.DataFrame(_tr_recovered_rows),
                            width="stretch",
                            hide_index=True,
                        )
                if (
                    _tr_adjacency_gate_withheld_records
                    or
                    _tr_withheld_candidates
                    or _tr_final_withheld_recommendations
                ):
                    _tr_withheld_rows = []
                    _tr_grounding_review_samples = []
                    for _tr_phase, _tr_withheld in (
                        [(
                            "Deterministic target/shared-path gate", item
                        ) for item in _tr_adjacency_gate_withheld_records]
                        + [("Draft before synthesis", item) for item in _tr_withheld_candidates]
                        + [("Final after synthesis", item) for item in _tr_final_withheld_recommendations]
                    ):
                        if not isinstance(_tr_withheld, dict):
                            continue
                        if _tr_withheld.get("human_review_sample"):
                            _tr_grounding_review_samples.append(_tr_withheld)
                        _tr_grounding_review_status = str(
                            _tr_withheld.get("grounding_review_status")
                            or "unreviewed"
                        )
                        _tr_withheld_rows.append({
                            "Stage": _tr_phase,
                            "Candidate": str(
                                _tr_withheld.get("candidate_id") or ""
                            ),
                            "Target category": str(
                                _tr_withheld.get("target_category") or ""
                            ),
                            "Area": str(_tr_withheld.get("area") or ""),
                            "Why withheld": _tr_grounding_reason_labels.get(
                                str(_tr_withheld.get("reason") or ""),
                                str(_tr_withheld.get("reason") or "")
                                .replace("_", " "),
                            ),
                            "Failed checks": " · ".join(
                                str(check).replace("_", " ")
                                for check in (
                                    _tr_withheld.get("failed_checks") or []
                                )
                            ),
                            "Repair outcome": str(
                                _tr_withheld.get("repair_status") or "not attempted"
                            ).replace("_", " "),
                            "Original candidate": str(
                                _tr_withheld.get("original_candidate_id") or ""
                            ),
                            "Changed fields": " · ".join(
                                str(field).replace("_", " ") for field in (
                                    _tr_withheld.get("changed_fields") or []
                                )
                            ),
                            "Recall sample": (
                                "Yes" if _tr_withheld.get("human_review_sample")
                                else "No"
                            ),
                            "Human review": _tr_grounding_review_status.replace(
                                "_", " "
                            ),
                            "Current-fix evidence": " · ".join(
                                f"{ref.get('bug_code')}[{ref.get('signal_index')}]"
                                for ref in (
                                    _tr_withheld.get("evidence_refs") or []
                                )
                                if isinstance(ref, dict)
                            ),
                        })
                    with st.expander(
                        "Candidates withheld by grounding audit",
                        expanded=False,
                    ):
                        st.caption(
                            "These structurally valid drafts or synthesized results "
                            "failed a deterministic target/shared-path gate or an "
                            "adversarial evidence pass and were never published as "
                            "recommendations. Historical analogs cannot rescue missing "
                            "current-fix grounding."
                        )
                        show_dataframe(
                            pd.DataFrame(_tr_withheld_rows),
                            width="stretch",
                            hide_index=True,
                        )
                        _tr_grounding_all_decisive = int(
                            _tr_grounding_review_summary.get("decisive_count") or 0
                        )
                        _tr_grounding_all_false_negative = int(
                            _tr_grounding_review_summary.get("false_negative_count")
                            or 0
                        )
                        if _tr_grounding_all_decisive:
                            _tr_grounding_rate = float(
                                _tr_grounding_review_summary.get(
                                    "false_negative_rate"
                                ) or 0
                            )
                            _tr_grounding_upper = float(
                                _tr_grounding_review_summary.get(
                                    "false_negative_wilson_upper_bound"
                                ) or 0
                            )
                            st.caption(
                                "Target-gate and grounding-critic recall reviews "
                                "across saved runs — "
                                f"{_tr_grounding_all_false_negative} false negatives "
                                f"among {_tr_grounding_all_decisive} decisive sampled "
                                f"reviews ({_tr_grounding_rate:.0%}; conservative 95% "
                                f"upper bound {_tr_grounding_upper:.0%}). Unclear reviews "
                                "are excluded. This ledger is observational and is never "
                                "used by prediction matching or guardrail calibration."
                            )
                            _tr_grounding_breakdown_rows = []
                            for _tr_dimension, _tr_groups in (
                                (
                                    "Stage",
                                    _tr_grounding_review_summary.get("by_stage")
                                    or {},
                                ),
                                (
                                    "Reason",
                                    _tr_grounding_review_summary.get("by_reason")
                                    or {},
                                ),
                            ):
                                for _tr_group_name, _tr_group in _tr_groups.items():
                                    if not isinstance(_tr_group, dict) or not int(
                                        _tr_group.get("reviewed_count") or 0
                                    ):
                                        continue
                                    _tr_group_rate = _tr_group.get(
                                        "false_negative_rate"
                                    )
                                    _tr_group_upper = _tr_group.get(
                                        "false_negative_wilson_upper_bound"
                                    )
                                    _tr_grounding_breakdown_rows.append({
                                        "Dimension": _tr_dimension,
                                        "Group": str(_tr_group_name).replace("_", " "),
                                        "Reviewed": int(
                                            _tr_group.get("reviewed_count") or 0
                                        ),
                                        "Decisive": int(
                                            _tr_group.get("decisive_count") or 0
                                        ),
                                        "False negatives": int(
                                            _tr_group.get("false_negative_count") or 0
                                        ),
                                        "Observed rate": (
                                            f"{float(_tr_group_rate):.0%}"
                                            if _tr_group_rate is not None else "—"
                                        ),
                                        "95% upper bound": (
                                            f"{float(_tr_group_upper):.0%}"
                                            if _tr_group_upper is not None else "—"
                                        ),
                                    })
                            if _tr_grounding_breakdown_rows:
                                show_dataframe(
                                    pd.DataFrame(_tr_grounding_breakdown_rows),
                                    width="stretch",
                                    hide_index=True,
                                )
                        _tr_alias_candidate_rows = (
                            _tr_target_alias_review_policy.get(
                                "candidate_aliases"
                            ) or []
                        )
                        if _tr_alias_candidate_rows:
                            st.markdown("#### Reviewed target vocabulary")
                            st.caption(
                                "Directional aliases remain inactive until the same "
                                "explicit term pair has four confirmed false-negative "
                                "reviews across at least three Close Builds and four "
                                "immutable source runs. Reviewer notes are never copied "
                                "into analysis input. Retirement requires at least 12 "
                                "explicit alias outcome reviews (eight effective) across "
                                "three cohorts and hypotheses, with a conservative lower "
                                "bound showing repeated alias-specific failures. Matcher "
                                "and other rejection reasons cannot retire an alias."
                            )
                            show_dataframe(pd.DataFrame([{
                                "Alias": (
                                    f"{row.get('target_term')} → "
                                    f"{row.get('evidence_term')}"
                                ),
                                "Status": str(
                                    row.get("status") or ""
                                ).replace("_", " "),
                                "Reviews": int(
                                    row.get("confirmed_review_count") or 0
                                ),
                                "Close Builds": int(
                                    row.get("distinct_close_build_count") or 0
                                ),
                                "Source runs": int(
                                    row.get("distinct_source_run_count") or 0
                                ),
                                "Candidates": int(
                                    row.get("distinct_candidate_count") or 0
                                ),
                                "Alias outcomes": int(
                                    (row.get("outcome_evidence") or {}).get(
                                        "decisive_review_count"
                                    ) or 0
                                ),
                                "Confirmed outcomes": int(
                                    (row.get("outcome_evidence") or {}).get(
                                        "confirmed_review_count"
                                    ) or 0
                                ),
                                "Alias-specific failures": int(
                                    (row.get("outcome_evidence") or {}).get(
                                        "alias_misleading_review_count"
                                    ) or 0
                                ),
                                "Other rejections excluded": int(
                                    (row.get("outcome_evidence") or {}).get(
                                        "excluded_other_rejection_count"
                                    ) or 0
                                ),
                                "Failure lower bound": (
                                    f"{float((row.get('outcome_evidence') or {}).get('alias_misleading_wilson_lower_bound')):.0%}"
                                    if (row.get("outcome_evidence") or {}).get(
                                        "decisive_review_count"
                                    ) else "—"
                                ),
                            } for row in _tr_alias_candidate_rows]),
                                width="stretch", hide_index=True,
                            )
                        if (
                            not _tr_grounding_all_decisive
                            and _tr_grounding_review_samples
                        ):
                            st.caption(
                                "No decisive grounding-recall reviews yet. The separate "
                                "ledger is observational and never calibrates prediction "
                                "matching or AI guardrails."
                            )
                        if _tr_grounding_review_samples:
                            st.markdown("#### Sampled recall review")
                            st.caption(
                                "Review only the deterministic reason-and-stage-diverse "
                                "sample. Mark False negative only when the withheld item "
                                "should have advanced on the frozen evidence."
                            )
                            _tr_grounding_decision_labels = {
                                "Correctly withheld": "correctly_withheld",
                                "False negative — should have advanced": "false_negative",
                                "Unclear": "unclear",
                            }
                            _tr_grounding_decision_labels_reverse = {
                                value: label for label, value
                                in _tr_grounding_decision_labels.items()
                            }
                            for _tr_sample in _tr_grounding_review_samples:
                                _tr_grounding_key = str(
                                    _tr_sample.get("grounding_review_key") or ""
                                )
                                if not _tr_grounding_key:
                                    continue
                                _tr_sample_id = str(
                                    _tr_sample.get("candidate_id") or "candidate"
                                )
                                _tr_existing_grounding_decision = str(
                                    _tr_sample.get("grounding_review_status") or ""
                                )
                                _tr_default_grounding_label = (
                                    _tr_grounding_decision_labels_reverse.get(
                                        _tr_existing_grounding_decision,
                                        "Correctly withheld",
                                    )
                                )
                                _tr_alias_none = "No reviewed alias"
                                _tr_alias_eligible = (
                                    _tr_sample.get("audit_stage")
                                    == "deterministic_adjacency_gate"
                                )
                                _tr_existing_alias_pair = (
                                    _tr_sample.get(
                                        "grounding_review_target_alias_pair"
                                    ) or {}
                                )
                                _tr_alias_generic_terms = {
                                    "another", "backed", "based", "behavior",
                                    "cache", "changed", "editor", "feature",
                                    "fixed", "flow", "manager", "other",
                                    "preview", "related", "renderer", "result",
                                    "service", "shared", "state", "surface",
                                    "system", "updated", "workflow",
                                }
                                _tr_alias_target_options = [
                                    _tr_alias_none,
                                    *sorted({
                                        str(term).casefold()
                                        for term in (
                                            _tr_sample.get(
                                                "adjacency_support_basis"
                                            ) or {}
                                        ).get("target_terms") or []
                                        if str(term).casefold()
                                        not in _tr_alias_generic_terms
                                    }),
                                ]
                                _tr_alias_evidence_options = [
                                    _tr_alias_none,
                                    *sorted({
                                        term
                                        for ref in (
                                            _tr_sample.get("evidence_refs") or []
                                        ) if isinstance(ref, dict)
                                        for term in _tr_re.findall(
                                            r"[a-z0-9]+",
                                            str(ref.get("quote") or "").casefold(),
                                        )
                                        if len(term) >= 2
                                        and not term.isdigit()
                                        and term not in _tr_alias_generic_terms
                                        and term not in _tr_alias_target_options
                                    }),
                                ]
                                with st.form(
                                    key=f"tr_grounding_review_{_tr_grounding_key}"
                                ):
                                    st.markdown(
                                        f"**{_tr_sample_id} · "
                                        f"{str(_tr_sample.get('audit_stage') or '').replace('_', ' ')}**"
                                    )
                                    _tr_grounding_decision_label = st.selectbox(
                                        "Decision",
                                        list(_tr_grounding_decision_labels),
                                        index=list(_tr_grounding_decision_labels).index(
                                            _tr_default_grounding_label
                                        ),
                                        key=(
                                            "tr_grounding_decision_"
                                            + _tr_grounding_key
                                        ),
                                    )
                                    _tr_failed_grounding_checks = list(
                                        _tr_sample.get("failed_checks") or []
                                    )
                                    _tr_grounding_overturned = st.multiselect(
                                        "Failed evidence checks to overturn",
                                        _tr_failed_grounding_checks,
                                        default=[
                                            check for check in (
                                                _tr_sample.get(
                                                    "grounding_review_overturned_checks"
                                                ) or []
                                            ) if check in _tr_failed_grounding_checks
                                        ],
                                        key=(
                                            "tr_grounding_overturned_"
                                            + _tr_grounding_key
                                        ),
                                    )
                                    if _tr_alias_eligible:
                                        st.caption(
                                            "Optional reviewed vocabulary bridge: "
                                            "select a directional target term → exact "
                                            "cited-signal term only when that wording "
                                            "difference caused the false negative."
                                        )
                                        _tr_alias_target_term = st.selectbox(
                                            "Prediction target term",
                                            _tr_alias_target_options,
                                            index=(
                                                _tr_alias_target_options.index(
                                                    _tr_existing_alias_pair.get(
                                                        "target_term"
                                                    )
                                                )
                                                if _tr_existing_alias_pair.get(
                                                    "target_term"
                                                ) in _tr_alias_target_options else 0
                                            ),
                                            key="tr_alias_target_" + _tr_grounding_key,
                                        )
                                        _tr_alias_evidence_term = st.selectbox(
                                            "Exact cited-signal term",
                                            _tr_alias_evidence_options,
                                            index=(
                                                _tr_alias_evidence_options.index(
                                                    _tr_existing_alias_pair.get(
                                                        "evidence_term"
                                                    )
                                                )
                                                if _tr_existing_alias_pair.get(
                                                    "evidence_term"
                                                ) in _tr_alias_evidence_options else 0
                                            ),
                                            key="tr_alias_evidence_" + _tr_grounding_key,
                                        )
                                    else:
                                        _tr_alias_target_term = _tr_alias_none
                                        _tr_alias_evidence_term = _tr_alias_none
                                    _tr_grounding_reviewer = st.text_input(
                                        "Reviewer (optional)",
                                        value=str(
                                            _tr_sample.get("grounding_reviewer") or ""
                                        ),
                                        key=(
                                            "tr_grounding_reviewer_"
                                            + _tr_grounding_key
                                        ),
                                    )
                                    _tr_grounding_note = st.text_area(
                                        "Review note (optional)",
                                        value=str(
                                            _tr_sample.get("grounding_review_note") or ""
                                        ),
                                        key=(
                                            "tr_grounding_note_"
                                            + _tr_grounding_key
                                        ),
                                    )
                                    _tr_save_grounding_review = st.form_submit_button(
                                        f"Save recall review for {_tr_sample_id}"
                                    )
                                if _tr_save_grounding_review:
                                    _tr_grounding_decision = (
                                        _tr_grounding_decision_labels[
                                            _tr_grounding_decision_label
                                        ]
                                    )
                                    if (
                                        _tr_grounding_decision == "false_negative"
                                        and not _tr_grounding_overturned
                                    ):
                                        st.error(
                                            "Select at least one failed evidence check "
                                            "to overturn for a false negative."
                                        )
                                    elif (
                                        (_tr_alias_target_term == _tr_alias_none)
                                        != (_tr_alias_evidence_term == _tr_alias_none)
                                    ):
                                        st.error(
                                            "Select both reviewed alias terms or leave "
                                            "both unselected."
                                        )
                                    else:
                                        try:
                                            upsert_grounding_review(
                                                _tr_grounding_review_fp,
                                                product=selected_slug or "legacy",
                                                analysis_source_sha256=str(
                                                    _tr_analysis.get("source_sha256") or ""
                                                ),
                                                analysis_engine_sha256=str(
                                                    _tr_analysis.get("engine_sha256") or ""
                                                ),
                                                analysis_model=str(
                                                    _tr_analysis.get("model") or ""
                                                ),
                                                close_build=_tr_build,
                                                candidate_id=_tr_sample_id,
                                                audit_stage=str(
                                                    _tr_sample.get("audit_stage") or ""
                                                ),
                                                withheld_reason=str(
                                                    _tr_sample.get("reason") or ""
                                                ),
                                                failed_checks=_tr_failed_grounding_checks,
                                                target_category=str(
                                                    _tr_sample.get("target_category") or ""
                                                ),
                                                area=str(_tr_sample.get("area") or ""),
                                                source_bug_codes=list(
                                                    _tr_sample.get("source_bug_codes") or []
                                                ),
                                                evidence_refs=list(
                                                    _tr_sample.get("evidence_refs") or []
                                                ),
                                                decision=_tr_grounding_decision,
                                                overturned_checks=(
                                                    _tr_grounding_overturned
                                                    if _tr_grounding_decision
                                                    == "false_negative" else []
                                                ),
                                                target_alias_pair=(
                                                    {
                                                        "target_term": (
                                                            _tr_alias_target_term
                                                        ),
                                                        "evidence_term": (
                                                            _tr_alias_evidence_term
                                                        ),
                                                    }
                                                    if (
                                                        _tr_grounding_decision
                                                        == "false_negative"
                                                        and _tr_alias_target_term
                                                        != _tr_alias_none
                                                        and _tr_alias_evidence_term
                                                        != _tr_alias_none
                                                    ) else None
                                                ),
                                                reviewer=_tr_grounding_reviewer,
                                                note=_tr_grounding_note,
                                            )
                                            st.success(
                                                f"Saved recall review for {_tr_sample_id}."
                                            )
                                            st.rerun()
                                        except ReviewLedgerError as _tr_save_error:
                                            st.error(
                                                "Grounding recall review was not saved: "
                                                + str(_tr_save_error)
                                            )
            _tr_historical_analog_count = int(
                _tr_analysis.get("historical_analog_count") or 0
            )
            if _tr_historical_analog_count:
                _tr_analog_label = (
                    "analog" if _tr_historical_analog_count == 1 else "analogs"
                )
                _tr_supported_recommendation_count = int(
                    _tr_analysis.get(
                        "recommendations_with_historical_support_count"
                    ) or 0
                )
                _tr_recommendation_label = (
                    "recommendation"
                    if _tr_supported_recommendation_count == 1
                    else "recommendations"
                )
                _tr_citation_verb = (
                    "cites"
                    if _tr_supported_recommendation_count == 1
                    else "cite"
                )
                _tr_run_notes.append(
                    "**Leakage-safe historical context** — "
                    f"{_tr_historical_analog_count} older resolved {_tr_analog_label}; "
                    f"{_tr_supported_recommendation_count} "
                    f"{_tr_recommendation_label} {_tr_citation_verb} "
                    "historical corroboration. "
                    "Analogs do not "
                    "replace current-fix evidence."
                )
            _tr_analog_policy = (
                _tr_analysis.get("historical_analog_policy") or {}
            )
            _tr_analog_calibration = (
                _tr_analog_policy.get("threshold_calibration") or {}
            )
            _tr_analog_global_calibration = (
                _tr_analog_calibration.get("global")
                if isinstance(_tr_analog_calibration.get("global"), dict)
                else {}
            )
            if _tr_analog_global_calibration:
                _tr_analog_floor = float(
                    _tr_analog_global_calibration.get("threshold") or 0.18
                )
                _tr_analog_calibration_status = str(
                    _tr_analog_global_calibration.get("status")
                    or "insufficient_reviews"
                )
                _tr_analog_status_labels = {
                    "insufficient_reviews": "default; insufficient reviewed outcomes",
                    "insufficient_diversity": "default; outcomes are too concentrated",
                    "default_performing_well": "default; reviewed quality is healthy",
                    "calibrated_higher": "raised from reviewed outcomes",
                    "no_safe_higher_threshold": "default; no supported higher floor",
                }
                _tr_run_notes.append(
                    f"**Historical-analog similarity floor:** {_tr_analog_floor:.2f} — "
                    + _tr_analog_status_labels.get(
                        _tr_analog_calibration_status,
                        _tr_analog_calibration_status,
                    )
                    + ". It can only rise after diverse confirmed/rejected reviews."
                )
                _tr_analog_excluded_rejections = int(
                    _tr_analog_global_calibration.get(
                        "excluded_ambiguous_rejection_count"
                    ) or 0
                )
                if _tr_analog_excluded_rejections:
                    _tr_run_notes.append(
                        "**Analog calibration** ignored "
                        f"{_tr_analog_excluded_rejections} rejected reviews whose "
                        "reason did not explicitly identify misleading historical "
                        "corroboration."
                    )

            if _tr_run_notes:
                with _tr_run_notes_slot:
                    with st.expander(
                        "📊 What happened when this report was built",
                        expanded=False,
                    ):
                        st.caption(
                            "Counts from this run only — how many claims the AI "
                            "proposed, how many survived each check, and what was "
                            "kept in provenance instead of shown. Zeros mean nothing "
                            "proposed could be tied to quoted text in these fixes, "
                            "which is a normal outcome, not an error."
                        )
                        st.markdown(linkify_bug_codes(
                            "\n".join(f"- {note}" for note in _tr_run_notes)
                        ))

            if _tr_missing_details:
                _tr_missing_links = " · ".join(
                    f"[{code}]({ECL_BUG_URL.format(bug_code=code)})"
                    for code in _tr_missing_details
                )
                st.warning(
                    "Full eBug status logs were unavailable after retries for: "
                    + _tr_missing_links
                    + ". List-level metadata was used for these bugs."
                )

            if _tr_report_fp and _tr_report_fp.exists():
                _tr_report_text = _tr_report_fp.read_text(encoding="utf-8")
                # Hide the machine-readable generation comment from the UI.
                _tr_report_text = _tr_re.sub(
                    r"^<!--.*?-->\s*", "", _tr_report_text,
                    count=1, flags=_tr_re.DOTALL,
                )
                _tr_sections = {}
                _tr_headings = list(_tr_re.finditer(
                    r"^#\s+(Recommended Regression Areas|Per-Bug Verification)\s*$",
                    _tr_report_text, flags=_tr_re.I | _tr_re.M,
                ))
                for _tr_idx, _tr_match in enumerate(_tr_headings):
                    _tr_end = (
                        _tr_headings[_tr_idx + 1].start()
                        if _tr_idx + 1 < len(_tr_headings)
                        else len(_tr_report_text)
                    )
                    _tr_sections[_tr_match.group(1).lower()] = (
                        _tr_report_text[_tr_match.end():_tr_end].strip()
                    )

                _tr_regression = _tr_sections.get("recommended regression areas", "")
                _tr_verification = _tr_sections.get("per-bug verification", "")
                # Older reports used plain newlines between bold module names;
                # Markdown renders those as one paragraph. Promote each entry
                # to a bullet so modules/features remain visually distinct.
                _tr_regression = _tr_re.sub(
                    r"(?m)^(?=\*\*[^*\n]+\*\*\s*[-–—:])",
                    "- ",
                    _tr_regression,
                )

                with _tr_work_slot:
                    _tr_structured_recommendations = (
                        _tr_analysis.get("recommendations") or []
                    )
                    # Publish-or-hide. An empty predictions section is noise on
                    # a page whose job is telling QA what to test, and this one
                    # is empty whenever the grounding gates do their job. The
                    # Focus List above carries the coverage; withheld drafts
                    # stay under *Candidates withheld by grounding audit* in the
                    # diagnostics expander for anyone tuning the gates.
                    # The narrative report is never empty: when nothing is
                    # grounded it still carries the NO_GROUNDED_REGRESSION
                    # sentence, so a plain truthiness test keeps the section
                    # alive showing exactly the empty state we are hiding.
                    _tr_regression_has_content = bool(
                        _tr_regression
                        and NO_GROUNDED_REGRESSION not in _tr_regression
                    )
                    if _tr_structured_recommendations or _tr_regression_has_content:
                        st.markdown("### 🎯 Recommended Regression Areas")
                        st.caption(
                            "Prioritized cross-feature checks. Risk is classified from "
                            "the observable failure and capped by evidence confidence; "
                            "confidence describes the deterministic strength and "
                            "independence of the supplied current-fix evidence."
                        )
                        if _tr_structured_recommendations:
                            render_recommendation_groups(
                                st, _tr_analysis, ECL_BUG_URL, _tr_changesets
                            )
                        else:
                            st.markdown(linkify_bug_codes(_tr_regression))
                    _tr_coverage_floor = (
                        _tr_analysis.get("regression_coverage_floor") or {}
                    )
                    _tr_floor_entries = [
                        entry for entry in (_tr_coverage_floor.get("entries") or [])
                        if isinstance(entry, dict)
                    ]
                    if _tr_floor_entries:
                        st.markdown("#### 🧭 Baseline Coverage")
                        st.caption(
                            "Not adjacency predictions, and deliberately unrated — "
                            "these carry no risk, confidence or priority. They name "
                            "where each fixed module sits and what its history shows, "
                            "so every fixed bug has a starting point even when no "
                            "adjacent area is supportable from the fix details. They "
                            "are never scored against the next build."
                        )
                        _tr_floor_label = {
                            "module_history": "Module history",
                            "fixed_surface": "Fixed surface",
                        }
                        show_dataframe([
                            {
                                "Area": str(entry.get("area") or ""),
                                "Basis": _tr_floor_label.get(
                                    str(entry.get("basis") or ""), "Baseline"
                                ),
                                "Category": str(entry.get("module_category") or ""),
                                "Action": str(entry.get("action") or ""),
                                "Why": str(entry.get("why") or ""),
                                "Source bug": str(entry.get("source_bug_code") or ""),
                                "Past bugs": ", ".join(
                                    str(ref.get("analog_bug_code") or "")
                                    for ref in entry.get("historical_analog_refs") or []
                                    if isinstance(ref, dict)
                                ),
                            }
                            for entry in _tr_floor_entries
                        ], hide_index=True, width="stretch")
                    st.markdown("### ✅ Per-Bug Verification")
                    _tr_filter = st.text_input(
                        "Filter verification checklists",
                        placeholder="Enter a BugCode, feature, locale, or keyword…",
                        key=f"tr_test_focus_filter_{selected_slug}",
                    ).strip().lower()
                    # Match only exact manifest BugCodes. Escaping the values keeps
                    # atypical legacy formats visible without treating arbitrary
                    # model-authored level-2 headings as source bugs.
                    _tr_manifest_blocks = extract_manifest_checklists(
                        _tr_verification, _tr_bug_codes
                    )
                    _tr_code_by_block = {block: code for code, block in _tr_manifest_blocks}
                    _tr_bug_blocks = [block for _code, block in _tr_manifest_blocks]
                    _tr_visible_blocks = [
                        block for block in _tr_bug_blocks
                        if not _tr_filter or _tr_filter in (
                            block + " " + checklist_label(_tr_code_by_block[block], _tr_analysis)
                        ).lower()
                    ]
                    st.caption(
                        f"Showing {len(_tr_visible_blocks)} of {len(_tr_bug_blocks)} checklists. "
                        "Expand a bug to see its steps."
                    )
                    for _tr_block in _tr_visible_blocks:
                        _tr_code = _tr_code_by_block.get(_tr_block, "Checklist")
                        _tr_body = _tr_re.sub(
                            r"^##[^\n]*\n?", "", _tr_block, count=1
                        ).strip()
                        with st.expander(checklist_label(_tr_code, _tr_analysis), expanded=False):
                            st.markdown(linkify_bug_codes(_tr_body))
                            if _tr_code != "Checklist":
                                st.markdown(
                                    f"[Open {_tr_code} in eCL ↗]"
                                    f"({ECL_BUG_URL.format(bug_code=_tr_code)})"
                                )

                    if _tr_filter and not _tr_visible_blocks:
                        st.info("No verification checklist matches that filter.")

                    if _tr_bug_codes:
                        with st.expander(f"🔗 Open source bugs ({len(_tr_bug_codes)})",
                                         expanded=False):
                            _tr_links = [
                                f"[{code}]({ECL_BUG_URL.format(bug_code=code)})"
                                for code in _tr_bug_codes
                            ]
                            st.markdown(" · ".join(_tr_links))
                if any(
                    item.get("evidence_refs")
                    for item in _tr_structured_recommendations
                    if isinstance(item, dict)
                ):
                    with st.expander(
                        "🔎 Evidence & reasoning", expanded=False
                    ):
                        st.caption("Original stored recommendations, including any instructions flagged for review.")
                        st.markdown(linkify_bug_codes(_tr_regression))
                        st.caption(
                            "Each reference was validated against the stored RD "
                            "root-cause, solution, or stated-risk evidence."
                        )
                        for _tr_rec in _tr_structured_recommendations:
                            if not isinstance(_tr_rec, dict):
                                continue
                            st.markdown(
                                f"**{_tr_html.escape(str(_tr_rec.get('area') or 'Recommendation'))}**"
                            )
                            _tr_hypothesis = str(
                                _tr_rec.get("hypothesis_id") or "Legacy prediction"
                            )
                            _tr_target = str(
                                _tr_rec.get("target_category") or "Unclassified"
                            )
                            st.caption(
                                f"{_tr_hypothesis} · target category: {_tr_target}"
                            )
                            _tr_priority_basis = (
                                _tr_rec.get("priority_basis") or {}
                            )
                            if _tr_priority_basis:
                                st.caption(
                                    "Priority #"
                                    + str(int(_tr_rec.get("priority_rank") or 0))
                                    + " · target-anchored independent groups: "
                                    + str(int(_tr_priority_basis.get(
                                        "target_anchored_independent_group_count"
                                    ) or 0))
                                    + " · high-quality independent groups: "
                                    + str(int(_tr_priority_basis.get(
                                        "high_quality_independent_group_count"
                                    ) or 0))
                                    + " · medium-or-better independent groups: "
                                    + str(int(_tr_priority_basis.get(
                                        "medium_or_better_independent_group_count"
                                    ) or 0))
                                    + " · total independent groups: "
                                    + str(int(_tr_priority_basis.get(
                                        "independent_source_group_count"
                                    ) or 0))
                                )
                            _tr_selection_origin = str(
                                _tr_rec.get("selection_origin") or ""
                            )
                            if _tr_selection_origin:
                                _tr_selection_origin_labels = {
                                    "model_synthesis": (
                                        "selected by AI synthesis, then deterministically ranked"
                                    ),
                                    "deterministic_synthesis_omission_backfill": (
                                        "recovered after AI synthesis omission, then passed "
                                        "the final adversarial audit"
                                    ),
                                }
                                st.caption(
                                    "Selection path: "
                                    + _tr_selection_origin_labels.get(
                                        _tr_selection_origin,
                                        _tr_selection_origin.replace("_", " "),
                                    )
                                )
                            _tr_generation_origins = list(
                                _tr_rec.get("candidate_generation_origins") or []
                            )
                            if _tr_generation_origins:
                                _tr_generation_origin_labels = {
                                    "primary_discovery": "primary discovery",
                                    "coverage_challenge": (
                                        "complementary coverage challenge"
                                    ),
                                }
                                st.caption(
                                    "Discovery path: "
                                    + " + ".join(
                                        _tr_generation_origin_labels.get(
                                            origin,
                                            str(origin).replace("_", " "),
                                        )
                                        for origin in _tr_generation_origins
                                    )
                                )
                            _tr_adjacency_support = (
                                _tr_rec.get("adjacency_support_basis") or {}
                            )
                            if _tr_adjacency_support:
                                _tr_reviewed_alias_matches = (
                                    _tr_adjacency_support.get(
                                        "reviewed_alias_matches"
                                    ) or []
                                )
                                _tr_adjacency_anchor_terms = ", ".join(
                                    str(term) for term in (
                                        _tr_adjacency_support.get(
                                            "target_anchor_terms"
                                        ) or []
                                    )
                                )
                                st.caption(
                                    "Deterministic adjacency support: "
                                    + (
                                        "reviewed directional alias "
                                        + " · ".join(
                                            f"{match.get('target_term')} → "
                                            f"{match.get('evidence_alias_term')}"
                                            for match in _tr_reviewed_alias_matches
                                            if isinstance(match, dict)
                                        )
                                        + "; anchored target term"
                                        if _tr_reviewed_alias_matches
                                        else "exact target term"
                                    )
                                    + (
                                        "s" if len(
                                            _tr_adjacency_support.get(
                                                "target_anchor_terms"
                                            ) or []
                                        ) != 1 else ""
                                    )
                                    + " "
                                    + (_tr_adjacency_anchor_terms or "none")
                                    + " found in "
                                    + str(int(_tr_adjacency_support.get(
                                        "target_anchor_reference_count"
                                    ) or 0))
                                    + " exact current-fix reference"
                                    + (
                                        "s" if int(_tr_adjacency_support.get(
                                            "target_anchor_reference_count"
                                        ) or 0) != 1 else ""
                                    )
                                    + "."
                                )
                            _tr_category_basis = (
                                _tr_rec.get("target_category_basis") or {}
                            )
                            if _tr_category_basis:
                                _tr_category_rule = str(
                                    _tr_category_basis.get("rule") or ""
                                ).replace("_", " ")
                                _tr_category_modules = ", ".join(
                                    str(module) for module in (
                                        _tr_category_basis.get("matched_modules")
                                        or []
                                    )
                                )
                                st.caption(
                                    "Deterministic category: "
                                    + _tr_target
                                    + " · "
                                    + _tr_category_rule
                                    + (
                                        " · matched module: "
                                        + _tr_category_modules
                                        if _tr_category_modules else ""
                                    )
                                    + " · model request: "
                                    + str(
                                        _tr_category_basis.get(
                                            "requested_category"
                                        ) or "none"
                                    )
                                )
                            _tr_confidence_basis = (
                                _tr_rec.get("confidence_basis") or {}
                            )
                            if _tr_confidence_basis:
                                _tr_confidence_rule_labels = {
                                    "multiple_current_sources_including_high_quality": (
                                        "multiple current sources including high quality"
                                    ),
                                    "multiple_independent_sources_including_high_quality": (
                                        "multiple independent evidence groups "
                                        "including high quality"
                                    ),
                                    "single_or_mixed_high_quality_current_source": (
                                        "single or mixed high-quality current source"
                                    ),
                                    "correlated_sources_do_not_raise_high": (
                                        "correlated sources do not raise High"
                                    ),
                                    "multiple_medium_quality_current_sources": (
                                        "multiple medium-quality current sources"
                                    ),
                                    "multiple_independent_medium_quality_sources": (
                                        "multiple independent medium-quality groups"
                                    ),
                                    "correlated_sources_do_not_create_independence": (
                                        "correlated sources do not create independence"
                                    ),
                                    "unanchored_target_caps_high": (
                                        "unanchored adjacent target caps High"
                                    ),
                                    "limited_or_low_quality_current_support": (
                                        "limited or low-quality current support"
                                    ),
                                }
                                _tr_confidence_rule = str(
                                    _tr_confidence_basis.get("rule") or ""
                                )
                                st.caption(
                                    "Deterministic confidence: "
                                    + str(_tr_rec.get("confidence") or "low").title()
                                    + " · "
                                    + _tr_confidence_rule_labels.get(
                                        _tr_confidence_rule,
                                        _tr_confidence_rule.replace("_", " "),
                                    )
                                    + " · current BugCodes: "
                                    + str(int(_tr_confidence_basis.get(
                                        "source_bug_count"
                                    ) or 0))
                                    + " · exact references: "
                                    + str(int(_tr_confidence_basis.get(
                                        "current_evidence_ref_count"
                                    ) or 0))
                                    + " · quality H/M/L: "
                                    + "/".join(str(int(
                                        _tr_confidence_basis.get(field) or 0
                                    )) for field in (
                                        "high_quality_source_count",
                                        "medium_quality_source_count",
                                        "low_quality_source_count",
                                    ))
                                    + " · historical analogs excluded from label"
                                    + (
                                        " · independent groups: "
                                        + str(int(_tr_confidence_basis.get(
                                            "independent_source_group_count"
                                        ) or 0))
                                        + " · correlated BugCodes: "
                                        + str(int(_tr_confidence_basis.get(
                                            "correlated_source_bug_count"
                                        ) or 0))
                                        if int(_tr_confidence_basis.get(
                                            "schema_version"
                                        ) or 0) >= 2 else ""
                                    )
                                )
                                if _tr_confidence_basis.get(
                                    "correlation_links"
                                ):
                                    st.caption(
                                        "Correlated evidence groups: "
                                        + " · ".join(
                                            ", ".join(
                                                str(code) for code in (
                                                    link.get(
                                                        "source_bug_codes"
                                                    ) or []
                                                )
                                            )
                                            + " ("
                                            + str(link.get("reason") or "")
                                            .replace("_", " ")
                                            + ")"
                                            for link in _tr_confidence_basis.get(
                                                "correlation_links"
                                            )
                                            if isinstance(link, dict)
                                        )
                                    )
                                if int(_tr_confidence_basis.get(
                                    "schema_version"
                                ) or 0) >= 3:
                                    _tr_anchor_terms = ", ".join(
                                        str(term) for term in (
                                            _tr_confidence_basis.get(
                                                "target_anchor_terms"
                                            ) or []
                                        )
                                    )
                                    st.caption(
                                        "Target anchoring: "
                                        + str(_tr_confidence_basis.get(
                                            "target_support_rule"
                                        ) or "").replace("_", " ")
                                        + " · anchored current BugCodes: "
                                        + str(int(_tr_confidence_basis.get(
                                            "target_anchor_source_count"
                                        ) or 0))
                                        + " · terms: "
                                        + (_tr_anchor_terms or "none")
                                        + " · confidence cap: "
                                        + str(_tr_confidence_basis.get(
                                            "target_support_cap"
                                        ) or "none").replace("_", " ")
                                    )
                            _tr_shared_path_basis = (
                                _tr_rec.get("shared_path_support_basis") or {}
                            )
                            if _tr_shared_path_basis:
                                _tr_shared_path_terms = ", ".join(
                                    str(term) for term in (
                                        _tr_shared_path_basis.get(
                                            "matched_path_terms"
                                        ) or []
                                    )
                                )
                                st.caption(
                                    "Deterministic shared-path grounding: exact "
                                    "non-target rationale terms found in cited current-"
                                    "fix evidence · terms: "
                                    + (_tr_shared_path_terms or "none")
                                    + " · model critic cannot override"
                                )
                            _tr_risk_basis = _tr_rec.get("risk_basis") or {}
                            if _tr_risk_basis:
                                _tr_risk_rule_labels = {
                                    "explicit_security_or_privacy_harm": (
                                        "explicit security or privacy harm"
                                    ),
                                    "explicit_data_loss_or_persistent_corruption": (
                                        "explicit data loss or persistent corruption"
                                    ),
                                    "explicit_billing_or_entitlement_harm": (
                                        "explicit billing or entitlement harm"
                                    ),
                                    "explicit_launch_or_startup_block": (
                                        "explicit launch or startup block"
                                    ),
                                    "explicit_crash_or_forced_termination": (
                                        "explicit crash or forced termination"
                                    ),
                                    "presentation_only_symptom": (
                                        "presentation-only symptom"
                                    ),
                                    "observable_functional_failure": (
                                        "observable functional failure"
                                    ),
                                }
                                _tr_risk_rule = str(
                                    _tr_risk_basis.get("rule") or ""
                                )
                                _tr_risk_cap = str(
                                    _tr_risk_basis.get("confidence_cap")
                                    or "none"
                                )
                                st.caption(
                                    "Deterministic risk: "
                                    + str(_tr_rec.get("risk") or "Medium")
                                    + " · impact: "
                                    + str(_tr_risk_basis.get("impact_basis")
                                          or "functional").replace("_", " ")
                                    + " · "
                                    + _tr_risk_rule_labels.get(
                                        _tr_risk_rule,
                                        _tr_risk_rule.replace("_", " "),
                                    )
                                    + " · confidence cap: "
                                    + _tr_risk_cap.replace("_", " ")
                                )
                                if _tr_risk_basis.get(
                                    "harm_evidence_required"
                                ):
                                    _tr_harm_matches = ", ".join(
                                        str(match.get("bug_code") or "")
                                        + " "
                                        + str(match.get("evidence_field")
                                              or "evidence")
                                        + ": "
                                        + str(match.get("signal") or "signal")
                                        for match in _tr_risk_basis.get(
                                            "harm_evidence_matches"
                                        ) or []
                                        if isinstance(match, dict)
                                    )
                                    st.caption(
                                        "Current severe-harm evidence: "
                                        + (
                                            "supported · " + _tr_harm_matches
                                            if _tr_risk_basis.get(
                                                "harm_evidence_supported"
                                            )
                                            else (
                                                "absent · prediction retained but "
                                                "capped to Medium"
                                            )
                                        )
                                        + " · historical analogs and the model "
                                        "critic cannot override"
                                    )
                                _tr_ignored_risk_signals = [
                                    str(signal.get("signal") or "signal")
                                    + " ("
                                    + str(signal.get("negation") or "negated")
                                    + ")"
                                    for signal in (
                                        _tr_risk_basis.get(
                                            "ignored_negated_signals"
                                        ) or []
                                    )
                                    if isinstance(signal, dict)
                                ]
                                if _tr_ignored_risk_signals:
                                    st.caption(
                                        "Ignored negated risk signals: "
                                        + ", ".join(_tr_ignored_risk_signals)
                                    )
                            _tr_prediction_origin_labels = {
                                "initial_audit_pass_only": "Initial audit pass only",
                                "recovered_draft_only": "Repaired draft only",
                                "mixed_initial_and_recovered": (
                                    "Mixed initial-pass and repaired sources"
                                ),
                            }
                            _tr_prediction_origin = str(
                                _tr_rec.get("prediction_origin") or ""
                            )
                            _tr_source_candidate_ids = [
                                str(candidate_id) for candidate_id in (
                                    _tr_rec.get("synthesis_source_candidate_ids")
                                    or []
                                )
                            ]
                            if (
                                _tr_prediction_origin
                                in _tr_prediction_origin_labels
                                and _tr_source_candidate_ids
                            ):
                                st.caption(
                                    (
                                        "Coherent claim-locked lineage: "
                                        if int(_tr_analysis.get("schema_version") or 0)
                                        >= 11
                                        else "Accepted-draft lineage: "
                                    )
                                    + _tr_prediction_origin_labels[
                                        _tr_prediction_origin
                                    ]
                                    + " · sources: "
                                    + ", ".join(_tr_source_candidate_ids)
                                )
                            if _tr_rec.get("trigger") or _tr_rec.get("expected_failure_mode"):
                                st.caption(
                                    f"Trigger: {_tr_rec.get('trigger') or 'Not supplied'} · "
                                    f"Watch for: "
                                    f"{_tr_rec.get('expected_failure_mode') or 'Not supplied'}"
                                )
                            _tr_scenario_specificity = (
                                _tr_rec.get("scenario_specificity_basis") or {}
                            )
                            if _tr_scenario_specificity:
                                _tr_specificity_rows = [
                                    row for row in (
                                        _tr_scenario_specificity.get(
                                            "claim_specificity"
                                        ) or []
                                    )
                                    if isinstance(row, dict)
                                ]
                                st.caption(
                                    "Exact scenario-detail grounding: "
                                    + (
                                        " · ".join(
                                            str(row.get("claim_field") or "claim")
                                            .replace("_", " ")
                                            + ": "
                                            + str(row.get("term") or "")
                                            + " ← "
                                            + ", ".join(
                                                str(support.get("bug_code") or "")
                                                + " "
                                                + str(support.get(
                                                    "evidence_field"
                                                ) or "evidence")
                                                for support in (
                                                    row.get("support") or []
                                                )
                                                if isinstance(support, dict)
                                            )
                                            for row in _tr_specificity_rows
                                        )
                                        if _tr_specificity_rows else (
                                            "no protected exact details claimed"
                                        )
                                    )
                                    + " · model critic cannot override"
                                )
                            if _tr_rec.get("scope_boundary"):
                                st.caption(
                                    "Boundary: "
                                    + str(_tr_rec.get("scope_boundary"))
                                )
                            if _tr_rec.get("disconfirming_observation"):
                                st.caption(
                                    "Disconfirming observation: "
                                    + str(
                                        _tr_rec.get(
                                            "disconfirming_observation"
                                        )
                                    )
                                )
                            for _tr_ref in (_tr_rec.get("evidence_refs") or []):
                                if not isinstance(_tr_ref, dict):
                                    continue
                                _tr_ref_code = (
                                    bug_link(_tr_ref.get("bug_code"), code_style=False)
                                    or "Unknown"
                                )
                                _tr_ref_quote = str(_tr_ref.get("quote") or "").strip()
                                _tr_ref_slot = int(_tr_ref.get("signal_index") or 0) + 1
                                st.caption(
                                    f"{_tr_ref_code} · evidence {_tr_ref_slot}: "
                                    f"{_tr_ref_quote}"
                                )
                            for _tr_analog in (
                                _tr_rec.get("historical_analog_refs") or []
                            ):
                                if not isinstance(_tr_analog, dict):
                                    continue
                                _tr_analog_code = (
                                    bug_link(
                                        _tr_analog.get("analog_bug_code"),
                                        code_style=False,
                                    )
                                    or "Unknown"
                                )
                                _tr_analog_module = str(
                                    _tr_analog.get("module") or "Unknown module"
                                )
                                _tr_analog_summary = str(
                                    _tr_analog.get("summary") or ""
                                )
                                _tr_analog_score = float(
                                    _tr_analog.get("similarity_score") or 0
                                )
                                st.caption(
                                    f"Historical corroboration {_tr_analog_code} · "
                                    f"{_tr_analog_module} · similarity "
                                    f"{_tr_analog_score:.2f}: {_tr_analog_summary}"
                                )

                _tr_prior_eval = _tr_analysis.get("prior_hypothesis_evaluation")
                _tr_prior_skip = _tr_analysis.get(
                    "prior_hypothesis_evaluation_skipped"
                )
                if (
                    not isinstance(_tr_prior_eval, dict)
                    and isinstance(_tr_prior_skip, dict)
                ):
                    st.markdown("### 📈 Previous TR Prediction Check")
                    _tr_skip_prior_build = str(
                        _tr_prior_skip.get("prior_close_build") or ""
                    )
                    if _tr_prior_skip.get("reason") == "same_close_build":
                        st.info(
                            "Nothing to score yet. This check compares one build's "
                            "predictions against the **next** build's bugs, and the "
                            "last stored report is for the same Close Build"
                            + (f" ({_tr_skip_prior_build})" if _tr_skip_prior_build else "")
                            + " — re-running the same build cannot confirm or refute "
                            "its own predictions. It populates when the next Close "
                            "Build is analyzed."
                        )
                    else:
                        st.info(
                            "No earlier report from a different Close Build is stored "
                            "for this product yet, so there are no prior predictions "
                            "to score."
                        )
                if isinstance(_tr_prior_eval, dict):
                    _tr_review_fp = (
                        Path(_product_dir) / "tr_reports" / "prediction_reviews.json"
                    )
                    _tr_review_summary = {
                        "reviewed_count": 0,
                        "confirmed_count": 0,
                        "rejected_count": 0,
                        "unclear_count": 0,
                        "decisive_count": 0,
                        "reviewed_confirmation_rate": None,
                        "prediction_reviewed_count": 0,
                        "prediction_confirmed_count": 0,
                        "prediction_rejected_count": 0,
                        "prediction_unclear_count": 0,
                        "prediction_decisive_count": 0,
                        "threshold_audit_reviewed_count": 0,
                        "threshold_audit_confirmed_count": 0,
                        "prediction_rejection_reason_counts": {},
                        "threshold_audit_rejection_reason_counts": {},
                        "rejected_without_reason_count": 0,
                    }
                    _tr_guardrail_policy = {
                        "active_guardrails": {}, "categories": {},
                    }
                    _tr_guardrail_effectiveness = {
                        "unknown_provenance_decisive_review_count": 0,
                        "comparisons": [],
                    }
                    _tr_repair_origin_effectiveness = {
                        "unknown_provenance_decisive_review_count": 0,
                        "comparisons": [],
                    }
                    try:
                        _tr_review_ledger = load_review_ledger(
                            _tr_review_fp, selected_slug or "legacy"
                        )
                        _tr_prior_eval, _tr_review_summary = overlay_reviews(
                            _tr_prior_eval, _tr_review_ledger
                        )
                        _tr_guardrail_policy = build_analysis_guardrail_policy(
                            _tr_review_ledger
                        )
                        _tr_guardrail_effectiveness = (
                            guardrail_effectiveness_comparison(
                                _tr_review_ledger
                            )
                        )
                        _tr_repair_origin_effectiveness = (
                            repair_origin_effectiveness_comparison(
                                _tr_review_ledger
                            )
                        )
                    except ReviewLedgerError as _tr_review_error:
                        st.warning(
                            "Prediction reviews could not be loaded; the ledger was "
                            f"left unchanged: {_tr_review_error}"
                        )
                    _tr_guardrail_categories = (
                        _tr_guardrail_policy.get("categories") or {}
                    )
                    if _tr_guardrail_categories:
                        _tr_active_live_guardrails = (
                            _tr_guardrail_policy.get("active_guardrails") or {}
                        )
                        if _tr_active_live_guardrails:
                            st.caption(
                                "Longitudinal review feedback now activates — "
                                + " · ".join(
                                    f"{category}: "
                                    + _tr_guardrail_labels.get(
                                        str(item.get("guardrail") or ""),
                                        str(item.get("guardrail") or "")
                                        .replace("_", " "),
                                    )
                                    for category, item in sorted(
                                        _tr_active_live_guardrails.items(),
                                        key=lambda pair: str(pair[0]).casefold(),
                                    )
                                    if isinstance(item, dict)
                                )
                                + ". It will constrain the next analysis run."
                            )
                        _tr_guardrail_status_labels = {
                            "active": "Active",
                            "insufficient_reviews": "Needs more reviews",
                            "insufficient_rejections": "Needs more rejections",
                            "insufficient_diversity": "Needs more diverse evidence",
                            "low_rejection_rate": "Rejection rate below gate",
                            "no_dominant_ai_failure": "No dominant AI failure",
                        }
                        _tr_guardrail_rows = []
                        for _tr_category, _tr_category_result in sorted(
                            _tr_guardrail_categories.items(),
                            key=lambda pair: str(pair[0]).casefold(),
                        ):
                            if not isinstance(_tr_category_result, dict):
                                continue
                            _tr_reason_counts = (
                                _tr_category_result.get("reason_counts") or {}
                            )
                            _tr_guardrail_rows.append({
                                "Target category": str(_tr_category),
                                "Status": _tr_guardrail_status_labels.get(
                                    str(_tr_category_result.get("status") or ""),
                                    str(_tr_category_result.get("status") or "")
                                    .replace("_", " "),
                                ),
                                "Decisive reviews": (
                                    f"{int(_tr_category_result.get('decisive_review_count') or 0)} raw · "
                                    f"{float(_tr_category_result.get('effective_decisive_review_count') or 0):g} effective"
                                ),
                                "Structured rejections": (
                                    f"{int(_tr_category_result.get('structured_rejection_count') or 0)} raw · "
                                    f"{float(_tr_category_result.get('effective_structured_rejection_count') or 0):g} effective"
                                ),
                                "Rejection rate": (
                                    f"{float(_tr_category_result.get('rejection_rate')):.0%}"
                                    if _tr_category_result.get("rejection_rate")
                                    is not None else "—"
                                ),
                                "Rejected diversity": (
                                    f"{int(_tr_category_result.get('rejected_cohort_count') or 0)} cohorts · "
                                    f"{int(_tr_category_result.get('rejected_hypothesis_count') or 0)} hypotheses"
                                ),
                                "Rejection reasons": " · ".join(
                                    f"{REJECTION_REASON_LABELS.get(code, code)}: {count}"
                                    for code, count in sorted(
                                        _tr_reason_counts.items(),
                                        key=lambda item: (
                                            -int(item[1]), str(item[0])
                                        ),
                                    )
                                ) or "—",
                                "Guardrail": _tr_guardrail_labels.get(
                                    str(_tr_category_result.get(
                                        "active_guardrail"
                                    ) or ""),
                                    str(_tr_category_result.get(
                                        "active_guardrail"
                                    ) or "—").replace("_", " "),
                                ),
                            })
                        with st.expander(
                            "Longitudinal category rejection trends",
                            expanded=False,
                        ):
                            st.caption(
                                "Only diverse, recurring structured AI-analysis "
                                "rejection codes can activate prompt constraints. "
                                "Matcher and analog reasons route to their own "
                                "calibrators. Free-form reviewer notes are never sent "
                                "to Ollama."
                            )
                            show_dataframe(
                                pd.DataFrame(_tr_guardrail_rows),
                                width="stretch",
                                hide_index=True,
                            )
                    _tr_effectiveness_comparisons = (
                        _tr_guardrail_effectiveness.get("comparisons") or []
                    )
                    if (
                        _tr_effectiveness_comparisons
                        or (_tr_guardrail_policy.get("active_guardrails") or {})
                    ):
                        _tr_effectiveness_status_labels = {
                            "insufficient_reviews": "Needs more reviews",
                            "insufficient_diversity": "Needs more diverse evidence",
                            "improved": "Lower target-failure rate",
                            "worse": "Higher target-failure rate",
                            "no_clear_difference": "No clear difference",
                        }
                        _tr_effectiveness_rows = []
                        for _tr_comparison in _tr_effectiveness_comparisons:
                            if not isinstance(_tr_comparison, dict):
                                continue
                            _tr_unguarded = _tr_comparison.get("unguarded") or {}
                            _tr_guarded = _tr_comparison.get("guarded") or {}
                            _tr_reduction = _tr_comparison.get(
                                "target_failure_rate_reduction"
                            )
                            _tr_effectiveness_rows.append({
                                "Target category": str(
                                    _tr_comparison.get("target_category") or ""
                                ),
                                "Guardrail": _tr_guardrail_labels.get(
                                    str(_tr_comparison.get("guardrail") or ""),
                                    str(_tr_comparison.get("guardrail") or "")
                                    .replace("_", " "),
                                ),
                                "Configuration": (
                                    f"{_tr_comparison.get('model') or 'Unknown'} · "
                                    f"{str(_tr_comparison.get('engine_sha256') or '')[:12]}"
                                ),
                                "Applied guardrail": _tr_guardrail_labels.get(
                                    str(_tr_comparison.get(
                                        "analysis_guardrail"
                                    ) or ""),
                                    str(_tr_comparison.get(
                                        "analysis_guardrail"
                                    ) or "None").replace("_", " "),
                                ),
                                "Status": _tr_effectiveness_status_labels.get(
                                    str(_tr_comparison.get("status") or ""),
                                    str(_tr_comparison.get("status") or "")
                                    .replace("_", " "),
                                ),
                                "Unguarded evidence": (
                                    f"{int(_tr_unguarded.get('decisive_review_count') or 0)} raw · "
                                    f"{float(_tr_unguarded.get('effective_decisive_review_count') or 0):g} effective · "
                                    f"{int(_tr_unguarded.get('reviewed_cohort_count') or 0)} cohorts"
                                ),
                                "Guarded evidence": (
                                    f"{int(_tr_guarded.get('decisive_review_count') or 0)} raw · "
                                    f"{float(_tr_guarded.get('effective_decisive_review_count') or 0):g} effective · "
                                    f"{int(_tr_guarded.get('reviewed_cohort_count') or 0)} cohorts"
                                ),
                                "Target failure before": (
                                    f"{float(_tr_unguarded.get('target_failure_rate')):.0%}"
                                    if _tr_unguarded.get("target_failure_rate")
                                    is not None else "—"
                                ),
                                "Target failure guarded": (
                                    f"{float(_tr_guarded.get('target_failure_rate')):.0%}"
                                    if _tr_guarded.get("target_failure_rate")
                                    is not None else "—"
                                ),
                                "Observed reduction": (
                                    f"{float(_tr_reduction):+.0%}"
                                    if _tr_reduction is not None else "—"
                                ),
                            })
                        with st.expander(
                            "AI guardrail effectiveness",
                            expanded=False,
                        ):
                            st.caption(
                                "Prospective observational check: guarded and "
                                "unguarded reviews are compared only when they share "
                                "the exact model and analysis-engine fingerprint. "
                                "Both groups need at least eight raw/effective reviews "
                                "across three cohorts and three hypotheses. This does "
                                "not establish causality or automatically change policy."
                            )
                            _tr_unknown_guardrail_provenance = int(
                                _tr_guardrail_effectiveness.get(
                                    "unknown_provenance_decisive_review_count"
                                ) or 0
                            )
                            if _tr_unknown_guardrail_provenance:
                                st.caption(
                                    f"Excluded {_tr_unknown_guardrail_provenance} "
                                    "decisive legacy reviews without exact model, "
                                    "engine, and applied-policy provenance."
                                )
                            if _tr_effectiveness_rows:
                                show_dataframe(
                                    pd.DataFrame(_tr_effectiveness_rows),
                                    width="stretch",
                                    hide_index=True,
                                )
                            else:
                                st.info(
                                    "No guarded prediction reviews with exact run "
                                    "provenance are available yet."
                                )
                    _tr_repair_comparisons = (
                        _tr_repair_origin_effectiveness.get("comparisons") or []
                    )
                    if _tr_repair_comparisons:
                        _tr_repair_status_labels = {
                            "insufficient_reviews": "Needs more reviews",
                            "insufficient_diversity": "Needs more diverse evidence",
                            "repaired_higher": "Repaired-origin higher",
                            "repaired_lower": "Repaired-origin lower",
                            "no_clear_difference": "No clear difference",
                        }
                        _tr_repair_rows = []
                        for _tr_comparison in _tr_repair_comparisons:
                            if not isinstance(_tr_comparison, dict):
                                continue
                            _tr_initial_group = (
                                _tr_comparison.get("initial_pass_only") or {}
                            )
                            _tr_repaired_group = (
                                _tr_comparison.get("repaired_origin") or {}
                            )
                            _tr_origin_difference = _tr_comparison.get(
                                "confirmation_rate_difference"
                            )
                            _tr_repair_rows.append({
                                "Target category": str(
                                    _tr_comparison.get("target_category") or ""
                                ),
                                "Configuration": (
                                    f"{_tr_comparison.get('model') or 'Unknown'} · "
                                    f"{str(_tr_comparison.get('engine_sha256') or '')[:12]}"
                                ),
                                "Status": _tr_repair_status_labels.get(
                                    str(_tr_comparison.get("status") or ""),
                                    str(_tr_comparison.get("status") or "")
                                    .replace("_", " "),
                                ),
                                "Initial-pass evidence": (
                                    f"{int(_tr_initial_group.get('decisive_review_count') or 0)} raw · "
                                    f"{float(_tr_initial_group.get('effective_decisive_review_count') or 0):g} effective · "
                                    f"{int(_tr_initial_group.get('reviewed_cohort_count') or 0)} cohorts"
                                ),
                                "Repaired-origin evidence": (
                                    f"{int(_tr_repaired_group.get('decisive_review_count') or 0)} raw · "
                                    f"{float(_tr_repaired_group.get('effective_decisive_review_count') or 0):g} effective · "
                                    f"{int(_tr_repaired_group.get('reviewed_cohort_count') or 0)} cohorts"
                                ),
                                "Initial confirmation": (
                                    f"{float(_tr_initial_group.get('confirmation_rate')):.0%}"
                                    if _tr_initial_group.get("confirmation_rate")
                                    is not None else "—"
                                ),
                                "Repaired confirmation": (
                                    f"{float(_tr_repaired_group.get('confirmation_rate')):.0%}"
                                    if _tr_repaired_group.get("confirmation_rate")
                                    is not None else "—"
                                ),
                                "Observed difference": (
                                    f"{float(_tr_origin_difference):+.0%}"
                                    if _tr_origin_difference is not None else "—"
                                ),
                            })
                        with st.expander(
                            "Bounded-repair outcome comparison",
                            expanded=False,
                        ):
                            st.caption(
                                "Repaired-origin and initial-pass predictions are "
                                "compared only within the same target category, model, "
                                "analysis-engine fingerprint, and applied category "
                                "guardrail state. Both groups need at "
                                "least eight raw/effective reviews across three cohorts "
                                "and three hypotheses. This is observational and never "
                                "changes the bounded-repair policy automatically."
                            )
                            _tr_unknown_lineage_provenance = int(
                                _tr_repair_origin_effectiveness.get(
                                    "unknown_provenance_decisive_review_count"
                                ) or 0
                            )
                            if _tr_unknown_lineage_provenance:
                                st.caption(
                                    f"Excluded {_tr_unknown_lineage_provenance} "
                                    "decisive reviews without exact schema-11 coherent "
                                    "claim lineage and run configuration."
                                )
                            show_dataframe(
                                pd.DataFrame(_tr_repair_rows),
                                width="stretch",
                                hide_index=True,
                            )
                    st.markdown("### 📈 Previous TR Prediction Check")
                    _tr_prior_build_label = str(
                        _tr_prior_eval.get("prior_close_build") or "Unknown"
                    )
                    _tr_current_build_label = str(
                        _tr_prior_eval.get("current_close_build") or _tr_build
                    )
                    st.caption(
                        "Did the areas predicted for the previous build actually "
                        "produce bugs in this one? Treat it as a tuning signal, not "
                        "a score."
                    )
                    st.caption(
                        f"Prior build {_tr_prior_build_label} "
                        f"→ current build {_tr_current_build_label}. "
                        + str(_tr_prior_eval.get("limitation") or "")
                    )
                    _tr_selection_note, _tr_comparison_warning = prior_comparison_context(_tr_prior_eval)
                    st.caption(_tr_selection_note)
                    st.warning(_tr_comparison_warning)
                    _tr_summary_count = int(_tr_prior_eval.get("category_evaluable_hypothesis_count")
                                            or _tr_prior_eval.get("scoreable_hypothesis_count") or 0)
                    _tr_summary_overlap = int(_tr_prior_eval.get("category_overlap_count") or 0)
                    st.info(
                        f"{_tr_summary_overlap} of {_tr_summary_count} category overlaps. "
                        + ("Insufficient evidence to judge prediction quality."
                           if _tr_summary_count < 2 else
                           "Category overlap does not confirm the predicted scenario; human review is required.")
                    )
                    if _tr_prior_build_label == _tr_current_build_label:
                        st.caption(
                            "Prior and current Close Build are the same, so there is "
                            "no new-bug window to score yet; the counts below stay at "
                            "zero until a later build is analyzed."
                        )
                    with st.expander("Matcher methodology", expanded=False):
                        _tr_match_threshold = float(
                            _tr_prior_eval.get("feature_match_threshold") or 0.24
                        )
                        _tr_calibration = _tr_prior_eval.get("threshold_calibration") or {}
                        _tr_global_calibration = (
                            _tr_calibration.get("global")
                            if isinstance(_tr_calibration.get("global"), dict)
                            else _tr_calibration
                        )
                        _tr_calibration_status = str(
                            _tr_global_calibration.get("status") or "legacy_default"
                        )
                        _tr_calibration_labels = {
                            "insufficient_reviews": "default; insufficient decisive reviews",
                            "insufficient_diversity": (
                                "default; reviews are too concentrated"
                            ),
                            "default_performing_well": "default; reviewed quality is healthy",
                            "default_recall_warning": (
                                "default; confirmed below-threshold match found"
                            ),
                            "calibrated_higher": "raised from reviewed evidence",
                            "no_safe_higher_threshold": "default; no supported higher cutoff",
                            "legacy_default": "default; legacy evaluation",
                        }
                        _tr_global_raw_reviews = int(
                            _tr_global_calibration.get("decisive_review_count") or 0
                        )
                        _tr_global_effective_reviews = _tr_global_calibration.get(
                            "effective_decisive_review_count"
                        )
                        _tr_global_cohorts = int(
                            _tr_global_calibration.get("reviewed_cohort_count") or 0
                        )
                        if _tr_global_effective_reviews is not None:
                            _tr_calibration_evidence = (
                                f" ({_tr_global_raw_reviews} raw; "
                                f"{float(_tr_global_effective_reviews):g} effective "
                                f"across {_tr_global_cohorts} cohorts)."
                            )
                        _tr_below_default_confirmed = int(
                            _tr_global_calibration.get(
                                "confirmed_below_default_audit_count"
                            ) or 0
                        )
                        if _tr_below_default_confirmed:
                            st.caption(
                                f"Recall safeguard: {_tr_below_default_confirmed} reviewed "
                                "audit sample below 0.24 was confirmed, so the matcher "
                                "threshold was not raised."
                            )
                        elif (
                            _tr_global_calibration
                            and _tr_global_effective_reviews is None
                        ):
                            _tr_calibration_evidence = (
                                f" ({_tr_global_raw_reviews} scored decisive reviews)."
                            )
                        else:
                            _tr_calibration_evidence = "."
                        st.caption(
                            f"Feature-match threshold: {_tr_match_threshold:.2f} — "
                            + _tr_calibration_labels.get(
                                _tr_calibration_status, _tr_calibration_status
                            )
                            + _tr_calibration_evidence
                        )
                        _tr_matcher_policy = str(
                            _tr_prior_eval.get("matcher_policy") or "legacy_union_v1"
                        )
                        st.caption(
                            "Matcher policy: "
                            + _tr_matcher_policy.replace("_", " ")
                            + (
                                " — candidates must overlap the predicted area and "
                                "at least one trigger or failure-mode facet."
                                if _tr_matcher_policy == "facet_coverage_v2"
                                else " — legacy pooled-token evaluation."
                            )
                        )
                        if _tr_global_effective_reviews is not None:
                            st.caption(
                                "Effective reviews discount correlated candidates: "
                                "at most four per build-pair cohort and two per "
                                "hypothesis within that cohort."
                            )
                        _tr_excluded_model_rejections = int(
                            _tr_global_calibration.get(
                                "excluded_non_matcher_rejection_count"
                            ) or 0
                        )
                        _tr_excluded_unroutable_rejections = int(
                            _tr_global_calibration.get(
                                "excluded_unroutable_rejection_count"
                            ) or 0
                        )
                        _tr_excluded_matcher_reviews = int(
                            _tr_global_calibration.get(
                                "excluded_matcher_policy_review_count"
                            ) or 0
                        )
                        if (
                            _tr_excluded_model_rejections
                            or _tr_excluded_unroutable_rejections
                        ):
                            st.caption(
                                "Calibration routing excluded "
                                f"{_tr_excluded_model_rejections} AI-analysis/analog "
                                "rejections and "
                                f"{_tr_excluded_unroutable_rejections} legacy or "
                                "reasonless rejections from lexical-threshold tuning."
                            )
                        if _tr_excluded_matcher_reviews:
                            st.caption(
                                f"Excluded {_tr_excluded_matcher_reviews} scored reviews "
                                "created by a different matcher policy; score cutoffs are "
                                "not transferred across incompatible scoring rules."
                            )
                        _tr_best_higher_attempt = _tr_global_calibration.get(
                            "best_higher_threshold_attempt"
                        )
                        if (
                            _tr_calibration_status == "no_safe_higher_threshold"
                            and isinstance(_tr_best_higher_attempt, dict)
                        ):
                            _tr_failed_gates = []
                            if float(
                                _tr_best_higher_attempt.get("confirmation_rate") or 0
                            ) < float(
                                _tr_global_calibration.get(
                                    "target_confirmation_rate"
                                ) or 0.80
                            ):
                                _tr_failed_gates.append("confirmation rate")
                            if float(
                                _tr_best_higher_attempt.get("wilson_lower_bound") or 0
                            ) < float(
                                _tr_global_calibration.get(
                                    "minimum_wilson_lower_bound"
                                ) or 0.50
                            ):
                                _tr_failed_gates.append("Wilson confidence")
                            if float(
                                _tr_best_higher_attempt.get(
                                    "confirmed_retention_rate"
                                ) or 0
                            ) < float(
                                _tr_global_calibration.get(
                                    "minimum_confirmed_retention_rate"
                                ) or 0.80
                            ):
                                _tr_failed_gates.append("confirmed-match retention")
                            st.caption(
                                f"Best reviewed higher-cutoff attempt: "
                                f"{float(_tr_best_higher_attempt.get('threshold') or 0):.2f}; "
                                "not adopted because it missed "
                                + ", ".join(_tr_failed_gates or ["a safety gate"])
                                + "."
                            )
                    _tr_eval_schema = int(
                        _tr_prior_eval.get("schema_version") or 0
                    )
                    _tr_eval_scoreable = int(
                        _tr_prior_eval.get(
                            "category_evaluable_hypothesis_count"
                        )
                        or _tr_prior_eval.get("scoreable_hypothesis_count")
                        or 0
                    )
                    _tr_eval_scenario = (
                        int(_tr_prior_eval.get(
                            "scenario_evaluable_hypothesis_count"
                        ) or 0)
                        if _tr_eval_schema >= 9 else None
                    )
                    _tr_eval_insufficient = int(
                        _tr_prior_eval.get(
                            "insufficient_prediction_detail_count"
                        ) or 0
                    )
                    _tr_eval_overlap = int(
                        _tr_prior_eval.get("category_overlap_count") or 0
                    )
                    _tr_eval_feature = int(
                        _tr_prior_eval.get("feature_candidate_hypothesis_count") or 0
                    )
                    _tr_eval_audit = int(
                        _tr_prior_eval.get("threshold_audit_candidate_count") or 0
                    )
                    _tr_eval_rate = _tr_prior_eval.get("category_overlap_rate")
                    _tr_eval_cols = st.columns(4)
                    _tr_eval_cols[0].metric(
                        "Category-evaluable", _tr_eval_scoreable
                    )
                    _tr_eval_cols[1].metric(
                        "Scenario-evaluable",
                        _tr_eval_scenario
                        if _tr_eval_scenario is not None else "—",
                    )
                    _tr_eval_cols[2].metric(
                        "Category overlaps", _tr_eval_overlap
                    )
                    _tr_eval_cols[3].metric(
                        "Feature candidates", _tr_eval_feature
                    )
                    _tr_observed_scenario_rate = _tr_prior_eval.get(
                        "feature_candidate_rate_given_category_observed"
                    )
                    if _tr_eval_scoreable >= 2:
                        st.caption(
                            "Coarse category-overlap rate: "
                            + (
                                f"{float(_tr_eval_rate):.0%}"
                                if _tr_eval_rate is not None else "—"
                            )
                            + " · Feature-candidate yield among scenario-evaluable "
                            "hypotheses whose category was observed: "
                            + (
                                f"{float(_tr_observed_scenario_rate):.0%}"
                                if _tr_observed_scenario_rate is not None else "—"
                            )
                            + "."
                        )
                    if _tr_eval_insufficient:
                        st.caption(
                            f"{_tr_eval_insufficient} category-evaluable prior "
                            "hypotheses lacked distinctive area/trigger/failure "
                            "detail and were excluded from scenario-match rates."
                        )
                    if _tr_eval_audit:
                        st.caption(
                            f"Boundary audit samples: {_tr_eval_audit}. These are "
                            "deliberately below the applied threshold and are not "
                            "prediction matches."
                        )
                    if _tr_review_summary["prediction_reviewed_count"]:
                        _tr_review_rate = _tr_review_summary[
                            "reviewed_confirmation_rate"
                        ]
                        st.caption(
                            "Prediction-candidate reviews — "
                            f"Confirmed: {_tr_review_summary['prediction_confirmed_count']} · "
                            f"Rejected: {_tr_review_summary['prediction_rejected_count']} · "
                            f"Unclear: {_tr_review_summary['prediction_unclear_count']} · "
                            "Reviewed confirmation rate: "
                            + (
                                f"{float(_tr_review_rate):.0%}"
                                if _tr_review_rate is not None else "—"
                            )
                        )
                    if _tr_review_summary["threshold_audit_reviewed_count"]:
                        st.caption(
                            "Boundary-audit reviews — "
                            f"Reviewed: {_tr_review_summary['threshold_audit_reviewed_count']} · "
                            f"Confirmed false negatives: "
                            f"{_tr_review_summary['threshold_audit_confirmed_count']}. "
                            "These are excluded from prediction confirmation rate."
                        )
                        _tr_audit_rejection_counts = _tr_review_summary.get(
                            "threshold_audit_rejection_reason_counts"
                        ) or {}
                        if _tr_audit_rejection_counts:
                            st.caption(
                                "Why boundary-audit samples were rejected — "
                                + " · ".join(
                                    f"{REJECTION_REASON_LABELS.get(code, code)}: {count}"
                                    for code, count in sorted(
                                        _tr_audit_rejection_counts.items(),
                                        key=lambda item: (
                                            -int(item[1]), str(item[0])
                                        ),
                                    )
                                )
                            )
                    _tr_rejection_counts = _tr_review_summary.get(
                        "prediction_rejection_reason_counts"
                    ) or {}
                    if _tr_rejection_counts:
                        st.caption(
                            "Why prediction candidates were rejected — "
                            + " · ".join(
                                f"{REJECTION_REASON_LABELS.get(code, code)}: {count}"
                                for code, count in sorted(
                                    _tr_rejection_counts.items(),
                                    key=lambda item: (
                                        -int(item[1]), str(item[0])
                                    ),
                                )
                            )
                        )
                    if _tr_review_summary.get("rejected_without_reason_count"):
                        st.caption(
                            f"Legacy rejected reviews without a structured reason: "
                            f"{int(_tr_review_summary['rejected_without_reason_count'])}. "
                            "They remain valid outcomes but cannot be routed to a "
                            "specific improvement layer."
                        )
                    _tr_historical_comparison = _tr_review_summary.get(
                        "historical_support_comparison"
                    )
                    if isinstance(_tr_historical_comparison, dict):
                        _tr_supported_group = (
                            _tr_historical_comparison.get("supported") or {}
                        )
                        _tr_unsupported_group = (
                            _tr_historical_comparison.get("unsupported") or {}
                        )
                        _tr_supported_rate = _tr_supported_group.get(
                            "confirmation_rate"
                        )
                        _tr_unsupported_rate = _tr_unsupported_group.get(
                            "confirmation_rate"
                        )
                        st.caption(
                            "Historical-corroboration outcome check — "
                            f"supported: {int(_tr_supported_group.get('decisive_review_count') or 0)} decisive"
                            + (
                                f" / {float(_tr_supported_rate):.0%} confirmed"
                                if _tr_supported_rate is not None else ""
                            )
                            + "; unsupported: "
                            f"{int(_tr_unsupported_group.get('decisive_review_count') or 0)} decisive"
                            + (
                                f" / {float(_tr_unsupported_rate):.0%} confirmed"
                                if _tr_unsupported_rate is not None else ""
                            )
                            + ". Status: "
                            + str(_tr_historical_comparison.get("status") or "unknown")
                                .replace("_", " ")
                            + ". This comparison is observational, not causal."
                        )
                    _tr_eval_rows = []
                    _tr_eval_labels = {
                        "feature_match_candidate": "Feature match candidate — review",
                        "category_overlap": "Category observed",
                        "category_overlap_insufficient_prediction_detail": (
                            "Category observed; scenario detail insufficient"
                        ),
                        "not_observed_in_current_cohort": "Not observed in this cohort",
                        "unscorable_cross_cutting": "Cross-cutting / unscorable",
                    }
                    _tr_review_candidates = []
                    _tr_threshold_source_labels = {
                        "category_reviewed": "Category reviewed",
                        "global_reviewed": "Global reviewed fallback",
                        "default": "Default; insufficient reviews",
                        "global_or_default": "Legacy global/default",
                    }
                    for _tr_hyp in (_tr_prior_eval.get("hypotheses") or []):
                        if not isinstance(_tr_hyp, dict):
                            continue
                        _tr_all_candidates = (
                            list(_tr_hyp.get("feature_match_candidates") or [])
                            + list(_tr_hyp.get("threshold_audit_candidates") or [])
                        )
                        _tr_eval_rows.append({
                            "Hypothesis": str(_tr_hyp.get("hypothesis_id") or ""),
                            "Target category": str(_tr_hyp.get("target_category") or ""),
                            "Predicted area": str(_tr_hyp.get("area") or ""),
                            "Historical corroboration": (
                                ", ".join(
                                    str(code) for code in (
                                        _tr_hyp.get("historical_analog_codes") or []
                                    )
                                )
                                if _tr_hyp.get("historical_support") is True
                                else (
                                    "None"
                                    if _tr_hyp.get("historical_support") is False
                                    else "Legacy unknown"
                                )
                            ),
                            "AI guardrail provenance": str(
                                _tr_hyp.get("analysis_guardrail_provenance")
                                or "legacy_unknown"
                            ).replace("_", " "),
                            "Applied AI guardrail": _tr_guardrail_labels.get(
                                str(_tr_hyp.get("analysis_guardrail") or ""),
                                str(_tr_hyp.get("analysis_guardrail") or "—")
                                .replace("_", " "),
                            ),
                            "Prediction origin": {
                                "initial_audit_pass_only": "Initial audit pass only",
                                "recovered_draft_only": "Repaired draft only",
                                "mixed_initial_and_recovered": (
                                    "Mixed initial and repaired"
                                ),
                                "legacy_unknown": "Legacy unknown",
                            }.get(
                                str(_tr_hyp.get("prediction_origin") or ""),
                                "Legacy unknown",
                            ),
                            "Claim lineage provenance": str(
                                _tr_hyp.get("prediction_origin_provenance")
                                or "unavailable_legacy"
                            ).replace("_", " "),
                            "Draft sources": ", ".join(
                                str(candidate_id) for candidate_id in (
                                    _tr_hyp.get("synthesis_source_candidate_ids")
                                    or []
                                )
                            ) or "—",
                            "Scope boundary": str(
                                _tr_hyp.get("scope_boundary") or "Legacy unavailable"
                            ),
                            "Disconfirming observation": str(
                                _tr_hyp.get("disconfirming_observation")
                                or "Legacy unavailable"
                            ),
                            "Scenario evaluability": {
                                "evaluable": "Evaluable",
                                "insufficient_prediction_detail": (
                                    "Insufficient prediction detail"
                                ),
                                "unscorable_target": "Unscorable target",
                            }.get(
                                str(_tr_hyp.get(
                                    "scenario_evaluation_status"
                                ) or ""),
                                "Legacy unavailable",
                            ),
                            "Scenario insufficiency reasons": ", ".join(
                                str(reason).replace("_", " ")
                                for reason in (
                                    _tr_hyp.get(
                                        "scenario_evaluation_reasons"
                                    ) or []
                                )
                            ) or "—",
                            "Match threshold": f"{float(_tr_hyp.get('feature_match_threshold') or _tr_match_threshold):.2f}",
                            "Threshold source": _tr_threshold_source_labels.get(
                                str(_tr_hyp.get("threshold_source") or "global_or_default"),
                                str(_tr_hyp.get("threshold_source") or "global_or_default"),
                            ),
                            "Threshold status": _tr_calibration_labels.get(
                                str(_tr_hyp.get("threshold_status") or "legacy_default"),
                                str(_tr_hyp.get("threshold_status") or "legacy_default"),
                            ),
                            "Reviewed evidence": (
                                (
                                    f"{int(_tr_hyp.get('threshold_decisive_review_count') or 0)} raw · "
                                    f"{float(_tr_hyp.get('threshold_effective_decisive_review_count') or 0):g} effective · "
                                    f"{int(_tr_hyp.get('threshold_reviewed_cohort_count') or 0)} cohorts"
                                )
                                if _tr_hyp.get(
                                    "threshold_effective_decisive_review_count"
                                ) is not None
                                else (
                                    f"{int(_tr_hyp.get('threshold_decisive_review_count') or 0)} decisive (legacy)"
                                    if _tr_hyp.get("threshold_decisive_review_count")
                                    else "—"
                                )
                            ),
                            "Result": _tr_eval_labels.get(
                                str(_tr_hyp.get("result") or ""),
                                str(_tr_hyp.get("result") or ""),
                            ),
                            "Current BugCodes": ", ".join(
                                str(code) for code in (
                                    _tr_hyp.get("matching_bug_codes") or []
                                )
                            ),
                            "Candidate evidence": " · ".join(
                                f"{candidate.get('bug_code')} "
                                f"({float(candidate.get('match_score') or 0):.2f}: "
                                f"{', '.join(str(term) for term in (candidate.get('overlap_terms') or []))})"
                                for candidate in (
                                    _tr_hyp.get("feature_match_candidates") or []
                                )
                                if isinstance(candidate, dict)
                            ),
                            "Candidate facets": " · ".join(
                                f"{candidate.get('bug_code')}: "
                                + "; ".join(
                                    f"{facet.replace('_', ' ')}="
                                    f"{', '.join(str(term) for term in terms) or 'none'}"
                                    for facet, terms in (
                                        candidate.get("overlap_facets") or {}
                                    ).items()
                                )
                                for candidate in (
                                    _tr_hyp.get("feature_match_candidates") or []
                                )
                                if isinstance(candidate, dict)
                                and candidate.get("overlap_facets")
                            ),
                            "Boundary audit": " · ".join(
                                f"{candidate.get('bug_code')} "
                                f"({float(candidate.get('match_score') or 0):.6f} "
                                f"< {float(candidate.get('applied_threshold') or _tr_match_threshold):.3f})"
                                for candidate in (
                                    _tr_hyp.get("threshold_audit_candidates") or []
                                )
                                if isinstance(candidate, dict)
                            ),
                            "Human review": " · ".join(
                                f"{candidate.get('bug_code')}: "
                                f"{candidate.get('review_status') or 'unreviewed'}"
                                + (
                                    " ("
                                    + REJECTION_REASON_LABELS.get(
                                        str(candidate.get(
                                            "review_rejection_reason"
                                        ) or ""),
                                        str(candidate.get(
                                            "review_rejection_reason"
                                        ) or ""),
                                    )
                                    + ")"
                                    if candidate.get(
                                        "review_rejection_reason"
                                    ) else ""
                                )
                                for candidate in _tr_all_candidates
                                if isinstance(candidate, dict)
                            ),
                        })
                        for _tr_candidate in _tr_all_candidates:
                            if isinstance(_tr_candidate, dict):
                                _tr_review_candidates.append((
                                    _tr_hyp, _tr_candidate
                                ))
                    if _tr_eval_rows:
                        with st.expander("Review hypothesis outcomes", expanded=False):
                            show_dataframe(
                                pd.DataFrame(_tr_eval_rows),
                                width="stretch",
                                hide_index=True,
                            )
                            if _tr_review_candidates:
                                st.markdown("#### Human review")
                                st.caption(
                                    "Confirm only when the candidate bug matches the "
                                    "predicted adjacent scenario. Reject category-only "
                                    "coincidences; use Unclear when evidence is insufficient. "
                                    "Boundary audits are sampled below the cutoff to measure "
                                    "false negatives and are not predictions."
                                )
                                _tr_decision_labels = {
                                    "confirmed": "Confirmed match",
                                    "rejected": "Rejected — unrelated scenario",
                                    "unclear": "Unclear — needs investigation",
                                }
                                for _tr_hyp, _tr_candidate in _tr_review_candidates:
                                    _tr_candidate_code = str(
                                        _tr_candidate.get("bug_code") or "Unknown"
                                    )
                                    _tr_hypothesis_id = str(
                                        _tr_hyp.get("hypothesis_id") or ""
                                    )
                                    _tr_candidate_kind = str(
                                        _tr_candidate.get("candidate_kind")
                                        or "feature_match"
                                    )
                                    _tr_kind_label = (
                                        "Boundary audit — below threshold"
                                        if _tr_candidate_kind == "threshold_audit"
                                        else "Feature match candidate"
                                    )
                                    _tr_review_key = str(
                                        _tr_candidate.get("review_key")
                                        or f"{_tr_hypothesis_id}_{_tr_candidate_code}"
                                    )
                                    _tr_current_decision = str(
                                        _tr_candidate.get("review_status") or ""
                                    )
                                    _tr_options = list(_tr_decision_labels)
                                    _tr_default_index = (
                                        _tr_options.index(_tr_current_decision)
                                        if _tr_current_decision in _tr_options else None
                                    )
                                    _tr_current_rejection_reason = str(
                                        _tr_candidate.get(
                                            "review_rejection_reason"
                                        ) or ""
                                    )
                                    _tr_rejection_options = list(
                                        REJECTION_REASON_LABELS
                                    )
                                    if _tr_hyp.get("historical_support") is not True:
                                        _tr_rejection_options.remove(
                                            "historical_corroboration_misleading"
                                        )
                                    if _tr_hyp.get(
                                        "prediction_boundary_provenance"
                                    ) != "schema7_validated":
                                        _tr_rejection_options = [
                                            code for code in _tr_rejection_options
                                            if code not in {
                                                "outside_scope",
                                                "disconfirming_observation_seen",
                                            }
                                        ]
                                    _tr_hyp_target_alias_pairs = (
                                        _tr_hyp.get("target_alias_pairs") or []
                                    )
                                    if (
                                        _tr_hyp.get("target_alias_provenance")
                                        != "schema27_exact_basis"
                                        or len(_tr_hyp_target_alias_pairs) != 1
                                    ):
                                        _tr_rejection_options = [
                                            code for code in _tr_rejection_options
                                            if code
                                            != "reviewed_target_alias_misleading"
                                        ]
                                    if (
                                        _tr_current_rejection_reason
                                        and _tr_current_rejection_reason
                                        not in _tr_rejection_options
                                    ):
                                        _tr_rejection_options.append(
                                            _tr_current_rejection_reason
                                        )
                                    _tr_rejection_default_index = (
                                        _tr_rejection_options.index(
                                            _tr_current_rejection_reason
                                        )
                                        if _tr_current_rejection_reason
                                        in _tr_rejection_options else None
                                    )
                                    with st.form(
                                        key=f"tr_prediction_review_{_tr_review_key}",
                                        border=True,
                                    ):
                                        st.markdown(
                                            f"**{_tr_kind_label}: {_tr_hypothesis_id} "
                                            f"→ {_tr_candidate_code}**"
                                        )
                                        st.caption(
                                            f"{_tr_hyp.get('area') or 'Unknown area'} · "
                                            f"match score "
                                            f"{float(_tr_candidate.get('match_score') or 0):.2f} · "
                                            f"applied threshold "
                                            f"{float(_tr_candidate.get('applied_threshold') or _tr_hyp.get('feature_match_threshold') or _tr_match_threshold):.2f} · "
                                            f"terms: {', '.join(str(term) for term in (_tr_candidate.get('overlap_terms') or []))}"
                                        )
                                        _tr_candidate_facets = (
                                            _tr_candidate.get("overlap_facets") or {}
                                        )
                                        if _tr_candidate_facets:
                                            st.caption(
                                                "Facet evidence — "
                                                + " · ".join(
                                                    f"{facet.replace('_', ' ')}: "
                                                    + (
                                                        ", ".join(
                                                            str(term) for term in terms
                                                        ) or "none"
                                                    )
                                                    for facet, terms in (
                                                        _tr_candidate_facets.items()
                                                    )
                                                )
                                            )
                                        if _tr_hyp.get("scope_boundary"):
                                            st.caption(
                                                "Boundary: "
                                                + str(_tr_hyp.get("scope_boundary"))
                                            )
                                        if _tr_hyp.get(
                                            "disconfirming_observation"
                                        ):
                                            st.caption(
                                                "Disconfirming observation: "
                                                + str(_tr_hyp.get(
                                                    "disconfirming_observation"
                                                ))
                                            )
                                        if _tr_hyp_target_alias_pairs:
                                            st.caption(
                                                "Reviewed target alias provenance: "
                                                + " · ".join(
                                                    f"{pair.get('target_term')} → "
                                                    f"{pair.get('evidence_term')}"
                                                    for pair in _tr_hyp_target_alias_pairs
                                                    if isinstance(pair, dict)
                                                )
                                            )
                                        _tr_decision = st.selectbox(
                                            f"Decision for {_tr_candidate_code}",
                                            _tr_options,
                                            index=_tr_default_index,
                                            placeholder="Choose a decision",
                                            format_func=lambda value: _tr_decision_labels[value],
                                            key=f"tr_review_decision_{_tr_review_key}",
                                        )
                                        _tr_rejection_reason = st.selectbox(
                                            f"Rejection reason for {_tr_candidate_code}",
                                            _tr_rejection_options,
                                            index=_tr_rejection_default_index,
                                            placeholder=(
                                                "Required only when Rejected"
                                            ),
                                            format_func=lambda value: (
                                                REJECTION_REASON_LABELS[value]
                                            ),
                                            key=(
                                                "tr_review_rejection_reason_"
                                                + _tr_review_key
                                            ),
                                            help=(
                                                "Routes a rejection to matcher, AI "
                                                "analysis, or historical-analog learning."
                                            ),
                                        )
                                        _tr_reviewer = st.text_input(
                                            f"Reviewer for {_tr_candidate_code}",
                                            value=str(_tr_candidate.get("reviewer") or ""),
                                            max_chars=100,
                                            key=f"tr_review_reviewer_{_tr_review_key}",
                                        )
                                        _tr_review_note = st.text_area(
                                            f"Review note for {_tr_candidate_code}",
                                            value=str(
                                                _tr_candidate.get("review_note") or ""
                                            ),
                                            max_chars=1000,
                                            key=f"tr_review_note_{_tr_review_key}",
                                        )
                                        _tr_save_review = st.form_submit_button(
                                            f"Save review for {_tr_candidate_code}"
                                        )
                                    if _tr_save_review:
                                        if _tr_decision is None:
                                            st.warning(
                                                f"Choose a decision for {_tr_candidate_code}."
                                            )
                                        elif (
                                            _tr_decision == "rejected"
                                            and _tr_rejection_reason is None
                                        ):
                                            st.warning(
                                                "Choose a structured rejection reason "
                                                f"for {_tr_candidate_code}."
                                            )
                                        elif (
                                            _tr_decision == "rejected"
                                            and _tr_rejection_reason == "other"
                                            and not _tr_review_note.strip()
                                        ):
                                            st.warning(
                                                "Add a review note when using Other reason."
                                            )
                                        else:
                                            try:
                                                upsert_review(
                                                    _tr_review_fp,
                                                    product=selected_slug or "legacy",
                                                    hypothesis_id=_tr_hypothesis_id,
                                                    prior_close_build=str(
                                                        _tr_prior_eval.get(
                                                            "prior_close_build"
                                                        ) or ""
                                                    ),
                                                    current_close_build=str(
                                                        _tr_prior_eval.get(
                                                            "current_close_build"
                                                        ) or _tr_build
                                                    ),
                                                    bug_code=_tr_candidate_code,
                                                    target_category=str(
                                                        _tr_hyp.get("target_category") or ""
                                                    ),
                                                    predicted_area=str(
                                                        _tr_hyp.get("area") or ""
                                                    ),
                                                    decision=_tr_decision,
                                                    rejection_reason=(
                                                        _tr_rejection_reason
                                                        if _tr_decision == "rejected"
                                                        else None
                                                    ),
                                                    reviewer=_tr_reviewer,
                                                    note=_tr_review_note,
                                                    match_score=float(
                                                        _tr_candidate.get(
                                                            "match_score"
                                                        ) or 0
                                                    ),
                                                    overlap_terms=[
                                                        str(term) for term in (
                                                            _tr_candidate.get(
                                                                "overlap_terms"
                                                            ) or []
                                                        )
                                                    ],
                                                    matcher_policy=(
                                                        str(_tr_candidate.get(
                                                            "matcher_policy"
                                                        ))
                                                        if _tr_candidate.get(
                                                            "matcher_policy"
                                                        ) else None
                                                    ),
                                                    overlap_facets=(
                                                        {
                                                            str(facet): [
                                                                str(term)
                                                                for term in terms
                                                            ]
                                                            for facet, terms in (
                                                                _tr_candidate.get(
                                                                    "overlap_facets"
                                                                ) or {}
                                                            ).items()
                                                        }
                                                        if _tr_candidate.get(
                                                            "overlap_facets"
                                                        ) else None
                                                    ),
                                                    trigger=str(
                                                        _tr_hyp.get("trigger") or ""
                                                    ),
                                                    expected_failure_mode=str(
                                                        _tr_hyp.get(
                                                            "expected_failure_mode"
                                                        ) or ""
                                                    ),
                                                    candidate_kind=_tr_candidate_kind,
                                                    applied_threshold=float(
                                                        _tr_candidate.get(
                                                            "applied_threshold"
                                                        )
                                                        or _tr_hyp.get(
                                                            "feature_match_threshold"
                                                        )
                                                        or _tr_match_threshold
                                                    ),
                                                    historical_analog_codes=(
                                                        [
                                                            str(code) for code in (
                                                                _tr_hyp.get(
                                                                    "historical_analog_codes"
                                                                ) or []
                                                            )
                                                        ]
                                                        if isinstance(
                                                            _tr_hyp.get(
                                                                "historical_support"
                                                            ), bool
                                                        )
                                                        else None
                                                    ),
                                                    historical_analog_max_similarity=(
                                                        float(
                                                            _tr_hyp.get(
                                                                "historical_analog_max_similarity"
                                                            )
                                                        )
                                                        if _tr_hyp.get(
                                                            "historical_support"
                                                        ) is True
                                                        and _tr_hyp.get(
                                                            "historical_analog_max_similarity"
                                                        ) is not None
                                                        else None
                                                    ),
                                                    scope_boundary=(
                                                        str(_tr_hyp.get(
                                                            "scope_boundary"
                                                        ))
                                                        if _tr_hyp.get(
                                                            "prediction_boundary_provenance"
                                                        ) == "schema7_validated"
                                                        else None
                                                    ),
                                                    disconfirming_observation=(
                                                        str(_tr_hyp.get(
                                                            "disconfirming_observation"
                                                        ))
                                                        if _tr_hyp.get(
                                                            "prediction_boundary_provenance"
                                                        ) == "schema7_validated"
                                                        else None
                                                    ),
                                                    analysis_guardrail_provenance=(
                                                        str(_tr_hyp.get(
                                                            "analysis_guardrail_provenance"
                                                        ) or "legacy_unknown")
                                                    ),
                                                    analysis_guardrail=(
                                                        str(_tr_hyp.get(
                                                            "analysis_guardrail"
                                                        ))
                                                        if _tr_hyp.get(
                                                            "analysis_guardrail"
                                                        ) else None
                                                    ),
                                                    analysis_guardrail_source_rejection_reason=(
                                                        str(_tr_hyp.get(
                                                            "analysis_guardrail_source_rejection_reason"
                                                        ))
                                                        if _tr_hyp.get(
                                                            "analysis_guardrail_source_rejection_reason"
                                                        ) else None
                                                    ),
                                                    analysis_model=(
                                                        str(_tr_hyp.get(
                                                            "analysis_model"
                                                        ))
                                                        if (
                                                            _tr_hyp.get(
                                                                "analysis_guardrail_provenance"
                                                            ) == "exact_policy"
                                                            or _tr_hyp.get(
                                                                "prediction_origin_provenance"
                                                            ) in {
                                                                "schema10_exact_synthesis_lineage",
                                                                "schema11_exact_claim_lineage",
                                                            }
                                                        )
                                                        else None
                                                    ),
                                                    analysis_engine_sha256=(
                                                        str(_tr_hyp.get(
                                                            "analysis_engine_sha256"
                                                        ))
                                                        if (
                                                            _tr_hyp.get(
                                                                "analysis_guardrail_provenance"
                                                            ) == "exact_policy"
                                                            or _tr_hyp.get(
                                                                "prediction_origin_provenance"
                                                            ) in {
                                                                "schema10_exact_synthesis_lineage",
                                                                "schema11_exact_claim_lineage",
                                                            }
                                                        )
                                                        else None
                                                    ),
                                                    prediction_origin_provenance=str(
                                                        _tr_hyp.get(
                                                            "prediction_origin_provenance"
                                                        ) or "unavailable_legacy"
                                                    ),
                                                    prediction_origin=str(
                                                        _tr_hyp.get(
                                                            "prediction_origin"
                                                        ) or "legacy_unknown"
                                                    ),
                                                    synthesis_source_candidate_ids=[
                                                        str(candidate_id)
                                                        for candidate_id in (
                                                            _tr_hyp.get(
                                                                "synthesis_source_candidate_ids"
                                                            ) or []
                                                        )
                                                    ],
                                                    recovered_source_candidate_ids=[
                                                        str(candidate_id)
                                                        for candidate_id in (
                                                            _tr_hyp.get(
                                                                "recovered_source_candidate_ids"
                                                            ) or []
                                                        )
                                                    ],
                                                    target_alias_provenance=str(
                                                        _tr_hyp.get(
                                                            "target_alias_provenance"
                                                        ) or "unavailable_legacy"
                                                    ),
                                                    target_alias_pairs=[
                                                        {
                                                            "target_term": str(
                                                                pair.get("target_term")
                                                            ),
                                                            "evidence_term": str(
                                                                pair.get("evidence_term")
                                                            ),
                                                        }
                                                        for pair in (
                                                            _tr_hyp.get(
                                                                "target_alias_pairs"
                                                            ) or []
                                                        )
                                                        if isinstance(pair, dict)
                                                    ],
                                                )
                                                st.success(
                                                    f"Saved {_tr_decision_labels[_tr_decision]} "
                                                    f"for {_tr_candidate_code}."
                                                )
                                                st.rerun()
                                            except ReviewLedgerError as _tr_review_error:
                                                st.error(
                                                    "Review was not saved: "
                                                    + str(_tr_review_error)
                                                )
                    else:
                        st.info(
                            "The prior report contained no scoreable adjacent "
                            "regression hypotheses — it published none. What it "
                            "did leave behind is the drafts its gates declined, "
                            "scored below as a tuning signal."
                        )
                    _tr_withheld_retro = (
                        _tr_prior_eval.get("withheld_draft_retrospective") or {}
                    )
                    _tr_retro_rows = [
                        row for row in (_tr_withheld_retro.get("drafts") or [])
                        if isinstance(row, dict)
                    ]
                    if _tr_retro_rows:
                        st.markdown("#### 🚧 Withheld Draft Check")
                        st.caption(
                            "Not predictions and not scored as such — these were "
                            "declined. This asks whether the bar was too high: "
                            "did a gate stop a draft in a category that then "
                            "produced a bug? A category overlap does not mean the "
                            "draft was right; the predicted area is often not the "
                            "area that failed, which is why the check that "
                            "stopped each draft is shown."
                        )
                        _tr_retro_overlaps = int(
                            _tr_withheld_retro.get("category_overlap_count") or 0
                        )
                        _tr_retro_evaluable = int(
                            _tr_withheld_retro.get("category_evaluable_count") or 0
                        )
                        _tr_retro_features = int(
                            _tr_withheld_retro.get("feature_candidate_count") or 0
                        )
                        _tr_retro_areas = [
                            str(area) for area in
                            _tr_withheld_retro.get("distinct_overlap_areas") or []
                        ]
                        _tr_retro_cols = st.columns(4)
                        _tr_retro_cols[0].metric(
                            "Withheld drafts",
                            int(_tr_withheld_retro.get("withheld_draft_count") or 0),
                        )
                        _tr_retro_cols[1].metric(
                            "Category-evaluable", _tr_retro_evaluable
                        )
                        _tr_retro_cols[2].metric(
                            "Category overlaps", _tr_retro_overlaps
                        )
                        _tr_retro_cols[3].metric(
                            "Feature candidates", _tr_retro_features
                        )
                        if _tr_retro_overlaps and not _tr_retro_features:
                            st.caption(
                                f"{_tr_retro_overlaps} declined draft"
                                + ("s" if _tr_retro_overlaps != 1 else "")
                                + " sat in a category this build produced a bug "
                                "in, across "
                                + str(len(_tr_retro_areas))
                                + " distinct area"
                                + ("s" if len(_tr_retro_areas) != 1 else "")
                                + (f" ({', '.join(_tr_retro_areas)})"
                                   if _tr_retro_areas else "")
                                + " — but none matched the failing feature. "
                                "Category alone is weak evidence that the bar is "
                                "too high; treat the repeated check below as the "
                                "thing to look at, not as a miss."
                            )
                        _tr_retro_check_counts = (
                            _tr_withheld_retro.get("overlap_failed_check_counts")
                            or {}
                        )
                        if _tr_retro_check_counts:
                            _tr_retro_top = max(
                                _tr_retro_check_counts.items(),
                                key=lambda pair: (pair[1], pair[0]),
                            )
                            st.caption(
                                "Most frequent check among overlapping drafts: "
                                f"**{str(_tr_retro_top[0]).replace('_', ' ')}** "
                                f"({_tr_retro_top[1]} of {_tr_retro_overlaps})."
                            )
                        show_dataframe([
                            {
                                "Area": str(row.get("area") or ""),
                                "Category": str(row.get("target_category") or ""),
                                "Result": str(row.get("result") or "")
                                .replace("_", " "),
                                "Stopped by": str(row.get("withheld_reason") or "")
                                .replace("_", " "),
                                "Failed check": ", ".join(
                                    str(check).replace("_", " ")
                                    for check in row.get("failed_checks") or []
                                ),
                                "Bugs in category": ", ".join(
                                    str(code) for code in
                                    row.get("observed_bug_codes") or []
                                ),
                            }
                            for row in _tr_retro_rows
                        ], hide_index=True, width="stretch")


            else:
                st.warning(
                    "The TR manifest exists, but its report path is missing, unreadable, "
                    "or outside the selected product directory."
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as _tr_err:
            st.warning(f"Could not load newest TR test-focus report: {_tr_err}")
