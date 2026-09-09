#!/usr/bin/env python3
"""PDR-I NLP Defect Clustering v3.0 — Severity stratification + cluster intelligence

New in v3.0:
  - Severity-stratified clustering (--stratify-severity):
      S1/S2 bugs are clustered separately from S3/S4. The global pass still
      runs on all bugs. Two extra columns are added to the output:
          cluster_id_s12 / cluster_label_s12  ← critical/major tier
          cluster_id_s34 / cluster_label_s34  ← normal/minor tier
      Stratified summaries are saved to:
          clusters/<stem>_cluster_summary_s12.csv
          clusters/<stem>_cluster_summary_s34.csv

  - Cluster velocity: for each cluster, compares bug count in the most recent
    VELOCITY_WINDOW versions against the prior VELOCITY_WINDOW versions.
    Exported as cluster_velocity_ratio and cluster_trend (growing/stable/declining)
    in the cluster summary CSV.

  - Module cluster entropy: Shannon entropy (base-2) of the cluster-assignment
    distribution per module. High entropy = broad instability (bugs spread across
    many themes). Low entropy = focused failure mode (bugs concentrated in one theme).
    Exported to clusters/<stem>_module_entropy.csv.

  - Cluster recurrence rate: fraction of bugs in each cluster whose source module
    also contributed to that cluster in the prior version window. High recurrence =
    the fix isn't holding — a strong predictor of future bugs of the same type.
    Exported as recurrence_rate in the cluster summary CSV.

  - Summary CSV columns (v3.0):
      cluster_id, cluster_label, count, modules, avg_sev,
      cluster_velocity_ratio, cluster_trend, recent_count, prior_count,
      recurrence_rate

All v2.4 features retained: embedding cache, fingerprint check, --force,
--no-cache, per-module DBSCAN, Ollama / TF-IDF provider choice.

Usage:
  python cluster_bugs.py <input_csv> <output_csv>
  python cluster_bugs.py <input_csv> <output_csv> --provider ollama
  python cluster_bugs.py <input_csv> <output_csv> --stratify-severity
  python cluster_bugs.py <input_csv> <output_csv> --provider ollama --force
  python cluster_bugs.py <input_csv> <output_csv> --provider ollama --no-cache
"""
import sys, json, hashlib, argparse, urllib.request, time
import pandas as pd, numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE            = "http://localhost:11434"

from ollama_compat import extract_json, response_text
EMBED_CHECKPOINT_EVERY = 50
VELOCITY_WINDOW        = 2    # versions per velocity comparison window
MIN_BUGS_STRATIFIED    = 20   # min bugs per severity tier to run a stratified cluster pass


# ─────────────────────────────────────────────────────────────────────────────
# Input fingerprint helpers  (unchanged from v2.4)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fingerprint(df: pd.DataFrame, embed_model: str = "", label_model: str = "") -> str:
    n = len(df)
    latest = ""
    if "Create Date" in df.columns:
        latest = str(pd.to_datetime(df["Create Date"], errors="coerce").max())
    raw = f"{n}|{latest}|{embed_model}|{label_model}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _fp_file(cluster_dir: Path) -> Path:
    return cluster_dir / "cluster_fingerprint.json"


def _load_fingerprint(cache_path: Path) -> str:
    for path in [cache_path, cache_path.parent / "cluster_fingerprint.json"]:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            fp = data.get("_fingerprint", "")
            if fp:
                return fp
        except Exception:
            pass
    return ""


def _save_fingerprint(cluster_dir: Path, fingerprint: str) -> None:
    fp_path = _fp_file(cluster_dir)
    tmp = fp_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"_fingerprint": fingerprint}, f)
    tmp.replace(fp_path)


def _save_cache(cache_path: Path, embeddings: dict, fingerprint: str) -> None:
    tmp = cache_path.with_suffix(".tmp")
    payload = {"_fingerprint": fingerprint, "embeddings": embeddings}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    tmp.replace(cache_path)


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("embeddings", {})
    except Exception as e:
        print(f"  Warning: could not load embedding cache ({e}) — starting fresh.")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Ollama helpers  (unchanged from v2.4)
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_embed(text: str, model: str = "nomic-embed-text") -> "list[float] | None":
    payload = json.dumps({"model": model, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode()).get("embedding")
    except Exception as e:
        print(f"  [embed] Ollama error: {e}", file=sys.stderr)
        return None


def _ollama_model_available(model: str) -> bool:
    """Return whether the requested model/tag is installed in local Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode())
        requested = model.strip()
        requested_base = requested.split(":", 1)[0]
        for entry in payload.get("models", []):
            installed = str(entry.get("name") or entry.get("model") or "").strip()
            if installed == requested or installed.split(":", 1)[0] == requested_base:
                return True
    except Exception as exc:
        print(f"  [embed] Could not query Ollama models: {exc}", file=sys.stderr)
    return False


def get_ollama_embeddings(
    texts: "list[str]",
    bug_codes: "list[str]",
    model: str = "nomic-embed-text",
    cache: "dict | None" = None,
    cache_path: "Path | None" = None,
    fingerprint: str = "",
    batch_delay: float = 0.1,
) -> "np.ndarray | None":
    if cache is None:
        cache = {}

    new_indices   = [i for i, code in enumerate(bug_codes) if str(code) not in cache]
    cached_count  = len(texts) - len(new_indices)

    if cached_count:
        print(f"  Embedding cache: {cached_count:,} reused, {len(new_indices):,} new bugs to embed.")
    else:
        print(f"  No cache hits — embedding all {len(texts):,} bugs (model={model}).")

    if new_indices:
        print(f"  Sending {len(new_indices):,} new bugs to Ollama...")

    if not new_indices:
        print(f"  All {len(texts):,} bugs served from cache — nothing to embed.")
        # fall through to assemble vectors from cache

    new_done = 0
    for i in tqdm(new_indices, desc="Embedding bugs", unit="bug",
                  file=sys.stderr, dynamic_ncols=True) if new_indices else []:
        code = str(bug_codes[i])
        vec = _ollama_embed(texts[i][:512], model=model)
        if vec is None:
            print(f"  Embedding failed at bug index {i} (BugCode={code}) — falling back to TF-IDF.")
            return None
        cache[code] = vec
        new_done += 1
        if batch_delay and new_done % 20 == 0:
            print(f"    {new_done}/{len(new_indices)} new embeddings done...")
            time.sleep(batch_delay)
        if cache_path and new_done % EMBED_CHECKPOINT_EVERY == 0:
            _save_cache(cache_path, cache, fingerprint)
            print(f"    Cache checkpointed at {new_done} new embeddings.")

    vecs = []
    for i, code in enumerate(bug_codes):
        v = cache.get(str(code))
        if v is None:
            print(f"  BugCode {code} still missing from cache after embedding — aborting Ollama path.")
            return None
        vecs.append(v)

    # Validate uniform embedding dimensions (cache may contain stale vectors from a different model)
    dims = {len(v) for v in vecs}
    if len(dims) > 1:
        print(f"  Inhomogeneous embedding dimensions {dims} — cache is stale (model changed?). Clearing and re-embedding.")
        # Drop all cached entries and re-embed everything
        if cache_path and cache_path.exists():
            cache_path.unlink()
        return None

    arr = np.array(vecs, dtype=np.float32)
    print(f"  Embeddings ready: shape={arr.shape}  (new={len(new_indices)}, cached={cached_count})")
    return arr


def _label_from_json(raw: str) -> str:
    """Read the label out of a {"label": "..."} answer.

    Asking for JSON keeps reasoning models from answering with commentary
    ("Subtitle glitches (but that\'s a bit broad)"), which no amount of line
    picking can clean up.
    """
    try:
        obj = extract_json(raw, expect="object")
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(obj, dict):
        return ""
    for key in ("label", "theme", "title", "name"):
        value = str(obj.get(key) or "").strip().strip('"').strip()
        if value:
            return value
    return ""


def _pick_label_line(raw: str, preamble_prefixes: tuple) -> str:
    """Pick the actual label out of a model answer.

    Reasoning models open with a sentence or two of restatement ("We are given
    three bug descriptions:") before answering, so the label is the last
    label-shaped line, not the first non-preamble one.
    """
    candidates = []
    for line in str(raw or "").splitlines():
        cleaned = line.strip().strip('"\'').strip("*#-•· ").strip()
        if not cleaned or cleaned.lower().startswith(preamble_prefixes):
            continue
        if cleaned.endswith(":") or len(cleaned.split()) > 8:
            continue  # a heading or a sentence, not a label
        candidates.append(cleaned)
    return candidates[-1] if candidates else ""


def _ollama_label_cluster(samples: "list[str]", model: str = "qwen3:4b",
                          retries: int = 3) -> str:
    joined = "\n".join(f"- {s}" for s in samples[:8])
    user_msg = (
        "Given these bug descriptions from the same cluster, name the common "
        "theme in 3-6 words.\n"
        'Return only JSON: {"label": "<3-6 word theme>"}\n'
        "No commentary, no caveats, no punctuation inside the label.\n\n"
        f"Bug descriptions:\n{joined}"
    )
    # Use /api/chat with an explicit system role — some models (e.g. gemma4
    # quantised variants) return empty content when no system message is present.
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a QA analyst. Always respond with a short text label."},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": "10m",
        "options": {"temperature": 0.2, "num_predict": 200},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    preamble_prefixes = (
        "here", "label:", "label ", "sure", "okay", "ok,", "certainly",
        "the label", "a concise", "based on",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                msg = data.get("message", {})
                # Some reasoning-enabled models emit the actual answer in a
                # "thinking" field, or inline it in <think> tags.
                raw = response_text(data)
                if not raw and attempt == 0:
                    print(f"  [label] DEBUG response keys: {list(data.keys())}, "
                          f"message keys: {list(msg.keys())}",
                          file=sys.stderr)
                label = _label_from_json(raw) or _pick_label_line(
                    raw, preamble_prefixes
                )
                if label:
                    return label
                if attempt < retries - 1:
                    print(f"  [label] Empty response on attempt {attempt + 1}/{retries} — retrying …",
                          file=sys.stderr)
                else:
                    snippet = raw.replace("\n", "\\n")[:200]
                    print(f"  [label] All {retries} attempts returned empty/unparseable response — "
                          f"trying /api/generate fallback. Last raw content: {snippet!r}",
                          file=sys.stderr)
        except Exception as e:
            print(f"  [label] Ollama error (attempt {attempt + 1}/{retries}): {e}", file=sys.stderr)
            if attempt == retries - 1:
                print(f"  [label] All {retries} attempts failed — trying /api/generate fallback.",
                      file=sys.stderr)

    # /api/generate fallback — simpler endpoint that works when /api/chat returns empty
    try:
        gen_prompt = (
            f"Bug descriptions:\n{joined}\n\n"
            "Respond with ONLY a 3-6 word label describing the common theme:"
        )
        gen_payload = json.dumps({
            "model": model,
            "prompt": gen_prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.3, "num_predict": 200},
        }).encode()
        gen_req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=gen_payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(gen_req, timeout=60) as resp:
            gdata = json.loads(resp.read().decode())
            fallback_raw = response_text(gdata)
            cleaned = _pick_label_line(fallback_raw, preamble_prefixes)
            if cleaned:
                print(f"  [label] /api/generate fallback succeeded: {cleaned!r}", file=sys.stderr)
                return cleaned
            print(f"  [label] /api/generate fallback also returned empty.", file=sys.stderr)
    except Exception as e:
        print(f"  [label] /api/generate fallback failed: {e}", file=sys.stderr)

    return "unlabelled"


# ─────────────────────────────────────────────────────────────────────────────
# Core clustering  (unchanged from v2.4)
# ─────────────────────────────────────────────────────────────────────────────

def _tfidf_matrix(texts: "list[str]"):
    for kwargs in [
        dict(max_features=3000, stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.8),
        dict(max_features=3000, stop_words="english", ngram_range=(1, 1), min_df=1, max_df=0.95),
    ]:
        try:
            vec = TfidfVectorizer(**kwargs)
            X = vec.fit_transform(texts)
            return X, vec
        except ValueError:
            continue
    return None, None


def cluster_bugs(
    df,
    text_col="parsed_description",
    method="kmeans",
    n_clusters=25,
    eps=0.7,
    min_samples=3,
    provider="tfidf",
    model="qwen3:4b",
    embed_model="nomic-embed-text",
    embed_cache: "dict | None" = None,
    cache_path: "Path | None" = None,
    fingerprint: str = "",
):
    df = df.copy()
    mask = df[text_col].notna() & (df[text_col].str.len() > 5)
    valid_idx = df[mask].index.tolist()

    if len(valid_idx) < 10:
        df["cluster_id"]    = -1
        df["cluster_label"] = "Unclustered"
        df["embed_source"]  = provider
        return df

    texts      = df.loc[valid_idx, text_col].tolist()
    X_dense    = None
    embed_source = "tfidf"
    new_embeds_generated = False

    if provider == "ollama":
        if "BugCode" in df.columns:
            bug_codes = df.loc[valid_idx, "BugCode"].fillna("").astype(str).tolist()
        else:
            bug_codes = [str(i) for i in valid_idx]

        new_indices_count = sum(1 for c in bug_codes if str(c) not in (embed_cache or {}))
        arr = get_ollama_embeddings(
            texts, bug_codes=bug_codes, model=embed_model,
            cache=embed_cache, cache_path=cache_path, fingerprint=fingerprint,
        )
        if arr is not None:
            X_dense      = normalize(arr)
            embed_source = "ollama"
            new_embeds_generated = new_indices_count > 0
        else:
            print("  Ollama embedding failed — falling back to TF-IDF.")

    if X_dense is None:
        X_sparse, vec = _tfidf_matrix(texts)
        if X_sparse is None:
            df["cluster_id"]    = -1
            df["cluster_label"] = "Unclustered"
            df["embed_source"]  = "tfidf"
            return df
        X_dense = X_sparse.toarray()
        vec_ref = vec
    else:
        vec_ref = None

    if method == "dbscan":
        dist   = cosine_distances(X_dense)
        labels = DBSCAN(eps=eps, min_samples=min_samples,
                        metric="precomputed").fit_predict(dist)
    else:
        labels = KMeans(
            n_clusters=min(n_clusters, len(valid_idx)),
            random_state=42, n_init=10,
        ).fit_predict(X_dense)

    unique_ids = sorted(set(labels))
    clabels = {}
    # If we just ran live embeddings (not all from cache), give Ollama a moment
    # to unload the embedding model before switching to the chat model for labelling.
    if embed_source == "ollama" and new_embeds_generated:
        time.sleep(3)
    for cid in unique_ids:
        if cid == -1:
            clabels[-1] = "Noise/Unclustered"
            continue
        cluster_texts = [texts[i] for i, l in enumerate(labels) if l == cid]

        if embed_source == "ollama":
            clabels[cid] = _ollama_label_cluster(cluster_texts[:8], model=model)
        else:
            mask_c = np.array(labels) == cid
            X_sp, vec_inner = _tfidf_matrix(texts)
            tfidf_mean = X_sp[mask_c].mean(axis=0)
            if hasattr(tfidf_mean, "A1"):
                tfidf_mean = tfidf_mean.A1
            else:
                tfidf_mean = np.asarray(tfidf_mean).flatten()
            fnames = vec_ref.get_feature_names_out()
            top = tfidf_mean.argsort()[-3:][::-1]
            clabels[cid] = " | ".join(fnames[i] for i in top)

    df["cluster_id"]    = -1
    df["cluster_label"] = "Unclustered"
    df["embed_source"]  = embed_source
    for i, idx in enumerate(valid_idx):
        df.at[idx, "cluster_id"]    = int(labels[i])
        df.at[idx, "cluster_label"] = clabels.get(int(labels[i]), "Unknown")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# NEW v3.0 — Cluster intelligence functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_cluster_velocity(df: pd.DataFrame,
                              build_col: str = "Build#",
                              cluster_col: str = "cluster_id",
                              window: int = VELOCITY_WINDOW) -> dict:
    """
    For each cluster, compare bug mass in the most recent `window` versions
    against the prior `window` versions. Bug mass uses ``status_weight`` when
    available so closed/fixed bugs contribute less than open bugs — a cluster
    made of ancient fixes shouldn't read as "growing".

    Returns
    -------
    dict: {cluster_id: {"recent_count", "prior_count", "velocity_ratio", "trend"}}
    trend is "growing" (ratio ≥ 1.5), "declining" (ratio ≤ 0.67), or "stable".
    """
    if build_col not in df.columns or cluster_col not in df.columns:
        return {}

    work = df.copy()
    work[build_col] = pd.to_numeric(work[build_col], errors="coerce")
    work = work.dropna(subset=[build_col, cluster_col])
    work = work[work[cluster_col] != -1]
    if work.empty:
        return {}

    if "status_weight" in work.columns:
        work["_w"] = pd.to_numeric(work["status_weight"], errors="coerce").fillna(1.0)
    else:
        work["_w"] = 1.0

    max_build  = work[build_col].max()
    recent_min = max_build - window + 1
    prior_min  = max_build - window * 2 + 1

    result = {}
    for cid in work[cluster_col].unique():
        cdf    = work[work[cluster_col] == cid]
        recent = float(cdf.loc[cdf[build_col] >= recent_min, "_w"].sum())
        prior  = float(cdf.loc[(cdf[build_col] >= prior_min) & (cdf[build_col] < recent_min), "_w"].sum())
        if prior == 0:
            ratio = float("nan")
            trend = "insufficient_history"
        else:
            ratio = recent / prior
            trend = ("growing"   if ratio >= 1.5 else
                     "declining" if ratio <= 0.67 else
                     "stable")
        result[int(cid)] = {
            "recent_count":   round(recent, 2),
            "prior_count":    round(prior, 2),
            "velocity_ratio": round(ratio, 2) if ratio == ratio else float("nan"),
            "trend":          trend,
        }
    return result


def compute_module_cluster_entropy(df: pd.DataFrame,
                                    cluster_col: str = "cluster_id",
                                    mod_col: str = "parsed_module") -> dict:
    """
    Shannon entropy (base-2) of the cluster-assignment distribution per module.

    High entropy → bugs spread across many themes (broad instability).
    Low entropy  → bugs concentrated in one theme (focused failure mode).

    Returns
    -------
    dict: {module_name: entropy_value}
    """
    if cluster_col not in df.columns or mod_col not in df.columns:
        return {}

    clustered = df[(df[cluster_col] != -1)].dropna(subset=[mod_col, cluster_col])
    result = {}
    for mod, grp in clustered.groupby(mod_col):
        counts = grp[cluster_col].value_counts().values.astype(float)
        total  = counts.sum()
        if total == 0:
            continue
        probs = counts / total
        ent   = float(-np.sum(probs * np.log2(probs + 1e-12)))
        result[str(mod)] = round(ent, 3)
    return result


def compute_cluster_recurrence_rate(df: pd.DataFrame,
                                     build_col: str = "Build#",
                                     cluster_col: str = "cluster_id",
                                     mod_col: str = "parsed_module",
                                     window: int = VELOCITY_WINDOW) -> dict:
    """
    For each cluster, compute the fraction of recent bug mass whose source
    module also contributed to that cluster in the prior window. Uses
    ``status_weight`` when available so a module whose only recent bugs are
    closed counts less than a module with fresh open bugs — high recurrence
    should reflect regressions that are still active, not fixed history.

    Returns
    -------
    dict: {cluster_id: recurrence_rate_0_to_1}
    """
    if build_col not in df.columns or cluster_col not in df.columns or mod_col not in df.columns:
        return {}

    work = df.copy()
    work[build_col] = pd.to_numeric(work[build_col], errors="coerce")
    work = work.dropna(subset=[build_col, cluster_col, mod_col])
    work = work[work[cluster_col] != -1]
    if work.empty:
        return {}

    if "status_weight" in work.columns:
        work["_w"] = pd.to_numeric(work["status_weight"], errors="coerce").fillna(1.0)
    else:
        work["_w"] = 1.0

    max_build  = work[build_col].max()
    recent_min = max_build - window + 1
    prior_min  = max_build - window * 2 + 1

    recent_df = work[work[build_col] >= recent_min]
    prior_df  = work[(work[build_col] >= prior_min) & (work[build_col] < recent_min)]

    result = {}
    for cid in work[cluster_col].unique():
        r_grp = recent_df[recent_df[cluster_col] == cid]
        p_grp = prior_df[prior_df[cluster_col]  == cid]
        r_mass = r_grp.groupby(mod_col)["_w"].sum()
        p_mods = set(p_grp[mod_col].unique())
        total = float(r_mass.sum())
        if total <= 0:
            result[int(cid)] = 0.0
        else:
            overlap = float(r_mass.loc[r_mass.index.isin(p_mods)].sum())
            result[int(cid)] = round(overlap / total, 3)
    return result


def cluster_bugs_stratified(df: pd.DataFrame,
                              method: str = "kmeans",
                              n_clusters: int = 15,
                              provider: str = "tfidf",
                              model: str = "qwen3:4b",
                              embed_model: str = "nomic-embed-text",
                              embed_cache: "dict | None" = None,
                              fingerprint: str = "") -> pd.DataFrame:
    """
    Run separate cluster passes for S1/S2 bugs and S3/S4 bugs.

    Adds four columns to the output DataFrame:
      cluster_id_s12     — cluster index within the critical/major tier (-1 if N/A)
      cluster_label_s12  — label for that cluster
      cluster_id_s34     — cluster index within the normal/minor tier (-1 if N/A)
      cluster_label_s34  — label for that cluster

    The global cluster_id / cluster_label columns from the main pass are unchanged.
    """
    df = df.copy()
    df["cluster_id_s12"]    = -1
    df["cluster_id_s34"]    = -1
    df["cluster_label_s12"] = "Unclustered"
    df["cluster_label_s34"] = "Unclustered"

    if "severity_num" not in df.columns:
        print("  NOTE: severity_num not found — skipping stratified clustering.")
        return df

    sev = pd.to_numeric(df["severity_num"], errors="coerce")

    # S1 / S2 tier
    s12_mask = sev.notna() & (sev <= 2)
    s12_df   = df[s12_mask].copy()
    if len(s12_df) >= MIN_BUGS_STRATIFIED:
        k = min(n_clusters, max(2, len(s12_df) // 5))
        print(f"  Stratified S1/S2: clustering {len(s12_df)} bugs into {k} clusters …")
        s12c = cluster_bugs(s12_df, method=method, n_clusters=k,
                            provider=provider, model=model, embed_model=embed_model,
                            embed_cache=embed_cache, cache_path=None,
                            fingerprint=fingerprint)
        df.loc[s12_mask, "cluster_id_s12"]    = s12c["cluster_id"].values
        df.loc[s12_mask, "cluster_label_s12"] = s12c["cluster_label"].values
    else:
        print(f"  Stratified S1/S2: only {len(s12_df)} bugs — skipping (need ≥ {MIN_BUGS_STRATIFIED}).")

    # S3 / S4 tier
    s34_mask = sev.notna() & (sev > 2)
    s34_df   = df[s34_mask].copy()
    if len(s34_df) >= MIN_BUGS_STRATIFIED:
        k = min(n_clusters, max(2, len(s34_df) // 5))
        print(f"  Stratified S3/S4: clustering {len(s34_df)} bugs into {k} clusters …")
        s34c = cluster_bugs(s34_df, method=method, n_clusters=k,
                            provider=provider, model=model, embed_model=embed_model,
                            embed_cache=embed_cache, cache_path=None,
                            fingerprint=fingerprint)
        df.loc[s34_mask, "cluster_id_s34"]    = s34c["cluster_id"].values
        df.loc[s34_mask, "cluster_label_s34"] = s34c["cluster_label"].values
    else:
        print(f"  Stratified S3/S4: only {len(s34_df)} bugs — skipping (need ≥ {MIN_BUGS_STRATIFIED}).")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Summary  (updated for v3.0)
# ─────────────────────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame,
              velocity_map: "dict | None" = None,
              recurrence_map: "dict | None" = None) -> pd.DataFrame:
    """Build a per-cluster summary DataFrame including velocity/trend/recurrence."""
    if "cluster_id" not in df.columns:
        print("No cluster_id column.")
        return pd.DataFrame()
    clustered = df[df["cluster_id"] != -1]
    if len(clustered) == 0:
        print("No clusters found.")
        return pd.DataFrame()

    agg_dict = dict(
        count=("cluster_id", "size"),
        modules=("parsed_module", lambda x: ", ".join(x.dropna().unique()[:5])),
        avg_sev=("severity_num", "mean"),
    )
    cl = (clustered
          .groupby(["cluster_id", "cluster_label"])
          .agg(**agg_dict)
          .sort_values("count", ascending=False)
          .reset_index())

    # Merge velocity columns. Clusters absent from the velocity map (e.g. too
    # few samples to compute) get NaN + "insufficient_history" so downstream
    # code can filter them out, rather than being silently labeled "stable 1.0×".
    if velocity_map:
        cl["cluster_velocity_ratio"] = cl["cluster_id"].map(
            lambda cid: velocity_map.get(int(cid), {}).get("velocity_ratio", float("nan")))
        cl["cluster_trend"] = cl["cluster_id"].map(
            lambda cid: velocity_map.get(int(cid), {}).get("trend", "insufficient_history"))
        cl["recent_count"] = cl["cluster_id"].map(
            lambda cid: velocity_map.get(int(cid), {}).get("recent_count", 0))
        cl["prior_count"] = cl["cluster_id"].map(
            lambda cid: velocity_map.get(int(cid), {}).get("prior_count", 0))
    else:
        cl["cluster_velocity_ratio"] = float("nan")
        cl["cluster_trend"]          = "insufficient_history"
        cl["recent_count"]           = 0
        cl["prior_count"]            = 0

    # Merge recurrence
    if recurrence_map:
        cl["recurrence_rate"] = cl["cluster_id"].map(
            lambda cid: recurrence_map.get(int(cid), 0.0))
    else:
        cl["recurrence_rate"] = 0.0

    n_c  = df["cluster_id"].nunique() - (1 if -1 in df["cluster_id"].values else 0)
    n_in = (df["cluster_id"] != -1).sum()
    src  = df["embed_source"].iloc[0] if ("embed_source" in df.columns and not df.empty) else "tfidf"
    print(f"Clusters: {n_c}, In clusters: {n_in}/{len(df)}, Embed source: {src}")

    display_cols = ["cluster_label", "count", "avg_sev",
                    "cluster_velocity_ratio", "cluster_trend", "recurrence_rate"]
    display_cols = [c for c in display_cols if c in cl.columns]
    print(cl.head(15)[display_cols].to_string(index=False))
    return cl


# ─────────────────────────────────────────────────────────────────────────────
# Relabel-only helper
# ─────────────────────────────────────────────────────────────────────────────

def relabel_clusters(
    df: pd.DataFrame,
    model: str,
    text_col: str = "parsed_description",
) -> pd.DataFrame:
    """Re-run only the Ollama label step on an already-clustered DataFrame.

    Cluster IDs and embeddings are left untouched — only cluster_label is updated.
    Falls back to TF-IDF top-keywords if Ollama returns unlabelled after retries.
    """
    df = df.copy()
    if "cluster_id" not in df.columns:
        print("  relabel: no cluster_id column — nothing to relabel.", file=sys.stderr)
        return df

    unique_ids = sorted(
        c for c in df["cluster_id"].unique() if c != -1
    )
    if not unique_ids:
        print("  relabel: no non-noise clusters found.", file=sys.stderr)
        return df

    print(f"  Re-labelling {len(unique_ids)} clusters using model={model} …", file=sys.stderr)
    for cid in unique_ids:
        mask  = df["cluster_id"] == cid
        texts = (
            df.loc[mask & df[text_col].notna(), text_col]
            .tolist()[:8]
        )
        if not texts:
            continue
        label = _ollama_label_cluster(texts, model=model)
        if label == "unlabelled":
            # TF-IDF keyword fallback
            try:
                cluster_texts = df.loc[mask & df[text_col].notna(), text_col].tolist()
                if len(cluster_texts) >= 2:
                    X_sp, vec_inner = _tfidf_matrix(cluster_texts)
                    if X_sp is not None:
                        mean_vec = X_sp.mean(axis=0)
                        mean_vec = mean_vec.A1 if hasattr(mean_vec, "A1") else np.asarray(mean_vec).flatten()
                        fnames = vec_inner.get_feature_names_out()
                        top = mean_vec.argsort()[-3:][::-1]
                        label = " | ".join(fnames[i] for i in top)
            except Exception:
                pass
        df.loc[mask, "cluster_label"] = label
        print(f"    Cluster {int(cid):>3}: {label}", file=sys.stderr)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PDR-I Defect Clustering v3.0")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--provider", choices=["tfidf", "ollama"], default="tfidf")
    parser.add_argument("--model", default="qwen3:4b",
                        help="Ollama model for cluster label generation (DEPRECATED: use --label-model)")
    parser.add_argument("--label-model", default="qwen3:4b",
                        help="Ollama model for cluster label generation")
    parser.add_argument("--embed-model", default="nomic-embed-text",
                        help="Ollama model for bug embeddings (must support /api/embeddings)")
    parser.add_argument("--force",    action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--relabel", action="store_true",
        help="Re-run only the Ollama label step on the existing clustered CSV (no re-embedding)"
    )
    parser.add_argument(
        "--stratify-severity", action="store_true",
        help="Also run separate cluster passes for S1/S2 and S3/S4 bugs"
    )
    args = parser.parse_args()

    inp = Path(args.input_csv)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    cluster_dir = inp.parent / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cluster_dir / "embedding_cache.json"

    # ── Load input ────────────────────────────────────────────────────────────
    df = pd.read_csv(str(inp))
    label_model = args.label_model
    print(f"Loaded {len(df):,} bugs | provider={args.provider} embed-model={args.embed_model} label-model={label_model}"
          + (" | stratify-severity=ON" if args.stratify_severity else ""))

    # Fail over once, before entering per-module embedding loops. Without this
    # check a missing model emits repeated HTTP 404s and misleadingly looks as
    # though Ollama clustering succeeded. TF-IDF remains a supported fallback.
    if args.provider == "ollama" and not _ollama_model_available(args.embed_model):
        print(
            f"\n  ⚠️  Ollama embedding model '{args.embed_model}' is not installed.\n"
            f"  Install it with: ollama pull {args.embed_model}\n"
            "  Falling back to TF-IDF clustering for this run.\n",
            file=sys.stderr,
        )
        args.provider = "tfidf"

    # ── Relabel-only mode ─────────────────────────────────────────────────────
    if args.relabel:
        if not out.exists():
            print(f"  --relabel: output file not found ({out}) — run a full cluster first.")
            sys.exit(1)
        existing = pd.read_csv(str(out))
        existing = relabel_clusters(existing, model=label_model)
        # Recompute summary with new labels
        velocity_map   = compute_cluster_velocity(existing)
        recurrence_map = compute_cluster_recurrence_rate(existing)
        summary = summarize(existing, velocity_map=velocity_map, recurrence_map=recurrence_map)
        if not summary.empty:
            summary_path = cluster_dir / (inp.stem + "_cluster_summary.csv")
            summary.to_csv(str(summary_path), encoding="utf-8-sig", index=False)
            print(f"  Cluster summary → {summary_path}")
        existing.to_csv(str(out), index=False, encoding="utf-8-sig")
        print(f"\nSaved to: {out}")
        return

    # ── Fingerprint check ─────────────────────────────────────────────────────
    embed_model_for_fp = args.embed_model if args.provider == "ollama" else ""
    fingerprint = _compute_fingerprint(df, embed_model=embed_model_for_fp,
                                       label_model=label_model if args.provider == "ollama" else "")

    if not args.force and not args.no_cache:
        cached_fp = _load_fingerprint(cache_path)
        if cached_fp == fingerprint and out.exists():
            # Validate: if the existing output has all-unlabelled labels, auto-relabel
            try:
                existing = pd.read_csv(str(out))
                if "cluster_label" in existing.columns:
                    unlabelled_pct = (existing["cluster_label"] == "unlabelled").mean()
                    if unlabelled_pct > 0.9 and args.provider == "ollama":
                        print(
                            f"\n  ⚠️  Cache hit but {unlabelled_pct:.0%} of labels are 'unlabelled'."
                            f"  Auto-running relabel step (no re-embedding needed)."
                        )
                        existing = relabel_clusters(existing, model=label_model)
                        velocity_map   = compute_cluster_velocity(existing)
                        recurrence_map = compute_cluster_recurrence_rate(existing)
                        summary = summarize(existing, velocity_map=velocity_map,
                                            recurrence_map=recurrence_map)
                        if not summary.empty:
                            summary_path = cluster_dir / (inp.stem + "_cluster_summary.csv")
                            summary.to_csv(str(summary_path), encoding="utf-8-sig", index=False)
                            print(f"  Cluster summary → {summary_path}")
                        existing.to_csv(str(out), index=False, encoding="utf-8-sig")
                        print(f"\nSaved to: {out}")
                        return
            except Exception as e:
                print(f"  Warning: could not validate existing output ({e}).")
            print(
                f"\n  ✅ Cache hit — input fingerprint unchanged ({fingerprint}).\n"
                f"  Provider={args.provider}  Output already exists → skipping re-cluster.\n"
                f"  Use --force to re-cluster regardless.\n"
                f"  Output: {out}"
            )
            return
        elif cached_fp:
            print(f"  Fingerprint changed: {cached_fp!r} → {fingerprint!r} — re-clustering.")
        else:
            print(f"  No cached fingerprint — running full cluster (provider={args.provider}).")

    if args.force:
        print("  --force: skipping fingerprint check, running full re-cluster.")

    # ── Load embedding cache (Ollama mode only) ───────────────────────────────
    embed_cache: dict = {}
    if args.provider == "ollama" and not args.no_cache:
        embed_cache = _load_cache(cache_path)
        if embed_cache:
            if "BugCode" in df.columns:
                current_codes = set(df["BugCode"].dropna().astype(str))
                cached_codes  = set(embed_cache.keys()) - {"_fingerprint"}
                stale = cached_codes - current_codes
                if stale:
                    print(f"  Note: {len(stale):,} BugCodes in cache absent from input — "
                          f"retained but won't affect output.")
            print(f"  Embedding cache loaded: {len(embed_cache):,} existing vectors.")
        else:
            print("  No embedding cache found — all bugs will be embedded from scratch.")

    # ── Global clustering ─────────────────────────────────────────────────────
    df = cluster_bugs(
        df, method="kmeans", n_clusters=25,
        provider=args.provider, model=label_model, embed_model=args.embed_model,
        embed_cache=embed_cache,
        cache_path=cache_path if not args.no_cache else None,
        fingerprint=fingerprint,
    )

    # ── Severity-stratified clustering (optional) ─────────────────────────────
    if args.stratify_severity:
        print("\nRunning severity-stratified clustering …")
        df = cluster_bugs_stratified(
            df, method="kmeans", n_clusters=15,
            provider=args.provider, model=label_model, embed_model=args.embed_model,
            embed_cache=embed_cache, fingerprint=fingerprint,
        )

    # ── Save final embedding cache ────────────────────────────────────────────
    if args.provider == "ollama" and not args.no_cache:
        _save_cache(cache_path, embed_cache, fingerprint)
        print(f"  Embedding cache saved: {len(embed_cache):,} vectors → {cache_path}")
    elif args.provider == "tfidf" and not args.no_cache:
        _save_fingerprint(cluster_dir, fingerprint)
        print(f"  Fingerprint saved → {_fp_file(cluster_dir)}")

    # ── Compute cluster intelligence ──────────────────────────────────────────
    print("\nComputing cluster velocity, recurrence, and module entropy …")
    velocity_map   = compute_cluster_velocity(df)
    recurrence_map = compute_cluster_recurrence_rate(df)
    entropy_map    = compute_module_cluster_entropy(df)

    # ── Global summary ────────────────────────────────────────────────────────
    summary = summarize(df, velocity_map=velocity_map, recurrence_map=recurrence_map)
    if not summary.empty:
        summary_path = cluster_dir / (inp.stem + "_cluster_summary.csv")
        summary.to_csv(str(summary_path), encoding="utf-8-sig", index=False)
        print(f"  Cluster summary → {summary_path}")

    # ── Module entropy CSV ────────────────────────────────────────────────────
    if entropy_map:
        ent_df = pd.DataFrame([
            {"module": mod, "cluster_entropy": ent}
            for mod, ent in sorted(entropy_map.items(), key=lambda x: -x[1])
        ])
        ent_path = cluster_dir / (inp.stem + "_module_entropy.csv")
        ent_df.to_csv(str(ent_path), encoding="utf-8-sig", index=False)
        print(f"  Module entropy   → {ent_path}  ({len(ent_df)} modules)")

    # ── Stratified summaries ──────────────────────────────────────────────────
    if args.stratify_severity:
        for tier, id_col, label_col in [
            ("s12", "cluster_id_s12", "cluster_label_s12"),
            ("s34", "cluster_id_s34", "cluster_label_s34"),
        ]:
            tier_df = df[df[id_col] != -1].copy()
            if tier_df.empty:
                continue
            # Build a temporary cluster_id/cluster_label view for summarize()
            tier_df["cluster_id"]    = tier_df[id_col]
            tier_df["cluster_label"] = tier_df[label_col]
            tier_vel  = compute_cluster_velocity(tier_df)
            tier_rec  = compute_cluster_recurrence_rate(tier_df)
            tier_sum  = summarize(tier_df, velocity_map=tier_vel, recurrence_map=tier_rec)
            if not tier_sum.empty:
                tier_path = cluster_dir / f"{inp.stem}_cluster_summary_{tier}.csv"
                tier_sum.to_csv(str(tier_path), encoding="utf-8-sig", index=False)
                print(f"  Stratified {tier} summary → {tier_path}")

    # ── Per-module DBSCAN clustering (top 5 modules) ──────────────────────────
    if "parsed_module" in df.columns:
        for mod in df["parsed_module"].value_counts().head(5).index.tolist():
            mdf = df[df["parsed_module"] == mod].copy()
            if len(mdf) >= 20:
                print(f"\nClustering within module: {mod} ({len(mdf):,} bugs)")
                mdf = cluster_bugs(
                    mdf, method="dbscan", eps=0.6, min_samples=2,
                    provider=args.provider, model=args.model,
                    embed_cache=embed_cache, cache_path=None,
                    fingerprint=fingerprint,
                )
                summarize(mdf)

    # ── Write output ──────────────────────────────────────────────────────────
    df.to_csv(str(out), index=False, encoding="utf-8-sig")
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
