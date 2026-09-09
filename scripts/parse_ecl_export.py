#!/usr/bin/env python3
"""PDR-I ECL Export Parser v2.4

Changes from v2.3:

  - JSON input support (n8n webhook integration):
      * Accepts .json files produced by fetch_from_n8n.py in addition to
        .xlsx / .csv files. The JSON must be a list of bug objects, each using
        the exact field names returned by the n8n "Get Columns_v3" Set node:
            Short Description, Severity, Priority, Status,
            Create Date, Closed Date, Version, Build#, Close Build#,
            BugCode, Creator, BugBelong, Handler
      * Column name normalisation on load: maps n8n output names to the
        internal names the rest of the pipeline expects (e.g. "Create Date",
        "Closed Date", "Build#", "Close Build#").  This means the JSON from
        the webhook can be fed in directly without any manual renaming.
      * Reproduce Probability is absent from the API — repro_rate defaults to
        0.5 for all JSON-sourced bugs (same as the "no repro column" fallback).

Changes from v2.2:

  - Issue 2:
      * Adds `status_active` boolean column.
      * NAB, Won't Fix, Not Reproducible, New Feature, etc. are still INCLUDED
        but flagged as inactive. Nothing is dropped; risk code can downweight or filter.

  - Issue 5:
      * Fuzzy matching (rapidfuzz) for module normalization.
      * Per-version mapping files under data/module_mappings/:
          permanent/mappings_global.json
          versions/<version>_pending.json
          versions/<version>_confirmed.json
        New / low-confidence strings go into <ver>_pending.json with suggested match.

  - Issue 8:
      * Adds `priority_num` and human-readable `priority_label` columns:
          1 = Fix Now, 2 = Must Fix, 3 = Better Fix, 4 = No Matter, 5 = N/A.

Usage (Excel / CSV — original):
  python scripts/parse_ecl_export.py data/ecl_export.xlsx data/ecl_parsed.csv

Usage (JSON from n8n webhook — new):
  python scripts/fetch_from_n8n.py --output data/ecl_raw.json
  python scripts/parse_ecl_export.py data/ecl_raw.json data/ecl_parsed.csv

  Or in one step:
  python scripts/fetch_from_n8n.py --then-parse
"""
import hashlib
import os
import re
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Optional

try:
    from rapidfuzz import process, fuzz

    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("WARNING: rapidfuzz not installed. Run: pip install rapidfuzz")
    print("         Falling back to exact alias matching only.")

# Ollama semantic matcher — re-ranks rapidfuzz top-K candidates with an LLM.
# Enabled via env var PDRI_OLLAMA_MATCHER=1 or CLI --ollama-matcher.
OLLAMA_MATCHER_URL = "http://localhost:11434/api/generate"

from ollama_compat import extract_json, response_text, with_thinking_disabled
OLLAMA_MATCHER_DEFAULT_MODEL = "qwen3:4b"
OLLAMA_MATCHER_TIMEOUT = 30
_OLLAMA_MATCHER_MEM_CACHE: dict[str, dict] = {}

# Ollama embedding-based candidate retrieval — replaces rapidfuzz as the
# primary source of top-K candidates fed to the LLM disambiguator.
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_EMBED_DEFAULT_MODEL = "nomic-embed-text"


class _CanonicalEmbedder:
    """Lazy singleton: embeds all canonical modules, supports top-K cosine retrieval.

    Initialized once per pipeline run. Falls back gracefully when Ollama is down
    (``ready`` stays False, callers fall through to rapidfuzz).
    """

    _instance: "_CanonicalEmbedder | None" = None

    def __init__(self):
        self._matrix: "np.ndarray | None" = None
        self._names: list[str] = []
        self._ready = False
        self._model = (
            os.environ.get("PDRI_EMBEDDING_MODEL", "").strip()
            or OLLAMA_EMBED_DEFAULT_MODEL
        )
        self._cache_dir: "Path | None" = None

    @classmethod
    def get(cls) -> "_CanonicalEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def ready(self) -> bool:
        return self._ready

    def initialize(self, cache_dir: "Path | None" = None):
        if self._ready:
            return

        texts, names = [], []
        for m in sorted(set(_CANONICAL_MODULES)):
            cat = _MODULE_TO_CATEGORY.get(m.lower(), "")
            texts.append(f"{m} ({cat})" if cat else m)
            names.append(m)

        fingerprint = hashlib.sha1(json.dumps(names).encode()).hexdigest()[:16]

        cache_path: "Path | None" = None
        if cache_dir:
            self._cache_dir = cache_dir
            cache_path = cache_dir / "canonical_embeddings.json"
            if cache_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    if (
                        cached.get("fingerprint") == fingerprint
                        and cached.get("model") == self._model
                        and len(cached.get("vectors", [])) == len(names)
                    ):
                        self._matrix = np.array(cached["vectors"], dtype=np.float32)
                        self._names = cached["names"]
                        self._ready = True
                        print(
                            f"  [embed-matcher] Loaded {len(names)} canonical embeddings from cache.",
                            file=sys.stderr,
                        )
                        return
                except Exception:
                    pass

        print(
            f"  [embed-matcher] Building {len(names)} canonical embeddings ({self._model})...",
            file=sys.stderr,
        )
        vectors = []
        for text in texts:
            vec = self._embed(text)
            if vec is None:
                print(
                    "  [embed-matcher] Ollama unreachable — falling back to rapidfuzz.",
                    file=sys.stderr,
                )
                return
            vectors.append(vec)

        self._matrix = np.array(vectors, dtype=np.float32)
        self._names = names
        self._ready = True

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fingerprint": fingerprint,
                        "model": self._model,
                        "dim": int(self._matrix.shape[1]),
                        "names": names,
                        "vectors": self._matrix.tolist(),
                    },
                    f,
                )
            print(
                f"  [embed-matcher] Cached {len(names)} embeddings to {cache_path.name}.",
                file=sys.stderr,
            )

    def _embed(self, text: str) -> "list[float] | None":
        payload = json.dumps({"model": self._model, "prompt": text}).encode()
        req = urllib.request.Request(
            OLLAMA_EMBED_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode()).get("embedding")
        except Exception:
            return None

    def top_k(self, raw: str, k: int = 8) -> "list[tuple[str, float]]":
        if not self._ready:
            return []
        vec = self._embed(raw)
        if vec is None:
            return []
        from sklearn.metrics.pairwise import cosine_similarity

        query = np.array([vec], dtype=np.float32)
        sims = cosine_similarity(query, self._matrix)[0]
        top_indices = np.argsort(sims)[::-1][:k]
        return [(self._names[i], float(sims[i])) for i in top_indices]


# ---------------------------------------------------------------------
# Issue 2 – Status classification
# ---------------------------------------------------------------------

# Bugs that were real but are now resolved. They count in predictions at half
# weight — they happened and cost engineering time, but the fix reduces ongoing risk.
CLOSED_STATUSES = {
    "close",
    "hqqa close",
    "fae close",
}

# Bugs that were never confirmed as real defects. Excluded from predictions entirely.
INVALID_STATUSES = {
    "need more info",
    "nab",
    "propose nab",
    "wont fix",
    "won't fix",
    "propose wont fix",
    "qa propose wont fix",
    "not reproducible",
    "notreproducible",
    "not a bug",
    "new feature",
    "external issue",
}

# Combined for backward-compat callers that only need active vs. inactive
INACTIVE_STATUSES = CLOSED_STATUSES | INVALID_STATUSES


def classify_status(status_val) -> bool:
    """True if active/open, False if resolved/invalidated."""
    if pd.isna(status_val):
        return True
    s = str(status_val).strip().lower()
    return s not in INACTIVE_STATUSES


def classify_status_weight(status_val) -> float:
    """
    Prediction weight for a bug based on its status.
      1.0 — open/active: full signal
      0.5 — closed/fixed: real bug, but resolved; counts at half weight
      0.0 — invalid (NAB, Won't Fix, etc.): excluded from predictions
    """
    if pd.isna(status_val):
        return 1.0
    s = str(status_val).strip().lower()
    if s in INVALID_STATUSES:
        return 0.0
    if s in CLOSED_STATUSES:
        return 0.5
    return 1.0


# Time-decay knobs for closed bugs. A fix is still informative the build after
# it lands (regression candidate), but its signal weakens as more clean builds
# pass. Decay is linear from INITIAL down to FLOOR over DECAY_SPAN builds; it
# never drops below FLOOR so long-settled modules still carry a faint echo of
# their bug history instead of looking pristine.
FIXED_BUG_INITIAL = 0.5
FIXED_BUG_FLOOR = 0.1
FIXED_BUG_DECAY_SPAN = 12


def decayed_fixed_weight(status_val, close_build, at_build) -> float:
    """
    Status weight for a bug evaluated as of build `at_build`.

    Invalid bugs return 0.0. Bugs not yet closed at `at_build` return 1.0
    (they were still open then). Closed bugs decay linearly from FIXED_BUG_INITIAL
    to FIXED_BUG_FLOOR over FIXED_BUG_DECAY_SPAN builds since close.

    Falls back to the static classify_status_weight when close_build or
    at_build is missing / unparseable.
    """
    if pd.isna(status_val):
        return 1.0
    s = str(status_val).strip().lower()
    if s in INVALID_STATUSES:
        return 0.0
    if s not in CLOSED_STATUSES:
        return 1.0
    try:
        cb = float(close_build)
        ab = float(at_build)
    except (TypeError, ValueError):
        return FIXED_BUG_INITIAL
    if pd.isna(cb) or pd.isna(ab):
        return FIXED_BUG_INITIAL
    if cb > ab:
        return 1.0
    age = ab - cb
    span = max(1, FIXED_BUG_DECAY_SPAN)
    decayed = FIXED_BUG_INITIAL - (FIXED_BUG_INITIAL - FIXED_BUG_FLOOR) * (age / span)
    return max(FIXED_BUG_FLOOR, decayed)


# ---------------------------------------------------------------------
# Aliases / Categories (truncated to what's needed for behaviour change)
# ---------------------------------------------------------------------

MODULE_ALIASES = {
    "HQ Auido Denoise": "HQ Audio Denoise",
    "Boay Effects": "Body Effects",
    "Speech Enhacner": "Speech Enhancer",
    "AI Anime Vide": "AI Anime Video",
    "AI Anime Vidoeo": "AI Anime Video",
    "AI Voice Change": "AI Voice Changer",
    "Auto Caption": "Auto Captions",
    "Facial Mosiac": "Facial Mosaic",
    "Voice-over": "Voice-Over",
    "voice-Over": "Voice-Over",
    "Voiceover": "Voice-Over",
    "Voice Over": "Voice-Over",
    "AI Storyteller": "AI Storytelling",
    "AI Storytelljng": "AI Storytelling",
    "Colour Board": "Color Board",
    "Crop&Rotate": "Crop & Rotate",
    "Crop(Rotate)": "Crop & Rotate",
    "Pan&Zoom": "Pan & Zoom",
    "Beat Marker": "Beats Marker",
    "Beats Maker": "Beats Marker",
    "Key Frame": "Transform Keyframe",
    "Talkling Avatar": "Talking Avatar",
    "TTS": "Text to Speech",
    "slpash": "Splash",
    "Shortuct": "Shortcut",
    "ShortCut": "Shortcut",
    "Skin Smoothness": "Skin Smoother",
    "Face smoother": "Skin Smoother",
    "Face Smoother": "Skin Smoother",
    "Msak": "Mask",
    "Transtion": "Transition",
    "Teansition": "Transition",
    "Clooud Storage": "Cloud Storage",
    "Cloud Stroage": "Cloud Storage",
    "sign in your account": "Sign In",
    "Sing in": "Sign In",
    "ripple editing": "Ripple Editing",
    "Add media": "Add Media",
    "MixPanel": "Mixpanel",
    "Mixpanel log": "Mixpanel",
    "Body Reshape": "Body Effects",
    "AI body Effect": "Body Effects",
    "AI Upscaler": "AI Video Upscaler",
    "AI FX": "AI Effect",
    "Layer Effect": "Effect Layer",
    "Effect layer": "Effect Layer",
    "FX Layer": "Effect Layer",
    "Fx Layer": "Effect Layer",
    "AI Cartoon (Cine Anime)": "AI Cartoon",
    "AI Cartoon (Crop)": "AI Cartoon",
    "Crop range panel": "Crop & Rotate",
}

MODULE_ALIASES.update(
    {
        "AI Video Enahncer": "Video Enhancer",
        "AI Video Enhacner": "Video Enhancer",
        "AI Video Upsacler": "AI Video Upscaler",
        "Video Enhacer(Compare)": "Video Enhancer",
        "Imageo to Video(Cash flow)": "Image to Video",
        "APNG Text": "Text",
        "APNG(style)": "Text",
        "Text(APNG)": "Text",
        "Title, Text, APNG": "Title, Text",
        "AI Music Creation": "AI Music Generator",
        "AI Music Generate": "AI Music Generator",
        "AI Music Genrator": "AI Music Generator",
        "AI Music Gnerator": "AI Music Generator",
        "AI Storytelljng(ENG cheerful Mia)": "AI Storytelling",
        "Shortcut(AI Anime Vidoeo)": "Shortcut(AI Anime Video)",
        "Shortcut(AI Video Anime)": "Shortcut(AI Anime Video)",
        "Clooud Storage(Upload)": "Cloud Storage(Upload)",
        "Cloud Stroage(Previe)": "Cloud Storage(Preview)",
        "Openging Intro": "Open intro",
        "New Open intro, Dark or Light mode": "Opening Intro",
        "Tex, Title": "Text, Title",
        "Text (Colorful": "Text (Colorful)",
        "Text (Gold": "Text (Gold_11~18)",
        "- MGT(Sparkle)": "MGT(Sparkle)",
    }
)

# Only 4 aliases are hardcoded — ones that automatic rules in normalize_module
# cannot handle: leading/trailing punctuation and pure acronyms.
# Everything else (sub-variants, comma lists, > separators, typos, locale/device
# suffixes) is resolved automatically by normalize_module's strip + split logic,
# or falls through to fuzzy matching -> pending.json for one-time confirmation.
MODULE_ALIASES.update({
    "]Launcher":    "Launcher",        # leading bracket corrupts name
    "IAP)":         "IAP",             # trailing bracket, nothing before it
    "IAP/B Group)": "IAP",             # mixed punctuation
    "STT":          "Text to Speech",  # pure acronym, no word overlap
})

# PHD-specific aliases: concatenated AI feature names (no spaces/underscores)
MODULE_ALIASES.update({
    "AIExpand":           "AI Expand",
    "AIOutfit":           "AI Outfit",
    "AIHeadshot":         "AI Headshot",
    "AIRelight":          "AI Relight",
    "AIHairstyle":        "AI Hairstyle",
    "AI face swap":       "AI Face Swap",
    "AI hairstyle":       "AI Hairstyle",
    "AI expand":          "AI Expand",
    "AI relight":         "AI Relight",
    "AI outfit":          "AI Outfit",
    "AI headshot":        "AI Headshot",
    "AI try on":          "AI Try On",
    "AI Try on":          "AI Try On",
    "AI creative studio": "AI Creative Studio",
    "AI Creative studio": "AI Creative Studio",
    "AI magic studio":    "AI Magic Studio",
    "AI Magic studio":    "AI Magic Studio",
    "AI anime video":     "AI Anime Video",
    "AI srt-custom":      "Auto Captions",
    "AI srt":             "Auto Captions",
    "Artistic avatar":    "Artistic Avatar",
    "My AI Artwork":      "My AI Artwork",
    "My AI artwork":      "My AI Artwork",
    "My artwork":         "My Artwork",
    "My Artwork":         "My Artwork",
    "Quick action":       "Quick Actions",
    "Quick Action":       "Quick Actions",
    "Quick Actions":      "Quick Actions",
    "Photo picker":       "Photo Picker",
    "Photo Picker":       "Photo Picker",
    "Add image":          "Add Image",
    "Add-image":          "Add Image",
    "Edit Room":          "Edit Room",
    "Draw to image":      "Draw to Image",
    "Draw to Image":      "Draw to Image",
    "ID photo":           "ID Photo",
    "ID Photo":           "ID Photo",
    "Image fusion":       "Image Fusion",
    "Image Fusion":       "Image Fusion",
    "Removal YOLO":       "Removal",
    "Gen AI":             "GAI",
    "Opening intro":      "Opening Intro",
})

# Canonical forms for modules that appear in data under multiple casings but
# aren't listed in MODULE_CATEGORIES. Each entry seeds the canonical set (via
# MODULE_ALIASES.values()) so step 2.5 case-fold in normalize_module collapses
# all case variants. Convention: Title Case with acronyms in caps (FX, AI).
MODULE_ALIASES.update({
    "Credit system":        "Credit System",
    "Tool menu":            "Tool Menu",
    "Media picker":         "Media Picker",
    "Video Effect layer":   "Video Effect Layer",
    "Filter layer":         "Filter Layer",
    "AI music":             "AI Music",
    "Video Fx":             "Video FX",
    "Trim before Edit":     "Trim Before Edit",
    "Trim before edit":     "Trim Before Edit",
    "Main tool menu":       "Main Tool Menu",
    "Sample project":       "Sample Project",
    "Media room":           "Media Room",
    "User Sign in":         "User Sign In",
    "AIsticker":            "AISticker",
    "shutterstock":         "Shutterstock",
    "Demo video":           "Demo Video",
    "File picker":          "File Picker",
    "Intro template":       "Intro Template",
    "Music/Sound Fx":       "Music/Sound FX",
    "Advanced cutout":      "Advanced Cutout",
    "Recent task":          "Recent Task",
    "New tag":              "New Tag",
    "Fx":                   "FX",
    "Color filter":         "Color Filter",
    "AI Audio tool":        "AI Audio Tool",
    # Typos & variants from ECL data
    "Launcer":              "Launcher",
    "Luancher":             "Launcher",
    "Launcher - Shortcut":  "Launcher Shortcut",
    "Launcher - Top Banner": "Launcher",
    "Shortcit":             "Shortcut",
    "Shorcut":              "Shortcut",
    "Shortut":              "Shortcut",
    "Sortcut":              "Shortcut",
    "Shortcuts":            "Shortcut",
    "Shortcut - Manage":    "Shortcut",
    "Speech to Tex":        "Speech to Text",
    "Summner Promotion":    "Summer Promotion",
    "OBON Prmotion":        "OBON Promotion",
    "Full Screen Preivew":  "Full Screen Preview",
    "ChromaKey":            "Chroma Key",
    "AISticker":            "AI Sticker",
    "AIColor":              "AI Color",
    "Add Medi":             "Add Media",
    "Media library":        "Media",
    "Fit & Fill":           "Fill",
    "My Project":           "My Projects",
    "Tempo Effects":        "Tempo Effect",
    "TalkingAvatar":        "Talking Avatar",
    "Social Sign-In":       "Social Sign In",
    "Log In page":          "Sign In",
    "Sign in dialog":       "Sign In",
    "Churn recover":        "Churn Recovery",
    "Open intro":           "Opening Intro",
    "Settings Transition":  "Settings",
    "Maxpanel":             "Mixpanel",
    "Sticker 's Series":    "Sticker",
    "Mixtape":              "Mixpanel",
    "Tools menu":           "Tool Menu",
    "Main Toolbar":         "Main Tool Menu",
    "Intro Designer":       "Intro/Outro",
    "Title Designer":       "Title",
    "Edit Toolbar":         "Editor",
    "Halloween Credit System": "Credit System",
    "Halloween Credits":    "Credit System",
    "Halloween Credit Page": "Credit System",
    "Halloween Credit Purchase": "Credit System",
    "VIP Benefits":         "VIP Benefit page",
    "VIP Benefit":          "VIP Benefit page",
    "iStock Premium":       "iStock",
    "iStock Pro":           "iStock",
    "Face Beautify":        "Beautify",
    "APP Store":            "More",
    "Download Data":        "Export",
    "Expand":               "AI Expand",
    "Promote":              "Cross Promote",
    "Deeplink":              "Launcher",
    "Result page":          "Export",
    # Edge-case typos/abbreviations
    "Speech to text":       "Speech to Text",
    "Shortcut - AI Anime":  "Shortcut",
    "Transition - Portrait": "Transition",
    "Permission Produce":   "Permission",
    "Opening":              "Opening Intro",
    "Settings Preferences Delete Date": "Preferences",
})

# Lowercase shadow for O(1) alias lookup — rebuilt after all updates
_MODULE_ALIASES_LOWER = {k.lower(): v for k, v in MODULE_ALIASES.items()}

MODULE_CATEGORIES = {
    "AI Features": [
        "AI Storytelling",
        "AI Art",
        "AI Smart Tools",
        "Image to Video",
        "Character Motion Swap",
        "Text to Video",
        "AI Anime Video",
        "AI Magic",
        "Talking Avatar",
        "AI Images",
        "Text to Image",
        "AI Style Transfer",
        "AI Sketch",
        "AI Cartoon",
        "AI Scene",
        "AI Effect",
        "AI Creation",
        "GAI",
        "AI Scene/Art",
        "Voice Cloning",
        "My Voices",
        "TTI Inspiration",
        "AI Face Swap",
        "AI Expand",
        "AI Outfit",
        "AI Headshot",
        "AI Relight",
        "AI Hairstyle",
        "AI Try On",
        "AI Creative Studio",
        "AI Magic Studio",
        "Artistic Avatar",
        "AI Avatar",
        "Draw to Image",
        "Image Fusion",
        "AI Sticker",
        "AI Video Enhancer",
        "AI Audio",
        "AI Audio Denoise",
        "AI Color",
        "TalkingAvatar",
        "TTI",
    ],
    "Editor Core": [
        "Project",
        "Auto Edit",
        "Auto Edit Project",
        "Recent Projects",
        "New Project",
        "Split",
        "Duplicate",
        "Trim",
        "Adjust Speed",
        "Speed",
        "Standard Speed",
        "Speed Curve",
        "Super Slow Motion",
        "Transform",
        "Crop & Rotate",
        "Pan & Zoom",
        "Fill",
        "Fit",
        "Flip",
        "Transform Keyframe",
        "Reduce Track Height",
        "Aspect Ratio",
        "Reverse",
        "Freeze Frame",
        "Range",
        "Replace",
        "Timeline",
        "Editor",
        "Edit Menu",
        "NewTimeline",
        "Ripple Editing",
        "PIP",
        "PiP",
        "Drag and Drop",
        "Edit Room",
        "Canvas",
        "Crop",
        "Toolbar",
        "Main Menu",
        "Main Tool Menu",
        "Tool Menu",
        "Main Toolbar",
        "Landscape",
    ],
    "Audio": [
        "Audio",
        "Music",
        "AI Music Generator",
        "Voice-Over",
        "Sound FX",
        "Extract Audio",
        "Text to Speech",
        "Speech to Text",
        "HQ Audio Denoise",
        "Denoise",
        "Speech Enhancer",
        "AI Voice Changer",
        "Audio Ducking",
        "Audio Mixing",
        "Beats Marker",
        "Volume",
        "Audio Tool",
        "Meta Sound",
        "My Voice",
        "AI Audio Tool",
        "AI Music",
        "BGM",
        "Creator Music",
        "Search Music",
        "Audio beats Marker",
        "Timeline audio",
    ],
    "Text & Captions": [
        "Text",
        "Add Text",
        "Auto Captions",
        "Edit Text",
        "Captions List",
        "Bilingual Subtitles",
        "Animation",
        "Template",
        "Style",
        "Font",
        "Backdrop",
        "MGT",
        "Title",
        "Auto Caption",
        "Intro/Outro",
        "Title Animation",
        "Text Animation",
        "Neon title",
    ],
    "Visual Effects": [
        "Effects",
        "Filter",
        "Video Effects",
        "Effect Overlay",
        "Body Effects",
        "Highlight",
        "Sticker",
        "Motion",
        "Blending",
        "Opacity",
        "Border & Shadow",
        "Mosaic & Blur",
        "Effect Layer",
        "AI Effect",
        "Magnifier",
        "Blur Tool",
        "Effect",
        "Video Effect Layer",
        "Filter Layer",
        "Video FX",
        "FX",
        "Tempo Effect",
        "Tempo Effects",
        "PiP Animation",
        "Color filter layer",
        "Decor",
    ],
    "Color & Adjust": [
        "Adjust",
        "AI Color",
        "Brightness",
        "Contrast",
        "Saturation",
        "HSL",
        "Hue",
        "Temp",
        "Tint",
        "Sharpness",
        "Color Board",
        "Color Selector",
        "HDR",
        "Adjustment",
    ],
    "Background & Cutout": [
        "Background",
        "Add Background",
        "Cutout",
        "Replace Background",
        "Replace BG",
        "Chroma Key",
        "ChromaKey",
        "Mask",
        "Advanced Cutout",
    ],
    "Enhance & Fix": [
        "Enhance",
        "Video Enhancer",
        "Photo Enhance",
        "Video Enhance",
        "Fix Shake",
        "AI Video Upscaler",
        "Video Upscaler",
        "Facial Mosaic",
        "Auto Mosaic",
        "Skin Smoother",
        "Portrait",
        "People Background",
        "Motion Tracking",
        "Face Reshape",
        "Body Reshape",
        "Stabilizer",
        "Makeup",
        "Beautify",
        "Removal",
        "ID Photo",
    ],
    "Export & Output": [
        "Export",
        "Produced Video",
        "Video Intro",
        "Video Outro",
        "Share Page",
        "Share extension",
        "Produce",
        "Preview",
        "Full Screen Preview",
        "Import",
    ],
    "UI & Settings": [
        "Settings",
        "Preference",
        "Preferences",
        "Tutorials & Tips",
        "VIP Benefit page",
        "VIP Benefits",
        "VIP Benefit",
        "Trending",
        "Camera",
        "Video",
        "Photo",
        "Overlay",
        "Transition",
        "Templates",
        "Duration",
        "Credit",
        "More",
        "My Artwork",
        "Feedback",
        "Send Feedback",
        "What's New",
        "Permission",
        "GDPR",
        "Privacy policy",
        "About",
        "Notice",
        "Home indicator",
        "App icon",
        "My AI Artwork",
        "My Artwork",
        "Mine",
        "Quick Actions",
        "Collage",
        "Watermark",
        "Premium",
        "Home",
        "Hint",
        "My Project",
        "My Projects",
        "Log Event",
        "APNG",
    ],
    "Launcher": ["Launcher", "Launch", "Opening Intro", "Splash", "Opening Tutorial", "Recent Task", "Launcher Shortcut"],
    "Media Picker": [
        "Add Media",
        "Add Image",
        "Media Picker",
        "Media Room",
        "File Picker",
        "Library",
        "Gettyimages",
        "Pexels",
        "iStock",
        "Photo Picker",
        "Cloud Storage (Media Picker)",
        "Pixabay",
        "Getty Image Premium",
        "Shutterstock",
        "Media",
        "Sample Project",
        "Demo Project",
        "Demo Video",
    ],
    "Cloud": ["Cloud Storage", "My Cloud", "Back Up to My Device", "Cloud"],
    "IAP": [
        "IAP",
        "Pro+",
        "Subscribe",
        "Upgrade Dialog",
        "Anniversary Sale",
        "Xmas",
        "Halloween",
        "Halloween IAP",
        "JP Golden Week",
        "Summer Sale",
        "New Year Sale",
        "Christmas",
        "Churn Recovery",
        "Subscription",
        "Credit IAP",
        "Credit System",
        "Pro",
        "IAP page",
        "Shortcut IAP",
        "JP IAP",
        "Summer Promotion",
        "OBON Promotion",
        "Cross Promote",
        "Cross-Promote",
        "Halloween Sale",
        "Halloween Interstitial",
        "Halloween Launcher",
        "Halloween Credit System",
        "Halloween Credits",
        "Halloween Credit Page",
        "Halloween Credit Purchase",
        "Christmas & New Year",
        "Redeem Code",
        "Try Before Buy",
        "Gamification",
    ],
    "Analytics": ["Mixpanel", "Appsflyer", "Flurry Log", "Flurry", "Meta", "Log Event"],
    "Sign In": ["Sign In", "Social Sign In", "Account", "User Sign In", "GDPR"],
    "Shortcut": ["Shortcut"],
    "Notification": ["Notification", "Push Notification", "Menu(Notification)"],
    "QA / Testing": ["Demo", "Setup", "Installer Package", "Third party", "CS", "App icon", "Independent Entry", "Shrink mode", "Deeplink"],
}

_FLAT_OVERRIDES = {
    # Enhance & Fix overrides
    "Body Effects": "Enhance & Fix",
    "Body Effect": "Enhance & Fix",
    "AI body Effect": "Enhance & Fix",
    "Body Reshape": "Enhance & Fix",
    "Body Effects (Slim)": "Enhance & Fix",
    "Body Effects(Slim)": "Enhance & Fix",
    # Analytics
    "Mixpanel": "Analytics",
    "MixPanel": "Analytics",
    "Mixpanel log": "Analytics",
    "Appsflyer": "Analytics",
    "Flurry Log": "Analytics",
    # IAP
    "IAP": "IAP",
    "IAP (7-day Free Trial)": "IAP",
    "IAP(7-day Free Trial)": "IAP",
    "IAP(B version)": "IAP",
    "IAP(C version)": "IAP",
    "IAP(A version)": "IAP",
    "IAP(Pro+)": "IAP",
    "IAP(Anniversary)": "IAP",
    "IAP(Xmas)": "IAP",
    "IAP(JP Golden Week)": "IAP",
    "Xmas": "IAP",
    "Xmas/New Year": "IAP",
    "Halloween": "IAP",
    "Halloween IAP": "IAP",
    "JP Golden Week": "IAP",
    "New Year Sale (IAP)": "IAP",
    "Anniversary Sale(IAP)": "IAP",
    "Churn Recovery": "IAP",
    "Cross Promote": "IAP",
    "Cross-Promote": "IAP",
    "Pro+": "IAP",
    "Subscribe": "IAP",
    # Launcher
    "Launcher": "Launcher",
    "Launch": "Launcher",
    "Opening Intro": "Launcher",
    "Opening intro": "Launcher",
    "Opening Tutorial": "Launcher",
    "Splash": "Launcher",
    "Recent Task": "Launcher",
    "Recent task": "Launcher",
    # Cloud
    "Cloud Storage": "Cloud",
    "My Cloud": "Cloud",
    "Back Up to My Device": "Cloud",
    "Cloud": "Cloud",
    # AI Features subset
    "AI Sketch": "AI Features",
    "AI Cartoon": "AI Features",
    "AI Scene": "AI Features",
    "AI Effect": "AI Features",
    "AI Creation": "AI Features",
    "GAI": "AI Features",
    "Voice Cloning": "AI Features",
    "My Voices": "AI Features",
    "TTI Inspiration": "AI Features",
    # Media Picker
    "Add Media": "Media Picker",
    "Media Picker": "Media Picker",
    # Editor core
    "Timeline": "Editor Core",
    "NewTimeline": "Editor Core",
    "Editor": "Editor Core",
    "PiP": "Editor Core",
    "PIP": "Editor Core",
    "Drag and Drop": "Editor Core",
    "Ripple Editing": "Editor Core",
    # Text
    "MGT": "Text & Captions",
    "Intro/Outro": "Text & Captions",
    # Export & Output
    "Share Page": "Export & Output",
    "Share page": "Export & Output",
    "Preview": "Export & Output",
    "Full Screen Preview": "Export & Output",
    # Visual Effects
    "Effect Layer": "Visual Effects",
    "Effect layer": "Visual Effects",
    # Enhance & Fix
    "Face Reshape": "Enhance & Fix",
    "Skin Smoother": "Enhance & Fix",
    "Stabilizer": "Enhance & Fix",
    # Sign In
    "Sign In": "Sign In",
    "Sign in": "Sign In",
    "Social Sign In": "Sign In",
    "User Sign In": "Sign In",
    "Account": "Sign In",
    # Shortcut
    "Shortcut": "Shortcut",
    # Notification
    "Notification": "Notification",
    "Push Notification": "Notification",
    # UI & Settings
    "Settings": "UI & Settings",
    "Menu": "UI & Settings",
    "What's New": "UI & Settings",
    "Watermark": "UI & Settings",
    "Feedback": "UI & Settings",
    "Permission": "UI & Settings",
    "GDPR": "UI & Settings",
    "About": "UI & Settings",
    # QA / Testing
    "Demo": "QA / Testing",
    "Setup": "QA / Testing",
    "CS": "QA / Testing",
    "Third party": "QA / Testing",
    # PDR desktop: Editor Core
    "Quick Editing": "Editor Core",
    "PiP Quick Editing": "Editor Core",
    "Title Quick Editing": "Editor Core",
    "Speed Quick Editing": "Editor Core",
    "Mask Quick Editing": "Editor Core",
    "Quick Editing for Title": "Editor Core",
    "On Screen Editing": "Editor Core",
    "PiP Designer": "Editor Core",
    "Mask Designer": "Editor Core",
    "Range Selection": "Editor Core",
    "Display Panel": "Editor Core",
    "Customize Toolbar": "Editor Core",
    "UI Revamp": "Editor Core",
    # PDR desktop: Text & Captions
    "Subtitle Room": "Text & Captions",
    "Subtitles Room": "Text & Captions",
    "Title Designer": "Text & Captions",
    "Title Designer Revamp": "Text & Captions",
    "Text-Based Editing": "Text & Captions",
    "Trim by Speech": "Text & Captions",
    # PDR desktop: AI Features
    "AI Copilot": "AI Features",
    "AI Video Generator": "AI Features",
    "Image Generator": "AI Features",
    "Video Generator": "AI Features",
    "AI Voice Translator": "AI Features",
    "Brand Kits": "AI Features",
    # PDR desktop: Visual Effects / Media
    "Effect Room": "Visual Effects",
    "Audio Room": "Audio",
    "Media Room": "Media Picker",
    # PDR desktop: QA / Testing
    "General": "QA / Testing",
    "Installation": "QA / Testing",
    # PHD-specific: Enhance & Fix
    "Makeup": "Enhance & Fix",
    "Beautify": "Enhance & Fix",
    "Removal": "Enhance & Fix",
    "ID Photo": "Enhance & Fix",
    # PHD-specific: AI Features
    "AI Face Swap": "AI Features",
    "AI Expand": "AI Features",
    "AI Outfit": "AI Features",
    "AI Headshot": "AI Features",
    "AI Relight": "AI Features",
    "AI Hairstyle": "AI Features",
    "AI Try On": "AI Features",
    "AI Creative Studio": "AI Features",
    "AI Magic Studio": "AI Features",
    "Artistic Avatar": "AI Features",
    "AI Avatar": "AI Features",
    "Draw to Image": "AI Features",
    "Image Fusion": "AI Features",
    # PHD-specific: UI & Settings
    "My AI Artwork": "UI & Settings",
    "My Artwork": "UI & Settings",
    "Mine": "UI & Settings",
    "Quick Actions": "UI & Settings",
    "Collage": "UI & Settings",
    "Blur Tool": "Visual Effects",
    # PHD-specific: Editor Core
    "Edit Room": "Editor Core",
    # PHD-specific: Media Picker
    "Add Image": "Media Picker",
    "Photo Picker": "Media Picker",
}

# Build canonical module list BEFORE update() which overwrites list values
# with string category names for flat overrides.
_CANONICAL_MODULES = list(
    {m for mods in MODULE_CATEGORIES.values() if isinstance(mods, list) for m in mods}
    | set(_FLAT_OVERRIDES.keys())
)

MODULE_CATEGORIES.update(_FLAT_OVERRIDES)

_MODULE_TO_CATEGORY = {}
for cat, modules in MODULE_CATEGORIES.items():
    if isinstance(modules, list):
        for m in modules:
            _MODULE_TO_CATEGORY[m.lower()] = cat
    elif isinstance(modules, str):
        _MODULE_TO_CATEGORY[cat.lower()] = modules

KNOWN_TAGS = [
    "EDF",
    "UX",
    "MX",
    "MUI",
    "Side Effect",
    "AT Found",
    "Function",
    "Stability",
    "Performance",
    "Regression",
    "Crash",
    "ANR",
    "UI",
    "Localization",
    "IAP",
]

SEVERITY_MAP = {
    "1": (1, 10, "1-Critical"),
    "2": (2, 5, "2-Major"),
    "3": (3, 2, "3-Normal"),
    "4": (4, 1, "4-Minor"),
}

# Issue 8 – RD priority labels
PRIORITY_LABEL_MAP = {
    1: "Fix Now",
    2: "Must Fix",
    3: "Better Fix",
    4: "No Matter",
    5: "N/A",
}

DESC_PATTERN = re.compile(
    r"^(?P<product>[A-Z]+(?:-[A-Za-z])?)\s*"
    r"(?:Mac\s+)?"
    r"(?P<version>[\d.]+(?:\s*\([\d.]+\))?)\s*"
    r"(?:Mac\s*)?"
    r"-\s*"
    r"(?P<tags>(?:\[[^\]]*\]\s*)*)"
    r"(?P<module>[^:]+?):\s*"
    r"(?P<description>.+)$",
    re.IGNORECASE,
)
TAG_PATTERN = re.compile(r"\[([^\]]+)\]")


# ---------------------------------------------------------------------
# Version Mapping Store (Issue 5)
# ---------------------------------------------------------------------


class VersionMappingStore:
    """Manages per-version pending/confirmed module mappings."""

    def __init__(self, mapping_dir: Path):
        self.mapping_dir = mapping_dir
        self.permanent_path = mapping_dir / "permanent" / "mappings_global.json"
        self.versions_dir = mapping_dir / "versions"

        self.mapping_dir.mkdir(parents=True, exist_ok=True)
        self.permanent_path.parent.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        self._permanent = self._load_json(self.permanent_path)
        self._confirmed_cache: dict[str, dict] = {}
        # Persistent Ollama matcher decisions across pipeline runs.
        self._ollama_disk_cache: dict = self._load_json(
            self.mapping_dir / "ollama_cache.json"
        )
        # Seed the in-process cache from disk so repeated rows in the same run
        # hit the warm cache even before the LLM is queried once.
        for k, v in self._ollama_disk_cache.items():
            _OLLAMA_MATCHER_MEM_CACHE.setdefault(k, v)

        if _embedding_matcher_enabled():
            embedder = _CanonicalEmbedder.get()
            if not embedder.ready:
                embedder.initialize(self.mapping_dir)

    def _load_json(self, path: Path) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_json(self, path: Path, data: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _get_confirmed(self, version: str) -> dict:
        if version not in self._confirmed_cache:
            p = self.versions_dir / f"{version}_confirmed.json"
            self._confirmed_cache[version] = self._load_json(p)
        return self._confirmed_cache[version]

    def lookup(self, raw: str, version: str) -> Optional[str]:
        """Lookup order: permanent -> all confirmed versions.
        Confirmed files loaded once and cached to avoid repeated disk
        scans across 10k+ bug rows.
        """
        if raw in self._permanent:
            return self._permanent[raw]

        if not hasattr(self, "_all_confirmed_cache"):
            self._all_confirmed_cache = {}
            for p in sorted(self.versions_dir.glob("*_confirmed.json")):
                self._all_confirmed_cache.update(self._load_json(p))
        if raw in self._all_confirmed_cache:
            return self._all_confirmed_cache[raw]
        return None

    def _invalidate_confirmed_cache(self):
        """Call after writing any confirmed file to force a cache rebuild."""
        if hasattr(self, "_all_confirmed_cache"):
            del self._all_confirmed_cache
    def add_pending(self, raw: str, suggested: str, version: str):
        p_path = self.versions_dir / f"{version}_pending.json"
        pending = self._load_json(p_path)
        if raw not in pending:
            pending[raw] = {"suggested": suggested, "confirmed": False}
            self._save_json(p_path, pending)

    def flush_pending(self, version: str) -> dict:
        p_path = self.versions_dir / f"{version}_pending.json"
        pending = self._load_json(p_path)
        return {k: v for k, v in pending.items() if not v.get("confirmed", False)}

    def promote_to_permanent(self, version: str):
        confirmed = self._get_confirmed(version)
        self._permanent.update(confirmed)
        self._save_json(self.permanent_path, self._permanent)
        print(
            f"  ✅ Promoted {len(confirmed)} mappings from {version} to permanent store."
        )

    def ollama_cache_path(self) -> Path:
        return self.mapping_dir / "ollama_cache.json"

    def load_ollama_cache(self) -> dict:
        return self._load_json(self.ollama_cache_path())

    def save_ollama_cache(self, cache: dict):
        self._save_json(self.ollama_cache_path(), cache)


def _ollama_disambiguate(raw: str, candidates: list[str], model: str) -> Optional[dict]:
    """Ask a local Ollama model to pick the best canonical match for `raw`
    from the supplied `candidates`. Returns {"match": str|None, "confidence": float,
    "reason": str} on success, or None if the call failed / produced invalid JSON.

    Fail-soft: network, JSON, and model errors all return None so the caller
    can fall back to the rapidfuzz pre-filter's top-1 suggestion.
    """
    numbered = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))
    prompt = (
        "You are matching raw bug-report module names to a canonical module list.\n"
        f'Raw module name: "{raw}"\n'
        "Candidates (choose one OR return null):\n"
        f"{numbered}\n\n"
        'Return STRICT JSON only: {"match": "<exact canonical or null>", '
        '"confidence": <0-1>, "reason": "<<=10 words>"}\n'
        "Rules:\n"
        "- If the raw name is semantically unrelated to all candidates, return null.\n"
        "- Do NOT match typo-neighbors that mean different things "
        "(e.g. Decor≠Demo, Hint≠Tint, Mixtape≠Mixpanel, Import≠Export).\n"
        "- Do NOT drop semantic qualifiers: 'AI Sticker' does NOT match 'Sticker' "
        "unless 'AI Sticker' is literally in the candidate list.\n"
        "- When in doubt, return null — a human will review.\n"
        "- confidence should be >=0.9 only when you are certain; otherwise lower.\n"
    )
    payload = json.dumps(with_thinking_disabled({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    })).encode()
    req = urllib.request.Request(
        OLLAMA_MATCHER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=OLLAMA_MATCHER_TIMEOUT)
        envelope = json.loads(resp.read().decode(errors="replace"))
        text = response_text(envelope)
        try:
            obj = extract_json(text, expect="object")
        except json.JSONDecodeError:
            return None
        match = obj.get("match")
        if match is not None:
            match = str(match).strip()
            if match.lower() in ("null", "none", ""):
                match = None
            elif match not in candidates:
                # LLM invented a name; treat as no-match so rapidfuzz top-1 wins.
                return None
        try:
            confidence = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "match": match,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(obj.get("reason", ""))[:120],
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None
    except Exception:
        return None


def _ollama_matcher_enabled() -> bool:
    val = os.environ.get("PDRI_OLLAMA_MATCHER", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _ollama_matcher_model() -> str:
    return os.environ.get("PDRI_OLLAMA_MATCHER_MODEL", OLLAMA_MATCHER_DEFAULT_MODEL).strip() \
        or OLLAMA_MATCHER_DEFAULT_MODEL


def _embedding_matcher_enabled() -> bool:
    val = os.environ.get("PDRI_EMBEDDING_MATCHER", "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    return _ollama_matcher_enabled() or val in ("1", "true", "yes", "on")


def suggest_canonical(
    raw: str, cache_dir: "Path | None" = None
) -> "tuple[str | None, float]":
    """Suggest a canonical module name for *raw* using embeddings + LLM.

    Public API for the Streamlit "AI Re-suggest blank rows" button.
    Returns ``(canonical_name_or_None, confidence)``.
    """
    embedder = _CanonicalEmbedder.get()
    if not embedder.ready:
        embedder.initialize(cache_dir)
    if not embedder.ready:
        return (None, 0.0)

    candidates_with_scores = embedder.top_k(raw, k=8)
    if not candidates_with_scores:
        return (None, 0.0)

    candidates = [name for name, _ in candidates_with_scores]

    cached = _OLLAMA_MATCHER_MEM_CACHE.get(raw)
    if cached is not None:
        pick = cached.get("match")
        conf = float(cached.get("confidence", 0.0))
        return (pick, conf)

    result = _ollama_disambiguate(raw, candidates, _ollama_matcher_model())
    if result is not None:
        _OLLAMA_MATCHER_MEM_CACHE[raw] = result
        pick = result.get("match")
        conf = float(result.get("confidence", 0.0))
        return (pick, conf)

    return (None, 0.0)


def normalize_module(raw: str, version: str = "unknown",
                     store: VersionMappingStore = None,
                     fuzzy_threshold: int = 85) -> str:
    """Normalize raw module string -> canonical.

    1. Strip parenthetical sub-variants e.g. "Auto Edit(Pet 02)" -> "Auto Edit"
       and square-bracket segments e.g. "Transition[Portrait]" -> "Transition"
    2. O(1) alias lookup via lowercase shadow dict
    2.5. Case-insensitive match against the canonical module list
    3. Mapping store (permanent + confirmed versions)
    4. Fuzzy match (>= threshold -> auto-confirm, 65-threshold -> pending)
    5. Raw string (Uncategorized; flagged for dashboard)
    """
    mod = raw.strip()

    # Step 0: replace underscores with spaces (PHD-A/PHD-i use underscore-separated
    # module names like "AI_Face_Swap" that should match canonical "AI Face Swap")
    mod = mod.replace("_", " ")

    # Step 1: normalise compound module strings automatically.
    #
    # (a) Strip parenthetical sub-variants:
    #     "Auto Edit(Pet 02)"          -> "Auto Edit"
    #     "AI Storytelling (Music)"    -> "AI Storytelling"
    #     "Text(Neon_01)"              -> "Text"
    stripped = re.sub(r'\s*\([^)]*\)', '', mod).strip()
    if stripped:
        mod = stripped
    # Dangling open paren/bracket/brace from truncated raw strings
    # (e.g. "Produce (16", "Shortcut[CHT", "Produce {16").
    stripped = re.sub(r'\s*[\(\[\{][^\)\]\}]*$', '', mod).strip()
    if stripped:
        mod = stripped

    # (b) Take first token of comma-separated multi-module strings:
    #     "Text, title, MGT"           -> "Text"
    #     "Add Media, Media picker"    -> "Add Media"
    #     "Launcher, Shortcut"         -> "Launcher"
    #     "Audio, Music, My Music"     -> "Audio"
    if "," in mod:
        first = mod.split(",")[0].strip()
        if first:  # guard against leading-comma edge cases
            mod = first

    # (c) Handle compound "Parent<sep>Child" notation by taking the first token.
    #     "Menu>Sign in"                 -> "Menu"
    #     "Audio> GettyImage Music"      -> "Audio"
    #     "Filter/Transition"            -> "Filter"
    #     "Text\\Animation"              -> "Text"
    #     "Cutout + Fx"                  -> "Cutout"
    #     "AI Effect/Video FX"           -> "AI Effect"
    for sep in (">", "/", "\\", "+"):
        if sep in mod:
            first = mod.split(sep)[0].strip()
            if first:
                mod = first
                break

    # (d) Strip square-bracket segments like [Portrait], [JPN], [iPhone 12 Pro Max].
    #     These encode test conditions (locale/device/orientation/content), not a
    #     different module:
    #       "Transition[Portrait]"          -> "Transition"
    #       "AI Storytelling [JPN]"         -> "AI Storytelling"
    #       "Shortcut[ iPhone 12 Pro Max]"  -> "Shortcut"
    stripped = re.sub(r'\s*\[[^\]]*\]\s*', ' ', mod).strip()
    stripped = re.sub(r' +', ' ', stripped)
    if stripped:
        mod = stripped

    # Step 2: O(1) alias lookup (case-insensitive via lowercase shadow dict)
    alias_result = _MODULE_ALIASES_LOWER.get(mod.lower())
    if alias_result:
        return alias_result

    # Step 2.5: Case-insensitive match against the canonical module list
    # ("text to speech" -> "Text to Speech"). Handles case variants that aren't
    # in MODULE_ALIASES, as long as the canonical form exists somewhere (either
    # in MODULE_CATEGORIES or as an alias target). Cached on the function object.
    if not hasattr(normalize_module, "_canonical_lower"):
        canonical_set = set(_CANONICAL_MODULES) | set(MODULE_ALIASES.values())
        normalize_module._canonical_lower = {
            m.lower(): m for m in canonical_set
        }
    canonical = normalize_module._canonical_lower.get(mod.lower())
    if canonical:
        return canonical

    # Step 3: Mapping store (permanent + confirmed versions)
    if store is not None:
        result = store.lookup(mod, version)
        if result:
            return result

    # Step 4: Candidate retrieval → Ollama LLM disambiguation.
    #
    # Three tiers:
    #   (A) Embedding-based (preferred): cosine-similarity against canonical
    #       embeddings, top-8 fed to _ollama_disambiguate().
    #   (B) rapidfuzz WRatio fallback: when embeddings are unavailable.
    #   (C) Raw passthrough: when neither works (mod too short, no candidates).
    #
    # LLM confidence gates:
    #   ≥ 0.85 → auto-confirm
    #   ≥ 0.3  → pending with populated suggestion (human reviews)
    #   < 0.3 or null → pending with blank suggestion
    if len(mod) < 4:
        return mod

    use_embeddings = _embedding_matcher_enabled()
    llm_enabled = _ollama_matcher_enabled()
    candidates: list[str] = []

    # (A) Embedding retrieval
    if use_embeddings:
        embedder = _CanonicalEmbedder.get()
        if not embedder.ready:
            cache_dir = store.mapping_dir if store is not None else None
            embedder.initialize(cache_dir)
        if embedder.ready:
            embed_hits = embedder.top_k(mod, k=8)
            if embed_hits:
                candidates = [name for name, _ in embed_hits]

    # (B) rapidfuzz fallback
    if not candidates and FUZZY_AVAILABLE:
        top_k = process.extract(
            mod, _CANONICAL_MODULES, scorer=fuzz.WRatio, limit=8
        )
        if top_k:
            candidates = [c for c, _, _ in top_k]

    if not candidates:
        return mod

    # LLM disambiguation
    llm_result = None
    if llm_enabled:
        llm_result = _OLLAMA_MATCHER_MEM_CACHE.get(mod)
        if llm_result is None and store is not None:
            llm_result = getattr(store, "_ollama_disk_cache", {}).get(mod)
        if llm_result is None:
            llm_result = _ollama_disambiguate(
                mod, candidates, _ollama_matcher_model()
            )
            if llm_result is not None:
                _OLLAMA_MATCHER_MEM_CACHE[mod] = llm_result
                if store is not None and hasattr(store, "_ollama_disk_cache"):
                    store._ollama_disk_cache[mod] = {
                        **llm_result,
                        "model": _ollama_matcher_model(),
                        "ts": int(time.time()),
                    }

    if llm_result is not None:
        pick = llm_result.get("match")
        conf = float(llm_result.get("confidence", 0.0))
        if pick and conf >= 0.85:
            if store is not None:
                confirmed_path = store.versions_dir / f"{version}_confirmed.json"
                confirmed = store._load_json(confirmed_path)
                if mod not in confirmed:
                    confirmed[mod] = pick
                    store._save_json(confirmed_path, confirmed)
                    store._confirmed_cache[version] = confirmed
                    store._invalidate_confirmed_cache()
            return pick
        if store is not None:
            suggested = pick or ""
            store.add_pending(mod, suggested, version)
        return pick if pick else mod

    # Legacy rapidfuzz-only path (LLM disabled or LLM call failed).
    if FUZZY_AVAILABLE and candidates:
        rf_top = process.extractOne(mod, candidates, scorer=fuzz.WRatio)
        if rf_top:
            rf_match, rf_score, _ = rf_top
            if rf_score >= fuzzy_threshold:
                if store is not None:
                    confirmed_path = store.versions_dir / f"{version}_confirmed.json"
                    confirmed = store._load_json(confirmed_path)
                    if mod not in confirmed:
                        confirmed[mod] = rf_match
                        store._save_json(confirmed_path, confirmed)
                        store._confirmed_cache[version] = confirmed
                        store._invalidate_confirmed_cache()
                return rf_match
            if rf_score >= 65 and store is not None:
                store.add_pending(mod, rf_match, version)

    return mod


def detect_case_variant_groups(
    modules, store: VersionMappingStore, bucket: str = "case_variants"
) -> int:
    """Surface case-only module duplicates to the pending review page.

    Groups parsed_module values by lowercase. For each group with >1 distinct
    casing, picks a winner (canonical form if any variant is in the canonical
    set, else the most-frequent casing) and writes every non-winner variant to
    `<bucket>_pending.json` via `store.add_pending`. Entries already present as
    aliases (MODULE_ALIASES), or already in the permanent store, are skipped.

    The fix takes effect after the user promotes them in the Pending Review
    page; on the next pipeline run, step 3 (mapping store lookup) resolves them.

    Returns the number of new pending entries written.
    """
    from collections import Counter

    counts = Counter(m for m in modules if isinstance(m, str) and m)
    groups: dict[str, list[tuple[str, int]]] = {}
    for m, c in counts.items():
        groups.setdefault(m.lower(), []).append((m, c))

    canonical_set = set(_CANONICAL_MODULES) | set(MODULE_ALIASES.values())
    permanent = store._permanent if store is not None else {}
    existing_pending: dict = {}
    if store is not None:
        existing_pending = store._load_json(
            store.versions_dir / f"{bucket}_pending.json"
        )
    written = 0

    for variants in groups.values():
        if len(variants) <= 1:
            continue
        known = [v for v, _ in variants if v in canonical_set]
        if known:
            winner = known[0]
        else:
            winner = max(variants, key=lambda x: x[1])[0]
        for v, _ in variants:
            if v == winner:
                continue
            if v.lower() in _MODULE_ALIASES_LOWER:
                continue
            if v in permanent or v in existing_pending:
                continue
            if store is not None:
                store.add_pending(v, winner, bucket)
                written += 1
    return written


def get_category(module_name: str) -> str:
    cat = _FLAT_OVERRIDES.get(module_name)
    if cat:
        return cat
    cat = _MODULE_TO_CATEGORY.get(module_name.lower())
    if cat:
        return cat

    mn = module_name.lower()
    for cat_name, modules in MODULE_CATEGORIES.items():
        if isinstance(modules, list):
            for m in modules:
                if len(m) >= 6 and (m.lower() in mn or mn in m.lower()):
                    return cat_name
    return "Uncategorized"


def parse_severity(sev_str):
    if pd.isna(sev_str):
        return 3, 2, "3-Normal"
    d = re.search(r"(\d)", str(sev_str).strip())
    if d:
        return SEVERITY_MAP.get(d.group(1), (3, 2, "3-Normal"))
    return 3, 2, "3-Normal"


def parse_repro(repro_str) -> float:
    if pd.isna(repro_str):
        return 0.5
    s = str(repro_str).strip().lower()
    m = re.search(r"(\d+)\s*(?:out\s*of|/)(\d+)", s)
    if m:
        return int(m.group(1)) / max(int(m.group(2)), 1)
    if "always" in s or "every" in s:
        return 1.0
    if "random" in s:
        return 0.3
    if "once" in s or "rare" in s:
        return 0.1
    return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Version catalogue helpers
# ─────────────────────────────────────────────────────────────────────────────

# Versions with fewer bugs than this are treated as sparse/typo entries.
# They are KEPT in the data (for full fidelity) but are flagged with
# version_is_sparse=True and excluded from auto-selected "recent" defaults.
VERSION_SPARSE_THRESHOLD = 4


def build_version_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """Build a version catalogue sorted by recency (latest Create Date DESC).

    Recency = max(Create Date) for all bugs sharing that parsed_version.
    This is robust against:
      - Version strings that don't sort lexicographically (e.g. "9.0" > "16.3.0")
      - Typo versions with only 1-2 bugs that appear at a random numeric position
      - Versions whose string prefix is misleading (e.g. a hotfix "16.2.5.1"
        that was filed after a newer "16.3.0" cycle)

    Columns returned:
        parsed_version      – the version string
        bug_count           – number of bugs in this version
        latest_create_date  – most recent Create Date among those bugs
        version_is_sparse   – True when bug_count < VERSION_SPARSE_THRESHOLD
        version_rank        – 1 = most recent, 2 = second most recent, …
                              sparse versions are ranked AFTER all real versions
    """
    if "parsed_version" not in df.columns:
        return pd.DataFrame(columns=[
            "parsed_version", "bug_count", "latest_create_date",
            "version_is_sparse", "version_rank",
        ])

    rows = []
    for ver, grp in df.groupby("parsed_version", dropna=True):
        n = len(grp)
        if "Create Date" in grp.columns:
            latest = pd.to_datetime(grp["Create Date"], errors="coerce").max()
        else:
            latest = pd.NaT
        rows.append({
            "parsed_version":     ver,
            "bug_count":          n,
            "latest_create_date": latest,
            "version_is_sparse":  n < VERSION_SPARSE_THRESHOLD,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "parsed_version", "bug_count", "latest_create_date",
            "version_is_sparse", "version_rank",
        ])

    cat = pd.DataFrame(rows)

    # Sort: real versions by date DESC first, sparse versions last (by date DESC within)
    cat["_sort_key"] = cat["latest_create_date"].fillna(pd.Timestamp.min)
    real   = cat[~cat["version_is_sparse"]].sort_values("_sort_key", ascending=False)
    sparse = cat[ cat["version_is_sparse"]].sort_values("_sort_key", ascending=False)
    cat = pd.concat([real, sparse], ignore_index=True)
    cat["version_rank"] = range(1, len(cat) + 1)
    cat = cat.drop(columns=["_sort_key"])

    return cat


def versions_by_recency(df: pd.DataFrame, exclude_sparse: bool = True):
    """Return parsed_version strings ordered by recency (newest first).

    When exclude_sparse=True (default), versions below VERSION_SPARSE_THRESHOLD
    bugs are omitted — useful for default sidebar selections and 'latest' scope.
    Pass exclude_sparse=False to get the complete list (e.g. for full exports).
    """
    cat = build_version_catalogue(df)
    if exclude_sparse:
        cat = cat[~cat["version_is_sparse"]]
    return cat["parsed_version"].tolist()


def parse_ecl_export(
    input_path: str,
    output_path: str,
    mapping_dir: str | None = None,
    fuzzy_threshold: int = 85,
):
    # Default mapping_dir: sibling of the output file's parent directory
    if mapping_dir is None:
        mapping_dir = str(Path(output_path).parent / "module_mappings")
    ext = Path(input_path).suffix.lower()
    if ext == ".json":
        with open(input_path, "r", encoding="utf-8") as _f:
            _raw = json.load(_f)
        # Accept either a plain list or n8n's wrapped {"json": {...}} shape
        if isinstance(_raw, list):
            _records = [
                r["json"] if (isinstance(r, dict) and "json" in r) else r
                for r in _raw
            ]
        else:
            _records = [_raw]
        df = pd.DataFrame(_records)

        # ── Normalise n8n column names to the names the parser expects ──
        # The n8n "Get Columns_v3" Set node outputs these exact names; map
        # them to canonical internal names so no downstream code changes.
        # All names happen to already match — this is an explicit contract so
        # future n8n renames are fixed in one place.
        _N8N_COL_MAP = {
            "Short Description" : "Short Description",
            "Severity"          : "Severity",
            "Priority"          : "Priority",
            "Status"            : "Status",
            "Create Date"       : "Create Date",
            "Closed Date"       : "Closed Date",
            "Version"           : "Version",
            "Build#"            : "Build#",
            "Close Build#"      : "Close Build#",
            "Creator"           : "Creator",
            "BugCode"           : "BugCode",
            "BugBelong"         : "BugBelong",
            "Handler"           : "Handler",
        }
        df.rename(
            columns={k: v for k, v in _N8N_COL_MAP.items() if k in df.columns},
            inplace=True,
        )
        print(f"Loaded {len(df)} bugs from JSON (n8n webhook output)")
        print(f"  Columns: {sorted(df.columns.tolist())}")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(input_path)
        print(f"Loaded {len(df)} bugs from Excel")
    else:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} bugs from CSV")

    store = VersionMappingStore(Path(mapping_dir))

    sd_col = next(
        (c for c in df.columns if "short" in c.lower() and "desc" in c.lower()), None
    )
    if sd_col is None:
        sd_col = "Short Description"
    if sd_col not in df.columns:
        print(f"ERROR: Short Description column not found (expected: {sd_col})")
        print("Available columns:", list(df.columns))
        sys.exit(1)

    # Resolve the "Version" column that will be used as the authoritative
    # parsed_version source (Change 6: prefer the dedicated Version field over
    # the regex-extracted version from Short Description).
    ver_col = next(
        (c for c in df.columns if c.strip().lower() == "version"), None
    )

    status_col = next((c for c in df.columns if c.lower() == "status"), None)

    print("Parsing Short Descriptions...")
    parsed = {
        "parsed_product": [],
        "parsed_version": [],
        "parsed_tags": [],
        "parsed_module_raw": [],   # original ECL string e.g. "Auto Edit(Pet 02)"
        "parsed_module": [],       # normalised + stripped e.g. "Auto Edit"
        "parsed_description": [],
        "module_category": [],
    }
    tag_cols = {t.lower().replace(" ", "_"): [] for t in KNOWN_TAGS}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing bugs",
                       unit="bug", file=sys.stderr):
        desc = str(row.get(sd_col, ""))

        # Change 6: derive version_hint from the dedicated Version column first;
        # fall back to a regex scan of Short Description only when the column is
        # absent or blank.  This is more reliable because the Short Description
        # prefix sometimes contains the wrong version or no version at all.
        if ver_col and pd.notna(row.get(ver_col)) and str(row.get(ver_col, "")).strip():
            version_hint = str(row[ver_col]).strip()
            # Normalise to the first dotted-number segment (e.g. "16.3.0.2847" → "16.3.0")
            vm2 = re.search(r"(\d+\.\d+(?:\.\d+)?)", version_hint)
            if vm2:
                version_hint = vm2.group(1)
        else:
            vm = re.search(r"(\d+\.\d+(?:\.\d+)?)", desc)
            version_hint = vm.group(1) if vm else "unknown"

        m = DESC_PATTERN.match(desc)
        if m:
            parsed["parsed_product"].append(m.group("product").strip())
            # Change 6: always store the Version-column value as parsed_version
            # (already computed above as version_hint); ignore the version
            # captured by DESC_PATTERN from the description prefix.
            parsed["parsed_version"].append(version_hint if version_hint != "unknown" else m.group("version").strip())
            tags_raw = TAG_PATTERN.findall(m.group("tags"))
            parsed["parsed_tags"].append(tags_raw)
            raw_mod = m.group("module").strip()
            mod = normalize_module(
                raw_mod,
                version=version_hint,
                store=store,
                fuzzy_threshold=fuzzy_threshold,
            )
            parsed["parsed_module_raw"].append(raw_mod)
            parsed["parsed_module"].append(mod)
            parsed["parsed_description"].append(m.group("description").strip())
            parsed["module_category"].append(get_category(mod))
            for tk in tag_cols:
                tag_cols[tk].append(
                    any(t.lower().replace(" ", "_") == tk for t in tags_raw)
                )
        else:
            parsed["parsed_product"].append(None)
            # Even for non-matching rows use the Version column value
            parsed["parsed_version"].append(version_hint if version_hint != "unknown" else None)
            parsed["parsed_tags"].append([])
            parsed["parsed_module_raw"].append(None)
            parsed["parsed_module"].append(None)
            parsed["parsed_description"].append(desc)
            parsed["module_category"].append("Uncategorized")
            for tk in tag_cols:
                tag_cols[tk].append(False)

    for k, v in parsed.items():
        df[k] = v
    for tk, tv in tag_cols.items():
        df[f"tag_{tk}"] = tv

    # Surface case-only module duplicates (e.g. "Credit System" vs "Credit system"
    # where no form is canonical) to the Pending Module Review page.
    new_case_pending = detect_case_variant_groups(df["parsed_module"], store)
    if new_case_pending:
        print(
            f"\n📝 Surfaced {new_case_pending} case-variant module mapping(s) to "
            f"pending review (bucket: case_variants). Review in the 'Pending "
            f"Module Review' page of the dashboard."
        )

    # status_active and status_weight
    if status_col:
        df["status_active"] = df[status_col].apply(classify_status)
        df["status_weight"] = df[status_col].apply(classify_status_weight)
        print("\nStatus classification:")
        print(f"  Active bugs:   {df['status_active'].sum():,}")
        print(f"  Closed bugs:   {((df['status_weight'] == 0.5)).sum():,}")
        print(f"  Invalid bugs:  {((df['status_weight'] == 0.0)).sum():,}")
    else:
        df["status_active"] = True
        df["status_weight"] = 1.0
        print("\nWARNING: No 'Status' column found — all bugs marked active.")

    # Severity
    sev_col = next((c for c in df.columns if c.lower() == "severity"), None)
    if sev_col:
        sev_data = df[sev_col].apply(parse_severity)
        df["severity_num"] = sev_data.apply(lambda x: x[0])
        df["severity_weight"] = sev_data.apply(lambda x: x[1])
        df["severity_label"] = sev_data.apply(lambda x: x[2])

    # Priority label
    pri_col = next((c for c in df.columns if c.lower() == "priority"), None)
    if pri_col:
        df["priority_num"] = (
            pd.to_numeric(df[pri_col], errors="coerce").fillna(5).astype(int)
        )
        df["priority_label"] = df["priority_num"].map(PRIORITY_LABEL_MAP).fillna("N/A")

    # Repro probability
    repro_col = next(
        (c for c in df.columns if "repro" in c.lower() and "prob" in c.lower()), None
    )
    if repro_col:
        df["repro_rate"] = df[repro_col].apply(parse_repro)

    # Dates / days_to_close
    for dc in ["Create Date", "Closed Date"]:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")

    if "Create Date" in df.columns and "Closed Date" in df.columns:
        df["days_to_close"] = (df["Closed Date"] - df["Create Date"]).dt.days

    # builds_to_fix
    build_col = next(
        (c for c in df.columns if c.lower().strip() in ["build#", "build"]), None
    )
    close_build_col = next(
        (c for c in df.columns if c.lower().strip() in ["close build#", "close build"]),
        None,
    )
    if build_col and close_build_col:
        df["builds_to_fix"] = (
            pd.to_numeric(df[close_build_col], errors="coerce")
            - pd.to_numeric(df[build_col], errors="coerce")
        ).clip(lower=0)

    # Summary
    parsed_ok = df["parsed_module"].notna().sum()
    total = len(df)
    print("\nPARSING SUMMARY")
    print(f"Total bugs:            {total:,}")
    print(f"Successfully parsed:   {parsed_ok} ({parsed_ok / total * 100:.1f}%)")
    print(f"Unique modules:        {df['parsed_module'].dropna().nunique()}")
    if FUZZY_AVAILABLE:
        print(f"Fuzzy threshold used:  {fuzzy_threshold}%")
        if _ollama_matcher_enabled():
            print(f"Ollama matcher:        ON (model={_ollama_matcher_model()})")

    if _ollama_matcher_enabled() and store._ollama_disk_cache:
        store.save_ollama_cache(store._ollama_disk_cache)

    # ── Backfill blank suggestions in pending files via Ollama ────────────
    # Previous runs (or low-confidence Ollama results with null match) may
    # have left pending entries with empty suggestions.  Re-process them
    # through Ollama now so the Pending Review page always shows a best-guess.
    if _ollama_matcher_enabled() and FUZZY_AVAILABLE:
        blank_entries: list[tuple[Path, str, str]] = []  # (file, raw, version)
        for p in store.versions_dir.glob("*_pending.json"):
            pdata = store._load_json(p)
            ver = p.stem.replace("_pending", "")
            for raw, info in pdata.items():
                if not info.get("confirmed") and not info.get("suggested"):
                    blank_entries.append((p, raw, ver))

        if blank_entries:
            print(f"\nBackfilling {len(blank_entries)} blank suggestion(s) via Ollama...")
            model = _ollama_matcher_model()
            backfilled = 0
            for p, raw, ver in tqdm(blank_entries, desc="Backfilling suggestions",
                                    unit="entry", file=sys.stderr):
                top_k = process.extract(
                    raw, _CANONICAL_MODULES, scorer=fuzz.WRatio, limit=8
                )
                if not top_k:
                    continue
                candidates = [c for c, _, _ in top_k]
                result = _ollama_disambiguate(raw, candidates, model)
                if result and result.get("match"):
                    pdata = store._load_json(p)
                    if raw in pdata:
                        pdata[raw]["suggested"] = result["match"]
                        store._save_json(p, pdata)
                        backfilled += 1
            if backfilled:
                print(f"  \u2705 Backfilled {backfilled}/{len(blank_entries)} suggestions")
            else:
                print(f"  No suggestions found for blank entries")

    total_pending = 0
    for p in store.versions_dir.glob("*_pending.json"):
        pending_data = store._load_json(p)
        unconfirmed = {k: v for k, v in pending_data.items() if not v.get("confirmed")}
        if unconfirmed:
            ver_name = p.stem.replace("_pending", "")
            print(
                f"\nPENDING review for version {ver_name} ({len(unconfirmed)} entries):"
            )
            for raw, info in list(unconfirmed.items())[:10]:
                print(f"  '{raw}' → suggested: '{info['suggested']}'")
            if len(unconfirmed) > 10:
                print(f"  ... and {len(unconfirmed) - 10} more (see {p})")
            total_pending += len(unconfirmed)

    print("\nTop 10 modules:")
    for mod, cnt in df["parsed_module"].value_counts().head(10).items():
        print(f"  {mod:35s} {cnt:>5}")

    uncat = (
        df[df["module_category"] == "Uncategorized"]["parsed_module"]
        .dropna()
        .unique()
    )
    if len(uncat) > 0:
        print(
            f"\nUNCATEGORIZED MODULES ({len(uncat)}) — add to MODULE_CATEGORIES or confirm in dashboard:"
        )
        for m in sorted(uncat):
            print(f"  - {m}")
    else:
        print("\n✅ No uncategorized modules!")

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to: {output_path}")

    # ── Version catalogue ────────────────────────────────────────────────────
    # Sorted by recency (max Create Date per version group), NOT by version
    # string.  Sparse versions (< VERSION_SPARSE_THRESHOLD bugs) are ranked
    # after all real versions so they don't pollute default selections in the
    # dashboard or the 'latest' scope in fetch_from_n8n.py.
    cat = build_version_catalogue(df)
    cat_path = str(Path(output_path).with_name("version_catalogue.csv"))
    cat.to_csv(cat_path, index=False, encoding="utf-8-sig")

    print("\nVERSION CATALOGUE (sorted by recency, typo/sparse versions last):")
    print(f"  {'#':>3}  {'Version':<18}  {'Bugs':>5}  {'Latest Create Date':<22}  Sparse?")
    print(f"  {'─'*3}  {'─'*18}  {'─'*5}  {'─'*22}  {'─'*7}")
    for _, row in cat.iterrows():
        flag = "⚠️  sparse" if row["version_is_sparse"] else ""
        date_str = str(row["latest_create_date"])[:10] if pd.notna(row["latest_create_date"]) else "unknown"
        print(f"  {int(row['version_rank']):>3}  {str(row['parsed_version']):<18}  "
              f"{int(row['bug_count']):>5}  {date_str:<22}  {flag}")
    print(f"\nSaved version catalogue to: {cat_path}")
    n_sparse = cat["version_is_sparse"].sum()
    if n_sparse:
        sparse_list = cat[cat["version_is_sparse"]]["parsed_version"].tolist()
        print(f"⚠️  {n_sparse} sparse version(s) found (< {VERSION_SPARSE_THRESHOLD} bugs each):")
        for sv in sparse_list:
            print(f"     '{sv}' — likely a typo or one-off entry; excluded from default selection")

    if total_pending > 0:
        print(
            f"⚠️  {total_pending} module strings pending dashboard review in {store.versions_dir}/"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PDR-I ECL Export Parser v2.4")
    parser.add_argument("input", help="Input file: .xlsx, .csv, or .json (n8n webhook output)")
    parser.add_argument("output", help="Output CSV file")
    parser.add_argument(
        "--mapping-dir",
        default=None,
        help="Directory for per-version mapping files (default: <output_dir>/module_mappings)",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=85,
        help="Fuzzy match confidence threshold 0–100 (default: 85)",
    )
    parser.add_argument(
        "--ollama-matcher",
        action="store_true",
        help="Enable Ollama semantic re-ranking of fuzzy module suggestions. "
             "Equivalent to PDRI_OLLAMA_MATCHER=1.",
    )
    parser.add_argument(
        "--ollama-matcher-model",
        default=None,
        help=f"Ollama model for the matcher (default: {OLLAMA_MATCHER_DEFAULT_MODEL}). "
             "Equivalent to PDRI_OLLAMA_MATCHER_MODEL.",
    )
    parser.add_argument(
        "--embedding-matcher",
        action="store_true",
        help="Enable embedding-based canonical retrieval (uses nomic-embed-text). "
             "On by default when --ollama-matcher is set. "
             "Equivalent to PDRI_EMBEDDING_MATCHER=1.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=f"Ollama embedding model (default: {OLLAMA_EMBED_DEFAULT_MODEL}). "
             "Equivalent to PDRI_EMBEDDING_MODEL.",
    )
    args = parser.parse_args()
    if args.ollama_matcher:
        os.environ["PDRI_OLLAMA_MATCHER"] = "1"
    if args.ollama_matcher_model:
        os.environ["PDRI_OLLAMA_MATCHER_MODEL"] = args.ollama_matcher_model
    if args.embedding_matcher:
        os.environ["PDRI_EMBEDDING_MATCHER"] = "1"
    if args.embedding_model:
        os.environ["PDRI_EMBEDDING_MODEL"] = args.embedding_model
    parse_ecl_export(
        args.input,
        args.output,
        mapping_dir=args.mapping_dir,
        fuzzy_threshold=args.fuzzy_threshold,
    )


if __name__ == "__main__":
    main()