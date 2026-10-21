"""AT_PDRi cases for a TR's changed modules — listed, not yet run.

The focus list says which modules changed; the AT_PDRi repo holds the real
iOS automation, one file per feature in two suites: ``tests/new_bft_ios``
(quick build check) and
``tests/new_sft_ios`` (the full checklist). This module joins the two, so the
TR Test Focus tab can say which cases a build should trigger and give the exact
command. Nothing here executes a test, and nothing in AT_PDRi is modified.

**The unit is a file.** Every case is a ``pytest-dependency`` chain link
(``test_002`` depends on ``test_001``), so a single case cannot be selected on
its own; chains never cross a class, and every class in a feature file tests
that feature (entry points: shortcut / AI Magic / timeline, plus its own
cleanup), so the whole file is the smallest safe selection. AT_PDRi has no
module marker to select on — its Allure ``epic/feature/story`` labels are case
*age* cohorts assigned at runtime, invisible to ``-m``.

**Matching** is on the feature named in each file's docstring ("iOS Text to
Video SFT …"), normalised the same way as eCL's ``parsed_module`` so "Video
Enhancer" meets "Video Enhance" and "Replace BG" meets "Replace Background".
``MODULE_ALIASES`` holds the few renames normalisation cannot see. A module
with no file is reported, never guessed.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shlex
import warnings
from pathlib import Path

#: Product slug → env var that overrides its saved AT repo folder.
AT_REPOS = {"pdri": "AT_PDRI_DIR"}

#: Repo-local and git-ignored, like ``.ecl_token``: the folder is a fact about
#: this machine. Set once from the TR Test Focus tab and reused by every later
#: render and process. There is no default path — unset says so on the page.
AT_SETTINGS_FILE = Path(__file__).resolve().parent.parent / ".at_repos.json"

SUITES = (("BFT", "tests/new_bft_ios"), ("SFT", "tests/new_sft_ios"))

#: Files that are run scaffolding, not a feature. Install puts the build under
#: test on the device; the SFT sweep clears My Artwork after generators ran.
INSTALL_STEM = "install_from_testflight"
TEARDOWN_STEM = "clear_my_artwork"
NOT_A_FEATURE = {INSTALL_STEM, TEARDOWN_STEM, "launch_preference", "second_launch"}

#: eCL module key → AT feature key, for renames normalisation cannot see.
#: Keys are ``match_key`` output. Add here rather than loosening the matcher:
#: a prefix match would pull "AI Anime Video" into "AI Anime".
MODULE_ALIASES = {
    "audiodenoise": "hqaudiodenoise",      # "AI Audio Denoise"
    "vipbenefitpage": "vipbenefit",        # "VIP Benefit page"
    "taklingavatar": "talkingavatar",      # typo in eCL
    "magic": "magicentrycheck",            # "AI Magic" menu → entry check
    "speed": "adjustspeed",
}

_WORD_MAP = {"bg": "background", "facial": "face", "enhancer": "enhance"}


def _read_settings(settings_file: Path) -> dict:
    try:
        data = json.loads(settings_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def configured_at_dir(slug: str,
                      settings_file: Path = AT_SETTINGS_FILE) -> str:
    """The folder set for ``slug`` — env var first, then the saved file."""
    if slug not in AT_REPOS:
        return ""
    return (os.environ.get(AT_REPOS[slug])
            or str(_read_settings(settings_file).get(slug) or "")).strip()


def at_repo_dir(slug: str, settings_file: Path = AT_SETTINGS_FILE) -> Path | None:
    """The AT repo for ``slug``, or None when unset or no longer there."""
    value = configured_at_dir(slug, settings_file)
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_dir() else None


def at_repo_problem(folder: str) -> str:
    """Why ``folder`` is not an AT repo; '' when it is."""
    path = Path(str(folder or "").strip()).expanduser()
    if not str(folder or "").strip():
        return "No folder given."
    if not path.is_dir():
        return f"{path} is not a folder."
    if not any((path / rel).is_dir() for _, rel in SUITES):
        return (f"{path} has neither " + " nor ".join(rel for _, rel in SUITES)
                + " — is it the AT repo root?")
    return ""


def save_at_repo_dir(slug: str, folder: str,
                     settings_file: Path = AT_SETTINGS_FILE) -> Path:
    """Remember ``folder`` for ``slug``. Raises ValueError if it is not one."""
    if slug not in AT_REPOS:
        raise ValueError(f"No AT repo is used for {slug}.")
    problem = at_repo_problem(folder)
    if problem:
        raise ValueError(problem)
    path = Path(folder.strip()).expanduser().resolve()
    settings = _read_settings(settings_file)
    settings[slug] = str(path)
    tmp = settings_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(settings_file)
    return path


def match_key(name: str) -> str:
    """Normalise a module / feature name for matching.

    Drops parentheticals and anything after " - ", the "AI" prefix eCL uses
    inconsistently, trailing plurals, and spacing ("VideoEnhancer").
    """
    name = re.sub(r"\([^)]*\)", " ", str(name or "")).split(" - ")[0]
    words = []
    for word in re.findall(r"[a-z0-9]+", name.lower()):
        if word == "ai":
            continue
        word = _WORD_MAP.get(word, word)
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    key = "".join(words).replace("enhancer", "enhance")
    return MODULE_ALIASES.get(key, key)


def feature_name(docstring: str) -> str:
    """"iOS Text to Video SFT (Standard …)" → "Text to Video"."""
    line = (str(docstring or "").strip().splitlines() or [""])[0]
    line = re.sub(r"^iOS\s+", "", line)
    cut = re.search(r"\s(?:Shortcut|SFT|tests?)\b|\s\(via\b", line)
    return (line[:cut.start()] if cut else line).strip(" .")


def _stem(path: Path) -> str:
    """test_16_text_to_video_ios.py → text_to_video."""
    return re.sub(r"^test_\d+_|_ios$", "", path.stem)


def _doc_line(node) -> str:
    lines = (ast.get_docstring(node) or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def index_at_suites(repo: Path) -> list[dict]:
    """One entry per test file in both suites, with its classes and cases."""
    files = []
    for suite, rel_dir in SUITES:
        for path in sorted((repo / rel_dir).glob("test_*.py")):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            rel = path.relative_to(repo).as_posix()
            docstring = ast.get_docstring(tree) or ""
            classes = []
            for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
                if not cls.name.startswith("Test"):
                    continue          # _TTVBase-style helpers hold no cases
                cases = [
                    {"node_id": f"{rel}::{cls.name}::{fn.name}",
                     "name": fn.name, "title": _doc_line(fn) or fn.name}
                    for fn in cls.body
                    if isinstance(fn, ast.FunctionDef) and fn.name.startswith("test")
                ]
                if cases:
                    classes.append({"name": cls.name,
                                    "node_id": f"{rel}::{cls.name}",
                                    "description": _doc_line(cls),
                                    "cases": cases})
            stem = _stem(path)
            feature = feature_name(docstring)
            files.append({
                "suite": suite, "file": rel, "stem": stem,
                "feature": feature,
                "key": "" if stem in NOT_A_FEATURE else match_key(feature),
                "cost_warning": "COST WARNING" in docstring,
                "classes": classes,
                "case_count": sum(len(c["cases"]) for c in classes),
            })
    return files


def at_cases_for_focus(focus: dict, files: list[dict]) -> dict:
    """Which AT files this build should run, and why.

    Modules changed in this build come first; modules fixed earlier in the
    same version follow as regression. Each module appears once, under its
    first reason. Unmatched modules are returned so the gap is visible.
    """
    wanted: list[tuple[str, str, bool]] = []
    for row in focus.get("change_surface") or []:
        wanted.append((row["module"], "changed in this build", True))
    version = focus.get("version") or "this version"
    for row in focus.get("fixed_earlier_in_version") or []:
        wanted.append((row["module"], f"fixed earlier in {version}", False))

    by_key: dict[str, list[dict]] = {}
    for entry in files:
        if entry["key"]:
            by_key.setdefault(entry["key"], []).append(entry)

    selected, unmatched, seen_modules, seen_files = [], [], set(), set()
    for module, reason, changed in wanted:
        if module.casefold() in seen_modules:
            continue
        seen_modules.add(module.casefold())
        matches = by_key.get(match_key(module)) or []
        if not matches:
            unmatched.append({"module": module, "reason": reason})
        for entry in matches:
            if entry["file"] in seen_files:
                continue
            seen_files.add(entry["file"])
            selected.append({**entry, "module": module, "reason": reason,
                             "changed_in_build": changed})
    return {"selected": selected, "unmatched": unmatched}


def split_close_build(close_build: str) -> tuple[str, str]:
    """16.8.0.189304 → ("16.8.0", "189304"), the TestFlight version + build."""
    match = re.fullmatch(r"\s*(\d+\.\d+\.\d+)\.(\d+)\s*", str(close_build or ""))
    return (match.group(1), match.group(2)) if match else ("", "")


def pytest_commands(selected: list[dict], files: list[dict], close_build: str,
                    device: str = "iphone17Air") -> dict[str, str]:
    """Suite → the command that would run it against this Close Build.

    Both suites cover every matched module: changed in this build, and fixed
    earlier in this version as regression. Each command installs the Close
    Build first; SFT ends with the My Artwork sweep its generators rely on.
    """
    version, build = split_close_build(close_build)
    scaffolding = {(f["suite"], f["stem"]): f["file"] for f in files}
    commands = {}
    for suite, _ in SUITES:
        chosen = [e["file"] for e in selected if e["suite"] == suite]
        if not chosen:
            continue
        paths = []
        install = scaffolding.get((suite, INSTALL_STEM))
        if install and version:
            paths.append(install)
        paths += chosen
        teardown = scaffolding.get((suite, TEARDOWN_STEM))
        if suite == "SFT" and teardown:
            paths.append(teardown)
        args = " ".join(shlex.quote(p) for p in paths)
        pinned = (f" --app-version={version} --build-number={build}"
                  if version else "")
        commands[suite] = (f"PLATFORM=ios python -m pytest {args} "
                           f"--device={shlex.quote(device)}{pinned}")
    return commands
