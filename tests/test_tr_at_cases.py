import json
from pathlib import Path

import pytest

from scripts.tr_at_cases import (
    AT_REPOS, at_cases_for_focus, at_repo_dir, configured_at_dir, feature_name,
    index_at_suites, match_key, pytest_commands, save_at_repo_dir,
    split_close_build,
)


def feature_file(doc: str, cls: str, cases: int, base: bool = False) -> str:
    body = [f'"""{doc}\n\nMore prose with a stray \\O escape."""', "import pytest", ""]
    if base:
        body += ["class _Base:", "    def _helper(self):", "        pass", ""]
    body += [f"class {cls}{'(_Base)' if base else ''}:",
             f'    """{cls} entry point."""']
    for n in range(1, cases + 1):
        body += ["    @pytest.mark.dependency()", f"    def test_{n:03d}(self):",
                 f'        """Step {n} / do thing - expect thing"""', "        pass"]
    return "\n".join(body) + "\n"


def make_at_repo(tmp_path: Path) -> Path:
    bft = tmp_path / "tests" / "new_bft_ios"
    sft = tmp_path / "tests" / "new_sft_ios"
    bft.mkdir(parents=True)
    sft.mkdir(parents=True)
    files = {
        bft / "test_00_install_from_testflight_ios.py":
            ("iOS TestFlight App Installation tests.", "Test_install_ios", 2),
        bft / "test_07_ai_anime_video_ios.py":
            ("iOS AI Anime Video Shortcut tests for PowerDirector app.",
             "Test_aav_ios", 3),
        bft / "test_22_ai_anime_ios.py":
            ("iOS AI Anime tests for PowerDirector app (via AI Magic).",
             "Test_anime_ios", 2),
        bft / "test_10_replace_bg_ios.py":
            ("iOS Replace BG (Background) Shortcut tests.", "Test_rbg_ios", 2),
        sft / "test_00_install_from_testflight_ios.py":
            ("iOS TestFlight App Installation tests.", "Test_install_ios", 2),
        sft / "test_16_text_to_video_ios.py":
            ("iOS Text to Video SFT (Standard Functional Test).\n\nCOST WARNING: "
             "credits.", "Test_ttv_sft_ios", 4),
        sft / "test_12_video_enhance_ios.py":
            ("iOS Video Enhance SFT (Standard Functional Test).",
             "Test_ve_sft_ios", 2),
        sft / "test_99_clear_my_artwork_ios.py":
            ("iOS SFT suite teardown: clear the launcher's My Artwork page.",
             "Test_cleanup_ios", 1),
    }
    for path, (doc, cls, n) in files.items():
        path.write_text(feature_file(doc, cls, n, base="ttv" in cls),
                        encoding="utf-8")
    return tmp_path


def test_feature_names_come_from_the_file_docstring():
    assert feature_name("iOS Text to Video SFT (Standard Functional Test)") == \
        "Text to Video"
    assert feature_name("iOS AI Anime tests for PowerDirector app (via AI Magic).") \
        == "AI Anime"
    assert feature_name("iOS Replace BG (Background) Shortcut tests") == \
        "Replace BG (Background)"


def test_ecl_module_names_meet_at_feature_names():
    assert match_key("Video Enhancer") == match_key("Video Enhance")
    assert match_key("VideoEnhancer") == match_key("Video Enhance")
    assert match_key("AI Video Enhancer") == match_key("Video Enhance")
    assert match_key("Replace BG (Background)") == match_key("Replace Background")
    assert match_key("Face Reshape") == match_key("Facial Reshape")
    assert match_key("Auto Caption") == match_key("Auto Captions")
    assert match_key("AI Audio Denoise") == match_key("HQ Audio Denoise")
    # Exact keys, never prefixes: AI Anime is not AI Anime Video.
    assert match_key("AI Anime") != match_key("AI Anime Video")


def test_index_counts_cases_per_class_and_skips_helper_bases(tmp_path):
    repo = make_at_repo(tmp_path)
    files = {f["file"]: f for f in index_at_suites(repo)}
    ttv = files["tests/new_sft_ios/test_16_text_to_video_ios.py"]
    assert [c["name"] for c in ttv["classes"]] == ["Test_ttv_sft_ios"]
    assert ttv["case_count"] == 4
    assert ttv["cost_warning"] is True
    case = ttv["classes"][0]["cases"][0]
    assert case["node_id"] == (
        "tests/new_sft_ios/test_16_text_to_video_ios.py::Test_ttv_sft_ios::test_001")
    assert case["title"] == "Step 1 / do thing - expect thing"
    # Scaffolding is indexed but never matched to a module.
    assert files["tests/new_sft_ios/test_99_clear_my_artwork_ios.py"]["key"] == ""


def focus():
    return {
        "version": "16.8.0",
        "change_surface": [{"module": "Text to Video"},
                           {"module": "AI Anime Video"},
                           {"module": "Timeline"}],
        "fixed_earlier_in_version": [{"module": "AI Anime Video"},
                                     {"module": "Video Enhancer"},
                                     {"module": "Replace BG"}],
    }


def test_changed_modules_first_regression_after_and_gaps_reported(tmp_path):
    files = index_at_suites(make_at_repo(tmp_path))
    result = at_cases_for_focus(focus(), files)
    assert [(e["module"], e["suite"], e["changed_in_build"])
            for e in result["selected"]] == [
        ("Text to Video", "SFT", True),
        ("AI Anime Video", "BFT", True),   # not AI Anime: exact match only
        ("Video Enhancer", "SFT", False),
        ("Replace BG", "BFT", False),
    ]
    assert result["unmatched"] == [
        {"module": "Timeline", "reason": "changed in this build"}]


def test_commands_install_the_close_build_and_cover_regression_too(tmp_path):
    files = index_at_suites(make_at_repo(tmp_path))
    selected = at_cases_for_focus(focus(), files)["selected"]
    commands = pytest_commands(selected, files, "16.8.0.189304")
    assert commands["BFT"] == (
        "PLATFORM=ios python -m pytest "
        "tests/new_bft_ios/test_00_install_from_testflight_ios.py "
        "tests/new_bft_ios/test_07_ai_anime_video_ios.py "
        "tests/new_bft_ios/test_10_replace_bg_ios.py "
        "--device=iphone17Air --app-version=16.8.0 --build-number=189304")
    # Video Enhance was fixed earlier, not changed now: SFT still covers it.
    assert commands["SFT"] == (
        "PLATFORM=ios python -m pytest "
        "tests/new_sft_ios/test_00_install_from_testflight_ios.py "
        "tests/new_sft_ios/test_16_text_to_video_ios.py "
        "tests/new_sft_ios/test_12_video_enhance_ios.py "
        "tests/new_sft_ios/test_99_clear_my_artwork_ios.py "
        "--device=iphone17Air --app-version=16.8.0 --build-number=189304")


def test_unparseable_build_skips_the_install_rather_than_guessing(tmp_path):
    files = index_at_suites(make_at_repo(tmp_path))
    selected = at_cases_for_focus(focus(), files)["selected"]
    assert split_close_build("v0908 [abc1234]") == ("", "")
    command = pytest_commands(selected, files, "v0908 [abc1234]")["BFT"]
    assert "install_from_testflight" not in command
    assert "--app-version" not in command


def test_no_folder_set_means_no_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("AT_PDRI_DIR", raising=False)
    settings = tmp_path / ".at_repos.json"
    assert configured_at_dir("pdri", settings) == ""
    assert at_repo_dir("pdri", settings) is None


def test_a_saved_folder_is_remembered_for_later_reads(tmp_path, monkeypatch):
    monkeypatch.delenv("AT_PDRI_DIR", raising=False)
    repo = make_at_repo(tmp_path / "AT_PDRi")
    settings = tmp_path / ".at_repos.json"
    saved = save_at_repo_dir("pdri", f"  {repo}  ", settings)
    assert saved == repo.resolve()
    # A fresh read — as a later render or another process would do.
    assert at_repo_dir("pdri", settings) == repo.resolve()
    assert json.loads(settings.read_text()) == {"pdri": str(repo.resolve())}


def test_saving_keeps_other_products_entries(tmp_path, monkeypatch):
    monkeypatch.setitem(AT_REPOS, "phdi", "AT_PHDI_DIR")
    repo = make_at_repo(tmp_path / "AT")
    settings = tmp_path / ".at_repos.json"
    settings.write_text(json.dumps({"phdi": "/elsewhere"}))
    save_at_repo_dir("pdri", str(repo), settings)
    assert json.loads(settings.read_text())["phdi"] == "/elsewhere"


def test_a_folder_that_is_not_an_at_repo_is_refused(tmp_path):
    settings = tmp_path / ".at_repos.json"
    for bad in ("", str(tmp_path / "missing"), str(tmp_path)):
        with pytest.raises(ValueError):
            save_at_repo_dir("pdri", bad, settings)
    assert not settings.exists()
    with pytest.raises(ValueError):
        save_at_repo_dir("phdi", str(make_at_repo(tmp_path / "x")), settings)


def test_a_saved_folder_that_disappeared_reads_as_unset_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("AT_PDRI_DIR", raising=False)
    settings = tmp_path / ".at_repos.json"
    settings.write_text(json.dumps({"pdri": str(tmp_path / "gone")}))
    assert configured_at_dir("pdri", settings) == str(tmp_path / "gone")
    assert at_repo_dir("pdri", settings) is None


def test_env_var_overrides_the_saved_folder(tmp_path, monkeypatch):
    repo = make_at_repo(tmp_path / "AT")
    settings = tmp_path / ".at_repos.json"
    settings.write_text(json.dumps({"pdri": str(tmp_path / "gone")}))
    monkeypatch.setenv("AT_PDRI_DIR", str(repo))
    assert at_repo_dir("pdri", settings) == repo
