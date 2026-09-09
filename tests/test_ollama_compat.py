"""Model-compatibility helpers: reasoning output must never be parsed as an answer."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ollama_compat as oc  # noqa: E402


def test_think_flag_is_added_but_never_overridden():
    assert oc.with_thinking_disabled({"model": "m"})["think"] is False
    assert oc.with_thinking_disabled({"model": "m", "think": True})["think"] is True


@pytest.mark.parametrize("raw, expected", [
    ('{"impact": 4}', {"impact": 4}),
    ('<think>weighing {"impact": 1}</think>{"impact": 4}', {"impact": 4}),
    ('<THINK>upper case</THINK>{"impact": 4}', {"impact": 4}),
    ('```json\n{"impact": 4}\n```', {"impact": 4}),
    ('Sure! {"impact": 4} hope that helps', {"impact": 4}),
    # An unterminated block must not swallow a usable answer.
    ('<think>reasoning that never closes\n{"impact": 4}', {"impact": 4}),
])
def test_extract_json_survives_reasoning_wrappers(raw, expected):
    assert oc.extract_json(raw, expect="object") == expected


def test_extract_json_handles_braces_inside_strings():
    assert oc.extract_json('[{"note": "a } brace"}]', expect="array") == [
        {"note": "a } brace"}
    ]


def test_extract_json_raises_when_there_is_no_json():
    with pytest.raises(json.JSONDecodeError):
        oc.extract_json("I could not complete that request")


def test_response_text_falls_back_to_the_thinking_field():
    # Some models leave `content` empty and put the answer in `thinking`.
    assert oc.response_text({"message": {"content": "", "thinking": "Subtitle bugs"}}) == "Subtitle bugs"
    assert oc.response_text({"response": "<think>x</think>Answer"}) == "Answer"
    assert oc.response_text({}) == ""


def test_strip_meta_preamble_removes_self_narration_only():
    assert oc.strip_meta_preamble(
        "Okay, the user wants a briefing.\n\nExport paths are unstable."
    ) == "Export paths are unstable."
    assert oc.strip_meta_preamble(
        "For testing priorities, I need to be specific. Clear is the common thread."
    ) == "Clear is the common thread."
    # Real content is never dropped, even when it opens with a stray phrase.
    assert oc.strip_meta_preamble("Export paths are unstable.") == "Export paths are unstable."
    assert oc.strip_meta_preamble("Let me think.") == "Let me think."


def test_every_ollama_caller_disables_thinking():
    """A stage that forgets this silently degrades: no error, no model output."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in (
        "ai_risk_scorer.py", "cluster_bugs.py", "predict_defects.py",
        "parse_ecl_export.py", "generate_release_pulse_scenarios.py",
        "summarize_tr_bugs.py",
    ):
        source = (scripts / name).read_text(encoding="utf-8")
        assert '"think": False' in source or "with_thinking_disabled" in source, name
