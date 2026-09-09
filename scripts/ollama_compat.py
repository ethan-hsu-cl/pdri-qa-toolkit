#!/usr/bin/env python3
"""Shared Ollama request/response helpers for reasoning-capable models.

Thinking models (qwen3, deepseek-r1, and gemma4's reasoning variants) spend
their first tokens on internal reasoning. Depending on the model and the Ollama
version that reasoning arrives in one of three shapes:

1. suppressed entirely, when the request sets ``"think": false``;
2. in a separate ``thinking`` field, leaving ``response``/``content`` empty;
3. inline in the answer, wrapped in ``<think>`` … ``</think>`` tags.

A caller that only does ``json.loads(envelope["response"])`` fails on shapes 2
and 3 — and in this pipeline that failure is silent: ``ai_risk_scorer`` catches
it and falls back to heuristic scoring, so a whole run looks successful while
no model was ever consulted. These helpers make every call site handle all
three, so any model can be pointed at any stage.

``"think": false`` is accepted and ignored by non-reasoning models, so sending
it unconditionally is safe.
"""

import json
import re

__all__ = [
    "THINK_TAG_RE",
    "extract_json",
    "response_text",
    "strip_meta_preamble",
    "strip_thinking",
    "with_thinking_disabled",
]

# Matches a complete <think>…</think> block, and also an unterminated one when
# the answer was truncated mid-reasoning.
THINK_TAG_RE = re.compile(
    r"<think(?:ing)?>.*?(?:</think(?:ing)?>|\Z)",
    flags=re.DOTALL | re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$",
    flags=re.DOTALL,
)


def with_thinking_disabled(payload: dict) -> dict:
    """Return the payload with reasoning output turned off.

    An explicit ``think`` already set by the caller is left alone.
    """
    if "think" not in payload:
        payload = {**payload, "think": False}
    return payload


def strip_thinking(text: object) -> str:
    """Remove inline reasoning blocks and markdown fences from model output."""
    cleaned = THINK_TAG_RE.sub(" ", str(text or "")).strip()
    fenced = _CODE_FENCE_RE.match(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    return cleaned


def response_text(envelope: dict) -> str:
    """Return the answer text from an /api/generate or /api/chat envelope.

    Falls back to the ``thinking`` field: some models leave the answer there
    when they are asked to reason and the answer itself is short.
    """
    if not isinstance(envelope, dict):
        return ""
    message = envelope.get("message")
    if isinstance(message, dict):
        candidates = (message.get("content"), message.get("thinking"))
    else:
        candidates = (envelope.get("response"), envelope.get("thinking"))
    for candidate in candidates:
        text = strip_thinking(candidate)
        if text:
            return text
    return ""


def _scan_balanced(source: str, opener: str, closer: str):
    """Return the first balanced opener…closer span that parses as JSON."""
    start = source.find(opener)
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(source)):
            char = source[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(source[start:index + 1])
                    except json.JSONDecodeError:
                        break
        start = source.find(opener, start + 1)
    return None


def extract_json(text: object, *, expect: str = "any"):
    """Parse JSON from model output, tolerating reasoning and prose wrappers.

    ``expect`` is "object", "array", or "any" and selects which bracket the
    salvage pass scans for. Raises ``json.JSONDecodeError`` when nothing
    parses, so callers keep their existing error handling.
    """
    raw = str(text or "").strip()
    cleaned = strip_thinking(text)
    if not cleaned and not raw:
        raise json.JSONDecodeError("no JSON content in model response", "", 0)
    # An unterminated <think> tag swallows everything after it, so always try
    # the raw text as well before giving up.
    sources = [source for source in dict.fromkeys((cleaned, raw)) if source]
    for source in sources:
        try:
            return json.loads(source)
        except json.JSONDecodeError:
            continue
    brackets = {"object": [("{", "}")], "array": [("[", "]")]}.get(
        expect, [("{", "}"), ("[", "]")]
    )
    for source in sources:
        for opener, closer in brackets:
            found = _scan_balanced(source, opener, closer)
            if found is not None:
                return found
    raise json.JSONDecodeError(
        "no parsable JSON in model response", sources[0], 0
    )


# Reasoning models often open a free-text answer by restating the task
# ("Okay, the user is a QA lead who needs…") before the answer proper.
_META_PREAMBLE_RE = re.compile(
    r"^\s*(okay|ok|alright|sure|certainly|hmm|let me|let's|first,? (?:i|we)|"
    r"i (?:need|should|will|'ll)|we (?:need|should|are asked)|"
    r"the user (?:is|wants|needs|asked)|as an? (?:ai|assistant)|"
    r"here(?:'s| is) (?:the|a)\b)",
    flags=re.IGNORECASE,
)


_SELF_REFERENCE_RE = re.compile(
    r"\b(i need to|i should|i'?ll|i will|i'?m going to|let me|we need to|"
    r"the user (?:is|wants|needs|asked)|as an? (?:ai|assistant)|"
    r"my (?:answer|response|task)|this (?:answer|response) (?:should|must))\b",
    flags=re.IGNORECASE,
)


def strip_meta_preamble(text: object) -> str:
    """Drop leading paragraphs where the model narrates the task to itself.

    Only leading meta paragraphs are removed, and never the whole answer: if
    every paragraph looks like preamble the original text is returned so a
    caller never silently loses content.
    """
    cleaned = strip_thinking(text)
    if not cleaned:
        return ""
    paragraphs = [para for para in re.split(r"\n\s*\n", cleaned) if para.strip()]
    kept = list(paragraphs)
    while kept and _META_PREAMBLE_RE.match(kept[0]):
        kept.pop(0)
    if not kept:
        return cleaned
    # Reasoning also leaks as a first sentence inside an otherwise fine
    # paragraph ("For testing priorities, I need to be specific.").
    sentences = re.split(r"(?<=[.!?])\s+", kept[0].strip())
    while len(sentences) > 1 and _SELF_REFERENCE_RE.search(sentences[0]):
        sentences.pop(0)
    kept[0] = " ".join(sentences).strip()
    result = "\n\n".join(part for part in kept if part).strip()
    return result or cleaned
