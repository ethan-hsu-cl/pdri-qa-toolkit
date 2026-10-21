"""One place that knows how a BugCode becomes an eCL eBug URL.

Bug codes are printed in a dozen places — the focus list file, the AI
narrative, checklist bodies, diagnostics tables — and each one used to decide
for itself whether to link. Some did, most did not, so the same code was a
link on one line of the page and dead text on the next.

Two entry points cover every case:

``bug_link`` when the caller already knows the string is a BugCode (it came out
of a ``bug_code`` field), and ``linkify_bug_codes`` when the codes are buried in
prose the caller did not write — LLM output, RD fix notes, or a ``focus_list.md``
archived by a run that predates linking.
"""
import re

ECL_BUG_URL = (
    "https://ecl.cyberlink.com/Ebug/eBugHandle/HandleMainEbug2.asp?BugCode={bug_code}"
)

#: ``DRI261882-0021``. Every product prefix seen in the parsed exports is two to
#: four letters (DRI, VDE, PRP, PDP, DRA, PHI, PHA, VVG, VCI, CDP, POA, CP …),
#: then a 6-7 digit date-ish serial, then a 4-digit sequence.
_BUG_CODE = r"[A-Z]{2,4}\d{6,7}-\d{4}"
BUG_CODE_RE = re.compile(_BUG_CODE)

#: eCL's own TR entry point. It routes by code: new-format TRs (4-digit serial,
#: ``TR260922-0013``) are redirected to the IF3 form view
#: ``/IF3/tr/BPM/FormView/<code>``, old ASP ones (``TR260821-053``) are served
#: in place. Linking the IF3 path directly would break every old-format TR.
ECL_TR_URL = "https://ecl.cyberlink.com/TR/TRHandle/HandleMainTR.asp?TRCode={tr_code}"

TR_CODE_RE = re.compile(r"TR\d{6}-\d{3,4}")

#: Spans ``linkify_bug_codes`` must not touch, in match order. A markdown link
#: is protected so the pass is idempotent: a code linked on a previous call sits
#: inside one and is skipped rather than nested. A backtick span is protected
#: *unless* it holds nothing but a bug code — that exact shape is how the focus
#: list writes them, and rewriting it to a linked code span is the whole point.
_PROTECTED = re.compile(
    r"```[\s\S]*?```"
    r"|~~~[\s\S]*?~~~"
    r"|!?\[[^\]\n]*\]\([^)\n]*\)"
    r"|<[^>\s]*>"
    r"|https?://\S+"
    rf"|`(?!{_BUG_CODE}`)[^`\n]*`"
)

#: Inside an unprotected gap: a lone code span, or a bare code.
_LINKABLE = re.compile(rf"`({_BUG_CODE})`|({_BUG_CODE})")


def bug_url(code: str) -> str:
    """The eCL eBug page for ``code``."""
    return ECL_BUG_URL.format(bug_code=str(code).strip())


def bug_link(code: str, code_style: bool = True) -> str:
    """``code`` as a markdown link to its eBug page.

    ``code_style`` keeps the backticks the focus list relies on to wrap a run
    of codes instead of forming one unbreakable line; the backticks sit inside
    the link label, so the code still reads as a code and still opens eCL.
    """
    text = "" if code is None else str(code).strip()
    if not text:
        return ""
    label = f"`{text}`" if code_style else text
    return f"[{label}]({bug_url(text)})"


def tr_url(code: str) -> str:
    """The eCL page for Testing Request ``code``."""
    return ECL_TR_URL.format(tr_code=str(code).strip())


def tr_link(code: str, code_style: bool = True) -> str:
    """``code`` as a markdown link to its TR page; same shape as ``bug_link``."""
    text = "" if code is None else str(code).strip()
    if not text:
        return ""
    label = f"`{text}`" if code_style else text
    return f"[{label}]({tr_url(text)})"


def linkify_bug_codes(text: str) -> str:
    """Link every bug code in ``text`` that is not already linked.

    Safe to run twice, and safe to run on prose nobody vetted: code fences,
    existing links, URLs and non-trivial backtick spans are left exactly as
    written, so a code quoted inside a snippet stays a snippet.
    """
    source = str(text or "")
    if not source:
        return source

    def _link(match: re.Match) -> str:
        return bug_link(match.group(1) or match.group(2))

    out, cursor = [], 0
    for guard in _PROTECTED.finditer(source):
        out.append(_LINKABLE.sub(_link, source[cursor:guard.start()]))
        out.append(guard.group(0))
        cursor = guard.end()
    out.append(_LINKABLE.sub(_link, source[cursor:]))
    return "".join(out)


def link_column_values(values) -> list[str] | None:
    """URLs for a table column, or ``None`` if it is not a column of bug codes.

    A dataframe cell carries one URL, not markdown, so a column only becomes a
    link when every filled value is a single bug code. ``"A · B · C"`` and free
    text that merely mentions a code both fall through and stay text.

    Kept here, away from Streamlit, because the decision is the interesting
    part: naming the columns instead is what let ``Source bug`` and
    ``analog_bug_code`` render as dead text for as long as they did.
    """
    cells = ["" if value is None else str(value).strip() for value in values]
    filled = [cell for cell in cells if cell and cell.lower() != "nan"]
    if not filled or not all(BUG_CODE_RE.fullmatch(cell) for cell in filled):
        return None
    return [bug_url(cell) if cell and cell.lower() != "nan" else "" for cell in cells]
