"""Every bug code printed on the dashboard opens its eBug page."""
import pandas as pd
import pytest

from scripts.ecl_links import (
    ECL_BUG_URL, bug_link, bug_url, link_column_values, linkify_bug_codes,
)

CODE = "DRI261882-0021"
LINK = f"[`{CODE}`]({ECL_BUG_URL.format(bug_code=CODE)})"


def test_bug_link_keeps_the_code_in_backticks_inside_the_label():
    """The focus list relies on the backticks to wrap a long run of codes."""
    assert bug_link(CODE) == LINK
    assert bug_link(CODE, code_style=False) == f"[{CODE}]({bug_url(CODE)})"


def test_bare_and_backticked_codes_both_become_links():
    """A run published before linking wrote its codes as plain code spans."""
    assert linkify_bug_codes(f"`{CODE}`") == LINK
    assert linkify_bug_codes(f"see {CODE} now") == f"see {LINK} now"


def test_linkifying_twice_does_not_nest_links():
    """The live page linkifies a focus list that already links its codes."""
    assert linkify_bug_codes(linkify_bug_codes(f"`{CODE}`")) == LINK


def test_an_existing_link_is_left_alone():
    text = f"[{CODE}](https://example.test/x)"
    assert linkify_bug_codes(text) == text


def test_a_url_already_carrying_the_code_is_not_rewritten():
    """The eCL URL contains the code; linking inside it would break the href."""
    text = f"https://ecl.cyberlink.com/x?BugCode={CODE}"
    assert linkify_bug_codes(text) == text


def test_code_fences_and_mixed_code_spans_stay_verbatim():
    """A code quoted inside a snippet is a snippet, not a link."""
    assert linkify_bug_codes(f"```\n{CODE}\n```") == f"```\n{CODE}\n```"
    assert linkify_bug_codes(f"`grep {CODE} log`") == f"`grep {CODE} log`"


@pytest.mark.parametrize(
    "code", ["VDE232308-0034", "PRP265229-0033", "CP1230907-0003", "PHI262604-0218"]
)
def test_every_product_prefix_in_the_exports_is_recognised(code):
    """Prefixes run two to four letters across the eight products."""
    assert linkify_bug_codes(code) == bug_link(code)


def test_prose_that_merely_looks_like_a_code_is_left_alone():
    for text in ("build 16.8.0.189063", "TR260910-067", "2026-09-21"):
        assert linkify_bug_codes(text) == text


def test_focus_list_links_every_code_it_prints():
    """No bug code on the focus list should need pasting into eCL by hand."""
    import re

    from scripts.tr_focus_view import build_focus_list, focus_list_markdown
    from tests.test_tr_focus_view import focus_details, focus_analysis

    md = focus_list_markdown(build_focus_list(focus_details(), focus_analysis()))
    for code in ("B1", "B2", "A-TODO"):
        assert re.search(rf"(?<!\[)`{re.escape(code)}`(?!\])", md) is None, (
            f"{code} is printed unlinked"
        )
        assert bug_link(code) in md


def test_a_single_code_column_links_and_a_multi_code_one_does_not():
    """``show_dataframe`` decides from the values, not the column name."""
    assert link_column_values([CODE, "", None]) == [bug_url(CODE), "", ""]
    assert link_column_values([f"{CODE} · VDE232308-0034", ""]) is None
    assert link_column_values(["Creator Music", "Timeline"]) is None
    assert link_column_values(["", None]) is None


def test_a_column_of_pandas_nan_is_not_treated_as_a_code():
    """``astype(str)`` turns a missing cell into the string "nan"."""
    column = pd.Series([CODE, float("nan")]).astype(str)
    assert link_column_values(column) == [bug_url(CODE), ""]
    assert link_column_values(pd.Series([float("nan")]).astype(str)) is None


def test_a_missing_code_links_to_nothing():
    """Callers fall back to "Unknown"; ``str(None)`` would link the word None."""
    assert bug_link(None) == ""
    assert bug_link("") == ""
    assert bug_link("  ") == ""


def test_tr_link_uses_ecls_own_router_for_both_code_formats():
    """The ASP handler redirects new codes to IF3 and serves old ones itself."""
    from scripts.ecl_links import ECL_TR_URL, TR_CODE_RE, tr_link, tr_url

    for code in ("TR260922-0013", "TR260821-053"):
        assert TR_CODE_RE.fullmatch(code)
        assert tr_url(code) == ECL_TR_URL.format(tr_code=code)
        assert tr_link(code) == f"[`{code}`]({tr_url(code)})"
    assert tr_link("") == "" and tr_link(None) == ""
