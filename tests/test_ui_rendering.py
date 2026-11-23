"""Markdown rendering traps in the Streamlit pages.

These exist because a bug of exactly this kind survived a visual review: the
adjuster workbench showed "Range 85,210" instead of "Range $10,653 – $85,210",
because Streamlit's markdown reads a pair of dollar signs as LaTeX maths and
silently swallowed everything between them. Nothing raised, nothing logged —
the number simply disappeared.

A page that renders without an exception is not the same as a page that
renders correctly, so this is checked statically across every page.
"""

from __future__ import annotations

import ast

import pytest

from claimpilot import config

PAGE_FILES = sorted((config.ROOT / "pages").glob("*.py")) + [
    config.ROOT / "streamlit_app.py"
]

# Calls whose first argument Streamlit renders as markdown.
MARKDOWN_CALLS = {
    "markdown",
    "write",
    "caption",
    "info",
    "warning",
    "error",
    "success",
    "title",
    "subheader",
    "header",
}


def _literal_dollar_count(source: str) -> int:
    """Count dollar signs that are not escaped for markdown."""
    return source.replace("\\\\$", "").replace("\\$", "").count("$")


def _markdown_arguments(path):
    """Yield (lineno, call name, source) for each markdown-rendered argument."""
    text = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", "")
        if name not in MARKDOWN_CALLS or not node.args:
            continue
        source = ast.get_source_segment(text, node.args[0]) or ""
        yield node.lineno, name, source


@pytest.mark.parametrize("path", PAGE_FILES, ids=[p.stem for p in PAGE_FILES])
def test_no_unescaped_dollar_pairs_in_markdown(path):
    """Two bare '$' in one markdown string become LaTeX and eat the text."""
    offenders = [
        f"{path.name}:{lineno} st.{name}()"
        for lineno, name, source in _markdown_arguments(path)
        if _literal_dollar_count(source) >= 2
    ]
    assert not offenders, (
        "Unescaped '$' pairs render as LaTeX maths and silently drop the "
        "text between them. Escape them as '\\\\$' in the f-string: "
        + "; ".join(offenders)
    )


def test_the_check_would_catch_the_bug_it_was_written_for():
    """Negative control: the real pre-fix string must be flagged."""
    broken = 'f"Range ${low:,.0f} – ${high:,.0f}"'
    fixed = 'f"Range \\\\${low:,.0f} – \\\\${high:,.0f}"'
    assert _literal_dollar_count(broken) >= 2
    assert _literal_dollar_count(fixed) == 0


def test_a_single_dollar_is_not_flagged():
    """One '$' cannot open and close a maths span, so it is fine."""
    assert _literal_dollar_count('f"Suggested reserve ${amount:,.0f}"') == 1
