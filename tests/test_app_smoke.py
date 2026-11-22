"""Deployment smoke tests: every page must load with no API key present.

These mirror what Streamlit Community Cloud does on a cold start, so a broken
page is caught by `make test` rather than after deploying.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from claimpilot import config

APP = config.ROOT / "streamlit_app.py"
PAGES = sorted((config.ROOT / "pages").glob("*.py"))

TIMEOUT = 60


def _run(path: Path) -> AppTest:
    app = AppTest.from_file(str(path), default_timeout=TIMEOUT)
    return app.run()


def test_entry_point_exists_at_repo_root():
    """Streamlit Community Cloud requires this exact layout."""
    assert APP.exists(), "streamlit_app.py must be at the repo root"


def test_main_app_runs_without_exception(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = _run(APP)
    assert not app.exception, f"app raised: {app.exception}"


def test_main_app_shows_the_disclaimer():
    app = _run(APP)
    captions = " ".join(c.value for c in app.caption)
    assert "Not a licensed insurance product" in captions


def test_main_app_warns_that_data_is_simulated():
    app = _run(APP)
    text = " ".join(w.value for w in app.warning) + " ".join(
        c.value for c in app.caption
    )
    assert "simulated" in text.lower()


def test_main_app_credits_the_dataset():
    app = _run(APP)
    captions = " ".join(c.value for c in app.caption)
    assert "CC BY 4.0" in captions
    assert "Mendeley" in captions


def test_main_app_reports_the_claim_count():
    app = _run(APP)
    values = [m.value for m in app.metric]
    assert "1,000" in values


@pytest.mark.skipif(not PAGES, reason="no pages/ yet (added from Phase 2)")
@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_each_page_runs_without_exception(page):
    """Parametrised over pages/, so new pages are covered automatically."""
    app = _run(page)
    assert not app.exception, f"{page.name} raised: {app.exception}"
