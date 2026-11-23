"""Shared fixtures. The committed Parquet sample is the test fixture, so the
tests assert on exactly the data the deployed app will read.
"""

from __future__ import annotations

import pandas as pd
import pytest

from claimpilot import config


@pytest.fixture(scope="session")
def claims() -> pd.DataFrame:
    from claimpilot.ingest import load_claims

    return load_claims()


@pytest.fixture(scope="session")
def narratives() -> pd.DataFrame:
    return pd.read_parquet(config.NARRATIVES_PARQUET)


@pytest.fixture(scope="session")
def policies() -> pd.DataFrame:
    return pd.read_parquet(config.POLICIES_PARQUET)


@pytest.fixture(scope="session")
def deadlines() -> pd.DataFrame:
    return pd.read_parquet(config.SAMPLE_DIR / "deadlines.parquet")


@pytest.fixture(scope="session")
def baseline() -> pd.DataFrame:
    return pd.read_parquet(config.SAMPLE_DIR / "baseline.parquet")


@pytest.fixture(scope="session")
def scored_portfolio() -> pd.DataFrame:
    from claimpilot.pipeline import load_scored

    return load_scored()


@pytest.fixture(scope="session")
def backtest_arms() -> pd.DataFrame:
    from claimpilot.backtest import load_backtest

    return load_backtest()
