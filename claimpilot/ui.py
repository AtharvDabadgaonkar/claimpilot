"""Shared Streamlit helpers: data loading, page chrome, disclaimers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from claimpilot import config
from claimpilot.llm import get_provider, provider_badge

PAGE_ICON = "🛡️"


@st.cache_data(show_spinner=False)
def load_tables() -> dict[str, pd.DataFrame]:
    """Load every committed Parquet table once per session."""
    from claimpilot.ingest import load_claims

    tables = {"claims": load_claims()}
    for name in ("policies", "narratives", "deadlines", "baseline"):
        path = config.SAMPLE_DIR / f"{name}.parquet"
        if path.exists():
            tables[name] = pd.read_parquet(path)
    return tables


@st.cache_resource(show_spinner=False)
def llm_provider():
    """One provider per server process."""
    return get_provider()


def page_setup(title: str, subtitle: str = "") -> None:
    """Standard page header. Call first in every page."""
    st.set_page_config(
        page_title=f"{title} · ClaimPilot",
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title(f"{PAGE_ICON} {title}")
    if subtitle:
        st.caption(subtitle)
    sidebar_badges()


def sidebar_badges() -> None:
    """Carrier, AI mode and simulated-data badges, on every page."""
    with st.sidebar:
        st.markdown(f"**{config.carrier_name()}**")
        st.caption(f"{config.assumption('meta', 'lob')}")
        st.info(provider_badge(llm_provider()), icon="🤖")
        st.warning("Contains simulated data", icon="🧪")
        st.divider()


def footer() -> None:
    """Required demo disclaimer. Call last in every page."""
    st.divider()
    st.caption(f"⚠️ {config.DISCLAIMER}")
    st.caption(f"Source data: {config.dataset_attribution()} — licensed CC BY 4.0.")


def friendly_error(exc: Exception, what: str) -> None:
    """Show a readable message instead of a traceback."""
    st.error(
        f"Could not {what}. The demo data may not be built yet — "
        "run `make data` locally to rebuild it.",
        icon="⚠️",
    )
    with st.expander("Technical detail"):
        st.code(f"{type(exc).__name__}: {exc}")


def how_calculated(text: str) -> None:
    """One-line 'how it's calculated' caption required under every chart."""
    st.caption(f"ℹ️ How it's calculated: {text}")


def simulated_badge(text: str = "Simulated") -> None:
    st.caption(f"🧪 {text}")
