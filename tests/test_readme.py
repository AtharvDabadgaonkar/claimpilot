"""The README must not contain hand-typed numbers.

Rule 6 of the project brief: `make metrics` writes every number to
reports/metrics.json and the README pulls its numbers from there. These tests
assert the headline figures in the README still agree with the metrics file, so
a stale README fails the build instead of misleading a reader.
"""

from __future__ import annotations

import re

from claimpilot import config
from claimpilot.kpis import load_metrics

README = config.ROOT / "README.md"


def _text() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_exists():
    assert README.exists()


def test_headline_dataset_numbers_match_metrics():
    m = load_metrics()["dataset"]
    text = _text()
    assert f"{m['claims']:,}" in text
    assert f"{m['reported_fraud_rate']:.1%}" in text
    assert f"${m['median_total_claim_usd']:,.0f}" in text
    assert f"{m['injury_claim_rate']:.1%}" in text
    assert f"{m['multi_party_rate']:.1%}" in text
    assert f"{m['missing_police_report_rate']:.1%}" in text


def test_masking_numbers_match_metrics():
    m = load_metrics()["pii_masking"]
    text = _text()
    assert f"{m['detection_rate']:.1%}" in text
    assert str(m["narratives_known_to_contain_pii"]) in text
    assert str(m["clean_narratives_checked"]) in text


def test_simulated_numbers_match_metrics():
    m = load_metrics()["simulated_layers"]
    text = _text()
    assert str(m["policies_expired_at_loss"]) in text
    assert str(m["policies_liability_only"]) in text
    assert str(m["narratives_vague_amount"]) in text
    assert f"{m['baseline_mean_cycle_days']}" in text
    assert f"${m['baseline_mean_lae_usd']:,.0f}" in text


def test_excluded_field_count_matches():
    g = load_metrics()["governance"]
    assert f"{g['fields_excluded_count']} fields excluded" in _text()


def test_readme_separates_real_from_simulated():
    text = _text()
    assert "Real — from the published dataset" in text
    assert "Simulated — derived from" in text


def test_readme_states_the_required_caveats():
    low = _text().lower()
    assert "recalibration" in low
    assert "proxy" in low
    assert "simulated" in low


def test_readme_credits_the_dataset_and_licence():
    text = _text()
    assert "CC BY 4.0" in text
    assert "10.17632/992mh7dk9y" in text
    assert "Aqqad" in text


def test_readme_carries_the_exact_disclaimer():
    """The required wording, verbatim."""
    normalised = re.sub(r"\s+", " ", _text())
    assert "Demo project using public and simulated data." in normalised
    assert "Not a licensed insurance product, not legal or compliance advice." in (
        normalised
    )


def test_readme_never_claims_compliance():
    """Careful wording: 'designed around', never 'compliant' or 'certified'."""
    low = _text().lower()
    assert "naic-compliant" not in low
    assert "fully compliant" not in low
    assert "is compliant" not in low
    assert "certified against" in low or "not certified" in low or (
        "nor certified" in low
    )
    assert "designed around" in low or "aligned with" in low


def test_readme_states_no_api_key_is_required():
    low = _text().lower()
    assert "no api key" in low
    assert "demo mode: rule-based ai" in low.lower()


def test_readme_warns_session_storage_is_not_permanent():
    low = _text().lower()
    assert "not permanent" in low


def test_readme_documents_the_no_auto_deny_rule():
    low = _text().lower()
    assert "never auto-deny" in low or "can never auto-deny" in low


def test_backtest_numbers_match_metrics():
    """Phase 5: the backtest table in the README is read from metrics.json."""
    bt = load_metrics()["backtest"]
    man, cp = bt["arms"]["manual"], bt["arms"]["claimpilot"]
    text = _text()
    assert f"{man['mean_handling_hours']} h" in text
    assert f"{cp['mean_handling_hours']} h" in text
    assert f"{man['mean_cycle_days']} d" in text
    assert f"{cp['mean_cycle_days']} d" in text
    assert f"${cp['mean_lae_usd']:,.0f}" in text
    assert f"${man['total_cost_usd']:,.0f}" in text
    assert f"${cp['total_cost_usd']:,.0f}" in text
    assert f"{bt['deltas']['total_cost_pct']:+.1%}" in text


def test_breakeven_numbers_match_metrics():
    be = load_metrics()["backtest"]["lae_breakeven"]
    text = _text()
    assert f"${be['automation_headroom_usd']:,.0f}" in text
    assert f"{be['breakeven_referral_rate']:.1%}" in text
    assert f"{be['actual_referral_rate']:.1%}" in text
    assert be["verdict"] in text


def test_kpi_tiles_in_readme_match_metrics():
    ops = load_metrics()["ops_dashboard"]
    text = _text()
    assert f"{ops['kpis']['stp_rate']:.1%}" in text
    assert f"{ops['kpis']['siu_hit_rate']:.1%}" in text
    assert f"{ops['guardrails']['worst_fairness_ratio']:.3f}" in text


def test_readme_says_the_backtest_is_not_evidence():
    """The headline savings must never be presented as a measured result."""
    low = _text().lower()
    assert "not evidence" in low
    assert "arithmetic consequence" in low
    assert "paired" in low


def test_readme_reports_the_expense_increase_not_just_the_saving():
    """The uncomfortable half of the backtest must survive an edit."""
    low = _text().lower()
    assert "more expensive" in low or "expense per claim went **up**" in low
    assert "break-even" in low


def test_readme_gives_the_github_and_streamlit_deploy_steps():
    text = _text()
    assert "git push -u origin main" in text
    assert "git remote add origin" in text
    assert "gh repo create" in text
    assert "share.streamlit.io" in text
    assert "streamlit_app.py" in text
    assert "3.11" in text


def test_readme_tells_the_reader_to_scan_for_secrets_before_pushing():
    low = _text().lower()
    assert "secrets.toml" in low
    assert ".gitignore" in low


def test_readme_links_the_prd():
    assert "docs/PRD.md" in _text()


def test_readme_has_a_place_for_the_live_link():
    """The brief requires the live Streamlit link at the top of the README."""
    text = _text()
    head = text[: text.find("## The four numbers")]
    assert "Live demo" in head
    assert "streamlit" in head.lower()
