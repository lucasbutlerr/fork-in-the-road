"""Tests for BreakoutScanner. Same monkeypatched-yfinance pattern as
test_feature_engine.py -- no network, real code otherwise."""
from __future__ import annotations

import pandas as pd
import pytest

from fitr.config_schemas.scanning_schema import ScanningConfig
from fitr.data import yf_proxy as yfp
from fitr.data.yf_proxy import YFProxy
from fitr.scanning.scanner import BreakoutScanner


def make_ohlcv(dates: pd.DatetimeIndex, base_price: float = 100.0, drift: float = 0.1) -> pd.DataFrame:
    n = len(dates)
    closes = [base_price + i * drift for i in range(n)]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": [c + 0.5 for c in closes],
            "Volume": [1_000_000] * n,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


@pytest.fixture
def business_days():
    return pd.bdate_range("2020-01-01", "2024-12-31")


@pytest.fixture
def mock_download(monkeypatch, business_days):
    """UPTREND climbs steadily (should trip a simple "positive momentum"
    setup); FLAT never moves (should trip nothing); MISSING has no data at
    all (simulates a delisted/not-yet-listed ticker)."""

    def fake_download(ticker, start=None, end=None, interval="1d", progress=False, auto_adjust=True):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        window = business_days[(business_days >= start_ts) & (business_days < end_ts)]
        if ticker == "UPTREND":
            return make_ohlcv(window, base_price=100.0, drift=0.4)
        if ticker == "FLAT":
            return make_ohlcv(window, base_price=100.0, drift=0.0)
        if ticker == "MISSING":
            return pd.DataFrame()
        # reference ticker (AAPL) used internally for the trading calendar
        return make_ohlcv(window, base_price=100.0, drift=0.1)

    monkeypatch.setattr(yfp.yf, "download", fake_download)
    return fake_download


@pytest.fixture
def proxy(tmp_path, mock_download):
    return YFProxy(cache_dir=tmp_path / "cache", reference_ticker="AAPL", calendar_lookback_years=5)


def _simple_config(logic="AND", conditions=None):
    return ScanningConfig.model_validate(
        {
            "metrics": [
                {"name": "return_20d", "function": "n_day_return", "params": {"window": 20}},
                {"name": "rsi_14", "function": "rsi", "params": {}},
            ],
            "setups": [
                {
                    "name": "momentum",
                    "logic": logic,
                    "conditions": conditions
                    or [
                        {"metric": "return_20d", "operator": ">=", "value": 0.02},
                        {"metric": "rsi_14", "operator": ">=", "value": 60},
                    ],
                }
            ],
        }
    )


def test_explain_matches_a_setup_when_all_and_conditions_pass(proxy):
    scanner = BreakoutScanner(proxy, _simple_config())
    explanation = scanner.explain("UPTREND", "2024-06-03")
    assert "momentum" in explanation.matched_setup_names
    assert all(c.passed for c in explanation.setups[0].conditions)


def test_explain_does_not_match_when_and_condition_fails(proxy):
    scanner = BreakoutScanner(proxy, _simple_config())
    explanation = scanner.explain("FLAT", "2024-06-03")
    assert explanation.matched_setup_names == []


def test_or_logic_matches_with_only_one_condition_passing(proxy):
    config = _simple_config(
        logic="OR",
        conditions=[
            {"metric": "return_20d", "operator": ">=", "value": 999},  # impossible -- always fails
            {"metric": "rsi_14", "operator": ">=", "value": 0},  # always true
        ],
    )
    scanner = BreakoutScanner(proxy, config)
    explanation = scanner.explain("FLAT", "2024-06-03")
    assert "momentum" in explanation.matched_setup_names
    setup_result = explanation.setups[0]
    assert setup_result.conditions[0].passed is False
    assert setup_result.conditions[1].passed is True


def test_between_operator_boundaries_are_inclusive(proxy):
    explanation = BreakoutScanner(
        proxy,
        _simple_config(conditions=[{"metric": "rsi_14", "operator": "between", "value": [0, 100]}]),
    ).explain("UPTREND", "2024-06-03")
    assert explanation.setups[0].passed is True


def test_explain_reports_actual_metric_values(proxy):
    scanner = BreakoutScanner(proxy, _simple_config())
    explanation = scanner.explain("UPTREND", "2024-06-03")
    assert "return_20d" in explanation.metrics
    assert "rsi_14" in explanation.metrics
    cond = explanation.setups[0].conditions[0]
    assert cond.actual_value == explanation.metrics["return_20d"]


def test_scan_one_returns_none_when_no_setup_matches(proxy):
    scanner = BreakoutScanner(proxy, _simple_config())
    assert scanner.scan_one("FLAT", "2024-06-03") is None


def test_scan_one_returns_result_when_a_setup_matches(proxy):
    scanner = BreakoutScanner(proxy, _simple_config())
    result = scanner.scan_one("UPTREND", "2024-06-03")
    assert result is not None
    assert result.ticker == "UPTREND"
    assert result.matched_setups == ["momentum"]


def test_scan_skips_tickers_with_no_data_without_crashing(proxy):
    scanner = BreakoutScanner(proxy, _simple_config())
    results = scanner.scan(["UPTREND", "MISSING", "FLAT"], "2024-06-03")
    tickers_found = {r.ticker for r in results}
    assert tickers_found == {"UPTREND"}


def test_scan_returns_empty_list_when_nothing_matches(proxy):
    scanner = BreakoutScanner(proxy, _simple_config())
    results = scanner.scan(["FLAT"], "2024-06-03")
    assert results == []


def test_explain_raises_for_ticker_with_no_data(proxy):
    from fitr.data.yf_proxy import TickerNotFoundError

    scanner = BreakoutScanner(proxy, _simple_config())
    with pytest.raises(TickerNotFoundError):
        scanner.explain("MISSING", "2024-06-03")


def test_nan_metric_fails_condition_without_crashing(proxy):
    # days_since_high needs 252 rows of history; with a short calendar
    # lookback most of that will be NaN for a fresh cache -- use a very
    # long window nothing in the fixture can satisfy to force a NaN metric.
    config = ScanningConfig.model_validate(
        {
            "metrics": [{"name": "impossible_high", "function": "relative_n_day_high", "params": {"window": 5000}}],
            "setups": [{"name": "s1", "conditions": [{"metric": "impossible_high", "operator": ">=", "value": -1}]}],
        }
    )
    scanner = BreakoutScanner(proxy, config)
    explanation = scanner.explain("UPTREND", "2024-06-03")
    assert explanation.metrics["impossible_high"] != explanation.metrics["impossible_high"]  # NaN
    assert explanation.setups[0].passed is False
