"""Tests for FeatureEngine. Same no-network pattern as test_yf_proxy.py:
yfinance.download is monkeypatched, everything else is real code."""
from __future__ import annotations

import pandas as pd
import pytest

from fitr.config_schemas.features_schema import FeaturesConfig
from fitr.data import yf_proxy as yfp
from fitr.data.yf_proxy import TickerNotFoundError, YFProxy
from fitr.features import feature_engine as fe_module
from fitr.features import indicators as ind
from fitr.features.feature_engine import FeatureEngine


def make_ohlcv(dates: pd.DatetimeIndex, base_price: float = 100.0) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": [base_price + i * 0.1 for i in range(n)],
            "High": [base_price + i * 0.1 + 1 for i in range(n)],
            "Low": [base_price + i * 0.1 - 1 for i in range(n)],
            "Close": [base_price + i * 0.1 + 0.5 for i in range(n)],
            "Volume": [1_000_000 + i * 1_000 for i in range(n)],
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


@pytest.fixture
def business_days():
    return pd.bdate_range("2020-01-01", "2024-12-31")


@pytest.fixture
def mock_download(monkeypatch, business_days):
    """Every ticker gets the same synthetic price path except SPY and XLK,
    which get a different base price -- that difference is what lets the
    source-ticker tests confirm they're really pulling a different series,
    not silently reusing the primary ticker's data."""

    def fake_download(ticker, start=None, end=None, interval="1d", progress=False, auto_adjust=True):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        window = business_days[(business_days >= start_ts) & (business_days < end_ts)]
        base = {"SPY": 400.0, "XLK": 180.0}.get(ticker, 100.0)
        return make_ohlcv(window, base_price=base)

    monkeypatch.setattr(yfp.yf, "download", fake_download)
    return fake_download


@pytest.fixture
def proxy(tmp_path, mock_download):
    return YFProxy(cache_dir=tmp_path / "cache", reference_ticker="AAPL", calendar_lookback_years=5)


def test_plain_feature_matches_direct_indicator_call(proxy):
    config = FeaturesConfig.model_validate(
        {"features": [{"name": "rsi_14", "function": "rsi", "params": {}}]}
    )
    engine = FeatureEngine(proxy, config)
    result = engine.compute("MSFT", "2024-06-03")

    resolved = proxy.previous_trading_day(pd.Timestamp("2024-06-03"))
    expected_df = proxy.get_history("MSFT", resolved - pd.Timedelta(days=500), resolved)
    assert result["rsi_14"] == pytest.approx(ind.rsi(expected_df))


def test_multiple_features_all_present_in_result(proxy):
    config = FeaturesConfig.model_validate(
        {
            "features": [
                {"name": "return_20d", "function": "n_day_return", "params": {"window": 20}},
                {"name": "rsi_14", "function": "rsi", "params": {}},
                {"name": "atr_norm", "function": "atr_normalized", "params": {}},
            ]
        }
    )
    engine = FeatureEngine(proxy, config)
    result = engine.compute("MSFT", "2024-06-03")
    assert set(result.keys()) == {"return_20d", "rsi_14", "atr_norm"}
    assert all(isinstance(v, float) for v in result.values())


def test_source_ticker_uses_benchmark_history_not_primary(proxy):
    config = FeaturesConfig.model_validate(
        {
            "benchmarks": {"market": "SPY"},
            "features": [
                {"name": "market_return_20d", "function": "n_day_return", "params": {"window": 20}, "source": "market"},
                {"name": "own_return_20d", "function": "n_day_return", "params": {"window": 20}},
            ],
        }
    )
    engine = FeatureEngine(proxy, config)
    result = engine.compute("MSFT", "2024-06-03")
    # MSFT and SPY have different base prices in the mock, but identical
    # percentage drift, so both n-day returns should actually be very close
    # in this fixture -- what matters is confirming the *data source*
    # differs, which we do directly against indicators.py below.
    resolved = proxy.previous_trading_day(pd.Timestamp("2024-06-03"))
    start = resolved - pd.Timedelta(days=500)
    spy_df = proxy.get_history("SPY", start, resolved)
    msft_df = proxy.get_history("MSFT", start, resolved)
    assert result["market_return_20d"] == pytest.approx(ind.n_day_return(spy_df, 20))
    assert result["own_return_20d"] == pytest.approx(ind.n_day_return(msft_df, 20))


def test_benchmark_relative_feature_matches_direct_calc(proxy):
    config = FeaturesConfig.model_validate(
        {
            "benchmarks": {"market": "SPY"},
            "features": [
                {
                    "name": "return_vs_market",
                    "function": "relative_return_vs_benchmark",
                    "params": {"window": 20},
                    "benchmark": "market",
                }
            ],
        }
    )
    engine = FeatureEngine(proxy, config)
    result = engine.compute("MSFT", "2024-06-03")

    resolved = proxy.previous_trading_day(pd.Timestamp("2024-06-03"))
    start = resolved - pd.Timedelta(days=500)
    msft_df = proxy.get_history("MSFT", start, resolved)
    spy_df = proxy.get_history("SPY", start, resolved)
    expected = ind.relative_return_vs_benchmark(msft_df, spy_df, 20)
    assert result["return_vs_market"] == pytest.approx(expected)


def test_unavailable_benchmark_degrades_that_feature_to_nan_without_crashing(proxy, monkeypatch):
    config = FeaturesConfig.model_validate(
        {
            "benchmarks": {"market": "GHOSTETF"},
            "features": [
                {"name": "rsi_14", "function": "rsi", "params": {}},
                {
                    "name": "return_vs_market",
                    "function": "relative_return_vs_benchmark",
                    "params": {"window": 20},
                    "benchmark": "market",
                },
            ],
        }
    )

    real_download = yfp.yf.download

    def failing_for_ghost(ticker, **kwargs):
        if ticker == "GHOSTETF":
            return pd.DataFrame()  # empty -> YFProxy raises TickerNotFoundError
        return real_download(ticker, **kwargs)

    monkeypatch.setattr(yfp.yf, "download", failing_for_ghost)

    engine = FeatureEngine(proxy, config)
    result = engine.compute("MSFT", "2024-06-03")

    assert result["return_vs_market"] != result["return_vs_market"]  # NaN != NaN
    assert result["rsi_14"] == result["rsi_14"]  # unaffected feature is a real number, not NaN


def test_primary_ticker_unavailable_raises(proxy, monkeypatch):
    config = FeaturesConfig.model_validate(
        {"features": [{"name": "rsi_14", "function": "rsi", "params": {}}]}
    )

    def empty_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(yfp.yf, "download", empty_download)
    engine = FeatureEngine(proxy, config)
    with pytest.raises(TickerNotFoundError):
        engine.compute("NOSUCHTICKER", "2024-06-03")


def test_optional_true_swallows_unexpected_error_as_nan(proxy, monkeypatch):
    config = FeaturesConfig.model_validate(
        {"features": [{"name": "flaky", "function": "rsi", "params": {}, "optional": True}]}
    )

    def broken_rsi(*args, **kwargs):
        raise RuntimeError("simulated unexpected failure")

    monkeypatch.setattr(fe_module.indicators, "rsi", broken_rsi)
    engine = FeatureEngine(proxy, config)
    result = engine.compute("MSFT", "2024-06-03")
    assert result["flaky"] != result["flaky"]  # NaN


def test_optional_false_propagates_unexpected_error(proxy, monkeypatch):
    config = FeaturesConfig.model_validate(
        {"features": [{"name": "flaky", "function": "rsi", "params": {}, "optional": False}]}
    )

    def broken_rsi(*args, **kwargs):
        raise RuntimeError("simulated unexpected failure")

    monkeypatch.setattr(fe_module.indicators, "rsi", broken_rsi)
    engine = FeatureEngine(proxy, config)
    with pytest.raises(RuntimeError):
        engine.compute("MSFT", "2024-06-03")


def test_weekend_as_of_date_resolves_to_prior_trading_day(proxy):
    config = FeaturesConfig.model_validate(
        {"features": [{"name": "rsi_14", "function": "rsi", "params": {}}]}
    )
    engine = FeatureEngine(proxy, config)
    saturday_result = engine.compute("MSFT", "2024-06-01")  # Saturday
    friday_result = engine.compute("MSFT", "2024-05-31")  # preceding Friday
    assert saturday_result["rsi_14"] == pytest.approx(friday_result["rsi_14"])
