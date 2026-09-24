"""Tests for YFProxy, PriceCache, and trading_calendar.

None of these hit the real network -- yfinance.download and yfinance.Ticker
are monkeypatched throughout. That keeps the suite fast and deterministic,
and safe to run in CI without hitting yfinance's rate limits.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fitr.data import yf_proxy as yfp
from fitr.data.yf_proxy import NoDataInRangeError, TickerNotFoundError, YFProxy


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
    # Five years of ordinary business days. Deliberately NOT filtering out
    # market holidays -- that's fine for these tests, since we only care
    # that "weekends aren't in the calendar" behavior works correctly.
    return pd.bdate_range("2020-01-01", "2024-12-31")


@pytest.fixture
def mock_download(monkeypatch, business_days):
    """Patch yfinance.download. Records every call made (ticker/start/end)
    and returns synthetic OHLCV rows for whatever business days fall in the
    requested window, regardless of which ticker was asked for."""
    calls = []

    def fake_download(ticker, start=None, end=None, interval="1d", progress=False, auto_adjust=True):
        calls.append({"ticker": ticker, "start": pd.Timestamp(start), "end": pd.Timestamp(end)})
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        window = business_days[(business_days >= start_ts) & (business_days < end_ts)]
        return make_ohlcv(window)

    monkeypatch.setattr(yfp.yf, "download", fake_download)
    return calls


@pytest.fixture
def proxy(tmp_path, mock_download):
    return YFProxy(cache_dir=tmp_path / "cache", reference_ticker="AAPL", calendar_lookback_years=5)


# ---- calendar ---------------------------------------------------------

def test_calendar_built_from_reference_ticker(proxy):
    assert proxy.is_trading_day("2024-06-03")  # a Monday
    assert not proxy.is_trading_day("2024-06-01")  # a Saturday


def test_get_price_on_trading_day_returns_that_day(proxy):
    row = proxy.get_price_on("AAPL", "2024-06-03")
    assert row.name == pd.Timestamp("2024-06-03")


def test_get_price_on_weekend_rolls_back_to_friday(proxy):
    row = proxy.get_price_on("AAPL", "2024-06-01")  # Saturday
    assert row.name == pd.Timestamp("2024-05-31")  # preceding Friday


# ---- caching ------------------------------------------------------------

def test_get_history_downloads_then_uses_cache(proxy, mock_download):
    calls_before = len(mock_download)

    df = proxy.get_history("MSFT", "2024-01-01", "2024-01-31")
    assert not df.empty
    assert len(mock_download) == calls_before + 1

    # Same range again -> should be served from cache, no new download.
    df2 = proxy.get_history("MSFT", "2024-01-01", "2024-01-31")
    assert len(mock_download) == calls_before + 1
    pd.testing.assert_frame_equal(df, df2)


def test_get_history_extends_cache_for_new_range(proxy, mock_download):
    proxy.get_history("MSFT", "2024-01-01", "2024-01-15")
    calls_before = len(mock_download)

    df = proxy.get_history("MSFT", "2024-02-01", "2024-02-15")
    assert not df.empty
    assert len(mock_download) == calls_before + 1

    # Both ranges should now be servable without further downloads.
    calls_before = len(mock_download)
    proxy.get_history("MSFT", "2024-01-01", "2024-01-15")
    proxy.get_history("MSFT", "2024-02-01", "2024-02-15")
    assert len(mock_download) == calls_before


def test_get_history_rejects_start_after_end(proxy):
    with pytest.raises(ValueError):
        proxy.get_history("MSFT", "2024-02-01", "2024-01-01")


def test_non_trading_day_boundary_does_not_trigger_a_refetch_every_call(proxy, mock_download):
    """Regression test for a real bug: requesting a start date that falls
    on a weekend/holiday used to trigger a fresh (empty) download on every
    single subsequent call, forever, because coverage was inferred from
    the cached data's own min/max index rather than the range actually
    requested. 2024-01-01 is a Monday holiday-observed range start in
    this fixture's business-day calendar in some years, but to keep this
    deterministic we pick an actual weekend: 2024-06-01 is a Saturday, so
    the first trading day on/after it is 2024-06-03.
    """
    proxy.get_history("MSFT", "2024-06-01", "2024-06-14")
    calls_after_first = len(mock_download)

    # Same range, repeated several times -- none of these should issue a
    # new download. Under the old (buggy) inference-from-data-bounds
    # logic, cached.index.min() would be 2024-06-03 (not 2024-06-01), so
    # every call below would conclude "the head is still missing" and
    # re-issue an empty fetch for it.
    for _ in range(3):
        proxy.get_history("MSFT", "2024-06-01", "2024-06-14")
    assert len(mock_download) == calls_after_first


def test_coverage_survives_a_new_proxy_instance(tmp_path, mock_download):
    cache_dir = tmp_path / "cache"
    proxy1 = YFProxy(cache_dir=cache_dir, reference_ticker="AAPL", calendar_lookback_years=5)
    proxy1.get_history("MSFT", "2024-06-01", "2024-06-14")
    calls_after_first = len(mock_download)

    # A fresh YFProxy instance pointed at the same cache_dir should read
    # the persisted coverage record from disk, not just in-memory state.
    proxy2 = YFProxy(cache_dir=cache_dir, reference_ticker="AAPL", calendar_lookback_years=5)
    proxy2.get_history("MSFT", "2024-06-01", "2024-06-14")
    assert len(mock_download) == calls_after_first


# ---- retry / error handling ---------------------------------------------

def test_download_retries_then_succeeds(tmp_path, monkeypatch, business_days):
    attempts = {"n": 0}

    def flaky_download(ticker, start=None, end=None, interval="1d", progress=False, auto_adjust=True):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("simulated network failure")
        window = business_days[
            (business_days >= pd.Timestamp(start)) & (business_days < pd.Timestamp(end))
        ]
        return make_ohlcv(window)

    monkeypatch.setattr(yfp.yf, "download", flaky_download)

    # Calendar construction itself exercises the retry path.
    YFProxy(
        cache_dir=tmp_path / "cache",
        reference_ticker="AAPL",
        calendar_lookback_years=5,
        max_retries=5,
        backoff_seconds=0,
    )
    assert attempts["n"] >= 3


def test_download_raises_ticker_not_found_after_max_retries(tmp_path, monkeypatch):
    def always_fail(*args, **kwargs):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(yfp.yf, "download", always_fail)
    with pytest.raises(TickerNotFoundError):
        YFProxy(
            cache_dir=tmp_path / "cache",
            reference_ticker="AAPL",
            calendar_lookback_years=5,
            max_retries=2,
            backoff_seconds=0,
        )


def test_empty_response_raises_ticker_not_found(proxy, monkeypatch):
    def empty_download(ticker, start=None, end=None, interval="1d", progress=False, auto_adjust=True):
        return pd.DataFrame()

    monkeypatch.setattr(yfp.yf, "download", empty_download)
    with pytest.raises(TickerNotFoundError):
        proxy.get_history("FAKETICKER", "2024-01-01", "2024-01-31")


# ---- fundamentals ---------------------------------------------------------

def test_get_fundamentals_returns_none_on_failure(proxy, monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            raise RuntimeError("network down")

    monkeypatch.setattr(yfp.yf, "Ticker", FakeTicker)
    assert proxy.get_fundamentals("MSFT") is None


def test_get_fundamentals_returns_info_dict(proxy, monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.info = {"regularMarketPrice": 123.45, "trailingPE": 30.2}

    monkeypatch.setattr(yfp.yf, "Ticker", FakeTicker)
    result = proxy.get_fundamentals("MSFT")
    assert result["trailingPE"] == 30.2
