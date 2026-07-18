"""YFProxy: the sole translation layer between yfinance and the rest of the
codebase. Nothing outside this module should import yfinance directly --
that keeps retry logic, caching, calendar handling, and error normalization
in one place.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from . import trading_calendar
from .cache import PriceCache

logger = logging.getLogger(__name__)

DateLike = str | date | datetime | pd.Timestamp


class TickerNotFoundError(Exception):
    """Raised when yfinance returns no usable data at all for a ticker,
    including after retries are exhausted."""


class NoDataInRangeError(Exception):
    """Raised when a ticker has data generally, but none in the specific
    requested date range."""


def _to_ts(d: DateLike) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


class YFProxy:
    """Translation layer between yfinance and the rest of the pipeline.

    Handles: local caching of price history, retry/backoff on flaky
    network calls, weekend/holiday resolution via a calendar derived from
    a reference ticker, and normalizing yfinance's various failure modes
    into a couple of well-defined exceptions.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/cache/prices",
        reference_ticker: str = "AAPL",
        calendar_lookback_years: int = 30,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
    ):
        self._cache = PriceCache(cache_dir)
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._reference_ticker = reference_ticker

        calendar_start = _to_ts(date.today()) - pd.DateOffset(years=calendar_lookback_years)
        calendar_df = self.get_history(reference_ticker, calendar_start, _to_ts(date.today()))
        self._calendar: pd.DatetimeIndex = pd.DatetimeIndex(calendar_df.index)
        logger.info(
            "Trading calendar built from %s: %d sessions, %s to %s",
            reference_ticker,
            len(self._calendar),
            self._calendar.min().date(),
            self._calendar.max().date(),
        )

    # ---- raw download, with retry --------------------------------------
    def _download_raw(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=end + pd.Timedelta(days=1),  # yfinance's end is exclusive
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )
                if isinstance(df.columns, pd.MultiIndex):
                    # yfinance sometimes returns a MultiIndex even for a
                    # single ticker depending on version/call shape.
                    df.columns = df.columns.get_level_values(0)
                return df
            except Exception as exc:  # yfinance raises a mix of exception types
                last_exc = exc
                logger.warning(
                    "yfinance download failed for %s (attempt %d/%d): %s",
                    ticker,
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries:
                    time.sleep(self._backoff_seconds * attempt)
        raise TickerNotFoundError(
            f"Failed to download {ticker} after {self._max_retries} attempts"
        ) from last_exc

    # ---- history ---------------------------------------------------------
    def get_history(self, ticker: str, start: DateLike, end: DateLike, interval: str = "1d") -> pd.DataFrame:
        """Return OHLCV rows for ticker within [start, end], inclusive.
        Transparently uses and extends the local cache."""
        if interval != "1d":
            raise NotImplementedError("Only daily interval is currently supported.")

        start_ts, end_ts = _to_ts(start), _to_ts(end)
        if start_ts > end_ts:
            raise ValueError(f"start ({start_ts.date()}) is after end ({end_ts.date()})")

        if not self._cache.covers(ticker, start_ts, end_ts):
            cached = self._cache.load(ticker)
            if cached is not None and not cached.empty:
                fetch_start = min(start_ts, cached.index.min())
                fetch_end = max(end_ts, cached.index.max())
            else:
                fetch_start, fetch_end = start_ts, end_ts
            raw = self._download_raw(ticker, fetch_start, fetch_end)
            if raw.empty:
                raise TickerNotFoundError(f"No data returned for {ticker}")
            self._cache.merge_and_save(ticker, raw)

        full = self._cache.load(ticker)
        window = full.loc[(full.index >= start_ts) & (full.index <= end_ts)]
        if window.empty:
            raise NoDataInRangeError(
                f"No data for {ticker} between {start_ts.date()} and {end_ts.date()}"
            )
        return window

    def get_price_on(self, ticker: str, on_date: DateLike) -> pd.Series:
        """Return the OHLCV row for ticker on on_date. If on_date isn't a
        trading day (weekend/holiday), rolls back to the most recent prior
        trading day and logs that it did so."""
        ts = _to_ts(on_date)
        resolved = self.previous_trading_day(ts)
        if resolved != ts:
            logger.info("%s is not a trading day; using %s instead.", ts.date(), resolved.date())
        lookback_start = resolved - pd.Timedelta(days=10)
        hist = self.get_history(ticker, lookback_start, resolved)
        return hist.loc[resolved]

    # ---- calendar passthroughs --------------------------------------------
    def is_trading_day(self, check_date: DateLike) -> bool:
        return trading_calendar.is_trading_day(self._calendar, check_date)

    def previous_trading_day(self, check_date: DateLike) -> pd.Timestamp:
        return trading_calendar.previous_trading_day(self._calendar, check_date)

    def trading_days_between(self, start: DateLike, end: DateLike) -> list[pd.Timestamp]:
        return trading_calendar.trading_days_between(self._calendar, start, end)

    # ---- fundamentals (best-effort, current snapshot only) ----------------
    def get_fundamentals(self, ticker: str) -> dict | None:
        """Best-effort current-snapshot fundamentals.

        IMPORTANT: yfinance exposes the *latest* fundamentals, not a
        point-in-time historical snapshot. Do not use this for features on
        historical candidate dates without accounting for that -- it's a
        source of lookahead bias if used carelessly.
        """
        try:
            info = yf.Ticker(ticker).info
            if not info or info.get("regularMarketPrice") is None:
                logger.warning("No fundamentals available for %s", ticker)
                return None
            return info
        except Exception as exc:
            logger.warning("Failed to fetch fundamentals for %s: %s", ticker, exc)
            return None
