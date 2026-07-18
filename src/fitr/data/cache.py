"""Local parquet-based cache for historical OHLCV price data.

This is a "widen and merge" cache rather than one that tracks exact date
gaps: a cache miss triggers a re-download of a range wide enough to cover
both what's already cached and what's newly requested, and the two frames
are merged and de-duplicated. That trades a bit of extra download volume
for a lot less bookkeeping, which is the right call for a small recreational
project pulling from yfinance rather than a production data pipeline.

Nothing in here touches yfinance directly -- this module only knows about
DataFrames and the filesystem. YFProxy is the only thing that talks to both
this and yfinance.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class PriceCache:
    """Per-ticker parquet cache of OHLCV history, indexed by date."""

    def __init__(self, cache_dir: str | Path = "data/cache/prices"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}.parquet"

    def load(self, ticker: str) -> pd.DataFrame | None:
        """Return the full cached history for a ticker, or None if there's
        nothing cached (or the cache file is unreadable, which is treated
        the same as a miss)."""
        path = self._path(ticker)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            logger.warning("Cache file for %s is unreadable; treating as a miss.", ticker)
            return None

    def covers(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """Whether the cached range fully covers [start, end]."""
        cached = self.load(ticker)
        if cached is None or cached.empty:
            return False
        return cached.index.min() <= start and cached.index.max() >= end

    def save(self, ticker: str, df: pd.DataFrame) -> None:
        df = df.sort_index()
        df.to_parquet(self._path(ticker))

    def merge_and_save(self, ticker: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """Merge new_df into whatever's already cached for this ticker,
        de-duplicate by date (keeping the newest values), save, and return
        the full merged frame."""
        existing = self.load(ticker)
        if existing is None or existing.empty:
            merged = new_df
        else:
            merged = pd.concat([existing, new_df])
            merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.sort_index()
        self.save(ticker, merged)
        return merged
