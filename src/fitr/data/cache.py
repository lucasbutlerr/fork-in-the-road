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

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class PriceCache:
    """Per-ticker parquet cache of OHLCV history, indexed by date.

    Coverage is tracked by REQUESTED date range, in a small JSON sidecar
    per ticker, rather than being inferred from the min/max of the rows
    actually returned. That distinction matters more than it looks:
    request 2023-01-01 and the earliest row you get back is 2023-01-03,
    because Jan 1 was a Sunday and Jan 2 a holiday. Inferring coverage
    from the data alone would conclude "the head is still missing" and
    re-issue the same futile head fetch on every subsequent call, forever
    -- once per ticker per call, for any request whose start or end lands
    on a weekend or holiday. Recording what was ASKED FOR lets the cache
    correctly answer "there is nothing before 2023-01-03; stop asking."
    """

    def __init__(self, cache_dir: str | Path = "data/cache/prices"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}.parquet"

    def _coverage_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}.coverage.json"

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

    def load_coverage(self, ticker: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """The widest [start, end] range previously requested for this
        ticker, or None if unknown (no sidecar yet, or unreadable -- both
        treated as a miss, falling back to data-range inference)."""
        path = self._coverage_path(ticker)
        if not path.exists():
            return None
        try:
            with path.open() as f:
                meta = json.load(f)
            return pd.Timestamp(meta["requested_start"]), pd.Timestamp(meta["requested_end"])
        except Exception:
            logger.warning("Coverage sidecar for %s is unreadable; treating as unknown.", ticker)
            return None

    def record_coverage(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Widen this ticker's recorded requested-range to include
        [start, end]. Only ever widens -- never narrows."""
        existing = self.load_coverage(ticker)
        if existing is not None:
            start = min(start, existing[0])
            end = max(end, existing[1])
        with self._coverage_path(ticker).open("w") as f:
            json.dump({"requested_start": str(start.date()), "requested_end": str(end.date())}, f)

    def covers(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """Whether [start, end] has already been fetched for this ticker.

        Prefers the recorded requested-range (see the class docstring for
        why); falls back to inferring from the cached rows themselves when
        no sidecar exists, so caches written before coverage tracking
        existed still work -- just without the futile-refetch fix until
        their next real fetch records a range."""
        cached = self.load(ticker)
        if cached is None or cached.empty:
            return False

        coverage = self.load_coverage(ticker)
        if coverage is not None:
            return coverage[0] <= start and coverage[1] >= end
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
