"""CandidateCache: persists raw (pre-cooldown) scan results, keyed by a
fingerprint of (scanning config, universe, date range). This is the
mechanism that lets DatasetBuilder skip the expensive part of a rebuild
-- the day-by-day scan across the whole universe -- when only
features.yaml, labeling.yaml, or the cooldown setting has changed.

Deliberately does NOT depend on features.yaml, labeling.yaml, or
cooldown_trading_days at all: changing any of those has no effect on
WHICH (ticker, date) pairs are candidates, only on what gets recorded
about them or how they're subsequently filtered. Changing
scanning_criteria.yaml, the universe, or the date range DOES change the
candidate set itself, so those three are exactly what invalidate this
cache -- there's no way around re-scanning when any of them changes,
since the candidate set is genuinely different.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


class CandidateCache:
    def __init__(self, cache_dir: str | Path = "data/cache/scans"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_key(scanning_fingerprint: str, universe: list[str], start_date, end_date) -> str:
        universe_fp = hashlib.sha256(",".join(sorted(universe)).encode()).hexdigest()[:16]
        composite = (
            f"{scanning_fingerprint}:{universe_fp}:"
            f"{pd.Timestamp(start_date).date()}:{pd.Timestamp(end_date).date()}"
        )
        return hashlib.sha256(composite.encode()).hexdigest()[:24]

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.cache_dir / f"{key}.parquet", self.cache_dir / f"{key}.meta.json"

    def load(self, key: str) -> pd.DataFrame | None:
        """Returns the cached raw scan DataFrame, or None on any cache
        miss -- including a corrupted/unreadable cache file, which is
        treated the same as a miss (triggers a fresh scan) rather than
        raising, since a stale or damaged cache file should never be
        able to block a rebuild from working."""
        data_path, _ = self._paths(key)
        if not data_path.exists():
            return None
        try:
            return pd.read_parquet(data_path)
        except Exception:
            return None

    def save(self, key: str, df: pd.DataFrame, meta: dict) -> None:
        data_path, meta_path = self._paths(key)
        df.to_parquet(data_path)
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2, default=str)
