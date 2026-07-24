"""DatasetBuilder: the first orchestrator in the pipeline, tying
BreakoutScanner + FeatureEngine + OutcomeLabeler together across a
historical date range and ticker universe to produce a labeled training
dataset.

Note that the FeatureEngine passed in here is expected to be built from
features.yaml (the full ML feature set) -- a DIFFERENT, separate
FeatureEngine instance from the smaller one BreakoutScanner builds
internally from scanning_criteria.yaml's `metrics` block. Scanning decides
*whether* a ticker is a candidate; this class decides what to *record*
about it. They're deliberately different feature sets computed by
deliberately different FeatureEngine instances, even though both go
through the same underlying indicators.py functions.

Per-ticker cooldown: once a ticker is flagged as a candidate on some date
(by ANY setup), it's excluded from becoming a new candidate again until
`cooldown_trading_days` trading days have passed -- regardless of which
setup(s) fire on those intervening days. This is deliberate, not an
oversight: cooldown exists to stop one underlying breakout move from
generating near-duplicate rows with heavily overlapping forward-return
labeling windows, and that redundancy is about the ticker's future price
path, which is the same regardless of which setup happened to trigger on
a given day. A different setup firing the next day on the same ticker is
still almost certainly describing the same underlying move, not an
independent event, so it doesn't get its own row -- though the row that
IS created only records whichever setup(s) matched on the day it was
actually created, not any later day's matches during the cooldown window.

Defaults to a 5-trading-day (one calendar week) cooldown.

Each output row carries `trigger_reason` and `forward_return` alongside
the binary `label` -- richer detail from LabelResult that a binary
SUCCESS/FAILURE column alone discards. Nothing in the current pipeline
reads these yet (ModelTrainer still only reads `label`), but they're what
makes computing *actual* realized returns possible later (position
sizing, return-aware backtesting) instead of only ever assuming a
worst-case fixed stop-loss for every failure. IMPORTANT: `forward_return`
directly encodes the outcome being predicted -- ModelTrainer's "auto"
feature-column resolution explicitly excludes both new columns for
exactly this reason (see trainer.py's _NON_FEATURE_COLUMNS). If you ever
hand-build a feature_columns list instead of using "auto", exclude both
yourself too, or the model will trivially "predict" success by reading
off its own future answer.

CANDIDATE CACHING: build() is split into two phases with very different
costs. Scanning -- walking every trading day and checking the whole
universe against scanning_criteria.yaml -- is the expensive phase.
Cooldown filtering, labeling, and feature computation are comparatively
cheap, since they only ever touch the (much smaller) set of candidates
scanning already found. CandidateCache persists the raw (pre-cooldown)
scan result, keyed by a fingerprint of (scanning_criteria.yaml content,
universe, date range) -- so changing ONLY features.yaml, labeling.yaml,
or cooldown_trading_days reuses the cached scan and skips straight to the
cheap phase. Changing scanning_criteria.yaml, the universe, or the date
range invalidates the cache and forces a fresh scan -- there's no way
around this, since those three are exactly what determine which
candidates exist in the first place.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fitr.data.yf_proxy import YFProxy
from fitr.dataset.candidate_cache import CandidateCache
from fitr.features.feature_engine import FeatureEngine
from fitr.labeling.labeler import INCONCLUSIVE, SUCCESS, OutcomeLabeler
from fitr.scanning.scanner import BreakoutScanner

logger = logging.getLogger(__name__)


@dataclass
class BuildStats:
    trading_days_scanned: int = 0
    total_scan_matches: int = 0
    excluded_by_cooldown: int = 0
    excluded_inconclusive: int = 0
    rows_written: int = 0
    success_count: int = 0
    failure_count: int = 0
    used_cached_scan: bool = False


class DatasetBuilder:
    def __init__(
        self,
        yf_proxy: YFProxy,
        scanner: BreakoutScanner,
        feature_engine: FeatureEngine,
        labeler: OutcomeLabeler,
        universe: list[str],
        cooldown_trading_days: int = 5,
        cache_dir: str | Path = "data/cache/scans",
    ):
        self._proxy = yf_proxy
        self._scanner = scanner
        self._feature_engine = feature_engine
        self._labeler = labeler
        self._universe = universe
        self._cooldown_trading_days = cooldown_trading_days
        self._candidate_cache = CandidateCache(cache_dir)
        self.last_build_stats: BuildStats | None = None

    def build(self, start_date, end_date, force_rescan: bool = False) -> pd.DataFrame:
        trading_days = self._proxy.trading_days_between(start_date, end_date)
        setup_names = self._scanner.setup_names

        cache_key = self._candidate_cache.compute_key(
            self._scanner.config_fingerprint(), self._universe, start_date, end_date
        )
        raw_df = None if force_rescan else self._candidate_cache.load(cache_key)
        used_cache = raw_df is not None

        if raw_df is None:
            raw_df = self._scan_raw(trading_days)
            self._candidate_cache.save(
                cache_key, raw_df,
                meta={
                    "start": str(start_date), "end": str(end_date),
                    "n_universe": len(self._universe), "n_trading_days": len(trading_days),
                },
            )
        else:
            logger.info("Reusing cached scan (%d raw candidates) -- skipping the expensive scan phase.", len(raw_df))

        if raw_df.empty:
            stats = BuildStats(trading_days_scanned=len(trading_days), used_cached_scan=used_cache)
            self.last_build_stats = stats
            return pd.DataFrame()

        cooled_df = self._apply_cooldown(raw_df, trading_days)
        final_df, excluded_inconclusive, success_count, failure_count = self._label_and_featurize(
            cooled_df, setup_names
        )

        stats = BuildStats(
            trading_days_scanned=len(trading_days),
            total_scan_matches=len(raw_df),
            excluded_by_cooldown=len(raw_df) - len(cooled_df),
            excluded_inconclusive=excluded_inconclusive,
            rows_written=len(final_df),
            success_count=success_count,
            failure_count=failure_count,
            used_cached_scan=used_cache,
        )
        self.last_build_stats = stats
        return final_df

    def _scan_raw(self, trading_days: list) -> pd.DataFrame:
        """The expensive phase: walks every trading day, scans the whole
        universe, and flattens every match into a row -- setup_* and
        metric_* columns, same flattening convention the final output
        already uses for setup_*. NOT cooldown-filtered; that happens
        separately in _apply_cooldown, on either a fresh or cached result,
        so cooldown_trading_days changes never need a re-scan either."""
        setup_names = self._scanner.setup_names
        metric_names = self._scanner.metric_names
        rows: list[dict] = []

        for day_index, day in enumerate(trading_days):
            scan_results = self._scanner.scan(self._universe, day)
            for result in scan_results:
                row: dict = {"ticker": result.ticker, "date": pd.Timestamp(day)}
                for name in setup_names:
                    row[f"setup_{name}"] = name in result.matched_setups
                for name in metric_names:
                    row[f"metric_{name}"] = result.metrics.get(name, float("nan"))
                rows.append(row)

            if logger.isEnabledFor(logging.INFO) and ((day_index + 1) % 20 == 0 or day_index == len(trading_days) - 1):
                logger.info(
                    "Day %d/%d (%s) -- %d raw candidates so far (cache hits: %d, downloaded: %d)",
                    day_index + 1, len(trading_days), pd.Timestamp(day).date(), len(rows),
                    self._proxy.stats.cache_hits, self._proxy.stats.downloads,
                )

        return pd.DataFrame(rows)

    def _apply_cooldown(self, raw_df: pd.DataFrame, trading_days: list) -> pd.DataFrame:
        """Cheap, in-memory re-application of the same per-ticker cooldown
        logic as before -- just operating on the (possibly cached) raw
        scan DataFrame instead of ScanResult objects live from the
        scanner. Same semantics: once a ticker is flagged (by ANY setup),
        it's excluded from becoming a new candidate again until
        cooldown_trading_days trading days have passed."""
        day_index = {pd.Timestamp(d): i for i, d in enumerate(trading_days)}
        sorted_df = raw_df.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)

        last_flagged_index: dict[str, int] = {}
        keep = []
        for row in sorted_df.itertuples(index=False):
            idx = day_index[pd.Timestamp(row.date)]
            prior = last_flagged_index.get(row.ticker)
            if prior is not None and (idx - prior) < self._cooldown_trading_days:
                keep.append(False)
                continue
            last_flagged_index[row.ticker] = idx
            keep.append(True)

        return sorted_df.loc[keep].reset_index(drop=True)

    def _label_and_featurize(
        self, candidates_df: pd.DataFrame, setup_names: list[str]
    ) -> tuple[pd.DataFrame, int, int, int]:
        """The cheap phase: labels and computes ML features for each
        (already cooldown-filtered) candidate. This is what re-runs on
        every build() call, cached scan or not -- it's what actually
        reflects the CURRENT features.yaml/labeling.yaml, so a cached scan
        never means stale features or labels."""
        rows: list[dict] = []
        excluded_inconclusive = 0
        success_count = failure_count = 0
        total = len(candidates_df)

        for i, row in enumerate(candidates_df.itertuples(index=False), start=1):
            label_result = self._labeler.label(row.ticker, row.date)
            if label_result.outcome == INCONCLUSIVE:
                excluded_inconclusive += 1
            else:
                features = self._feature_engine.compute(row.ticker, row.date)
                out_row: dict = {"ticker": row.ticker, "date": pd.Timestamp(row.date)}
                out_row.update(features)
                for name in setup_names:
                    out_row[f"setup_{name}"] = bool(getattr(row, f"setup_{name}"))
                out_row["label"] = label_result.outcome
                out_row["trigger_reason"] = label_result.trigger_reason
                out_row["forward_return"] = label_result.forward_return
                rows.append(out_row)

                if label_result.outcome == SUCCESS:
                    success_count += 1
                else:
                    failure_count += 1

            if logger.isEnabledFor(logging.INFO) and (i % 200 == 0 or i == total):
                logger.info("Labeling/featurizing candidate %d/%d -- %d rows written so far", i, total, len(rows))

        return pd.DataFrame(rows), excluded_inconclusive, success_count, failure_count

    def save(self, df: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def split_by_date(df: pd.DataFrame, split_date, date_col: str = "date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based train/test split: rows with date < split_date go to
    train, rows with date >= split_date go to test. Deliberately not a
    random split -- randomly splitting time series data leaks information,
    since adjacent-in-time rows (especially pre-cooldown-era neighbors of
    the same move) are correlated with each other."""
    split_ts = pd.Timestamp(split_date)
    train = df.loc[df[date_col] < split_ts].reset_index(drop=True)
    test = df.loc[df[date_col] >= split_ts].reset_index(drop=True)
    return train, test
