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
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fitr.data.yf_proxy import YFProxy
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


class DatasetBuilder:
    def __init__(
        self,
        yf_proxy: YFProxy,
        scanner: BreakoutScanner,
        feature_engine: FeatureEngine,
        labeler: OutcomeLabeler,
        universe: list[str],
        cooldown_trading_days: int = 5,
    ):
        self._proxy = yf_proxy
        self._scanner = scanner
        self._feature_engine = feature_engine
        self._labeler = labeler
        self._universe = universe
        self._cooldown_trading_days = cooldown_trading_days
        self.last_build_stats: BuildStats | None = None

    def build(self, start_date, end_date) -> pd.DataFrame:
        trading_days = self._proxy.trading_days_between(start_date, end_date)
        setup_names = self._scanner.setup_names

        rows: list[dict] = []
        last_flagged_index: dict[str, int] = {}
        stats = BuildStats(trading_days_scanned=len(trading_days))

        for day_index, day in enumerate(trading_days):
            scan_results = self._scanner.scan(self._universe, day)
            stats.total_scan_matches += len(scan_results)

            for result in scan_results:
                ticker = result.ticker
                prior_index = last_flagged_index.get(ticker)
                if prior_index is not None and (day_index - prior_index) < self._cooldown_trading_days:
                    stats.excluded_by_cooldown += 1
                    continue

                # Cooldown clock resets here regardless of what the label
                # ends up being -- the candidate event happened on this
                # day either way.
                last_flagged_index[ticker] = day_index

                label_result = self._labeler.label(ticker, day)
                if label_result.outcome == INCONCLUSIVE:
                    stats.excluded_inconclusive += 1
                    continue

                features = self._feature_engine.compute(ticker, day)
                row: dict = {"ticker": ticker, "date": pd.Timestamp(day)}
                row.update(features)
                for name in setup_names:
                    row[f"setup_{name}"] = name in result.matched_setups
                row["label"] = label_result.outcome
                rows.append(row)

                if label_result.outcome == SUCCESS:
                    stats.success_count += 1
                else:
                    stats.failure_count += 1

            if logger.isEnabledFor(logging.INFO) and ((day_index + 1) % 20 == 0 or day_index == len(trading_days) - 1):
                logger.info("Scanned %d/%d trading days -- %d rows so far", day_index + 1, len(trading_days), len(rows))

        stats.rows_written = len(rows)
        self.last_build_stats = stats
        return pd.DataFrame(rows)

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
