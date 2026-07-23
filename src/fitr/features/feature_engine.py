"""FeatureEngine: turns a (ticker, as_of_date) pair into a flat feature
dict, driven entirely by a FeaturesConfig (i.e. by features.yaml). This is
the class BreakoutScanner and DatasetBuilder will both build on in later
steps -- neither of them should ever call an indicators.py function
directly; they go through this class so scanning logic, training features,
and live-prediction features are all guaranteed to be computed identically.
"""
from __future__ import annotations

import logging

import pandas as pd

from fitr.config_schemas.features_schema import FeaturesConfig
from fitr.data.yf_proxy import NoDataInRangeError, TickerNotFoundError, YFProxy
from fitr.features import indicators

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Computes a features.yaml-defined feature vector for one candidate.

    Point-in-time discipline: every DataFrame this class hands to an
    indicator function is fetched with end=as_of_date, so nothing beyond
    the as-of date is ever visible to a calculation. This is the single
    enforcement point for the project's point-in-time-correctness
    requirement -- callers never need to truncate data themselves.
    """

    def __init__(self, yf_proxy: YFProxy, config: FeaturesConfig, history_buffer_days: int = 500):
        """history_buffer_days: how much calendar history to pull before
        as_of_date. 500 comfortably covers every window used in the
        current indicator set (the largest is 252 trading days, i.e. a
        52-week lookback) with room to spare for weekends/holidays and
        each indicator's own internal warm-up period (e.g. RSI/MACD's EMA
        smoothing). Raise this if a future indicator needs a longer
        lookback than that."""
        self._proxy = yf_proxy
        self._config = config
        self._history_buffer_days = history_buffer_days

    def _history_for(self, ticker: str, as_of: pd.Timestamp) -> pd.DataFrame:
        start = as_of - pd.Timedelta(days=self._history_buffer_days)
        return self._proxy.get_history(ticker, start, as_of)

    @property
    def feature_names(self) -> list[str]:
        """All configured feature names, in config order -- used by
        LivePredictor to validate a loaded model's expected columns
        against the currently configured features.yaml before making any
        predictions, the same way BreakoutScanner.setup_names is used to
        build stable one-hot columns."""
        return [spec.name for spec in self._config.features]

    def compute(self, ticker: str, as_of_date) -> dict[str, float]:
        """Returns {feature_name: value} for every feature in the config.

        Raises TickerNotFoundError / NoDataInRangeError if the *primary*
        ticker itself has no data as of as_of_date -- there's nothing
        meaningful to compute without it, so this is a hard failure the
        caller (e.g. DatasetBuilder) should catch and skip the candidate.

        A benchmark that fails to fetch degrades gracefully instead: every
        feature depending on it becomes NaN (with a logged warning),
        rather than taking down every other, unrelated feature for this
        candidate. A missing SPY/sector data point is a data-availability
        problem, not a reason to discard an otherwise-good candidate.
        """
        as_of = pd.Timestamp(as_of_date).normalize()
        primary_df = self._history_for(ticker, as_of)

        benchmark_dfs: dict[str, pd.DataFrame | None] = {}

        def benchmark_df(key: str) -> pd.DataFrame | None:
            if key not in benchmark_dfs:
                benchmark_ticker = self._config.benchmarks[key]
                try:
                    benchmark_dfs[key] = self._history_for(benchmark_ticker, as_of)
                except (TickerNotFoundError, NoDataInRangeError) as exc:
                    logger.warning(
                        "Benchmark '%s' (%s) unavailable as of %s: %s -- features using it will be NaN.",
                        key,
                        benchmark_ticker,
                        as_of.date(),
                        exc,
                    )
                    benchmark_dfs[key] = None
            return benchmark_dfs[key]

        result: dict[str, float] = {}
        for spec in self._config.features:
            func = getattr(indicators, spec.function)
            df = primary_df if spec.source is None else benchmark_df(spec.source)
            if df is None:
                result[spec.name] = float("nan")
                continue

            args = [df]
            if spec.benchmark is not None:
                bdf = benchmark_df(spec.benchmark)
                if bdf is None:
                    result[spec.name] = float("nan")
                    continue
                args.append(bdf)

            try:
                result[spec.name] = float(func(*args, **spec.params))
            except Exception:
                if spec.optional:
                    logger.warning("Feature '%s' raised unexpectedly; using NaN.", spec.name, exc_info=True)
                    result[spec.name] = float("nan")
                else:
                    raise

        return result
