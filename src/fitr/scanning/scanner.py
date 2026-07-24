"""BreakoutScanner: given a ticker universe and an as-of date, flags which
tickers match at least one configured breakout setup from
scanning_criteria.yaml.

Every metric is computed via FeatureEngine, reusing the exact same
indicators.py functions and point-in-time-safe history fetching that
features.yaml-driven ML training features use. Scanning logic and training
features can never quietly diverge, since both go through the same path --
this is the guarantee referenced in FeatureEngine's own docstring.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

import pandas as pd

from fitr.config_schemas.features_schema import FeaturesConfig
from fitr.config_schemas.scanning_schema import ConditionSpec, ScanningConfig, SetupSpec
from fitr.data.yf_proxy import NoDataInRangeError, TickerNotFoundError, YFProxy
from fitr.features.feature_engine import FeatureEngine

logger = logging.getLogger(__name__)

_OPERATORS = {
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
    ">": lambda v, t: v > t,
    "<": lambda v, t: v < t,
    "==": lambda v, t: v == t,
    "!=": lambda v, t: v != t,
    "between": lambda v, t: t[0] <= v <= t[1],
}


@dataclass
class ConditionResult:
    metric: str
    operator: str
    threshold: float | list[float]
    actual_value: float
    passed: bool


@dataclass
class SetupResult:
    name: str
    logic: str
    conditions: list[ConditionResult]
    passed: bool


@dataclass
class ScanExplanation:
    """Full per-condition, per-setup breakdown for one ticker on one date --
    what tools/debug_scanner.py uses to show its work."""

    ticker: str
    as_of_date: pd.Timestamp
    metrics: dict[str, float]
    setups: list[SetupResult]

    @property
    def matched_setup_names(self) -> list[str]:
        return [s.name for s in self.setups if s.passed]


@dataclass
class ScanResult:
    """Lean result for a normal scan -- just enough for a caller like
    DatasetBuilder (added in a later step) to work with. Use
    BreakoutScanner.explain() instead when the full per-condition
    breakdown is needed."""

    ticker: str
    as_of_date: pd.Timestamp
    matched_setups: list[str]
    metrics: dict[str, float]


class BreakoutScanner:
    def __init__(self, yf_proxy: YFProxy, config: ScanningConfig):
        self._proxy = yf_proxy
        self._config = config
        # Metrics reuse FeatureSpec's exact shape, so a plain FeatureEngine
        # computes them identically to how ML training features are
        # computed. Constructed once here so every scan() / explain() call
        # gets that guarantee without re-deriving it.
        self._engine = FeatureEngine(
            yf_proxy, FeaturesConfig(benchmarks=config.benchmarks, features=config.metrics)
        )

    @property
    def setup_names(self) -> list[str]:
        """All configured setup names, in config order -- used by callers
        like DatasetBuilder to build a stable set of one-hot columns
        regardless of which setups happen to match in a given run."""
        return [s.name for s in self._config.setups]

    @property
    def metric_names(self) -> list[str]:
        """All configured metric names (from scanning_criteria.yaml's
        `metrics` block), in config order -- used by DatasetBuilder to
        build stable columns for its raw-scan cache regardless of which
        metrics happen to appear in any single day's results."""
        return [m.name for m in self._config.metrics]

    def config_fingerprint(self) -> str:
        """A short, deterministic hash of the scanning config this
        scanner was built from -- used by DatasetBuilder's candidate
        cache to detect whether scanning_criteria.yaml has changed since
        a cached raw scan was produced. Two BreakoutScanner instances
        built from equivalent configs (even loaded from different file
        paths, or constructed in-memory) produce the same fingerprint;
        any change to setups, conditions, or metrics changes it."""
        canonical = json.dumps(self._config.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _evaluate_condition(self, cond: ConditionSpec, metrics: dict[str, float]) -> ConditionResult:
        value = metrics.get(cond.metric, float("nan"))
        if value != value:  # NaN check -- a missing/NaN metric fails the condition, doesn't crash the scan
            passed = False
        else:
            passed = _OPERATORS[cond.operator](value, cond.value)
        return ConditionResult(
            metric=cond.metric, operator=cond.operator, threshold=cond.value, actual_value=value, passed=passed
        )

    def _evaluate_setup(self, setup: SetupSpec, metrics: dict[str, float]) -> SetupResult:
        conditions = [self._evaluate_condition(c, metrics) for c in setup.conditions]
        if setup.logic == "AND":
            passed = all(c.passed for c in conditions)
        else:
            passed = any(c.passed for c in conditions)
        return SetupResult(name=setup.name, logic=setup.logic, conditions=conditions, passed=passed)

    def explain(self, ticker: str, as_of_date) -> ScanExplanation:
        """Full breakdown for one ticker. Raises TickerNotFoundError /
        NoDataInRangeError if the ticker has no data as of as_of_date --
        unlike scan(), which skips such tickers silently, explain() is for
        deliberately inspecting one specific ticker, so a data problem
        should surface rather than disappear."""
        as_of = pd.Timestamp(as_of_date).normalize()
        metrics = self._engine.compute(ticker, as_of_date)
        setups = [self._evaluate_setup(s, metrics) for s in self._config.setups]
        return ScanExplanation(ticker=ticker, as_of_date=as_of, metrics=metrics, setups=setups)

    def scan_one(self, ticker: str, as_of_date) -> ScanResult | None:
        """Returns a ScanResult if `ticker` matches at least one setup on
        as_of_date, else None. Swallows data-availability problems (returns
        None) rather than raising -- routine and expected when scanning a
        large universe that inevitably includes delisted or
        not-yet-listed tickers on any given historical date."""
        try:
            explanation = self.explain(ticker, as_of_date)
        except (TickerNotFoundError, NoDataInRangeError) as exc:
            logger.debug("Skipping %s on %s: %s", ticker, as_of_date, exc)
            return None
        matched = explanation.matched_setup_names
        if not matched:
            return None
        return ScanResult(
            ticker=ticker, as_of_date=explanation.as_of_date, matched_setups=matched, metrics=explanation.metrics
        )

    def scan(self, tickers: list[str], as_of_date, progress_every: int = 250) -> list[ScanResult]:
        """Scans every ticker in `tickers` and returns a ScanResult for
        each one that matched at least one setup, in the same order as
        the input list.

        Logs periodic progress at INFO level every `progress_every`
        tickers (enable with -v on the pipeline scripts) -- scanning a
        large universe against yfinance, even with caching, can take a
        while, and total silence for minutes at a time is worse than a
        handful of extra log lines. Reads YFProxy.stats for a running
        cache-hit/download count, which is a good proxy for "how much
        network work is actually left" without needing a separate
        up-front cache audit."""
        results = []
        total = len(tickers)
        for i, ticker in enumerate(tickers, start=1):
            result = self.scan_one(ticker, as_of_date)
            if result is not None:
                results.append(result)
            if logger.isEnabledFor(logging.INFO) and (i % progress_every == 0 or i == total):
                logger.info(
                    "  [%s] scanned %d/%d tickers (%.0f%%) -- %d candidates so far "
                    "(cache hits: %d, downloaded: %d)",
                    as_of_date, i, total, 100 * i / total, len(results),
                    self._proxy.stats.cache_hits, self._proxy.stats.downloads,
                )
        return results
