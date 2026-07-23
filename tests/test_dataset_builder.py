"""Tests for DatasetBuilder. Per the roadmap's plan for this step, these
use plain fake Scanner/FeatureEngine/Labeler/YFProxy stand-ins rather than
real YFProxy-backed ones -- DatasetBuilder is pure orchestration, and its
job is to assemble rows correctly and apply cooldown/exclusion logic
correctly, neither of which needs a real network-backed component to
verify."""
from __future__ import annotations

import pandas as pd
import pytest

from fitr.dataset.builder import DatasetBuilder, split_by_date
from fitr.labeling.labeler import FAILURE, INCONCLUSIVE, SUCCESS, LabelResult
from fitr.scanning.scanner import ScanResult

TRADING_DAYS = pd.bdate_range("2024-01-01", "2024-02-29").tolist()  # ~42 trading days
D = TRADING_DAYS  # shorthand for readability below


class FakeProxy:
    def trading_days_between(self, start, end):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        return [d for d in TRADING_DAYS if start_ts <= d <= end_ts]


class FakeScanner:
    def __init__(self, setup_names, results_by_day):
        self.setup_names = setup_names
        self._results_by_day = results_by_day  # {date: [ScanResult, ...]}

    def scan(self, tickers, as_of_date):
        return self._results_by_day.get(pd.Timestamp(as_of_date), [])


class FakeFeatureEngine:
    def __init__(self, feature_map=None):
        self._feature_map = feature_map or {}

    def compute(self, ticker, as_of_date):
        return self._feature_map.get((ticker, pd.Timestamp(as_of_date)), {"feat_a": 1.0, "feat_b": 2.0})


class FakeLabeler:
    def __init__(self, outcomes):
        self._outcomes = outcomes  # {(ticker, date): LabelResult}

    def label(self, ticker, as_of_date):
        return self._outcomes[(ticker, pd.Timestamp(as_of_date))]


def _label(ticker, date, outcome, trigger_reason=None, forward_return=None):
    return LabelResult(
        ticker=ticker, candidate_date=pd.Timestamp(date), outcome=outcome,
        trigger_reason=trigger_reason, forward_return=forward_return,
    )


def test_build_produces_a_row_with_features_setups_and_label():
    scanner = FakeScanner(["setup_a", "setup_b"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {"m": 1.0})]})
    labeler = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], SUCCESS)})
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA"])

    df = builder.build(D[0], D[0])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "AAA"
    assert row["feat_a"] == 1.0 and row["feat_b"] == 2.0
    assert row["label"] == SUCCESS
    assert row["setup_setup_a"] == True  # noqa: E712
    assert row["setup_setup_b"] == False  # noqa: E712


def test_row_carries_trigger_reason_and_forward_return():
    scanner = FakeScanner(["setup_a"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {})]})
    labeler = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], SUCCESS, trigger_reason="target_hit", forward_return=0.083)})
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA"])

    df = builder.build(D[0], D[0])

    assert df.iloc[0]["trigger_reason"] == "target_hit"
    assert df.iloc[0]["forward_return"] == pytest.approx(0.083)


def test_inconclusive_candidates_are_excluded_from_the_dataset():
    scanner = FakeScanner(["setup_a"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {})]})
    labeler = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], INCONCLUSIVE)})
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA"])

    df = builder.build(D[0], D[0])

    assert len(df) == 0
    assert builder.last_build_stats.excluded_inconclusive == 1


def test_cooldown_suppresses_a_reflag_within_the_window():
    results = {
        D[0]: [ScanResult("AAA", D[0], ["setup_a"], {})],
        D[2]: [ScanResult("AAA", D[2], ["setup_a"], {})],  # 2 trading days later -- within default 5-day cooldown
    }
    scanner = FakeScanner(["setup_a"], results)
    labeler = FakeLabeler(
        {("AAA", D[0]): _label("AAA", D[0], SUCCESS), ("AAA", D[2]): _label("AAA", D[2], SUCCESS)}
    )
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA"], cooldown_trading_days=5)

    df = builder.build(D[0], D[5])

    assert len(df) == 1
    assert builder.last_build_stats.excluded_by_cooldown == 1


def test_cooldown_allows_a_reflag_once_the_window_has_elapsed():
    results = {
        D[0]: [ScanResult("AAA", D[0], ["setup_a"], {})],
        D[5]: [ScanResult("AAA", D[5], ["setup_a"], {})],  # exactly 5 trading days later -- cooldown has elapsed
    }
    scanner = FakeScanner(["setup_a"], results)
    labeler = FakeLabeler(
        {("AAA", D[0]): _label("AAA", D[0], SUCCESS), ("AAA", D[5]): _label("AAA", D[5], FAILURE)}
    )
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA"], cooldown_trading_days=5)

    df = builder.build(D[0], D[6])

    assert len(df) == 2


def test_cooldown_applies_regardless_of_which_setup_matched():
    # Different setup fires on the very next day for the same ticker --
    # this should STILL be suppressed by cooldown, per the design decision
    # that cooldown is per-ticker, not per-(ticker, setup).
    results = {
        D[0]: [ScanResult("AAA", D[0], ["setup_a"], {})],
        D[1]: [ScanResult("AAA", D[1], ["setup_b"], {})],
    }
    scanner = FakeScanner(["setup_a", "setup_b"], results)
    labeler = FakeLabeler(
        {("AAA", D[0]): _label("AAA", D[0], SUCCESS), ("AAA", D[1]): _label("AAA", D[1], SUCCESS)}
    )
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA"], cooldown_trading_days=5)

    df = builder.build(D[0], D[2])

    assert len(df) == 1
    # The surviving row only reflects what matched on the day it was
    # actually created -- setup_b's later match isn't merged in.
    assert df.iloc[0]["setup_setup_a"] == True  # noqa: E712
    assert df.iloc[0]["setup_setup_b"] == False  # noqa: E712


def test_different_tickers_have_independent_cooldowns():
    results = {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {}), ScanResult("BBB", D[0], ["setup_a"], {})]}
    scanner = FakeScanner(["setup_a"], results)
    labeler = FakeLabeler(
        {("AAA", D[0]): _label("AAA", D[0], SUCCESS), ("BBB", D[0]): _label("BBB", D[0], FAILURE)}
    )
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA", "BBB"])

    df = builder.build(D[0], D[0])

    assert len(df) == 2
    assert set(df["ticker"]) == {"AAA", "BBB"}


def test_no_matches_produces_empty_dataframe_and_zeroed_stats():
    scanner = FakeScanner(["setup_a"], {})
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), FakeLabeler({}), ["AAA"])

    df = builder.build(D[0], D[3])

    assert df.empty
    assert builder.last_build_stats.rows_written == 0


def test_stats_track_success_and_failure_counts_separately():
    results = {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {}), ScanResult("BBB", D[0], ["setup_a"], {})]}
    scanner = FakeScanner(["setup_a"], results)
    labeler = FakeLabeler(
        {("AAA", D[0]): _label("AAA", D[0], SUCCESS), ("BBB", D[0]): _label("BBB", D[0], FAILURE)}
    )
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA", "BBB"])

    builder.build(D[0], D[0])

    assert builder.last_build_stats.success_count == 1
    assert builder.last_build_stats.failure_count == 1


def test_split_by_date_partitions_on_the_correct_side():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01"), pd.Timestamp("2024-12-01")],
            "x": [1, 2, 3],
        }
    )
    train, test = split_by_date(df, "2024-06-01")
    assert list(train["x"]) == [1]
    assert list(test["x"]) == [2, 3]
