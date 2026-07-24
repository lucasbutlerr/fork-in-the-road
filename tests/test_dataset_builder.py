"""Tests for DatasetBuilder. Per the roadmap's plan for this step, these
use plain fake Scanner/FeatureEngine/Labeler/YFProxy stand-ins rather than
real YFProxy-backed ones -- DatasetBuilder is pure orchestration, and its
job is to assemble rows correctly and apply cooldown/exclusion logic
correctly, neither of which needs a real network-backed component to
verify. Every test passes its own tmp_path as cache_dir, isolating it from
both other tests and the real project's data/cache/scans directory."""
from __future__ import annotations

import uuid

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
    def __init__(self, setup_names, results_by_day, metric_names=None, fingerprint=None):
        self.setup_names = setup_names
        self.metric_names = metric_names or []
        self._results_by_day = results_by_day  # {date: [ScanResult, ...]}
        # A fresh, unique fingerprint per instance by default -- so tests
        # that don't care about caching never accidentally collide with
        # each other. Tests that specifically test cache reuse pass a
        # matching fingerprint explicitly instead.
        self._fingerprint = fingerprint or str(uuid.uuid4())
        self.scan_call_count = 0

    def scan(self, tickers, as_of_date):
        self.scan_call_count += 1
        return self._results_by_day.get(pd.Timestamp(as_of_date), [])

    def config_fingerprint(self):
        return self._fingerprint


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


def test_build_produces_a_row_with_features_setups_and_label(tmp_path):
    scanner = FakeScanner(["setup_a", "setup_b"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {"m": 1.0})]})
    labeler = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], SUCCESS)})
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA"], cache_dir=tmp_path)

    df = builder.build(D[0], D[0])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "AAA"
    assert row["feat_a"] == 1.0 and row["feat_b"] == 2.0
    assert row["label"] == SUCCESS
    assert row["setup_setup_a"] == True  # noqa: E712
    assert row["setup_setup_b"] == False  # noqa: E712


def test_row_carries_trigger_reason_and_forward_return(tmp_path):
    scanner = FakeScanner(["setup_a"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {})]})
    labeler = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], SUCCESS, trigger_reason="target_hit", forward_return=0.083)})
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA"], cache_dir=tmp_path)

    df = builder.build(D[0], D[0])

    assert df.iloc[0]["trigger_reason"] == "target_hit"
    assert df.iloc[0]["forward_return"] == pytest.approx(0.083)


def test_inconclusive_candidates_are_excluded_from_the_dataset(tmp_path):
    scanner = FakeScanner(["setup_a"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {})]})
    labeler = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], INCONCLUSIVE)})
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA"], cache_dir=tmp_path)

    df = builder.build(D[0], D[0])

    assert len(df) == 0
    assert builder.last_build_stats.excluded_inconclusive == 1


def test_cooldown_suppresses_a_reflag_within_the_window(tmp_path):
    results = {
        D[0]: [ScanResult("AAA", D[0], ["setup_a"], {})],
        D[2]: [ScanResult("AAA", D[2], ["setup_a"], {})],  # 2 trading days later -- within default 5-day cooldown
    }
    scanner = FakeScanner(["setup_a"], results)
    labeler = FakeLabeler(
        {("AAA", D[0]): _label("AAA", D[0], SUCCESS), ("AAA", D[2]): _label("AAA", D[2], SUCCESS)}
    )
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA"], cooldown_trading_days=5, cache_dir=tmp_path)

    df = builder.build(D[0], D[5])

    assert len(df) == 1
    assert builder.last_build_stats.excluded_by_cooldown == 1


def test_cooldown_allows_a_reflag_once_the_window_has_elapsed(tmp_path):
    results = {
        D[0]: [ScanResult("AAA", D[0], ["setup_a"], {})],
        D[5]: [ScanResult("AAA", D[5], ["setup_a"], {})],  # exactly 5 trading days later -- cooldown has elapsed
    }
    scanner = FakeScanner(["setup_a"], results)
    labeler = FakeLabeler(
        {("AAA", D[0]): _label("AAA", D[0], SUCCESS), ("AAA", D[5]): _label("AAA", D[5], FAILURE)}
    )
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA"], cooldown_trading_days=5, cache_dir=tmp_path)

    df = builder.build(D[0], D[6])

    assert len(df) == 2


def test_cooldown_applies_regardless_of_which_setup_matched(tmp_path):
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
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA"], cooldown_trading_days=5, cache_dir=tmp_path)

    df = builder.build(D[0], D[2])

    assert len(df) == 1
    # The surviving row only reflects what matched on the day it was
    # actually created -- setup_b's later match isn't merged in.
    assert df.iloc[0]["setup_setup_a"] == True  # noqa: E712
    assert df.iloc[0]["setup_setup_b"] == False  # noqa: E712


def test_different_tickers_have_independent_cooldowns(tmp_path):
    results = {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {}), ScanResult("BBB", D[0], ["setup_a"], {})]}
    scanner = FakeScanner(["setup_a"], results)
    labeler = FakeLabeler(
        {("AAA", D[0]): _label("AAA", D[0], SUCCESS), ("BBB", D[0]): _label("BBB", D[0], FAILURE)}
    )
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA", "BBB"], cache_dir=tmp_path)

    df = builder.build(D[0], D[0])

    assert len(df) == 2
    assert set(df["ticker"]) == {"AAA", "BBB"}


def test_no_matches_produces_empty_dataframe_and_zeroed_stats(tmp_path):
    scanner = FakeScanner(["setup_a"], {})
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), FakeLabeler({}), ["AAA"], cache_dir=tmp_path)

    df = builder.build(D[0], D[3])

    assert df.empty
    assert builder.last_build_stats.rows_written == 0


def test_stats_track_success_and_failure_counts_separately(tmp_path):
    results = {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {}), ScanResult("BBB", D[0], ["setup_a"], {})]}
    scanner = FakeScanner(["setup_a"], results)
    labeler = FakeLabeler(
        {("AAA", D[0]): _label("AAA", D[0], SUCCESS), ("BBB", D[0]): _label("BBB", D[0], FAILURE)}
    )
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA", "BBB"], cache_dir=tmp_path)

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


# ---- Candidate-cache tests --------------------------------------------------

def test_first_build_performs_a_real_scan(tmp_path):
    scanner = FakeScanner(["setup_a"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {"m1": 1.0})]})
    labeler = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], SUCCESS)})
    builder = DatasetBuilder(FakeProxy(), scanner, FakeFeatureEngine(), labeler, ["AAA"], cache_dir=tmp_path)

    builder.build(D[0], D[0])

    assert scanner.scan_call_count == 1
    assert builder.last_build_stats.used_cached_scan is False


def test_second_build_with_same_fingerprint_reuses_the_cache_without_rescanning(tmp_path):
    fingerprint = "shared-fp"
    scanner1 = FakeScanner(["setup_a"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {"m1": 1.0})]}, fingerprint=fingerprint)
    labeler1 = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], SUCCESS)})
    DatasetBuilder(FakeProxy(), scanner1, FakeFeatureEngine(), labeler1, ["AAA"], cache_dir=tmp_path).build(D[0], D[0])

    scanner2 = FakeScanner(["setup_a"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {"m1": 1.0})]}, fingerprint=fingerprint)
    labeler2 = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], SUCCESS)})
    builder2 = DatasetBuilder(FakeProxy(), scanner2, FakeFeatureEngine(), labeler2, ["AAA"], cache_dir=tmp_path)
    df2 = builder2.build(D[0], D[0])

    assert scanner2.scan_call_count == 0
    assert builder2.last_build_stats.used_cached_scan is True
    assert len(df2) == 1


def test_cached_scan_still_reflects_the_current_labeler_not_a_stale_one(tmp_path):
    fingerprint = "shared-fp-2"
    scanner1 = FakeScanner(["setup_a"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {"m1": 1.0})]}, fingerprint=fingerprint)
    labeler1 = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], SUCCESS)})
    DatasetBuilder(FakeProxy(), scanner1, FakeFeatureEngine(), labeler1, ["AAA"], cache_dir=tmp_path).build(D[0], D[0])

    # Same scan fingerprint, but labeling.yaml (simulated here by the fake
    # labeler) has changed -- e.g. a different target/stop/horizon.
    scanner2 = FakeScanner(["setup_a"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {"m1": 1.0})]}, fingerprint=fingerprint)
    labeler2 = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], FAILURE)})
    builder2 = DatasetBuilder(FakeProxy(), scanner2, FakeFeatureEngine(), labeler2, ["AAA"], cache_dir=tmp_path)
    df2 = builder2.build(D[0], D[0])

    assert scanner2.scan_call_count == 0  # scan itself was skipped
    assert df2.iloc[0]["label"] == FAILURE  # but the label reflects the NEW labeler, not the cached run's


def test_cached_scan_still_reflects_the_current_feature_engine(tmp_path):
    fingerprint = "shared-fp-3"
    scanner1 = FakeScanner(["setup_a"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {"m1": 1.0})]}, fingerprint=fingerprint)
    labeler1 = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], SUCCESS)})
    DatasetBuilder(FakeProxy(), scanner1, FakeFeatureEngine(), labeler1, ["AAA"], cache_dir=tmp_path).build(D[0], D[0])

    scanner2 = FakeScanner(["setup_a"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {"m1": 1.0})]}, fingerprint=fingerprint)
    labeler2 = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], SUCCESS)})
    fe2 = FakeFeatureEngine({("AAA", D[0]): {"brand_new_feature": 42.0}})
    builder2 = DatasetBuilder(FakeProxy(), scanner2, fe2, labeler2, ["AAA"], cache_dir=tmp_path)
    df2 = builder2.build(D[0], D[0])

    assert scanner2.scan_call_count == 0
    assert df2.iloc[0]["brand_new_feature"] == 42.0


def test_different_fingerprint_forces_a_fresh_scan(tmp_path):
    scanner1 = FakeScanner(["setup_a"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {"m1": 1.0})]}, fingerprint="fp-A")
    labeler1 = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], SUCCESS)})
    DatasetBuilder(FakeProxy(), scanner1, FakeFeatureEngine(), labeler1, ["AAA"], cache_dir=tmp_path).build(D[0], D[0])

    scanner2 = FakeScanner(["setup_a"], {D[0]: [ScanResult("BBB", D[0], ["setup_a"], {"m1": 2.0})]}, fingerprint="fp-B")
    labeler2 = FakeLabeler({("BBB", D[0]): _label("BBB", D[0], SUCCESS)})
    builder2 = DatasetBuilder(FakeProxy(), scanner2, FakeFeatureEngine(), labeler2, ["AAA"], cache_dir=tmp_path)
    df2 = builder2.build(D[0], D[0])

    assert scanner2.scan_call_count == 1
    assert df2.iloc[0]["ticker"] == "BBB"


def test_force_rescan_bypasses_an_existing_cache(tmp_path):
    fingerprint = "shared-fp-4"
    scanner1 = FakeScanner(["setup_a"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {"m1": 1.0})]}, fingerprint=fingerprint)
    labeler1 = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], SUCCESS)})
    DatasetBuilder(FakeProxy(), scanner1, FakeFeatureEngine(), labeler1, ["AAA"], cache_dir=tmp_path).build(D[0], D[0])

    scanner2 = FakeScanner(["setup_a"], {D[0]: [ScanResult("AAA", D[0], ["setup_a"], {"m1": 1.0})]}, fingerprint=fingerprint)
    labeler2 = FakeLabeler({("AAA", D[0]): _label("AAA", D[0], SUCCESS)})
    builder2 = DatasetBuilder(FakeProxy(), scanner2, FakeFeatureEngine(), labeler2, ["AAA"], cache_dir=tmp_path)
    builder2.build(D[0], D[0], force_rescan=True)

    assert scanner2.scan_call_count == 1


def test_cooldown_still_correct_when_re_derived_from_a_cached_scan(tmp_path):
    fingerprint = "shared-fp-5"
    results = {
        D[0]: [ScanResult("AAA", D[0], ["setup_a"], {"m1": 1.0})],
        D[2]: [ScanResult("AAA", D[2], ["setup_a"], {"m1": 1.0})],  # within default 5-day cooldown
    }
    scanner1 = FakeScanner(["setup_a"], results, fingerprint=fingerprint)
    labeler1 = FakeLabeler(
        {("AAA", D[0]): _label("AAA", D[0], SUCCESS), ("AAA", D[2]): _label("AAA", D[2], SUCCESS)}
    )
    DatasetBuilder(FakeProxy(), scanner1, FakeFeatureEngine(), labeler1, ["AAA"], cache_dir=tmp_path).build(D[0], D[5])

    scanner2 = FakeScanner(["setup_a"], results, fingerprint=fingerprint)
    labeler2 = FakeLabeler(
        {("AAA", D[0]): _label("AAA", D[0], SUCCESS), ("AAA", D[2]): _label("AAA", D[2], SUCCESS)}
    )
    builder2 = DatasetBuilder(FakeProxy(), scanner2, FakeFeatureEngine(), labeler2, ["AAA"], cache_dir=tmp_path)
    df2 = builder2.build(D[0], D[5])

    assert scanner2.scan_call_count == 0  # scan was cached
    assert len(df2) == 1  # cooldown still correctly suppressed the reflag
