"""Tests for LivePredictor. Same fake-component-injection style as
test_dataset_builder.py: a fake scanner returning known candidates, a fake
model returning known probabilities, confirming ranking, setup one-hot
construction, graceful skipping, and the config-mismatch guard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fitr.data.yf_proxy import TickerNotFoundError
from fitr.modeling.position_sizer import PositionSizeResult
from fitr.modeling.predictor import LivePredictor, format_predictions
from fitr.scanning.scanner import ScanResult


class FakeScanner:
    def __init__(self, setup_names, results):
        self.setup_names = setup_names
        self._results = results

    def scan(self, universe, as_of_date):
        return self._results


class FakeFeatureEngine:
    def __init__(self, feature_names, feature_map=None, fail_for=()):
        self.feature_names = feature_names
        self._feature_map = feature_map or {}
        self._fail_for = set(fail_for)

    def compute(self, ticker, as_of_date):
        if ticker in self._fail_for:
            raise TickerNotFoundError(f"no data for {ticker}")
        return self._feature_map.get(ticker, {"feat_a": 1.0, "feat_b": 2.0})


class _FakeModelInner:
    def __init__(self, importances):
        self.feature_importances_ = importances


class FakeTrainedModel:
    """predict_proba_success is called once per candidate, in scan-result
    order -- probs_in_call_order maps directly onto that sequence."""

    def __init__(self, feature_columns, probs_in_call_order, importances=None):
        self.feature_columns = feature_columns
        self._probs = list(probs_in_call_order)
        self._call_count = 0
        self.model = _FakeModelInner(np.array(importances) if importances is not None else np.array([0.5, 0.5]))
        self.seen_dfs: list[pd.DataFrame] = []

    def predict_proba_success(self, df):
        self.seen_dfs.append(df.copy())
        p = self._probs[self._call_count]
        self._call_count += 1
        return np.array([p])

    def feature_importances(self):
        # Mirrors TrainedModel.feature_importances()'s public contract --
        # format_predictions() calls this, not .model.feature_importances_
        # directly (that distinction is exactly what the real bug was).
        raw = self.model.feature_importances_
        if raw is None:
            return {}
        pairs = sorted(zip(self.feature_columns, raw), key=lambda p: p[1], reverse=True)
        return {name: float(value) for name, value in pairs}


def test_predictions_ranked_by_probability_descending():
    scanner = FakeScanner(
        ["setup_a", "setup_b"],
        [
            ScanResult("AAA", pd.Timestamp("2024-06-03"), ["setup_a"], {"m1": 0.1}),
            ScanResult("BBB", pd.Timestamp("2024-06-03"), ["setup_b"], {"m1": 0.2}),
            ScanResult("CCC", pd.Timestamp("2024-06-03"), ["setup_a", "setup_b"], {"m1": 0.3}),
        ],
    )
    fe = FakeFeatureEngine(["feat_a", "feat_b"])
    model = FakeTrainedModel(
        feature_columns=["feat_a", "feat_b", "setup_setup_a", "setup_setup_b"],
        probs_in_call_order=[0.3, 0.9, 0.6],  # AAA, BBB, CCC in scan order
    )
    predictions = LivePredictor(scanner, fe, model).predict(["AAA", "BBB", "CCC"], "2024-06-03")

    assert [p.ticker for p in predictions] == ["BBB", "CCC", "AAA"]
    assert [p.rank for p in predictions] == [1, 2, 3]
    assert predictions[0].predicted_probability == 0.9


def test_matched_setups_carried_through_to_result():
    scanner = FakeScanner(
        ["setup_a", "setup_b"],
        [ScanResult("CCC", pd.Timestamp("2024-06-03"), ["setup_a", "setup_b"], {})],
    )
    fe = FakeFeatureEngine(["feat_a"])
    model = FakeTrainedModel(feature_columns=["feat_a", "setup_setup_a", "setup_setup_b"], probs_in_call_order=[0.5])
    predictions = LivePredictor(scanner, fe, model).predict(["CCC"], "2024-06-03")
    assert set(predictions[0].matched_setups) == {"setup_a", "setup_b"}


def test_setup_one_hot_columns_built_correctly():
    scanner = FakeScanner(
        ["setup_a", "setup_b"],
        [
            ScanResult("AAA", pd.Timestamp("2024-06-03"), ["setup_a"], {}),
            ScanResult("CCC", pd.Timestamp("2024-06-03"), ["setup_a", "setup_b"], {}),
        ],
    )
    fe = FakeFeatureEngine(["feat_a"])
    model = FakeTrainedModel(
        feature_columns=["feat_a", "setup_setup_a", "setup_setup_b"], probs_in_call_order=[0.5, 0.5]
    )
    LivePredictor(scanner, fe, model).predict(["AAA", "CCC"], "2024-06-03")

    aaa_df = model.seen_dfs[0]
    assert bool(aaa_df["setup_setup_a"].iloc[0]) is True
    assert bool(aaa_df["setup_setup_b"].iloc[0]) is False

    ccc_df = model.seen_dfs[1]
    assert bool(ccc_df["setup_setup_a"].iloc[0]) is True
    assert bool(ccc_df["setup_setup_b"].iloc[0]) is True


def test_ticker_with_failed_feature_computation_is_skipped_not_fatal():
    scanner = FakeScanner(
        ["setup_a"],
        [
            ScanResult("GOOD", pd.Timestamp("2024-06-03"), ["setup_a"], {}),
            ScanResult("BAD", pd.Timestamp("2024-06-03"), ["setup_a"], {}),
        ],
    )
    fe = FakeFeatureEngine(["feat_a"], fail_for=["BAD"])
    model = FakeTrainedModel(feature_columns=["feat_a", "setup_setup_a"], probs_in_call_order=[0.7])
    predictions = LivePredictor(scanner, fe, model).predict(["GOOD", "BAD"], "2024-06-03")
    assert len(predictions) == 1
    assert predictions[0].ticker == "GOOD"


def test_constructor_raises_on_model_config_mismatch():
    scanner = FakeScanner(["setup_a"], [])
    fe = FakeFeatureEngine(["feat_a"])  # doesn't know about feat_never_configured
    model = FakeTrainedModel(feature_columns=["feat_a", "feat_never_configured"], probs_in_call_order=[])
    with pytest.raises(ValueError, match="feat_never_configured"):
        LivePredictor(scanner, fe, model)


def test_empty_scan_results_in_empty_predictions():
    scanner = FakeScanner(["setup_a"], [])
    fe = FakeFeatureEngine(["feat_a"])
    model = FakeTrainedModel(feature_columns=["feat_a", "setup_setup_a"], probs_in_call_order=[])
    assert LivePredictor(scanner, fe, model).predict(["AAA"], "2024-06-03") == []


def test_as_of_date_reflected_in_results():
    scanner = FakeScanner(["setup_a"], [ScanResult("AAA", pd.Timestamp("2024-06-03"), ["setup_a"], {})])
    fe = FakeFeatureEngine(["feat_a"])
    model = FakeTrainedModel(feature_columns=["feat_a", "setup_setup_a"], probs_in_call_order=[0.5])
    predictions = LivePredictor(scanner, fe, model).predict(["AAA"], "2024-06-03")
    assert predictions[0].as_of_date == pd.Timestamp("2024-06-03")


def test_format_predictions_handles_empty_list():
    model = FakeTrainedModel(feature_columns=["feat_a"], probs_in_call_order=[])
    text = format_predictions([], model)
    assert "No candidates" in text


def test_format_predictions_does_not_crash_and_includes_key_fields():
    # This is the test that should have existed before the real bug shipped:
    # format_predictions() calls trained_model.feature_importances(), and no
    # prior test in this file actually called format_predictions() at all.
    scanner = FakeScanner(["setup_a"], [ScanResult("AAA", pd.Timestamp("2024-06-03"), ["setup_a"], {"m1": 1.23})])
    fe = FakeFeatureEngine(["feat_a", "feat_b"])
    model = FakeTrainedModel(
        feature_columns=["feat_a", "feat_b", "setup_setup_a"],
        probs_in_call_order=[0.73],
        importances=[0.9, 0.1, 0.05],
    )
    predictions = LivePredictor(scanner, fe, model).predict(["AAA"], "2024-06-03")
    text = format_predictions(predictions, model, top_n=10, top_features=2)

    assert "AAA" in text
    assert "73.0%" in text
    assert "setup_a" in text
    assert "feat_a" in text  # the highest-importance feature should be surfaced


def test_format_predictions_can_hide_scan_metrics_and_model_features():
    scanner = FakeScanner(["setup_a"], [ScanResult("AAA", pd.Timestamp("2024-06-03"), ["setup_a"], {"m1": 1.23})])
    fe = FakeFeatureEngine(["feat_a"])
    model = FakeTrainedModel(feature_columns=["feat_a", "setup_setup_a"], probs_in_call_order=[0.5], importances=[0.9, 0.1])
    predictions = LivePredictor(scanner, fe, model).predict(["AAA"], "2024-06-03")

    text = format_predictions(predictions, model, show_scan_metrics=False, show_model_features=False)

    assert "Scan metrics" not in text
    assert "Top model features" not in text
    assert "AAA" in text  # the candidate itself is still shown


def test_format_predictions_shows_position_size_when_attached():
    scanner = FakeScanner(["setup_a"], [ScanResult("AAA", pd.Timestamp("2024-06-03"), ["setup_a"], {})])
    fe = FakeFeatureEngine(["feat_a"])
    model = FakeTrainedModel(feature_columns=["feat_a", "setup_setup_a"], probs_in_call_order=[0.5])
    predictions = LivePredictor(scanner, fe, model).predict(["AAA"], "2024-06-03")

    predictions[0].position_size = PositionSizeResult(
        ticker="AAA", predicted_probability=0.5, method="fixed_fractional",
        raw_fraction=0.1, recommended_fraction=0.1, recommended_shares=40,
        position_value=1000.0, entry_price=25.0, equity=10_000.0, capped=False,
    )

    text = format_predictions(predictions, model)
    assert "Suggested allocation" in text
    assert "10.0%" in text
    assert "40 shares" in text


def test_format_predictions_omits_position_size_section_when_not_attached():
    scanner = FakeScanner(["setup_a"], [ScanResult("AAA", pd.Timestamp("2024-06-03"), ["setup_a"], {})])
    fe = FakeFeatureEngine(["feat_a"])
    model = FakeTrainedModel(feature_columns=["feat_a", "setup_setup_a"], probs_in_call_order=[0.5])
    predictions = LivePredictor(scanner, fe, model).predict(["AAA"], "2024-06-03")

    text = format_predictions(predictions, model)
    assert "Suggested allocation" not in text
