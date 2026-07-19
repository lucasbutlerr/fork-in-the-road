"""Tests for ModelEvaluator. Uses a fake TrainedModel-like double with
hand-specified predict_proba_success() output rather than a real fitted
XGBoost model -- every confusion-matrix cell, accuracy, and precision
value below is computed by hand and asserted exactly, which the roadmap
called out as the easiest test file in the project for exactly this
reason: it's pure arithmetic once predictions are fixed and known."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fitr.modeling.evaluator import ModelEvaluator


class _FakeModel:
    def __init__(self, importances):
        self.feature_importances_ = importances


class FakeTrainedModel:
    """Stands in for a real TrainedModel: same interface ModelEvaluator
    actually uses (label_column, feature_columns, predict_proba_success,
    model.feature_importances_), but with fully controlled, known output."""

    def __init__(self, probs, label_column="label", feature_columns=None, importances=None):
        self._probs = np.array(probs, dtype=float)
        self.label_column = label_column
        self.feature_columns = feature_columns or []
        self.model = _FakeModel(np.array(importances) if importances is not None else np.array([]))

    def predict_proba_success(self, df):
        return self._probs


def test_perfect_predictions_give_accuracy_one_everywhere():
    df = pd.DataFrame({"label": ["SUCCESS", "SUCCESS", "FAILURE", "FAILURE"]})
    model = FakeTrainedModel(probs=[0.9, 0.8, 0.1, 0.2])  # all correctly separated at threshold 0.5
    report = ModelEvaluator().evaluate(model, df)

    assert report.overall_accuracy == 1.0
    assert report.success_accuracy == 1.0
    assert report.failure_accuracy == 1.0
    assert report.success_precision == 1.0
    assert report.failure_precision == 1.0
    assert report.confusion_matrix == {
        "true_success_pred_success": 2,
        "true_success_pred_failure": 0,
        "true_failure_pred_success": 0,
        "true_failure_pred_failure": 2,
    }


def test_all_wrong_predictions_give_accuracy_zero():
    df = pd.DataFrame({"label": ["SUCCESS", "SUCCESS", "FAILURE", "FAILURE"]})
    model = FakeTrainedModel(probs=[0.1, 0.2, 0.9, 0.8])  # every prediction flipped
    report = ModelEvaluator().evaluate(model, df)

    assert report.overall_accuracy == 0.0
    assert report.success_accuracy == 0.0
    assert report.failure_accuracy == 0.0


def test_mixed_case_matches_hand_computed_values():
    # true:  SUCCESS SUCCESS FAILURE FAILURE FAILURE
    # probs: 0.9     0.3     0.2     0.6     0.4
    # pred:  1       0       0       1       0     (threshold 0.5)
    #
    # confusion: TP=1 (row0), FN=1 (row1), FP=1 (row3), TN=2 (rows 2,4)
    # success_accuracy (recall) = 1/2 = 0.5
    # failure_accuracy (recall) = 2/3
    # success_precision = 1/(1+1) = 0.5
    # failure_precision = 2/(2+1) = 2/3
    # overall_accuracy = 3/5 = 0.6
    df = pd.DataFrame({"label": ["SUCCESS", "SUCCESS", "FAILURE", "FAILURE", "FAILURE"]})
    model = FakeTrainedModel(probs=[0.9, 0.3, 0.2, 0.6, 0.4])
    report = ModelEvaluator().evaluate(model, df)

    assert report.confusion_matrix == {
        "true_success_pred_success": 1,
        "true_success_pred_failure": 1,
        "true_failure_pred_success": 1,
        "true_failure_pred_failure": 2,
    }
    assert report.overall_accuracy == pytest.approx(0.6)
    assert report.success_accuracy == pytest.approx(0.5)
    assert report.failure_accuracy == pytest.approx(2 / 3)
    assert report.success_precision == pytest.approx(0.5)
    assert report.failure_precision == pytest.approx(2 / 3)


def test_custom_threshold_changes_classification():
    df = pd.DataFrame({"label": ["SUCCESS", "FAILURE"]})
    model = FakeTrainedModel(probs=[0.6, 0.6])  # both above 0.5, both below 0.7

    report_low = ModelEvaluator().evaluate(model, df, threshold=0.5)
    assert report_low.confusion_matrix["true_failure_pred_success"] == 1  # FAILURE row predicted SUCCESS

    report_high = ModelEvaluator().evaluate(model, df, threshold=0.7)
    assert report_high.confusion_matrix["true_failure_pred_success"] == 0  # now correctly predicted FAILURE


def test_missing_class_in_test_set_gives_nan_not_crash():
    df = pd.DataFrame({"label": ["FAILURE", "FAILURE", "FAILURE"]})  # zero SUCCESS rows
    model = FakeTrainedModel(probs=[0.1, 0.2, 0.9])  # row 2 (prob=0.9) is a false-positive SUCCESS prediction
    report = ModelEvaluator().evaluate(model, df)

    # success_accuracy (recall) is genuinely undefined: there are no actual
    # SUCCESS rows to compute a "fraction correctly found" over.
    assert report.success_accuracy != report.success_accuracy  # NaN != NaN
    # success_precision is NOT undefined here, even though it looks similar
    # at a glance: precision's denominator is PREDICTED positives, not
    # actual ones, and the model did predict SUCCESS once (row 2) -- so
    # precision is a well-defined (if unfortunate) 0/(0+1) = 0.0, a real
    # false-alarm rate, not something to report as NaN.
    assert report.success_precision == pytest.approx(0.0)
    assert report.failure_accuracy == pytest.approx(2 / 3)  # failure metrics still computable


def test_precision_is_nan_when_neither_true_nor_predicted_positives_exist():
    df = pd.DataFrame({"label": ["FAILURE", "FAILURE", "FAILURE"]})
    model = FakeTrainedModel(probs=[0.1, 0.2, 0.3])  # model never predicts SUCCESS either
    report = ModelEvaluator().evaluate(model, df)
    assert report.success_precision != report.success_precision  # NaN: 0 predicted positives, 0/0


def test_unexpected_label_value_raises():
    df = pd.DataFrame({"label": ["SUCCESS", "MAYBE"]})
    model = FakeTrainedModel(probs=[0.9, 0.5])
    with pytest.raises(ValueError, match="MAYBE"):
        ModelEvaluator().evaluate(model, df)


def test_empty_test_set_raises():
    df = pd.DataFrame({"label": []})
    model = FakeTrainedModel(probs=[])
    with pytest.raises(ValueError):
        ModelEvaluator().evaluate(model, df)


def test_feature_importances_zipped_and_sorted_descending():
    df = pd.DataFrame({"label": ["SUCCESS", "FAILURE"]})
    model = FakeTrainedModel(
        probs=[0.9, 0.1],
        feature_columns=["feat_a", "feat_b", "feat_c"],
        importances=[0.2, 0.5, 0.3],
    )
    report = ModelEvaluator().evaluate(model, df)
    assert list(report.feature_importances.keys()) == ["feat_b", "feat_c", "feat_a"]
    assert report.feature_importances["feat_b"] == pytest.approx(0.5)


def test_setup_breakdown_computed_per_matched_setup():
    # setup_A covers rows 0,1,2 ; setup_B covers rows 2,3
    df = pd.DataFrame(
        {
            "label": ["SUCCESS", "SUCCESS", "FAILURE", "FAILURE"],
            "setup_A": [True, True, True, False],
            "setup_B": [False, False, True, True],
        }
    )
    # probs -> pred at 0.5: [1, 0, 0, 1] (row1 and row3 wrong)
    model = FakeTrainedModel(probs=[0.9, 0.3, 0.2, 0.6])
    report = ModelEvaluator().evaluate(model, df)

    # setup_A: true=[S,S,F], pred=[S,F,F] -> 2/3 correct, success recall 1/2, failure recall 1/1
    assert report.setup_breakdown["A"].n_rows == 3
    assert report.setup_breakdown["A"].accuracy == pytest.approx(2 / 3)
    assert report.setup_breakdown["A"].success_accuracy == pytest.approx(0.5)
    assert report.setup_breakdown["A"].failure_accuracy == pytest.approx(1.0)

    # setup_B: true=[F,F], pred=[F,S] -> 1/2 correct, no SUCCESS rows -> success_accuracy None
    assert report.setup_breakdown["B"].n_rows == 2
    assert report.setup_breakdown["B"].accuracy == pytest.approx(0.5)
    assert report.setup_breakdown["B"].success_accuracy is None
    assert report.setup_breakdown["B"].failure_accuracy == pytest.approx(0.5)


def test_setup_with_zero_matching_rows_excluded_from_breakdown():
    df = pd.DataFrame(
        {
            "label": ["SUCCESS", "FAILURE"],
            "setup_never_matches": [False, False],
        }
    )
    model = FakeTrainedModel(probs=[0.9, 0.1])
    report = ModelEvaluator().evaluate(model, df)
    assert "never_matches" not in report.setup_breakdown
