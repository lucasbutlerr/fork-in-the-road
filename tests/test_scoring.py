"""Tests for fitr.tuning.scoring. Every EvaluationReport here is
hand-built with a known confusion matrix, and every expected score is
computed by hand -- same philosophy as test_position_sizer.py."""
from __future__ import annotations

import pytest

from fitr.modeling.evaluator import EvaluationReport
from fitr.modeling.walk_forward import Fold, FoldResult, WalkForwardReport
from fitr.tuning.scoring import (
    GATED_SCORE,
    expected_value_score_from_report,
    expected_value_score_from_walk_forward,
    score_from_report,
    score_from_walk_forward,
)


def make_report(precision: float, predicted_success: int) -> EvaluationReport:
    cm = {
        "true_success_pred_success": predicted_success // 2,
        "true_failure_pred_success": predicted_success - predicted_success // 2,
        "true_success_pred_failure": 0,
        "true_failure_pred_failure": 0,
    }
    return EvaluationReport(
        n_rows=100, threshold=0.5, overall_accuracy=0.5, success_accuracy=0.5, failure_accuracy=0.5,
        success_precision=precision, failure_precision=0.5, confusion_matrix=cm, feature_importances={},
    )


def make_fold_result(fold_number: int, report: EvaluationReport) -> FoldResult:
    fold = Fold(fold_number=fold_number, train_start=None, train_end=None, test_start=None, test_end=None)
    return FoldResult(fold=fold, n_train_rows=100, n_test_rows=100, low_confidence=False, report=report)


def test_below_min_trades_is_gated():
    report = make_report(precision=0.5, predicted_success=10)
    assert score_from_report(report, min_trades=30) == GATED_SCORE


def test_at_or_above_min_trades_returns_real_precision():
    report = make_report(precision=0.42, predicted_success=60)
    assert score_from_report(report, min_trades=30) == pytest.approx(0.42)


def test_exactly_at_min_trades_boundary_is_not_gated():
    report = make_report(precision=0.35, predicted_success=30)
    assert score_from_report(report, min_trades=30) == pytest.approx(0.35)


def test_nan_precision_from_zero_predicted_success_is_gated_even_with_min_trades_zero():
    report = make_report(precision=float("nan"), predicted_success=0)
    assert score_from_report(report, min_trades=0) == GATED_SCORE


def test_walk_forward_aggregate_is_plain_mean_of_per_fold_scores():
    wf = WalkForwardReport(
        folds=[
            make_fold_result(1, make_report(0.4, 60)),
            make_fold_result(2, make_report(0.3, 50)),
        ]
    )
    assert score_from_walk_forward(wf, min_trades=30) == pytest.approx((0.4 + 0.3) / 2)


def test_walk_forward_aggregate_is_dragged_down_hard_by_a_gated_fold():
    # The gated fold's OWN precision (0.9) is the highest of the three --
    # but with only 5 predicted-success rows, it should still pull the
    # aggregate down, not up. This is deliberate: instability shouldn't
    # hide behind a lucky small sample.
    wf = WalkForwardReport(
        folds=[
            make_fold_result(1, make_report(0.4, 60)),
            make_fold_result(2, make_report(0.3, 50)),
            make_fold_result(3, make_report(0.9, 5)),
        ]
    )
    expected = (0.4 + 0.3 + GATED_SCORE) / 3
    assert score_from_walk_forward(wf, min_trades=30) == pytest.approx(expected)
    assert score_from_walk_forward(wf, min_trades=30) < 0.3  # worse than either real fold alone


def test_walk_forward_with_no_folds_is_gated():
    assert score_from_walk_forward(WalkForwardReport(folds=[]), min_trades=30) == GATED_SCORE


def test_ev_score_matches_hand_calc():
    # EV = p*T - (1-p)*S. p=0.419, T=0.08, S=0.08 -> 0.419*0.08 - 0.581*0.08 = -0.01296
    report = make_report(precision=0.419, predicted_success=60)
    score = expected_value_score_from_report(report, target_return=0.08, stop_loss_magnitude=0.08, min_trades=30)
    assert score == pytest.approx(-0.01296)


def test_ev_score_below_min_trades_is_gated():
    report = make_report(precision=0.9, predicted_success=5)
    score = expected_value_score_from_report(report, target_return=0.08, stop_loss_magnitude=0.04, min_trades=30)
    assert score == GATED_SCORE


def test_ev_score_correctly_ranks_a_balanced_payout_above_a_lopsided_one_despite_lower_raw_precision():
    # The whole point of this fix: a config needing very high precision to
    # break even (tiny target, huge stop) should NOT beat a balanced
    # config just because its raw precision number is bigger.
    lopsided = make_report(precision=0.70, predicted_success=60)   # target=0.04, stop=0.10 -> needs 71.4% to break even
    balanced = make_report(precision=0.35, predicted_success=60)   # target=0.08, stop=0.04 -> needs 33.3% to break even

    lopsided_score = expected_value_score_from_report(lopsided, target_return=0.04, stop_loss_magnitude=0.10, min_trades=30)
    balanced_score = expected_value_score_from_report(balanced, target_return=0.08, stop_loss_magnitude=0.04, min_trades=30)

    assert lopsided.success_precision > balanced.success_precision  # raw precision says lopsided "wins"
    assert balanced_score > lopsided_score  # EV correctly says balanced wins instead
    assert lopsided_score < 0  # lopsided is genuinely unprofitable
    assert balanced_score > 0  # balanced genuinely isn't


def test_ev_walk_forward_aggregate_is_plain_mean_across_folds():
    wf = WalkForwardReport(
        folds=[
            make_fold_result(1, make_report(0.40, 60)),
            make_fold_result(2, make_report(0.30, 50)),
        ]
    )
    score = expected_value_score_from_walk_forward(wf, target_return=0.08, stop_loss_magnitude=0.04, min_trades=30)
    expected = (
        expected_value_score_from_report(make_report(0.40, 60), 0.08, 0.04, min_trades=30)
        + expected_value_score_from_report(make_report(0.30, 50), 0.08, 0.04, min_trades=30)
    ) / 2
    assert score == pytest.approx(expected)


def test_ev_walk_forward_with_no_folds_is_gated():
    assert expected_value_score_from_walk_forward(WalkForwardReport(folds=[]), 0.08, 0.04, min_trades=30) == GATED_SCORE
