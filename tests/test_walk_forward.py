"""Tests for WalkForwardValidator. Uses synthetic in-memory data (not a
real dataset) -- this is orchestration over ModelTrainer/ModelEvaluator/
BacktestSimulator, all already tested in isolation, so what needs
verifying here is fold generation correctness (expanding vs. rolling,
non-overlapping test windows) and that backtest/low-confidence wiring
behaves as documented."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fitr.config_schemas.model_schema import ModelConfig
from fitr.config_schemas.position_sizing_schema import PositionSizingConfig
from fitr.config_schemas.walk_forward_schema import WalkForwardConfig
from fitr.modeling.position_sizer import PositionSizer
from fitr.modeling.walk_forward import WalkForwardValidator, format_walk_forward_report


def make_full_dataset(n=600, seed=0, n_success_frac=0.25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)
    n_success = int(n * n_success_frac)
    labels = ["SUCCESS"] * n_success + ["FAILURE"] * (n - n_success)
    rng.shuffle(labels)
    forward_returns = [
        0.08 + rng.normal(0, 0.01) if lbl == "SUCCESS" else -0.04 + rng.normal(0, 0.01) for lbl in labels
    ]
    return pd.DataFrame(
        {
            "ticker": [f"T{i % 20:03d}" for i in range(n)],
            "date": dates,
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(0, 1, n),
            "setup_classic": rng.random(n) < 0.5,
            "label": labels,
            "trigger_reason": ["target_hit" if lbl == "SUCCESS" else "stop_hit" for lbl in labels],
            "forward_return": forward_returns,
        }
    )


@pytest.fixture
def full_df():
    return make_full_dataset()


def test_expanding_window_shares_the_same_train_start_across_folds(full_df):
    config = WalkForwardConfig.model_validate({"n_folds": 4, "window_type": "expanding"})
    report = WalkForwardValidator(ModelConfig.model_validate({})).run(full_df, config)
    train_starts = {fr.fold.train_start for fr in report.folds}
    assert len(train_starts) == 1


def test_expanding_window_train_end_strictly_grows(full_df):
    config = WalkForwardConfig.model_validate({"n_folds": 4, "window_type": "expanding"})
    report = WalkForwardValidator(ModelConfig.model_validate({})).run(full_df, config)
    train_ends = [fr.fold.train_end for fr in report.folds]
    assert train_ends == sorted(train_ends)
    assert len(set(train_ends)) == len(train_ends)


def test_rolling_window_train_start_advances_and_respects_window_size(full_df):
    config = WalkForwardConfig.model_validate(
        {"n_folds": 3, "window_type": "rolling", "rolling_train_window_days": 200}
    )
    report = WalkForwardValidator(ModelConfig.model_validate({})).run(full_df, config)
    train_starts = [fr.fold.train_start for fr in report.folds]
    assert len(set(train_starts)) > 1
    for fr in report.folds:
        span_days = (fr.fold.train_end - fr.fold.train_start).days
        assert span_days <= 201  # rolling_train_window_days + a day of slack


def test_test_windows_are_non_overlapping_and_sequential(full_df):
    config = WalkForwardConfig.model_validate({"n_folds": 4, "window_type": "expanding"})
    report = WalkForwardValidator(ModelConfig.model_validate({})).run(full_df, config)
    windows = [(fr.fold.test_start, fr.fold.test_end) for fr in report.folds]
    for (start_a, end_a), (start_b, end_b) in zip(windows, windows[1:]):
        assert end_a <= start_b


def test_backtest_attached_per_fold_when_position_sizer_given(full_df):
    config = WalkForwardConfig.model_validate({"n_folds": 4})
    sizing_config = PositionSizingConfig.model_validate(
        {"method": "fixed_fractional", "risk_fraction": 0.01, "target_return": 0.08, "stop_loss_return": -0.04}
    )
    validator = WalkForwardValidator(ModelConfig.model_validate({}), position_sizer=PositionSizer(sizing_config))
    report = validator.run(full_df, config)
    assert all(fr.backtest is not None for fr in report.folds)


def test_no_backtest_when_position_sizer_not_given(full_df):
    config = WalkForwardConfig.model_validate({"n_folds": 4})
    report = WalkForwardValidator(ModelConfig.model_validate({})).run(full_df, config)
    assert all(fr.backtest is None for fr in report.folds)


def test_low_confidence_flag_set_when_min_test_rows_not_met(full_df):
    config = WalkForwardConfig.model_validate({"n_folds": 4, "min_test_rows": 100_000})
    report = WalkForwardValidator(ModelConfig.model_validate({})).run(full_df, config)
    assert all(fr.low_confidence for fr in report.folds)
    # low-confidence folds are still reported, never silently dropped
    config_normal = WalkForwardConfig.model_validate({"n_folds": 4})
    report_normal = WalkForwardValidator(ModelConfig.model_validate({})).run(full_df, config_normal)
    assert len(report.folds) == len(report_normal.folds)


def test_empty_dataset_raises():
    config = WalkForwardConfig.model_validate({"n_folds": 4})
    with pytest.raises(ValueError):
        WalkForwardValidator(ModelConfig.model_validate({})).run(pd.DataFrame({"date": [], "label": []}), config)


def test_format_report_includes_fold_and_aggregate_info(full_df):
    config = WalkForwardConfig.model_validate({"n_folds": 4})
    report = WalkForwardValidator(ModelConfig.model_validate({})).run(full_df, config)
    text = format_walk_forward_report(report)
    assert "folds" in text
    assert "accuracy=" in text
    assert f"Fold {report.folds[0].fold.fold_number}" in text


def test_format_report_includes_calibration_and_setup_diagnostics(full_df):
    # These are what actually let you diagnose a fold that makes zero
    # predictions -- without them, you can only guess at the cause.
    config = WalkForwardConfig.model_validate({"n_folds": 3})
    report = WalkForwardValidator(ModelConfig.model_validate({})).run(full_df, config, threshold=0.5)
    text = format_walk_forward_report(report)
    assert "predicted_success_count=" in text
    assert "calibration (mean predicted" in text
    assert "highest bucket reached" in text
    assert "per-setup (base_rate" in text
    assert "setup_classic"[len("setup_"):] in text  # "classic" -- the setup_ prefix is stripped in the report


def test_format_report_can_hide_calibration_and_setup_sections(full_df):
    config = WalkForwardConfig.model_validate({"n_folds": 3})
    report = WalkForwardValidator(ModelConfig.model_validate({})).run(full_df, config)
    text = format_walk_forward_report(report, show_calibration=False, show_setups=False)
    assert "calibration (mean predicted" not in text
    assert "per-setup (base_rate" not in text
