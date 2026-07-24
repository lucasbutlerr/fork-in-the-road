"""Tests for fitr.config_schemas.walk_forward_schema."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from fitr.config_schemas.walk_forward_schema import WalkForwardConfig


def test_valid_config_loads_with_defaults():
    config = WalkForwardConfig.model_validate({"n_folds": 4})
    assert config.n_folds == 4
    assert config.window_type == "expanding"
    assert config.rolling_train_window_days is None
    assert config.initial_train_fraction == 0.5
    assert config.min_test_rows == 20


def test_rolling_without_window_size_rejected():
    with pytest.raises(ValidationError, match="rolling_train_window_days"):
        WalkForwardConfig.model_validate({"n_folds": 4, "window_type": "rolling"})


def test_rolling_with_window_size_accepted():
    config = WalkForwardConfig.model_validate(
        {"n_folds": 4, "window_type": "rolling", "rolling_train_window_days": 365}
    )
    assert config.rolling_train_window_days == 365


def test_expanding_does_not_require_window_size():
    config = WalkForwardConfig.model_validate({"n_folds": 4, "window_type": "expanding"})
    assert config.rolling_train_window_days is None


def test_n_folds_must_be_positive():
    with pytest.raises(ValidationError):
        WalkForwardConfig.model_validate({"n_folds": 0})
    with pytest.raises(ValidationError):
        WalkForwardConfig.model_validate({"n_folds": -1})


def test_initial_train_fraction_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        WalkForwardConfig.model_validate({"n_folds": 4, "initial_train_fraction": 0.0})
    with pytest.raises(ValidationError):
        WalkForwardConfig.model_validate({"n_folds": 4, "initial_train_fraction": 1.0})


def test_unknown_window_type_rejected():
    with pytest.raises(ValidationError):
        WalkForwardConfig.model_validate({"n_folds": 4, "window_type": "sliding"})
