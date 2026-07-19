"""Tests for fitr.config_schemas.labeling_schema."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from fitr.config_schemas.labeling_schema import LabelingConfig


def _valid(**overrides):
    base = {"horizon_trading_days": 10, "target_return": 0.08, "stop_loss_return": -0.04, "evaluation_method": "first_touch"}
    base.update(overrides)
    return base


def test_valid_config_loads():
    config = LabelingConfig.model_validate(_valid())
    assert config.horizon_trading_days == 10
    assert config.target_return == 0.08
    assert config.stop_loss_return == -0.04
    assert config.evaluation_method == "first_touch"


def test_end_of_horizon_is_accepted():
    config = LabelingConfig.model_validate(_valid(evaluation_method="end_of_horizon"))
    assert config.evaluation_method == "end_of_horizon"


def test_unknown_evaluation_method_rejected():
    with pytest.raises(ValidationError):
        LabelingConfig.model_validate(_valid(evaluation_method="halfway_there"))


def test_evaluation_method_has_no_default_and_is_required():
    data = _valid()
    del data["evaluation_method"]
    with pytest.raises(ValidationError):
        LabelingConfig.model_validate(data)


def test_zero_horizon_rejected():
    with pytest.raises(ValidationError):
        LabelingConfig.model_validate(_valid(horizon_trading_days=0))


def test_negative_horizon_rejected():
    with pytest.raises(ValidationError):
        LabelingConfig.model_validate(_valid(horizon_trading_days=-5))


def test_excessive_horizon_rejected():
    with pytest.raises(ValidationError):
        LabelingConfig.model_validate(_valid(horizon_trading_days=10_000))


def test_non_positive_target_return_rejected():
    with pytest.raises(ValidationError):
        LabelingConfig.model_validate(_valid(target_return=0.0))
    with pytest.raises(ValidationError):
        LabelingConfig.model_validate(_valid(target_return=-0.05))


def test_non_negative_stop_loss_rejected():
    with pytest.raises(ValidationError):
        LabelingConfig.model_validate(_valid(stop_loss_return=0.0))
    with pytest.raises(ValidationError):
        LabelingConfig.model_validate(_valid(stop_loss_return=0.04))
