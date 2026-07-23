"""Tests for fitr.config_schemas.position_sizing_schema."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from fitr.config_schemas.position_sizing_schema import PositionSizingConfig


def _valid(**overrides):
    base = {"method": "fixed_fractional", "target_return": 0.08, "stop_loss_return": -0.04}
    base.update(overrides)
    return base


def test_valid_config_loads_with_defaults():
    config = PositionSizingConfig.model_validate(_valid())
    assert config.method == "fixed_fractional"
    assert config.risk_fraction == 0.01
    assert config.kelly_multiplier == 0.01
    assert config.max_position_fraction == 0.10
    assert config.min_position_fraction == 0.0


def test_fractional_kelly_method_accepted():
    config = PositionSizingConfig.model_validate(_valid(method="fractional_kelly"))
    assert config.method == "fractional_kelly"


def test_unknown_method_rejected():
    with pytest.raises(ValidationError):
        PositionSizingConfig.model_validate(_valid(method="martingale"))


def test_non_positive_target_return_rejected():
    with pytest.raises(ValidationError):
        PositionSizingConfig.model_validate(_valid(target_return=0.0))
    with pytest.raises(ValidationError):
        PositionSizingConfig.model_validate(_valid(target_return=-0.05))


def test_non_negative_stop_loss_rejected():
    with pytest.raises(ValidationError):
        PositionSizingConfig.model_validate(_valid(stop_loss_return=0.0))
    with pytest.raises(ValidationError):
        PositionSizingConfig.model_validate(_valid(stop_loss_return=0.04))


def test_min_exceeding_max_rejected():
    with pytest.raises(ValidationError):
        PositionSizingConfig.model_validate(_valid(min_position_fraction=0.5, max_position_fraction=0.1))


def test_min_equal_to_max_is_allowed():
    config = PositionSizingConfig.model_validate(_valid(min_position_fraction=0.1, max_position_fraction=0.1))
    assert config.min_position_fraction == config.max_position_fraction


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("risk_fraction", 0.0),
        ("risk_fraction", 1.5),
        ("kelly_multiplier", 0.0),
        ("kelly_multiplier", 1.5),
        ("max_position_fraction", 0.0),
        ("max_position_fraction", 1.5),
        ("min_position_fraction", -0.1),
    ],
)
def test_out_of_range_fields_rejected(field, bad_value):
    with pytest.raises(ValidationError):
        PositionSizingConfig.model_validate(_valid(**{field: bad_value}))
