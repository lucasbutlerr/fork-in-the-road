"""Tests for fitr.config_schemas.scanning_schema. All in-memory dicts, no
network, no real config/ files required."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from fitr.config_schemas.scanning_schema import ConditionSpec, ScanningConfig


def _minimal_metric(name="rsi_14"):
    return {"name": name, "function": "rsi", "params": {}}


def test_valid_minimal_config_loads():
    config = ScanningConfig.model_validate(
        {
            "metrics": [_minimal_metric()],
            "setups": [{"name": "s1", "conditions": [{"metric": "rsi_14", "operator": ">=", "value": 60}]}],
        }
    )
    assert len(config.metrics) == 1
    assert len(config.setups) == 1


def test_unknown_operator_is_rejected():
    with pytest.raises(ValidationError):
        ConditionSpec(metric="rsi_14", operator="~=", value=60)


def test_between_requires_two_item_list():
    with pytest.raises(ValidationError):
        ConditionSpec(metric="rsi_14", operator="between", value=60)
    with pytest.raises(ValidationError):
        ConditionSpec(metric="rsi_14", operator="between", value=[60])


def test_between_requires_ascending_bounds():
    with pytest.raises(ValidationError):
        ConditionSpec(metric="rsi_14", operator="between", value=[80, 60])


def test_between_accepts_valid_pair():
    cond = ConditionSpec(metric="rsi_14", operator="between", value=[60, 80])
    assert cond.value == [60, 80]


def test_non_between_operator_rejects_list_value():
    with pytest.raises(ValidationError):
        ConditionSpec(metric="rsi_14", operator=">=", value=[60, 80])


def test_default_logic_is_and():
    config = ScanningConfig.model_validate(
        {
            "metrics": [_minimal_metric()],
            "setups": [{"name": "s1", "conditions": [{"metric": "rsi_14", "operator": ">=", "value": 60}]}],
        }
    )
    assert config.setups[0].logic == "AND"


def test_duplicate_metric_names_rejected():
    with pytest.raises(ValidationError):
        ScanningConfig.model_validate(
            {
                "metrics": [_minimal_metric("dup"), {"name": "dup", "function": "atr_normalized", "params": {}}],
                "setups": [{"name": "s1", "conditions": [{"metric": "dup", "operator": ">=", "value": 1}]}],
            }
        )


def test_duplicate_setup_names_rejected():
    with pytest.raises(ValidationError):
        ScanningConfig.model_validate(
            {
                "metrics": [_minimal_metric()],
                "setups": [
                    {"name": "dup", "conditions": [{"metric": "rsi_14", "operator": ">=", "value": 60}]},
                    {"name": "dup", "conditions": [{"metric": "rsi_14", "operator": "<=", "value": 40}]},
                ],
            }
        )


def test_condition_referencing_undefined_metric_is_rejected():
    with pytest.raises(ValidationError) as exc_info:
        ScanningConfig.model_validate(
            {
                "metrics": [_minimal_metric("rsi_14")],
                "setups": [{"name": "s1", "conditions": [{"metric": "not_defined", "operator": ">=", "value": 1}]}],
            }
        )
    assert "not_defined" in str(exc_info.value)


def test_empty_setups_list_rejected():
    with pytest.raises(ValidationError):
        ScanningConfig.model_validate({"metrics": [_minimal_metric()], "setups": []})


def test_setup_with_empty_conditions_rejected():
    with pytest.raises(ValidationError):
        ScanningConfig.model_validate(
            {"metrics": [_minimal_metric()], "setups": [{"name": "s1", "conditions": []}]}
        )


def test_reuses_feature_spec_validation_for_metrics():
    # metrics go through the exact same FeatureSpec validators as
    # features.yaml -- an unknown function should be rejected the same way.
    with pytest.raises(ValidationError):
        ScanningConfig.model_validate(
            {
                "metrics": [{"name": "bad", "function": "not_a_real_function", "params": {}}],
                "setups": [{"name": "s1", "conditions": [{"metric": "bad", "operator": ">=", "value": 1}]}],
            }
        )
