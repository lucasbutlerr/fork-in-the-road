"""Tests for fitr.config_schemas.model_schema."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from fitr.config_schemas.model_schema import ModelConfig


def test_all_defaults_load_with_no_input():
    config = ModelConfig.model_validate({})
    assert config.n_estimators == 300
    assert config.max_depth == 5
    assert config.class_weight_mode == "balanced"
    assert config.early_stopping_rounds is None
    assert config.feature_columns == "auto"


def test_every_named_field_is_independently_overridable():
    overrides = {
        "n_estimators": 500,
        "max_depth": 8,
        "max_leaves": 64,
        "grow_policy": "lossguide",
        "max_bin": 128,
        "tree_method": "approx",
        "booster": "dart",
        "learning_rate": 0.1,
        "gamma": 0.5,
        "min_child_weight": 2.0,
        "max_delta_step": 1.0,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 0.9,
        "colsample_bynode": 0.9,
        "class_weight_mode": "none",
        "scale_pos_weight": 2.0,
        "objective": "binary:logitraw",
        "eval_metric": "auc",
        "early_stopping_rounds": 20,
        "early_stopping_validation_fraction": 0.2,
        "n_jobs": 4,
        "random_state": 7,
        "base_score": 0.5,
        "verbosity": 2,
        "feature_columns": ["feat_a", "feat_b"],
        "label_column": "outcome",
    }
    config = ModelConfig.model_validate(overrides)
    for key, value in overrides.items():
        assert getattr(config, key) == value


def test_extra_params_accepts_arbitrary_xgboost_kwargs():
    config = ModelConfig.model_validate({"extra_params": {"monotone_constraints": "(1,0,-1)"}})
    assert config.extra_params == {"monotone_constraints": "(1,0,-1)"}


def test_extra_params_colliding_with_named_field_is_rejected():
    with pytest.raises(ValidationError) as exc_info:
        ModelConfig.model_validate({"extra_params": {"max_depth": 10}})
    assert "max_depth" in str(exc_info.value)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("n_estimators", 0),
        ("max_depth", 0),
        ("learning_rate", 0.0),
        ("learning_rate", 1.5),
        ("subsample", 0.0),
        ("subsample", 1.5),
        ("colsample_bytree", 1.5),
        ("gamma", -1.0),
        ("min_child_weight", -1.0),
        ("reg_alpha", -1.0),
        ("reg_lambda", -1.0),
        ("scale_pos_weight", 0.0),
        ("early_stopping_validation_fraction", 0.0),
        ("early_stopping_validation_fraction", 1.0),
        ("verbosity", -1),
        ("verbosity", 4),
    ],
)
def test_out_of_range_numeric_fields_rejected(field, bad_value):
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({field: bad_value})


def test_early_stopping_rounds_none_is_valid_default():
    config = ModelConfig.model_validate({"early_stopping_rounds": None})
    assert config.early_stopping_rounds is None


def test_early_stopping_rounds_zero_or_negative_rejected():
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"early_stopping_rounds": 0})
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"early_stopping_rounds": -5})


def test_unknown_grow_policy_rejected():
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"grow_policy": "sideways"})


def test_unknown_tree_method_rejected():
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"tree_method": "quantum"})


def test_unknown_booster_rejected():
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"booster": "gblasso"})


def test_unknown_class_weight_mode_rejected():
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"class_weight_mode": "aggressive"})


def test_eval_metric_accepts_single_string_or_list():
    config1 = ModelConfig.model_validate({"eval_metric": "auc"})
    assert config1.eval_metric == "auc"
    config2 = ModelConfig.model_validate({"eval_metric": ["auc", "logloss"]})
    assert config2.eval_metric == ["auc", "logloss"]


def test_feature_columns_accepts_auto_or_explicit_list():
    assert ModelConfig.model_validate({"feature_columns": "auto"}).feature_columns == "auto"
    explicit = ModelConfig.model_validate({"feature_columns": ["a", "b", "c"]})
    assert explicit.feature_columns == ["a", "b", "c"]


def test_calibration_defaults_to_none():
    config = ModelConfig.model_validate({})
    assert config.calibration_method == "none"
    assert config.calibration_fraction == 0.15


def test_calibration_method_accepts_sigmoid_and_isotonic():
    assert ModelConfig.model_validate({"calibration_method": "sigmoid"}).calibration_method == "sigmoid"
    assert ModelConfig.model_validate({"calibration_method": "isotonic"}).calibration_method == "isotonic"


def test_unknown_calibration_method_rejected():
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"calibration_method": "platt"})


@pytest.mark.parametrize("bad_value", [0.0, 1.0, -0.1, 1.1])
def test_calibration_fraction_out_of_range_rejected(bad_value):
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"calibration_fraction": bad_value})
