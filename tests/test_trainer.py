"""Tests for ModelTrainer and TrainedModel. Trains on small synthetic
in-memory DataFrames (not Step 4's real pipeline output) -- this is a pure
orchestration layer around XGBoost, so what needs verifying is that the
right data reaches XGBoost in the right shape, not XGBoost's own training
math (a well-tested external library, not this project's job to re-verify)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from fitr.config_schemas.model_schema import ModelConfig
from fitr.modeling.trainer import ModelTrainer, TrainedModel


def make_train_df(n=200, n_success_frac=0.3, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    n_success = int(n * n_success_frac)
    labels = ["SUCCESS"] * n_success + ["FAILURE"] * (n - n_success)
    rng.shuffle(labels)
    return pd.DataFrame(
        {
            "ticker": [f"T{i % 20:03d}" for i in range(n)],
            "date": dates,
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(5, 2, n),
            "feat_c": np.where(rng.random(n) < 0.1, np.nan, rng.normal(0, 1, n)),  # NaN like real features
            "setup_classic": rng.random(n) < 0.5,
            "label": labels,
        }
    )


def test_auto_feature_columns_excludes_ticker_date_label():
    trainer = ModelTrainer(ModelConfig.model_validate({}))
    trained = trainer.fit(make_train_df())
    assert set(trained.feature_columns) == {"feat_a", "feat_b", "feat_c", "setup_classic"}


def test_auto_feature_columns_excludes_trigger_reason_and_forward_return():
    # Safety-critical: forward_return directly encodes the outcome being
    # predicted. If "auto" ever treated it as a feature, the model would
    # trivially "predict" success by reading off its own future answer.
    # DatasetBuilder writes both of these columns alongside `label`.
    df = make_train_df()
    df["trigger_reason"] = "target_hit"
    df["forward_return"] = 0.08
    trained = ModelTrainer(ModelConfig.model_validate({})).fit(df)
    assert "trigger_reason" not in trained.feature_columns
    assert "forward_return" not in trained.feature_columns


def test_positive_and_negative_class_are_fixed_to_success_failure():
    trained = ModelTrainer(ModelConfig.model_validate({})).fit(make_train_df())
    assert trained.positive_class == "SUCCESS"
    assert trained.negative_class == "FAILURE"


def test_explicit_feature_columns_are_respected():
    config = ModelConfig.model_validate({"feature_columns": ["feat_a", "feat_b"]})
    trained = ModelTrainer(config).fit(make_train_df())
    assert trained.feature_columns == ["feat_a", "feat_b"]


def test_explicit_feature_columns_referencing_missing_column_raises():
    config = ModelConfig.model_validate({"feature_columns": ["feat_a", "not_a_real_column"]})
    with pytest.raises(ValueError, match="not_a_real_column"):
        ModelTrainer(config).fit(make_train_df())


def test_unexpected_label_value_raises():
    df = make_train_df()
    df.loc[0, "label"] = "MAYBE"
    with pytest.raises(ValueError, match="MAYBE"):
        ModelTrainer(ModelConfig.model_validate({})).fit(df)


def test_balanced_scale_pos_weight_matches_failure_over_success_ratio():
    df = make_train_df(n=200, n_success_frac=0.2)  # 40 success, 160 failure
    n_success = (df["label"] == "SUCCESS").sum()
    n_failure = (df["label"] == "FAILURE").sum()
    expected = n_failure / n_success

    trained = ModelTrainer(ModelConfig.model_validate({})).fit(df)
    assert trained.model.get_params()["scale_pos_weight"] == pytest.approx(expected)


def test_class_weight_mode_none_uses_configured_value_literally():
    config = ModelConfig.model_validate({"class_weight_mode": "none", "scale_pos_weight": 3.5})
    trained = ModelTrainer(config).fit(make_train_df(n_success_frac=0.2))
    assert trained.model.get_params()["scale_pos_weight"] == 3.5


def test_single_class_training_data_falls_back_to_scale_pos_weight_one():
    df = make_train_df(n=50, n_success_frac=0.0)  # all FAILURE
    trained = ModelTrainer(ModelConfig.model_validate({})).fit(df)
    assert trained.model.get_params()["scale_pos_weight"] == 1.0


def test_extra_params_reach_the_underlying_classifier():
    config = ModelConfig.model_validate({"extra_params": {"monotone_constraints": "(1,0,0,0)"}})
    trained = ModelTrainer(config).fit(make_train_df())
    assert trained.model.get_params().get("monotone_constraints") == "(1,0,0,0)"


def test_early_stopping_uses_a_time_based_tail_slice():
    config = ModelConfig.model_validate({"early_stopping_rounds": 10, "early_stopping_validation_fraction": 0.2})
    trained = ModelTrainer(config).fit(make_train_df(n=200))
    assert trained.model.best_iteration is not None
    assert len(trained.model.evals_result()) > 0


def test_early_stopping_falls_back_gracefully_on_a_tiny_dataset(caplog):
    config = ModelConfig.model_validate({"early_stopping_rounds": 10, "early_stopping_validation_fraction": 0.15})
    # Deliberately checks the trainer's OWN logged behavior rather than
    # introspecting XGBoost's internal state (best_iteration / evals_result())
    # for whether early stopping ran -- those have shown two different real
    # exception behaviors across environments (AttributeError, then
    # XGBoostError) that this sandbox can't reliably verify against the real
    # library. What actually needs confirming is fitr's own logic: did it
    # correctly detect the too-small validation slice and fall back to a
    # plain fit? That's fully within this project's control and is what
    # this checks instead.
    with caplog.at_level(logging.WARNING):
        trained = ModelTrainer(config).fit(make_train_df(n=20))  # 0.15 * 20 = 3 rows, below the 10-row floor
    assert trained.model is not None
    assert any("fitting on the full training set without early stopping" in r.message for r in caplog.records)


def test_empty_training_data_raises():
    with pytest.raises(ValueError):
        ModelTrainer(ModelConfig.model_validate({})).fit(make_train_df().iloc[0:0])


def test_save_and_load_round_trip_produces_identical_predictions(tmp_path):
    train_df = make_train_df()
    trained = ModelTrainer(ModelConfig.model_validate({})).fit(train_df)

    model_path = tmp_path / "model.json"
    trained.save(model_path)
    assert model_path.exists()
    assert model_path.with_suffix(".json.meta.json").exists()

    reloaded = TrainedModel.load(model_path)
    assert reloaded.feature_columns == trained.feature_columns
    assert reloaded.label_column == trained.label_column
    assert reloaded.positive_class == trained.positive_class
    assert reloaded.config.n_estimators == trained.config.n_estimators

    original_probs = trained.predict_proba_success(train_df)
    reloaded_probs = reloaded.predict_proba_success(train_df)
    assert np.allclose(original_probs, reloaded_probs)


def test_predict_proba_success_is_column_order_independent():
    train_df = make_train_df()
    trained = ModelTrainer(ModelConfig.model_validate({})).fit(train_df)

    shuffled = train_df[["label", "feat_c", "ticker", "feat_a", "date", "feat_b", "setup_classic"]]
    assert np.allclose(trained.predict_proba_success(shuffled), trained.predict_proba_success(train_df))


def test_feature_importances_on_raw_model():
    trained = ModelTrainer(ModelConfig.model_validate({})).fit(make_train_df())
    importances = trained.feature_importances()
    assert set(importances.keys()) == set(trained.feature_columns)
    assert all(isinstance(v, float) for v in importances.values())


def test_feature_importances_on_calibrated_model_does_not_raise():
    # This is the exact scenario that crashed in production: a calibrated
    # model wrapped in CalibratedClassifierCV doesn't expose
    # feature_importances_ directly, and this is what both ModelEvaluator
    # and LivePredictor's format_predictions() rely on to not crash.
    config = ModelConfig.model_validate({"calibration_method": "sigmoid", "calibration_fraction": 0.2})
    trained = ModelTrainer(config).fit(make_train_df(n=400))
    importances = trained.feature_importances()
    assert len(importances) > 0
    assert set(importances.keys()) == set(trained.feature_columns)


def test_calibration_disabled_by_default_leaves_raw_classifier():
    import xgboost as xgb

    trained = ModelTrainer(ModelConfig.model_validate({})).fit(make_train_df())
    assert isinstance(trained.model, xgb.XGBClassifier)


def test_sigmoid_calibration_wraps_model_in_calibrated_classifier():
    from sklearn.calibration import CalibratedClassifierCV

    config = ModelConfig.model_validate({"calibration_method": "sigmoid", "calibration_fraction": 0.2})
    trained = ModelTrainer(config).fit(make_train_df(n=400))
    assert isinstance(trained.model, CalibratedClassifierCV)


def test_isotonic_calibration_also_wraps_model():
    from sklearn.calibration import CalibratedClassifierCV

    config = ModelConfig.model_validate({"calibration_method": "isotonic", "calibration_fraction": 0.25})
    trained = ModelTrainer(config).fit(make_train_df(n=400))
    assert isinstance(trained.model, CalibratedClassifierCV)


def test_calibrated_model_predict_proba_success_still_works():
    config = ModelConfig.model_validate({"calibration_method": "sigmoid", "calibration_fraction": 0.2})
    trained = ModelTrainer(config).fit(make_train_df(n=400))
    probs = trained.predict_proba_success(make_train_df(n=50, seed=99))
    assert len(probs) == 50
    assert np.all((probs >= 0) & (probs <= 1))


def test_calibration_falls_back_to_raw_model_on_tiny_slice():
    import xgboost as xgb

    config = ModelConfig.model_validate({"calibration_method": "sigmoid", "calibration_fraction": 0.15})
    trained = ModelTrainer(config).fit(make_train_df(n=20))  # 0.15*20=3 rows, below the 10-row floor
    assert isinstance(trained.model, xgb.XGBClassifier)


def test_early_stopping_and_calibration_together_do_not_crash():
    config = ModelConfig.model_validate(
        {
            "calibration_method": "sigmoid",
            "calibration_fraction": 0.15,
            "early_stopping_rounds": 5,
            "early_stopping_validation_fraction": 0.15,
        }
    )
    trained = ModelTrainer(config).fit(make_train_df(n=500))
    assert trained.model is not None


def test_calibrated_save_load_round_trip_reproduces_predictions(tmp_path):
    from sklearn.calibration import CalibratedClassifierCV

    config = ModelConfig.model_validate({"calibration_method": "sigmoid", "calibration_fraction": 0.2})
    train_df = make_train_df(n=400)
    trained = ModelTrainer(config).fit(train_df)

    model_path = tmp_path / "model.joblib"
    trained.save(model_path)
    assert model_path.exists()
    assert model_path.with_suffix(".joblib.meta.json").exists()

    reloaded = TrainedModel.load(model_path)
    assert isinstance(reloaded.model, CalibratedClassifierCV)
    assert reloaded.config.calibration_method == "sigmoid"

    test_df = make_train_df(n=50, seed=99)
    assert np.allclose(trained.predict_proba_success(test_df), reloaded.predict_proba_success(test_df))
