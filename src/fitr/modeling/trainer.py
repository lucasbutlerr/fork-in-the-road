"""ModelTrainer: config-driven wrapper around xgboost.XGBClassifier for the
breakout success/failure classifier, plus TrainedModel, the artifact it
produces (a trained classifier bundled with the metadata needed to use it
correctly later: which columns it expects, in what order, and how its
output maps back to SUCCESS/FAILURE).

Label encoding is explicit and controlled here, not left to XGBoost's own
internal label encoder: SUCCESS -> 1, FAILURE -> 0, always, regardless of
string sort order or library version behavior. This matters because
`class 1 = positive class = SUCCESS` is relied on directly by
predict_proba_success() below and will be relied on again by
ModelEvaluator and LivePredictor in later steps -- it needs to be a fact
about this code, not an accident of how a library happens to sort strings.

Class imbalance is handled via scale_pos_weight (XGBoost's mechanism for
this, since it doesn't have sklearn's class_weight='balanced' string
option) rather than oversampling, per the project's earlier design
decision: oversampling would need to fabricate rows in a domain (market
data) where a fabricated row has no real meaning, whereas scale_pos_weight
just reweights the loss function using the real data as-is.

Persistence note: TrainedModel.save()/load() use joblib rather than
XGBoost's own native save_model()/load_model() format, as of calibration
support being added. That's a deliberate change, not an oversight --
CalibratedClassifierCV (a scikit-learn meta-estimator wrapping XGBoost)
doesn't support XGBoost's native format at all, and hand-rolling custom
serialization for its internals would be fragile across scikit-learn
versions. joblib is scikit-learn's own recommended way to persist any
fitted sklearn-API estimator (including third-party ones like XGBoost's
classifier) and handles the raw and calibrated cases identically. The
tradeoff: artifact portability is now tied to the combination of XGBoost +
scikit-learn versions used when saving, not just XGBoost's version alone
-- pin both in your environment if long-term portability across library
upgrades matters to you.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from fitr.config_schemas.model_schema import ModelConfig
from fitr.labeling.labeler import FAILURE, SUCCESS

logger = logging.getLogger(__name__)

_NON_FEATURE_COLUMNS = {
    "ticker",
    # trigger_reason and forward_return are DatasetBuilder metadata about
    # HOW a row was labeled (added for future return-aware analysis), not
    # predictive inputs -- forward_return in particular directly encodes
    # the outcome being predicted, so including it as a feature would be
    # severe label leakage (the model would trivially "predict" success
    # by reading off its own future answer). date_col and label_col are
    # also excluded, handled separately below since their column names
    # are configurable.
    "trigger_reason",
    "forward_return",
}


@dataclass
class TrainedModel:
    model: xgb.XGBClassifier | CalibratedClassifierCV  # calibrated iff calibration_method != "none"
    feature_columns: list[str]
    label_column: str
    positive_class: str  # always SUCCESS -- kept as a field (not a hardcoded assumption at every call
    negative_class: str  # site) so callers can label output without importing labeler.py's constants
    trained_at: pd.Timestamp
    config: ModelConfig

    def predict_proba_success(self, df: pd.DataFrame) -> np.ndarray:
        """Returns P(SUCCESS) for each row of df. Reindexes df to
        feature_columns in the trained order -- raises a clear KeyError-
        style message (via pandas) if a required column is missing rather
        than silently misaligning columns, which is a hard-to-debug
        failure mode for exactly the kind of columns-must-match-training
        assumption this method relies on."""
        X = df[self.feature_columns]
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self) -> dict[str, float]:
        """Feature importances as {feature_name: importance}, sorted
        descending. The single source of truth for this across the
        project (ModelEvaluator and LivePredictor both call this rather
        than each having their own copy of the same fallback logic --
        that duplication is exactly how format_predictions() in
        predictor.py ended up with an unguarded feature_importances_
        access that a raw XGBClassifier survives but a calibrated model
        doesn't).

        Handles both a raw XGBClassifier and a CalibratedClassifierCV-
        wrapped one transparently: CalibratedClassifierCV doesn't expose
        feature_importances_ directly, so this reaches into the underlying
        frozen base estimator instead. Verified this exact attribute path
        (calibrated_classifiers_[0].estimator) against the real
        scikit-learn CalibratedClassifierCV/FrozenEstimator, but it's
        still reaching into another library's internals, so this falls
        back to an empty dict (never raises) if a future scikit-learn
        version changes that structure."""
        raw = getattr(self.model, "feature_importances_", None)
        if raw is None:
            try:
                raw = self.model.calibrated_classifiers_[0].estimator.feature_importances_
            except (AttributeError, IndexError):
                return {}
        if raw is None:
            return {}
        pairs = sorted(zip(self.feature_columns, raw), key=lambda p: p[1], reverse=True)
        return {name: float(value) for name, value in pairs}

    def save(self, path: str | Path) -> None:
        """Saves the fitted model via joblib (see module docstring for why
        this isn't XGBoost's native format anymore) plus a small JSON
        sidecar with everything needed to use it correctly later: feature
        column order, the label encoding convention, and the config that
        produced it, for provenance."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta = {
            "feature_columns": self.feature_columns,
            "label_column": self.label_column,
            "positive_class": self.positive_class,
            "negative_class": self.negative_class,
            "trained_at": self.trained_at.isoformat(),
            "config": dict(vars(self.config)),
        }
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "TrainedModel":
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        with meta_path.open("r") as f:
            meta = json.load(f)

        model = joblib.load(path)

        return cls(
            model=model,
            feature_columns=meta["feature_columns"],
            label_column=meta["label_column"],
            positive_class=meta["positive_class"],
            negative_class=meta["negative_class"],
            trained_at=pd.Timestamp(meta["trained_at"]),
            config=ModelConfig.model_validate(meta["config"]),
        )


class ModelTrainer:
    def __init__(self, config: ModelConfig):
        self._config = config

    def _resolve_feature_columns(self, df: pd.DataFrame, date_column: str) -> list[str]:
        if self._config.feature_columns != "auto":
            missing = set(self._config.feature_columns) - set(df.columns)
            if missing:
                raise ValueError(f"feature_columns references column(s) not present in the training data: {sorted(missing)}")
            return list(self._config.feature_columns)
        excluded = _NON_FEATURE_COLUMNS | {date_column, self._config.label_column}
        return [c for c in df.columns if c not in excluded]

    def _encode_labels(self, series: pd.Series) -> pd.Series:
        encoded = series.map({SUCCESS: 1, FAILURE: 0})
        if encoded.isna().any():
            bad = sorted(series[encoded.isna()].unique())
            raise ValueError(
                f"Unexpected label value(s) {bad} in column '{self._config.label_column}'; "
                f"expected only {SUCCESS!r} or {FAILURE!r}."
            )
        return encoded.astype(int)

    def _build_classifier(self, scale_pos_weight: float) -> xgb.XGBClassifier:
        params = dict(
            n_estimators=self._config.n_estimators,
            max_depth=self._config.max_depth,
            learning_rate=self._config.learning_rate,
            min_child_weight=self._config.min_child_weight,
            gamma=self._config.gamma,
            subsample=self._config.subsample,
            colsample_bytree=self._config.colsample_bytree,
            colsample_bylevel=self._config.colsample_bylevel,
            colsample_bynode=self._config.colsample_bynode,
            reg_alpha=self._config.reg_alpha,
            reg_lambda=self._config.reg_lambda,
            max_delta_step=self._config.max_delta_step,
            scale_pos_weight=scale_pos_weight,
            grow_policy=self._config.grow_policy,
            max_leaves=self._config.max_leaves,
            max_bin=self._config.max_bin,
            tree_method=self._config.tree_method,
            booster=self._config.booster,
            objective=self._config.objective,
            eval_metric=self._config.eval_metric,
            n_jobs=self._config.n_jobs,
            random_state=self._config.random_state,
            base_score=self._config.base_score,
            verbosity=self._config.verbosity,
            missing=float("nan"),  # matches this project's NaN convention throughout -- deliberately
            # not configurable; every upstream NaN means exactly "insufficient history",
            # and that should always be XGBoost's missing-value sentinel, never anything else.
            **self._config.extra_params,
        )
        return xgb.XGBClassifier(**params)

    def _resolve_scale_pos_weight(self, y: pd.Series) -> float:
        if self._config.class_weight_mode != "balanced":
            return self._config.scale_pos_weight
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            logger.warning(
                "Training data has only one class present (SUCCESS=%d, FAILURE=%d) -- "
                "scale_pos_weight left at 1.0 instead of computing a ratio.",
                n_pos, n_neg,
            )
            return 1.0
        weight = n_neg / n_pos
        logger.info("class_weight_mode=balanced: SUCCESS=%d, FAILURE=%d -> scale_pos_weight=%.3f", n_pos, n_neg, weight)
        return weight

    def fit(self, train_df: pd.DataFrame, date_column: str = "date") -> TrainedModel:
        if train_df.empty:
            raise ValueError("Cannot fit on an empty training set.")

        feature_columns = self._resolve_feature_columns(train_df, date_column)

        # Calibration slice is carved off FIRST, before anything else --
        # chronological order ends up [fit] -> [early-stopping validation]
        # -> [calibration] -> (test.csv, never touched here). scale_pos_weight
        # and early stopping both then operate only on `main_df`, so the
        # calibration slice never leaks into the base model's own fitting.
        main_df, calibration_df = self._carve_calibration_slice(train_df, feature_columns, date_column)

        y_all = self._encode_labels(main_df[self._config.label_column])
        scale_pos_weight = self._resolve_scale_pos_weight(y_all)
        classifier = self._build_classifier(scale_pos_weight)

        if self._config.early_stopping_rounds is not None:
            classifier = self._fit_with_early_stopping(classifier, main_df, feature_columns, y_all, date_column)
        else:
            logger.info("Fitting on %d rows, %d features, no early stopping.", len(main_df), len(feature_columns))
            classifier.fit(main_df[feature_columns], y_all)

        if calibration_df is not None:
            classifier = self._calibrate(classifier, calibration_df, feature_columns)

        return TrainedModel(
            model=classifier,
            feature_columns=feature_columns,
            label_column=self._config.label_column,
            positive_class=SUCCESS,
            negative_class=FAILURE,
            trained_at=pd.Timestamp.now(),
            config=self._config,
        )

    def _carve_calibration_slice(
        self, train_df: pd.DataFrame, feature_columns: list[str], date_column: str
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Returns (main_df, calibration_df). calibration_df is None if
        calibration is disabled, or if the requested slice would be too
        small/single-class to fit a meaningful calibration on -- in either
        case main_df is just the original train_df, unchanged."""
        if self._config.calibration_method == "none":
            return train_df, None

        sorted_df = train_df.sort_values(date_column)
        split_idx = int(len(sorted_df) * (1 - self._config.calibration_fraction))
        main_df, calibration_df = sorted_df.iloc[:split_idx], sorted_df.iloc[split_idx:]

        calib_y = self._encode_labels(calibration_df[self._config.label_column])
        if len(calibration_df) < 10 or calib_y.nunique() < 2:
            logger.warning(
                "Calibration slice too small or single-class (%d rows) -- skipping calibration, "
                "the model will return raw (uncalibrated) probabilities instead.",
                len(calibration_df),
            )
            return train_df, None

        return main_df, calibration_df

    def _calibrate(self, classifier: xgb.XGBClassifier, calibration_df: pd.DataFrame, feature_columns: list[str]):
        calib_y = self._encode_labels(calibration_df[self._config.label_column])
        logger.info(
            "Calibrating (%s) on %d held-out rows (%.0f%% of train_df, most recent by date).",
            self._config.calibration_method, len(calibration_df), self._config.calibration_fraction * 100,
        )
        # FrozenEstimator marks the already-fitted classifier as "don't
        # refit me" -- CalibratedClassifierCV then only fits the
        # calibration mapping on calibration_df, never touching the base
        # model's own training data again. (This replaces the older
        # cv="prefit" parameter, removed in modern scikit-learn -- verified
        # directly against the installed scikit-learn version rather than
        # assumed, given how many XGBoost API surprises turned up earlier
        # in this project.)
        calibrated = CalibratedClassifierCV(FrozenEstimator(classifier), method=self._config.calibration_method)
        calibrated.fit(calibration_df[feature_columns], calib_y)
        return calibrated

    def _fit_with_early_stopping(
        self, classifier: xgb.XGBClassifier, train_df: pd.DataFrame, feature_columns: list[str],
        y_all: pd.Series, date_column: str,
    ) -> xgb.XGBClassifier:
        sorted_df = train_df.sort_values(date_column)
        sorted_y = y_all.loc[sorted_df.index]
        split_idx = int(len(sorted_df) * (1 - self._config.early_stopping_validation_fraction))
        fit_df, eval_df = sorted_df.iloc[:split_idx], sorted_df.iloc[split_idx:]
        fit_y, eval_y = sorted_y.iloc[:split_idx], sorted_y.iloc[split_idx:]

        if len(eval_df) < 10 or eval_y.nunique() < 2:
            logger.warning(
                "Early-stopping validation slice too small or single-class (%d rows) -- "
                "fitting on the full training set without early stopping instead.",
                len(eval_df),
            )
            classifier.fit(train_df[feature_columns], y_all)
            return classifier

        logger.info(
            "Fitting with early stopping: %d fit rows, %d validation rows (tail %.0f%% by date), patience=%d.",
            len(fit_df), len(eval_df), self._config.early_stopping_validation_fraction * 100,
            self._config.early_stopping_rounds,
        )
        classifier.set_params(early_stopping_rounds=self._config.early_stopping_rounds)
        classifier.fit(
            fit_df[feature_columns], fit_y,
            eval_set=[(eval_df[feature_columns], eval_y)],
            verbose=False,
        )
        return classifier
