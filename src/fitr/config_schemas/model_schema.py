"""Pydantic schema for model.yaml -- configuration for ModelTrainer's
XGBoost classifier.

On "every parameter configurable": every XGBoost hyperparameter someone
would realistically want to tune for a binary tabular classifier is a
named, validated field below (24 of them, covering tree structure,
regularization, sampling, learning rate, and class-imbalance handling).
A handful of XGBoost's more exotic/rarely-used parameters aren't given
their own named field -- things like monotone_constraints or
interaction_constraints, which need per-feature specification and are a
different kind of config entirely, or GPU device selection, which isn't
relevant to a CPU-based recreational setup. Nothing is actually
unreachable, though: `extra_params` is a passthrough dict merged directly
into the XGBClassifier constructor, so any XGBoost keyword argument at all
can be set through it, even ones this schema has no explicit opinion
about. The 24 named fields exist because they're worth documenting and
validating individually, not because they're the only ones accessible.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Every field name below that maps directly to an XGBClassifier
# constructor keyword argument -- used to reject an extra_params entry
# that collides with one of these (which would otherwise blow up as a
# duplicate-keyword TypeError when the classifier is actually built).
_XGBOOST_NAMED_FIELDS = {
    "n_estimators", "max_depth", "learning_rate", "min_child_weight", "gamma",
    "subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode",
    "reg_alpha", "reg_lambda", "max_delta_step", "scale_pos_weight",
    "grow_policy", "max_leaves", "max_bin", "tree_method", "booster",
    "objective", "eval_metric", "n_jobs", "random_state", "base_score", "verbosity",
}


class ModelConfig(BaseModel):
    # ---- tree structure -----------------------------------------------
    n_estimators: int = Field(300, gt=0, description="Number of boosting rounds (trees).")
    max_depth: int = Field(5, gt=0, description="Max depth per tree.")
    max_leaves: int = Field(0, ge=0, description="Max leaves per tree; 0 = no limit. Only used with grow_policy=lossguide.")
    grow_policy: Literal["depthwise", "lossguide"] = Field(
        "depthwise", description="depthwise splits level by level; lossguide splits whichever leaf reduces loss most."
    )
    max_bin: int = Field(256, gt=1, description="Max number of bins for continuous feature histograms (tree_method=hist/approx).")
    tree_method: Literal["auto", "exact", "approx", "hist"] = Field("hist", description="hist is the modern default -- fast, low memory.")
    booster: Literal["gbtree", "dart", "gblinear"] = Field("gbtree", description="gbtree is the standard GBT booster.")

    # ---- learning rate / regularization --------------------------------
    learning_rate: float = Field(0.05, gt=0, le=1, description="Shrinkage applied to each tree's contribution (aka eta).")
    gamma: float = Field(0.0, ge=0, description="Min loss reduction required to make a further split (aka min_split_loss).")
    min_child_weight: float = Field(1.0, ge=0, description="Min sum of instance weight needed in a child to keep splitting.")
    max_delta_step: float = Field(0.0, ge=0, description="Max delta step each tree's weight estimate is allowed; 0 = no constraint.")
    reg_alpha: float = Field(0.0, ge=0, description="L1 regularization on leaf weights.")
    reg_lambda: float = Field(1.0, ge=0, description="L2 regularization on leaf weights.")

    # ---- sampling -----------------------------------------------------
    subsample: float = Field(0.8, gt=0, le=1, description="Fraction of rows sampled per tree.")
    colsample_bytree: float = Field(0.8, gt=0, le=1, description="Fraction of features sampled per tree.")
    colsample_bylevel: float = Field(1.0, gt=0, le=1, description="Fraction of features sampled per tree level.")
    colsample_bynode: float = Field(1.0, gt=0, le=1, description="Fraction of features sampled per split.")

    # ---- class imbalance ------------------------------------------------
    class_weight_mode: Literal["none", "balanced"] = Field(
        "balanced",
        description="'balanced': ModelTrainer computes scale_pos_weight from the training data's actual "
        "class counts (count(FAILURE)/count(SUCCESS)) and overrides the value below. 'none': use "
        "scale_pos_weight literally as given.",
    )
    scale_pos_weight: float = Field(
        1.0, gt=0, description="Only used when class_weight_mode is 'none' -- ignored/overridden under 'balanced'."
    )

    # ---- objective / evaluation -----------------------------------------
    objective: str = Field("binary:logistic", min_length=1, description="XGBoost objective. binary:logistic is standard for this project.")
    eval_metric: str | list[str] = Field(["logloss", "auc"], description="Metric(s) XGBoost tracks/reports during training.")

    # ---- early stopping (fitr-level orchestration, not a raw XGBoost kwarg directly) --
    early_stopping_rounds: int | None = Field(
        None,
        gt=0,
        description="If set, ModelTrainer carves a time-based validation slice off the TAIL of the "
        "training data itself (see early_stopping_validation_fraction) and stops boosting once this "
        "many rounds pass without improvement. Never uses test.csv for this -- that would leak the "
        "held-out set into model selection.",
    )
    early_stopping_validation_fraction: float = Field(
        0.15, gt=0, lt=1, description="Fraction of train_df (by date, most recent rows) held out for early stopping."
    )

    # ---- reproducibility / runtime --------------------------------------
    n_jobs: int = Field(-1, description="-1 = use all available cores.")
    random_state: int = Field(42, description="Seed for reproducible training runs.")
    base_score: float | None = Field(None, description="Initial prediction score; None = XGBoost's own default.")
    verbosity: int = Field(1, ge=0, le=3, description="0=silent, 1=warning, 2=info, 3=debug.")

    # ---- fitr-level orchestration (not passed to XGBClassifier at all) --
    feature_columns: Literal["auto"] | list[str] = Field(
        "auto", description="'auto': every train_df column except ticker/date/label. Or an explicit list to override."
    )
    label_column: str = Field("label", min_length=1)

    # ---- escape hatch ----------------------------------------------------
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Passthrough kwargs merged directly into the XGBClassifier constructor, for any "
        "XGBoost parameter not named above. Must not duplicate a named field.",
    )

    @model_validator(mode="after")
    def extra_params_must_not_shadow_named_fields(self) -> "ModelConfig":
        overlap = set(self.extra_params) & _XGBOOST_NAMED_FIELDS
        if overlap:
            raise ValueError(
                f"extra_params contains field(s) already defined by name: {sorted(overlap)}. "
                "Set them directly via their named field instead of through extra_params."
            )
        return self
