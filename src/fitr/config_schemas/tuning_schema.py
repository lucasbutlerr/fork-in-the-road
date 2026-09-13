"""Pydantic schema for tuning.yaml -- the search space and trial budget
for the joint model+features+labeling tuning study (Phase 3 Step 5).
scanning_criteria.yaml is deliberately NOT part of this: it's the one
config that's genuinely expensive to vary (any change forces a real
rescan), so it gets its own, separately-budgeted study later rather than
sharing this one's per-trial cost.

Every tunable parameter is expressed as a ParamRange (low/high, optionally
log-scaled) rather than hardcoded in Python -- keeping with this
project's "only the config files should need editing" principle. Search
space changes are config edits, not code edits.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ParamRange(BaseModel):
    low: float
    high: float
    log: bool = Field(False, description="Sample on a log scale -- appropriate for parameters like "
                       "learning_rate where the useful range spans multiple orders of magnitude.")

    @model_validator(mode="after")
    def low_below_high(self) -> "ParamRange":
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be less than high ({self.high})")
        if self.log and self.low <= 0:
            raise ValueError(f"log-scaled range requires low > 0, got {self.low}")
        return self


class JointTuningConfig(BaseModel):
    n_trials: int = Field(..., gt=0)
    seed: int = Field(0, description="For reproducibility -- same seed, same sequence of trials.")

    universe_path: str = Field("config/universe.txt")
    scanning_config_path: str = Field(
        "config/scanning_criteria.yaml",
        description="FIXED across every trial in this study -- scanning_criteria.yaml is deliberately "
        "not part of this search space (see fitr/tuning/scoring.py's module docstring for why).",
    )
    features_config_path: str = Field(
        "config/features.yaml",
        description="FIXED across every trial -- the feature DEFINITIONS don't change, only which "
        "columns get used varies, via feature_groups below.",
    )

    scan_start_date: str = Field(
        ..., description="Should match whatever scripts/build_dataset.py was last run with, so the "
        "first trial's scan is a cache hit rather than a fresh (slow) scan."
    )
    scan_end_date: str = Field(..., description="Same as scan_start_date -- match your last build_dataset.py run.")
    tuning_end_date: str = Field(
        ..., description="The train/test boundary -- should match build_dataset.py's --split-date. Every "
        "trial only ever sees data strictly before this date; nothing at or after it is visible to "
        "tuning at all, preserving it as a genuinely untouched final holdout.",
    )
    scan_cache_dir: str = Field("data/cache/scans")

    walk_forward_config_path: str = Field("config/walk_forward.yaml")
    min_trades_for_scoring: int = Field(30, gt=0, description="See fitr.tuning.scoring -- gates folds with too few predicted-SUCCESS candidates.")

    # ---- labeling.yaml search ranges ------------------------------------------
    horizon_trading_days: ParamRange
    target_return: ParamRange
    stop_loss_magnitude: ParamRange = Field(
        ..., description="Positive magnitude, e.g. {low: 0.02, high: 0.10} -- negated internally when "
        "building the actual LabelingConfig, whose own stop_loss_return must be negative."
    )
    cooldown_trading_days: ParamRange

    # ---- model.yaml search ranges ----------------------------------------------
    n_estimators: ParamRange
    max_depth: ParamRange
    learning_rate: ParamRange

    # ---- feature selection -------------------------------------------------------
    feature_groups: dict[str, list[str]] = Field(
        default_factory=dict,
        description="group_name -> list of feature column names in that group. Each group is toggled "
        "on/off as a single search dimension -- searching individual features independently is an "
        "impractically large space (2^n for n features). Features not listed in any group are always "
        "included. Group names must be unique (enforced by dict keys); feature names should not repeat "
        "across groups, or the search's meaning becomes ambiguous.",
    )
