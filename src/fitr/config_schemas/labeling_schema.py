"""Pydantic schema for labeling.yaml.

Return thresholds are fractions (0.08 for +8%), matching the convention
used everywhere else in the project (n_day_return and friends return
fractions too) -- not "_pct" percent-scale numbers like 8.0, to keep this
directly comparable to feature values without a mental unit conversion.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class LabelingConfig(BaseModel):
    """Top-level schema for labeling.yaml -- defines what counts as a
    successful vs. failed breakout for a candidate."""

    horizon_trading_days: int = Field(..., gt=0, le=500)
    target_return: float = Field(
        ..., description="Fractional return that counts as a successful breakout, e.g. 0.08 for +8%."
    )
    stop_loss_return: float = Field(
        ..., description="Fractional return (negative) that counts as a failed breakout, e.g. -0.04 for -4%."
    )
    # No default: this materially changes what a label means, so it's
    # required to be stated explicitly rather than silently assumed.
    evaluation_method: Literal["first_touch", "end_of_horizon"] = Field(
        ...,
        description="first_touch: whichever of target_return/stop_loss_return is hit first within the "
        "horizon wins. end_of_horizon: ignores intra-horizon touches, judges only the return on the "
        "horizon's last day.",
    )

    @model_validator(mode="after")
    def target_positive_and_stop_negative(self) -> "LabelingConfig":
        if self.target_return <= 0:
            raise ValueError(f"target_return must be positive (a return threshold to succeed), got {self.target_return}")
        if self.stop_loss_return >= 0:
            raise ValueError(f"stop_loss_return must be negative (a return threshold to fail), got {self.stop_loss_return}")
        return self
