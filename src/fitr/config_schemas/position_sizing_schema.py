"""Pydantic schema for position_sizing.yaml.

Both methods' parameters (risk_fraction, kelly_multiplier) are always
present with defaults, same convention as ModelConfig's class_weight_mode
-- only the one matching `method` is actually used, the other is simply
unused rather than conditionally required.

target_return/stop_loss_return mirror LabelingConfig's fields exactly (same
names, same validation) but are NOT imported from there -- position sizing
is kept independently configurable in case you ever want to size for a
different planned exit than whatever the model was trained/labeled
against. In the common case they should just match labeling.yaml.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PositionSizingConfig(BaseModel):
    method: Literal["fixed_fractional", "fractional_kelly"]

    # ---- fixed_fractional --------------------------------------------------
    risk_fraction: float = Field(
        0.01, gt=0, le=1,
        description="Fraction of equity risked per trade (fixed_fractional only) -- sized so a full "
        "stop-out loses exactly this fraction of equity, regardless of predicted probability.",
    )

    # ---- fractional_kelly ---------------------------------------------------
    kelly_multiplier: float = Field(
        0.01, gt=0, le=1,
        description="Fraction of full Kelly actually bet (fractional_kelly only). NOT the commonly-cited "
        "'quarter Kelly' (0.25) -- that convention comes from gambling contexts with much larger, more "
        "symmetric payoffs. For a tight stop relative to target (this project's default 4%/8%), full "
        "Kelly is already huge for almost any probability with real edge, so 0.25 saturates "
        "max_position_fraction almost immediately, making position size look constant regardless of "
        "confidence. Re-tune this if target_return/stop_loss_return change -- the right multiplier "
        "scales with how tight the stop is. Full Kelly (1.0) is rarely used in practice regardless -- "
        "extremely sensitive to probability estimation error, large drawdowns even with a real edge. "
        "Requires a genuinely calibrated probability to mean what it claims to mean (see model.yaml's "
        "calibration_method) -- an overconfident probability will recommend an oversized position.",
    )

    # ---- trade economics, used by both methods -------------------------------
    target_return: float = Field(
        ..., gt=0, description="Planned profit target as a fraction, e.g. 0.08 for +8%. Should generally "
        "match labeling.yaml's target_return."
    )
    stop_loss_return: float = Field(
        ..., lt=0, description="Planned stop loss as a fraction, e.g. -0.04 for -4%. Should generally "
        "match labeling.yaml's stop_loss_return."
    )

    # ---- portfolio-level guardrails, applied regardless of method -------------
    max_position_fraction: float = Field(
        0.10, gt=0, le=1, description="Hard cap on any single position as a fraction of equity, "
        "regardless of what the sizing method alone would recommend."
    )
    min_position_fraction: float = Field(
        0.0, ge=0, description="Below this recommended fraction, round down to zero -- not worth the "
        "transaction-cost/complexity of a tiny position."
    )

    @model_validator(mode="after")
    def target_positive_and_stop_negative(self) -> "PositionSizingConfig":
        if self.target_return <= 0:
            raise ValueError(f"target_return must be positive, got {self.target_return}")
        if self.stop_loss_return >= 0:
            raise ValueError(f"stop_loss_return must be negative, got {self.stop_loss_return}")
        return self

    @model_validator(mode="after")
    def min_below_max(self) -> "PositionSizingConfig":
        if self.min_position_fraction > self.max_position_fraction:
            raise ValueError(
                f"min_position_fraction ({self.min_position_fraction}) cannot exceed "
                f"max_position_fraction ({self.max_position_fraction})"
            )
        return self
