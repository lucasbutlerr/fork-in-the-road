"""Pydantic schema for walk_forward.yaml."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class WalkForwardConfig(BaseModel):
    n_folds: int = Field(..., gt=0, description="Number of sequential test windows to evaluate.")
    window_type: Literal["expanding", "rolling"] = Field(
        "expanding",
        description="'expanding': every fold's training window starts at the very beginning of the "
        "data and grows fold over fold -- maximizes training data per fold, at the cost of older data "
        "never 'falling off'. 'rolling': training window is a fixed size (rolling_train_window_days) "
        "that slides forward with each fold -- bounds how much (possibly stale) old data is used, at "
        "the cost of less data per fold.",
    )
    rolling_train_window_days: int | None = Field(
        None, gt=0, description="Required if window_type is 'rolling': calendar days of training data per fold."
    )
    initial_train_fraction: float = Field(
        0.5, gt=0, lt=1,
        description="Fraction of the full date range reserved as a minimum training period before the "
        "first fold's test window begins -- without this, fold 1 would have almost no training data.",
    )
    min_test_rows: int = Field(
        20, gt=0,
        description="Folds with fewer test rows than this are still evaluated and reported, just flagged "
        "as low-confidence -- never silently dropped, since that would hide exactly the kind of small-fold "
        "noise this whole mechanism exists to surface.",
    )

    @model_validator(mode="after")
    def rolling_window_requires_a_window_size(self) -> "WalkForwardConfig":
        if self.window_type == "rolling" and self.rolling_train_window_days is None:
            raise ValueError("rolling_train_window_days is required when window_type is 'rolling'.")
        return self
