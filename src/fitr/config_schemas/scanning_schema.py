"""Pydantic schema for scanning_criteria.yaml.

Reuses FeatureSpec from features_schema.py for the `metrics` block --
scanning criteria are computed exactly the same way ML training features
are (same indicators.py functions, same point-in-time-safe FeatureEngine
path), just a smaller, purpose-specific set of things to compute. This is
the reason scanning_schema.py exists as a separate file from
features_schema.py rather than both configs' models living in one place:
the reuse is a real dependency, not just a filing convenience.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from fitr.config_schemas.features_schema import FeatureSpec

_OPERATORS = {">=", "<=", ">", "<", "==", "!=", "between"}


class ConditionSpec(BaseModel):
    """One threshold check within a setup: `metric` <operator> `value`."""

    metric: str = Field(..., description="Must match a `name` in the top-level `metrics` block.")
    operator: str = Field(..., description=f"One of {sorted(_OPERATORS)}.")
    value: float | list[float] = Field(
        ..., description="A single number for most operators; a [low, high] pair for 'between'."
    )

    @field_validator("operator")
    @classmethod
    def operator_must_be_known(cls, v: str) -> str:
        if v not in _OPERATORS:
            raise ValueError(f"Unknown operator '{v}'. Must be one of {sorted(_OPERATORS)}.")
        return v

    @model_validator(mode="after")
    def value_shape_must_match_operator(self) -> "ConditionSpec":
        if self.operator == "between":
            if not isinstance(self.value, list) or len(self.value) != 2:
                raise ValueError("operator 'between' requires `value` to be a two-item list [low, high].")
            if self.value[0] >= self.value[1]:
                raise ValueError(f"operator 'between' requires value[0] < value[1], got {self.value}.")
        else:
            if isinstance(self.value, list):
                raise ValueError(f"operator '{self.operator}' requires a single numeric `value`, not a list.")
        return self


class SetupSpec(BaseModel):
    """One named breakout setup: a combination of conditions, AND'd or
    OR'd together."""

    name: str
    description: str = ""
    logic: Literal["AND", "OR"] = "AND"
    conditions: list[ConditionSpec] = Field(..., min_length=1)


class ScanningConfig(BaseModel):
    """Top-level schema for scanning_criteria.yaml.

    A ticker is a breakout candidate on a given date if it matches at
    least one setup (setups are always OR'd together at this top level --
    only the conditions *within* a setup use the configurable AND/OR
    `logic`)."""

    benchmarks: dict[str, str] = Field(
        default_factory=dict,
        description="Same shape as FeaturesConfig.benchmarks -- only needed if a metric below uses "
        "`source` or `benchmark` (e.g. a setup that requires outperforming the market).",
    )
    metrics: list[FeatureSpec] = Field(..., min_length=1)
    setups: list[SetupSpec] = Field(..., min_length=1)

    @field_validator("metrics")
    @classmethod
    def metric_names_must_be_unique(cls, v: list[FeatureSpec]) -> list[FeatureSpec]:
        names = [m.name for m in v]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"Duplicate metric name(s): {dupes}")
        return v

    @field_validator("setups")
    @classmethod
    def setup_names_must_be_unique(cls, v: list[SetupSpec]) -> list[SetupSpec]:
        names = [s.name for s in v]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"Duplicate setup name(s): {dupes}")
        return v

    @model_validator(mode="after")
    def condition_metrics_must_be_defined(self) -> "ScanningConfig":
        known = {m.name for m in self.metrics}
        for setup in self.setups:
            for cond in setup.conditions:
                if cond.metric not in known:
                    raise ValueError(
                        f"Setup '{setup.name}' references undefined metric '{cond.metric}'. "
                        f"Known metrics: {sorted(known)}"
                    )
        return self
