"""Pydantic schemas for the project's YAML config files.

Each config file gets its own top-level model here. This file grows
additively as later development steps introduce new config files
(scanning_criteria.yaml -> ScanningConfig, labeling.yaml -> LabelingConfig,
model.yaml -> ModelConfig, etc.) -- once a step is done with a class, that
class doesn't get modified again, only new ones get appended, per the
project roadmap.

Validation philosophy: catch as many config mistakes as possible at load
time, with an error message that says exactly what's wrong and where,
rather than letting a typo in a YAML file surface as a confusing exception
partway through a pipeline run. In particular, FeatureSpec below validates
both that the referenced indicator function actually exists AND that every
parameter name in the config is a real parameter of that function -- a
misspelled `windwo: 20` fails immediately at config-load time with a
message naming the bad key and the valid alternatives, rather than either
crashing deep inside indicators.py or (worse) silently being ignored.
"""
from __future__ import annotations

import inspect
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from fitr.features import indicators


# ============================================================================
# features.yaml -> FeaturesConfig
# ============================================================================

class FeatureSpec(BaseModel):
    """One entry in features.yaml: which indicator function to call, with
    what parameters, and which DataFrame(s) to call it against."""

    name: str = Field(..., description="Output key for this feature. Must be unique within the config.")
    function: str = Field(..., description="Name of a public function in fitr.features.indicators.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the function, beyond the price-history DataFrame(s).",
    )
    source: str | None = Field(
        default=None,
        description="Which DataFrame to compute this feature against. None (default) means the "
        "primary ticker being evaluated; otherwise a key from the top-level `benchmarks` map, meaning "
        "'run this function against that benchmark's own history instead' (e.g. n_day_return with "
        "source='market' produces the market benchmark's own N-day return).",
    )
    benchmark: str | None = Field(
        default=None,
        description="For functions that take a second, benchmark DataFrame as an argument (currently "
        "just relative_return_vs_benchmark) -- a key from the top-level `benchmarks` map.",
    )
    optional: bool = Field(
        default=False,
        description="If true, an unexpected runtime error computing this specific feature is logged "
        "and the feature is set to NaN instead of failing the whole compute() call. This does NOT "
        "cover config mistakes (unknown function, bad parameter names) -- those always fail at load "
        "time regardless of this flag. It's a narrow safety net for genuinely unexpected errors, since "
        "indicators.py functions are designed to return NaN rather than raise on insufficient data.",
    )

    @field_validator("function")
    @classmethod
    def function_must_be_a_known_public_indicator(cls, v: str) -> str:
        func = getattr(indicators, v, None)
        if v.startswith("_") or func is None or not callable(func):
            available = sorted(
                n for n in dir(indicators) if not n.startswith("_") and callable(getattr(indicators, n))
            )
            raise ValueError(
                f"'{v}' is not a known public function in fitr.features.indicators. "
                f"Available functions: {available}"
            )
        return v

    @model_validator(mode="after")
    def benchmark_flag_must_match_function_signature(self) -> "FeatureSpec":
        # Checked before params_must_match_function_signature below, and
        # deliberately kept as its own validator rather than folded in:
        # params_must_match_function_signature *assumes* a correct
        # `benchmark` flag when deciding how many positional slots to skip
        # (2 if benchmark is set, else 1) -- if that assumption is wrong,
        # its error message misattributes the mistake to an innocent
        # parameter name (e.g. flags a perfectly valid `window` as
        # "unknown") instead of naming the actual problem, which is the
        # `benchmark` field itself. Catching the mismatch explicitly, here,
        # first, keeps the error message pointed at what's actually wrong.
        func = getattr(indicators, self.function)
        function_takes_benchmark = "benchmark_df" in inspect.signature(func).parameters
        if function_takes_benchmark and self.benchmark is None:
            raise ValueError(
                f"Feature '{self.name}': function '{self.function}' compares against a benchmark "
                "series and requires a `benchmark` reference (e.g. benchmark: market)."
            )
        if not function_takes_benchmark and self.benchmark is not None:
            raise ValueError(
                f"Feature '{self.name}': function '{self.function}' does not take a benchmark "
                f"DataFrame, but `benchmark: {self.benchmark}` was set. Remove it."
            )
        return self

    @model_validator(mode="after")
    def params_must_match_function_signature(self) -> "FeatureSpec":
        func = getattr(indicators, self.function)
        positional_names = list(inspect.signature(func).parameters.keys())
        # The primary DataFrame is always supplied positionally by
        # FeatureEngine, never from params -- and so is the benchmark
        # DataFrame, for the handful of functions that take one. Both are
        # excluded from what's checkable against `params` here. Safe to
        # rely on self.benchmark's correctness at this point: the validator
        # above already guarantees it agrees with the function's actual
        # signature before this one runs.
        n_supplied_positionally = 2 if self.benchmark is not None else 1
        checkable_names = set(positional_names[n_supplied_positionally:])
        unknown = set(self.params.keys()) - checkable_names
        if unknown:
            raise ValueError(
                f"Feature '{self.name}': unknown parameter(s) {sorted(unknown)} for function "
                f"'{self.function}'. Valid parameters: {sorted(checkable_names)}"
            )
        return self


class FeaturesConfig(BaseModel):
    """Top-level schema for features.yaml."""

    benchmarks: dict[str, str] = Field(
        default_factory=dict,
        description="Named benchmark tickers, e.g. {'market': 'SPY', 'sector': 'XLK'}, referenced by "
        "FeatureSpec.source and FeatureSpec.benchmark.",
    )
    features: list[FeatureSpec] = Field(..., min_length=1)

    @field_validator("features")
    @classmethod
    def names_must_be_unique(cls, v: list[FeatureSpec]) -> list[FeatureSpec]:
        names = [f.name for f in v]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"Duplicate feature name(s): {dupes}")
        return v

    @model_validator(mode="after")
    def source_and_benchmark_refs_must_exist(self) -> "FeaturesConfig":
        known = set(self.benchmarks.keys())
        for f in self.features:
            if f.source is not None and f.source not in known:
                raise ValueError(
                    f"Feature '{f.name}': source '{f.source}' is not defined in `benchmarks` "
                    f"(known: {sorted(known)})"
                )
            if f.benchmark is not None and f.benchmark not in known:
                raise ValueError(
                    f"Feature '{f.name}': benchmark '{f.benchmark}' is not defined in `benchmarks` "
                    f"(known: {sorted(known)})"
                )
        return self
