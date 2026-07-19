"""Tests for fitr.config_schemas.feature_schema (FeatureSpec/FeaturesConfig) and
fitr.config_schemas.loader (load_config). All operate on in-memory dicts or
tmp_path-written YAML -- no network, no real config/ files required."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from fitr.config_schemas.loader import ConfigError, load_config
from fitr.config_schemas.features_schema import FeatureSpec, FeaturesConfig


def test_valid_minimal_config_loads():
    config = FeaturesConfig.model_validate(
        {"features": [{"name": "rsi_14", "function": "rsi", "params": {}}]}
    )
    assert len(config.features) == 1
    assert config.features[0].name == "rsi_14"


def test_unknown_function_is_rejected():
    with pytest.raises(ValidationError):
        FeaturesConfig.model_validate(
            {"features": [{"name": "bad", "function": "not_a_real_function"}]}
        )


def test_private_function_is_rejected():
    # _linreg_slope exists in indicators.py but is a private helper, not a
    # feature-computing entry point -- it must not be referenceable.
    with pytest.raises(ValidationError):
        FeaturesConfig.model_validate(
            {"features": [{"name": "bad", "function": "_linreg_slope"}]}
        )


def test_unknown_parameter_name_is_rejected():
    with pytest.raises(ValidationError) as exc_info:
        FeaturesConfig.model_validate(
            {"features": [{"name": "bad", "function": "n_day_return", "params": {"windwo": 20}}]}
        )
    assert "windwo" in str(exc_info.value)


def test_valid_parameter_name_is_accepted():
    config = FeaturesConfig.model_validate(
        {"features": [{"name": "ret20", "function": "n_day_return", "params": {"window": 20}}]}
    )
    assert config.features[0].params == {"window": 20}


def test_duplicate_feature_names_are_rejected():
    with pytest.raises(ValidationError):
        FeaturesConfig.model_validate(
            {
                "features": [
                    {"name": "dup", "function": "rsi"},
                    {"name": "dup", "function": "atr_normalized"},
                ]
            }
        )


def test_empty_features_list_is_rejected():
    with pytest.raises(ValidationError):
        FeaturesConfig.model_validate({"features": []})


def test_source_referencing_undefined_benchmark_is_rejected():
    with pytest.raises(ValidationError) as exc_info:
        FeaturesConfig.model_validate(
            {
                "benchmarks": {"market": "SPY"},
                "features": [
                    {"name": "bad", "function": "n_day_return", "params": {"window": 20}, "source": "sector"}
                ],
            }
        )
    assert "sector" in str(exc_info.value)


def test_benchmark_referencing_undefined_key_is_rejected():
    with pytest.raises(ValidationError):
        FeaturesConfig.model_validate(
            {
                "benchmarks": {"market": "SPY"},
                "features": [
                    {
                        "name": "bad",
                        "function": "relative_return_vs_benchmark",
                        "params": {"window": 20},
                        "benchmark": "sector",
                    }
                ],
            }
        )


def test_valid_source_and_benchmark_reference_are_accepted():
    config = FeaturesConfig.model_validate(
        {
            "benchmarks": {"market": "SPY", "sector": "XLK"},
            "features": [
                {"name": "mkt_ret", "function": "n_day_return", "params": {"window": 20}, "source": "market"},
                {
                    "name": "vs_mkt",
                    "function": "relative_return_vs_benchmark",
                    "params": {"window": 20},
                    "benchmark": "market",
                },
            ],
        }
    )
    assert config.features[0].source == "market"
    assert config.features[1].benchmark == "market"


def test_params_check_excludes_benchmark_df_positional_slot():
    # relative_return_vs_benchmark(df, benchmark_df, window, price_col) --
    # `window` and `price_col` are valid param keys; `df`/`benchmark_df`
    # (the first two positional slots) must NOT be treated as valid keys.
    with pytest.raises(ValidationError):
        FeatureSpec(
            name="bad",
            function="relative_return_vs_benchmark",
            params={"benchmark_df": "something"},
            benchmark="market",
        )
    spec = FeatureSpec(
        name="ok", function="relative_return_vs_benchmark", params={"window": 20}, benchmark="market"
    )
    assert spec.params == {"window": 20}


def test_benchmark_set_on_non_benchmark_function_is_rejected_with_clear_message():
    # rsi() doesn't take a benchmark_df -- setting `benchmark` on it anyway
    # must be caught explicitly, not surface as a confusing "unknown
    # parameter 'window'" error (window is perfectly valid for rsi; the
    # actual mistake is the benchmark field).
    with pytest.raises(ValidationError) as exc_info:
        FeatureSpec(name="bad", function="rsi", params={"window": 20}, benchmark="market")
    message = str(exc_info.value)
    assert "does not take a benchmark" in message
    assert "window" not in message


def test_benchmark_function_missing_benchmark_flag_is_rejected():
    with pytest.raises(ValidationError) as exc_info:
        FeatureSpec(name="bad", function="relative_return_vs_benchmark", params={"window": 20})
    assert "requires a `benchmark`" in str(exc_info.value)


def test_optional_flag_defaults_false():
    config = FeaturesConfig.model_validate({"features": [{"name": "rsi_14", "function": "rsi"}]})
    assert config.features[0].optional is False


# ---- loader.py --------------------------------------------------------------

def test_load_config_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.yaml", FeaturesConfig)


def test_load_config_malformed_yaml_raises_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("features: [this is not: valid: yaml: [[[")
    with pytest.raises(ConfigError):
        load_config(path, FeaturesConfig)


def test_load_config_valid_yaml_returns_model(tmp_path):
    path = tmp_path / "features.yaml"
    path.write_text(
        "features:\n"
        "  - name: rsi_14\n"
        "    function: rsi\n"
        "    params: {}\n"
    )
    config = load_config(path, FeaturesConfig)
    assert isinstance(config, FeaturesConfig)
    assert config.features[0].name == "rsi_14"


def test_load_config_invalid_schema_raises_config_error_with_context(tmp_path):
    path = tmp_path / "features.yaml"
    path.write_text(
        "features:\n"
        "  - name: bad\n"
        "    function: not_a_real_function\n"
    )
    with pytest.raises(ConfigError) as exc_info:
        load_config(path, FeaturesConfig)
    assert str(path) in str(exc_info.value)
