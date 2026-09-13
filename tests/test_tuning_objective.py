"""Tests for JointTuningObjective. Uses fake Scanner/FeatureEngine/Proxy/
OutcomeLabeler (injected via labeler_factory) rather than real
network-backed ones -- this is orchestration logic (suggest params,
rebuild dataset, walk-forward, score), and the components it orchestrates
are already tested in isolation elsewhere. What needs verifying here is
that the wiring is correct: every trial's suggested values actually reach
the configs they should, feature-group toggling actually changes the
model's feature_columns, and the scan cache is genuinely reused across
trials with different labeling parameters."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fitr.config_schemas.tuning_schema import JointTuningConfig
from fitr.config_schemas.walk_forward_schema import WalkForwardConfig
from fitr.labeling.labeler import LabelResult
from fitr.scanning.scanner import ScanResult
from fitr.tuning.objective import JointTuningObjective

try:
    import optuna
except ImportError:
    optuna = None

pytestmark = pytest.mark.skipif(optuna is None, reason="optuna not installed")


class FakeScanner:
    """Deterministic ~30% daily hit rate across the universe -- enough
    density that walk-forward folds have real material regardless of how
    selective a real scanning_criteria.yaml's conditions would be."""

    def __init__(self, seed=0):
        self.setup_names = ["classic_high_breakout"]
        self.metric_names = ["m1"]
        self._rng = np.random.default_rng(seed)
        self.scan_call_count = 0

    def scan(self, tickers, as_of_date):
        self.scan_call_count += 1
        if self._rng.random() < 0.3:
            n = self._rng.integers(1, 4)
            chosen = self._rng.choice(tickers, size=min(n, len(tickers)), replace=False)
            return [ScanResult(t, pd.Timestamp(as_of_date), ["classic_high_breakout"], {"m1": 1.0}) for t in chosen]
        return []

    def config_fingerprint(self):
        return "fake-fp-fixed"


class FakeProxy:
    def trading_days_between(self, start, end):
        return list(pd.bdate_range(start, end))


class FakeFeatureEngine:
    def __init__(self):
        self.feature_names = ["feat_a", "feat_b", "rsi_14", "rsi_avg_5d", "macd_value", "macd_hist_avg_5d"]
        self._rng = np.random.default_rng(1)

    def compute(self, ticker, as_of_date):
        return {name: float(self._rng.normal(0, 1)) for name in self.feature_names}


class FakeOutcomeLabeler:
    """Injected via labeler_factory -- avoids needing a real network-backed
    OutcomeLabeler or monkeypatching the module under test."""

    def __init__(self, proxy, config):
        self._config = config
        self._rng = np.random.default_rng(2)

    def label(self, ticker, as_of_date):
        outcome = "SUCCESS" if self._rng.random() < 0.25 else "FAILURE"
        fr = 0.08 + self._rng.normal(0, 0.01) if outcome == "SUCCESS" else -0.04 + self._rng.normal(0, 0.01)
        return LabelResult(
            ticker=ticker, candidate_date=pd.Timestamp(as_of_date), outcome=outcome,
            trigger_reason="target_hit" if outcome == "SUCCESS" else "stop_hit", forward_return=fr,
        )


def make_tuning_config(cache_dir: str, **overrides) -> JointTuningConfig:
    base = {
        "n_trials": 3, "seed": 0,
        "scan_start_date": "2023-01-01", "scan_end_date": "2023-06-30", "tuning_end_date": "2023-06-30",
        "scan_cache_dir": cache_dir,
        "horizon_trading_days": {"low": 5, "high": 15},
        "target_return": {"low": 0.05, "high": 0.12},
        "stop_loss_magnitude": {"low": 0.02, "high": 0.06},
        "cooldown_trading_days": {"low": 1, "high": 8},
        "n_estimators": {"low": 50, "high": 150},
        "max_depth": {"low": 3, "high": 6},
        "learning_rate": {"low": 0.05, "high": 0.2, "log": True},
        "min_trades_for_scoring": 5,
        "feature_groups": {"rsi_family": ["rsi_14", "rsi_avg_5d"], "macd_family": ["macd_value", "macd_hist_avg_5d"]},
    }
    base.update(overrides)
    return JointTuningConfig.model_validate(base)


@pytest.fixture
def tmp_cache_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def make_objective(tmp_cache_dir, n_trials=3):
    universe = [f"T{i:03d}" for i in range(25)]
    scanner = FakeScanner()
    proxy = FakeProxy()
    feature_engine = FakeFeatureEngine()
    tuning_config = make_tuning_config(tmp_cache_dir, n_trials=n_trials)
    wf_config = WalkForwardConfig.model_validate({"n_folds": 2, "initial_train_fraction": 0.5, "min_test_rows": 1})
    objective = JointTuningObjective(
        tuning_config, universe, scanner, feature_engine, proxy, wf_config, labeler_factory=FakeOutcomeLabeler
    )
    return objective, scanner


def test_study_completes_and_produces_real_scores(tmp_cache_dir):
    objective, _ = make_objective(tmp_cache_dir)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=3)

    assert len(study.trials) == 3
    assert all(t.value == t.value for t in study.trials)  # no NaN scores
    assert len(objective.trial_log) == 3


def test_scan_is_reused_across_trials_with_different_labeling(tmp_cache_dir):
    objective, scanner = make_objective(tmp_cache_dir, n_trials=4)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=4)

    # scan() is called once per trading day PER trial only if the scan
    # cache misses -- with a fixed scanning fingerprint and identical
    # scan_start/end across every trial, only the first trial should
    # actually invoke the scanner.
    trading_days = len(list(pd.bdate_range("2023-01-01", "2023-06-30")))
    assert scanner.scan_call_count == trading_days  # exactly one trial's worth, not four


def test_feature_group_toggles_are_reflected_in_trial_params(tmp_cache_dir):
    objective, _ = make_objective(tmp_cache_dir, n_trials=5)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=5)

    for trial in study.trials:
        assert "include_rsi_family" in trial.params
        assert "include_macd_family" in trial.params
        assert isinstance(trial.params["include_rsi_family"], (bool, np.bool_))


def test_suggested_labeling_params_are_within_configured_ranges(tmp_cache_dir):
    objective, _ = make_objective(tmp_cache_dir, n_trials=5)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=5)

    for trial in study.trials:
        assert 5 <= trial.params["horizon_trading_days"] <= 15
        assert 0.05 <= trial.params["target_return"] <= 0.12
        assert 0.02 <= trial.params["stop_loss_magnitude"] <= 0.06


def test_best_trial_is_selectable_from_the_study(tmp_cache_dir):
    objective, _ = make_objective(tmp_cache_dir, n_trials=5)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=5)

    assert study.best_value == max(t.value for t in study.trials)
    assert study.best_params == study.best_trial.params


def test_scores_are_ev_shaped_not_precision_shaped():
    # A raw precision score is bounded in [0, 1] (or GATED_SCORE). An EV
    # score is NOT -- it's a return, and can legitimately be negative for
    # a real (non-gated) trial whose achieved precision doesn't clear
    # THAT trial's own breakeven bar. This is the direct, end-to-end
    # confirmation that JointTuningObjective is actually using
    # expected_value_score_from_walk_forward and not the old
    # score_from_walk_forward -- not just that scoring.py's formula is
    # correct in isolation (test_scoring.py covers that already).
    objective, _ = make_objective(tempfile.mkdtemp(), n_trials=8)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1))
    study.optimize(objective, n_trials=8)

    real_scores = [t.value for t in study.trials if t.value != -1.0]
    assert any(s < 0 for s in real_scores), (
        "expected at least one real (non-gated) negative score across 8 trials with varying "
        "target/stop -- if every real score is in [0, 1], this is silently back to precision-shaped scoring"
    )
