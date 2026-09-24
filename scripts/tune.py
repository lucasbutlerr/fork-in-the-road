"""Production pipeline script: Phase 3 Step 5 -- runs the joint
model + feature-selection + labeling tuning study.

Every trial scores against the tuning-visible portion of the data only
(strictly before tuning_end_date, via WalkForwardValidator) -- test.csv,
or anything at/after tuning_end_date, is NEVER touched by any trial. That
data is a one-time, final, honest check to run by hand (scripts/
evaluate_model.py or scripts/backtest.py) against whatever config wins
here -- not something to feed back into tuning itself.

scanning_criteria.yaml is fixed throughout this study; it isn't part of
the search space here (see fitr/tuning/scoring.py's module docstring).

Usage:
    python -m scripts.tune --tuning-config config/tuning.yaml

    python -m scripts.tune \\
        --tuning-config config/tuning.yaml \\
        --output-dir data/tuning_runs

Requires an editable install (`pip install -e ".[dev]"`), and a live
network connection unless the scan for tuning.yaml's scan_start_date/
scan_end_date/scanning_config_path/universe_path is already cached from a
prior scripts/build_dataset.py run.
"""
from __future__ import annotations

import argparse

from scripts._config_defaults import resolve_config_default
import csv
import json
from pathlib import Path

import optuna

from fitr.config_schemas.features_schema import FeaturesConfig
from fitr.config_schemas.loader import ConfigError, load_config
from fitr.config_schemas.scanning_schema import ScanningConfig
from fitr.config_schemas.tuning_schema import JointTuningConfig
from fitr.config_schemas.walk_forward_schema import WalkForwardConfig
from fitr.data.yf_proxy import YFProxy
from fitr.dataset.universe import UniverseError, load_universe
from fitr.features.feature_engine import FeatureEngine
from fitr.scanning.scanner import BreakoutScanner
from fitr.tuning.objective import JointTuningObjective


def write_trial_log_csv(trial_log: list[dict], path: str) -> None:
    if not trial_log:
        return
    param_names = sorted({k for entry in trial_log for k in entry["params"]})
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_number", "score", "note", *param_names])
        for entry in trial_log:
            row = [entry["trial_number"], f"{entry['score']:.6f}", entry.get("note", "")]
            row += [entry["params"].get(name, "") for name in param_names]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tuning-config", default=resolve_config_default("tuning"))
    parser.add_argument("--output-dir", default="data/tuning_runs")
    args = parser.parse_args()

    try:
        tuning_config = load_config(args.tuning_config, JointTuningConfig)
        scanning_config = load_config(tuning_config.scanning_config_path, ScanningConfig)
        features_config = load_config(tuning_config.features_config_path, FeaturesConfig)
        wf_config = load_config(tuning_config.walk_forward_config_path, WalkForwardConfig)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        raise SystemExit(1)

    try:
        universe = load_universe(tuning_config.universe_path)
    except UniverseError as exc:
        print(f"Universe error: {exc}")
        raise SystemExit(1)

    proxy = YFProxy()
    scanner = BreakoutScanner(proxy, scanning_config)
    feature_engine = FeatureEngine(proxy, features_config)

    print(f"Tuning: {tuning_config.n_trials} trials, seed={tuning_config.seed}")
    print(f"Scan range: {tuning_config.scan_start_date} to {tuning_config.scan_end_date}")
    print(f"Tuning-visible boundary: strictly before {tuning_config.tuning_end_date} (test.csv is never touched)")
    print(f"Universe: {len(universe)} tickers\n")

    objective = JointTuningObjective(tuning_config, universe, scanner, feature_engine, proxy, wf_config)

    sampler = optuna.samplers.TPESampler(seed=tuning_config.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=tuning_config.n_trials)

    print(f"\nCompleted {len(study.trials)} trials.")
    print(f"Best score: {study.best_value:.4f}")
    print("Best params:")
    for name, value in sorted(study.best_params.items()):
        print(f"  {name}: {value}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_log_path = out_dir / "trial_log.csv"
    write_trial_log_csv(objective.trial_log, str(trial_log_path))
    print(f"\nWrote all {len(objective.trial_log)} trials to {trial_log_path}")

    best_params_path = out_dir / "best_params.json"
    with best_params_path.open("w") as f:
        json.dump({"score": study.best_value, "params": study.best_params}, f, indent=2, default=str)
    print(f"Wrote best_params.json to {best_params_path}")
    print(
        "\nThis is a SUGGESTION, not an automatic update -- review best_params.json and manually apply "
        "the values you want to model.yaml/labeling.yaml before retraining for real. Once you do, run "
        "scripts/evaluate_model.py or scripts/backtest.py against test.csv exactly once, by hand, as the "
        "final honest check -- test.csv was never touched by any trial above."
    )


if __name__ == "__main__":
    main()
