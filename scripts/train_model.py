"""Production pipeline script: trains an XGBoost classifier on Step 4's
train.csv, saves the model artifact, and (if --test is given) immediately
evaluates it against test.csv and prints a report -- a single command that
runs the train-then-evaluate workflow end to end on already-built data.

Only model.yaml should need editing to change the model's behavior. For
re-evaluating an already-saved model without retraining, use
scripts/evaluate_model.py instead.

Usage:
    python -m scripts.train_model --train data/processed/train.csv

    python -m scripts.train_model \\
        --train data/processed/train.csv --test data/processed/test.csv \\
        --model-config config/model.yaml \\
        --output-dir data/models -v

Requires an editable install (`pip install -e ".[dev]"`) and xgboost
installed (`pip install xgboost`, or via the project's [dev] extra).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fitr.config_schemas.loader import ConfigError, load_config
from fitr.config_schemas.model_schema import ModelConfig
from fitr.modeling.evaluator import ModelEvaluator, format_evaluation_report
from fitr.modeling.trainer import ModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, help="Path to train.csv (Step 4's DatasetBuilder output)")
    parser.add_argument("--test", default=None, help="Optional path to test.csv -- if given, evaluates immediately after training")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for the evaluation report (default 0.5)")
    parser.add_argument(
        "--predictions-output",
        default="data/evaluations/predictions.csv",
        help="Only used with --test. Where to write row-level predictions. Pass '' to skip.",
    )
    parser.add_argument("--model-config", default="config/model.yaml")
    parser.add_argument("--output-dir", default="data/models")
    parser.add_argument("--model-name", default=None, help="Base filename (default: model_<timestamp>)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    try:
        model_config = load_config(args.model_config, ModelConfig)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        raise SystemExit(1)

    try:
        train_df = pd.read_csv(args.train, parse_dates=["date"])
    except FileNotFoundError:
        print(f"Training file not found: {args.train}")
        raise SystemExit(1)

    if train_df.empty:
        print(f"{args.train} has no rows -- nothing to train on.")
        raise SystemExit(1)

    label_counts = train_df[model_config.label_column].value_counts().to_dict()
    print(f"Training data: {len(train_df)} rows, label distribution: {label_counts}")
    print(
        f"Model: {model_config.n_estimators} trees, max_depth={model_config.max_depth}, "
        f"learning_rate={model_config.learning_rate}, class_weight_mode={model_config.class_weight_mode}"
    )
    if model_config.early_stopping_rounds is not None:
        print(
            f"Early stopping: {model_config.early_stopping_rounds} rounds patience, "
            f"{model_config.early_stopping_validation_fraction:.0%} validation tail"
        )
    if model_config.calibration_method != "none":
        print(
            f"Calibration: {model_config.calibration_method}, "
            f"{model_config.calibration_fraction:.0%} held-out tail"
        )

    trainer = ModelTrainer(model_config)
    print("\nTraining...\n")
    trained = trainer.fit(train_df)

    print(f"\nTrained on {len(trained.feature_columns)} features: {trained.feature_columns}")

    model_name = args.model_name or f"model_{trained.trained_at.strftime('%Y%m%d_%H%M%S')}"
    model_path = f"{args.output_dir}/{model_name}.joblib"
    trained.save(model_path)
    print(f"\nWrote {model_path}")
    print(f"Wrote {model_path}.meta.json")

    if args.test is None:
        print()
        return

    try:
        test_df = pd.read_csv(args.test, parse_dates=["date"])
    except FileNotFoundError:
        print(f"\nTest file not found: {args.test} -- model was saved, but no evaluation was run.")
        raise SystemExit(1)

    if test_df.empty:
        print(f"\n{args.test} has no rows -- model was saved, but no evaluation was run.")
        return

    print(f"\n{'=' * 60}\nEvaluating on {args.test}\n{'=' * 60}\n")
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(trained, test_df, threshold=args.threshold)
    print(format_evaluation_report(report))
    print()

    if args.predictions_output:
        detail = evaluator.predictions_detail(trained, test_df, threshold=args.threshold)
        out_path = Path(args.predictions_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(out_path, index=False)
        print(f"Wrote {len(detail)} row-level predictions to {out_path}\n")


if __name__ == "__main__":
    main()
