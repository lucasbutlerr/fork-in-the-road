"""Production pipeline script: evaluates an already-saved model against a
test set and prints a report. Use this to re-evaluate a previous run's
model (e.g. against a different test set, or a different decision
threshold) without retraining -- for training-then-immediately-evaluating
in one command instead, use scripts/train_model.py's --test flag.

Usage:
    python -m scripts.evaluate_model --model data/models/model_20240601_120000.json --test data/processed/test.csv

    python -m scripts.evaluate_model \\
        --model data/models/model_20240601_120000.json \\
        --test data/processed/test.csv \\
        --threshold 0.6

Requires an editable install (`pip install -e ".[dev]"`) and xgboost
installed.
"""
from __future__ import annotations

import argparse

import pandas as pd

from fitr.modeling.evaluator import ModelEvaluator, format_evaluation_report
from fitr.modeling.trainer import TrainedModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a saved model .json (from scripts/train_model.py)")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default 0.5)")
    args = parser.parse_args()

    try:
        trained = TrainedModel.load(args.model)
    except FileNotFoundError as exc:
        print(f"Model file not found: {exc}")
        raise SystemExit(1)

    try:
        test_df = pd.read_csv(args.test, parse_dates=["date"])
    except FileNotFoundError:
        print(f"Test file not found: {args.test}")
        raise SystemExit(1)

    if test_df.empty:
        print(f"{args.test} has no rows -- nothing to evaluate.")
        raise SystemExit(1)

    print(f"Model: {args.model} (trained {trained.trained_at})")
    print(f"Features ({len(trained.feature_columns)}): {trained.feature_columns}\n")

    report = ModelEvaluator().evaluate(trained, test_df, threshold=args.threshold)
    print(format_evaluation_report(report))
    print()


if __name__ == "__main__":
    main()
