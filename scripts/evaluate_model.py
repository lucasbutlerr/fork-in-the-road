"""Production pipeline script: evaluates an already-saved model against a
test set and prints a report. Use this to re-evaluate a previous run's
model (e.g. against a different test set, or a different decision
threshold) without retraining -- for training-then-immediately-evaluating
in one command instead, use scripts/train_model.py's --test flag.

Usage:
    python -m scripts.evaluate_model --model data/models/model_20240601_120000.joblib --test data/processed/test.csv

    python -m scripts.evaluate_model \\
        --model data/models/model_20240601_120000.joblib \\
        --test data/processed/test.csv \\
        --threshold 0.6 --predictions-output data/evaluations/predictions.csv

Requires an editable install (`pip install -e ".[dev]"`) and xgboost
installed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fitr.modeling.evaluator import ModelEvaluator, format_evaluation_report
from fitr.modeling.trainer import TrainedModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a saved model .joblib (from scripts/train_model.py)")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default 0.5)")
    parser.add_argument(
        "--predictions-output",
        default="data/evaluations/predictions.csv",
        help="Where to write row-level predictions (ticker, date, true/predicted label, probability, correct). "
        "Default: data/evaluations/predictions.csv. Pass an empty string ('') to skip writing this file.",
    )
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
