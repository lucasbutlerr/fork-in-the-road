"""Production pipeline script -- the capstone of the whole project. Given
only the config files and a saved model, produces a ranked list of today's
(or any given date's) breakout candidates with predicted success
probability. Nothing about this run is hardcoded beyond the CLI args
below: change what counts as a breakout, what features feed the model, or
retrain the model itself, and this script's behavior changes with it.

Usage:
    python -m scripts.scan_live --model data/models/model_20240601_120000.joblib

    python -m scripts.scan_live \\
        --model data/models/model_20240601_120000.joblib \\
        --date 2024-06-03 --universe config/universe.txt \\
        --top 20 --threshold 0.6 --output predictions.csv

Requires an editable install (`pip install -e ".[dev]"`), xgboost
installed, and a live network connection for yfinance.
"""
from __future__ import annotations

import argparse
import csv

import pandas as pd

from fitr.config_schemas.features_schema import FeaturesConfig
from fitr.config_schemas.loader import ConfigError, load_config
from fitr.config_schemas.scanning_schema import ScanningConfig
from fitr.data.yf_proxy import YFProxy
from fitr.dataset.universe import UniverseError, load_universe
from fitr.features.feature_engine import FeatureEngine
from fitr.modeling.predictor import LivePredictor, format_predictions
from fitr.modeling.trainer import TrainedModel
from fitr.scanning.scanner import BreakoutScanner


def write_predictions_csv(predictions, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "ticker", "date", "predicted_probability", "matched_setups"])
        for p in predictions:
            writer.writerow(
                [p.rank, p.ticker, p.as_of_date.date(), f"{p.predicted_probability:.6f}", "|".join(p.matched_setups)]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a saved model .joblib (from scripts/train_model.py)")
    parser.add_argument("--date", default=None, help="Date to scan as-of (default: today)")
    parser.add_argument("--universe", default="config/universe.txt")
    parser.add_argument("--scanning-config", default="config/scanning_criteria.yaml")
    parser.add_argument("--features-config", default="config/features.yaml")
    parser.add_argument("--top", type=int, default=20, help="How many top candidates to show in full detail (default 20)")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Only show/count candidates with P(success) at or above this value (default: show all)",
    )
    parser.add_argument("--output", default=None, help="Optional path to write the full ranked list as CSV")
    args = parser.parse_args()

    try:
        scanning_config = load_config(args.scanning_config, ScanningConfig)
        features_config = load_config(args.features_config, FeaturesConfig)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        raise SystemExit(1)

    try:
        universe = load_universe(args.universe)
    except UniverseError as exc:
        print(f"Universe error: {exc}")
        raise SystemExit(1)

    try:
        trained_model = TrainedModel.load(args.model)
    except FileNotFoundError as exc:
        print(f"Model file not found: {exc}")
        raise SystemExit(1)

    proxy = YFProxy()
    scanner = BreakoutScanner(proxy, scanning_config)
    feature_engine = FeatureEngine(proxy, features_config)

    try:
        predictor = LivePredictor(scanner, feature_engine, trained_model)
    except ValueError as exc:
        print(f"Model/config mismatch: {exc}")
        raise SystemExit(1)

    as_of_date = args.date or pd.Timestamp.today().normalize()
    print(f"Model: {args.model} (trained {trained_model.trained_at})")
    print(f"Universe: {len(universe)} tickers")
    print(f"Scanning as of {as_of_date}...\n")

    predictions = predictor.predict(universe, as_of_date)

    if args.threshold is not None:
        predictions = [p for p in predictions if p.predicted_probability >= args.threshold]
        for i, p in enumerate(predictions, start=1):
            p.rank = i

    print(format_predictions(predictions, trained_model, top_n=args.top))

    if args.output:
        write_predictions_csv(predictions, args.output)
        print(f"\nWrote {len(predictions)} rows to {args.output}")
    print()


if __name__ == "__main__":
    main()
