"""Production pipeline script -- the capstone of the whole project. Given
only the config files and a saved model, produces a ranked list of today's
(or any given date's) breakout candidates with predicted success
probability, and (if --position-sizing-config is given) a suggested
equity allocation per candidate. Nothing about this run is hardcoded
beyond the CLI args below: change what counts as a breakout, what
features feed the model, or retrain the model itself, and this script's
behavior changes with it.

Usage:
    python -m scripts.scan_live --model data/models/model_20240601_120000.joblib

    python -m scripts.scan_live \\
        --model data/models/model_20240601_120000.joblib \\
        --date 2024-06-03 --universe config/universe.txt \\
        --position-sizing-config config/position_sizing.yaml --equity 100000 \\
        --top 20 --threshold 0.6 --output predictions.csv

Requires an editable install (`pip install -e ".[dev]"`), xgboost
installed, and a live network connection for yfinance.
"""
from __future__ import annotations

import argparse
import csv

import pandas as pd

from fitr.config_schemas.features_schema import FeaturesConfig
from fitr.config_schemas.labeling_schema import LabelingConfig
from fitr.config_schemas.loader import ConfigError, load_config
from fitr.config_schemas.position_sizing_schema import PositionSizingConfig
from fitr.config_schemas.scanning_schema import ScanningConfig
from fitr.data.yf_proxy import NoDataInRangeError, TickerNotFoundError, YFProxy
from fitr.dataset.universe import UniverseError, load_universe
from fitr.features.feature_engine import FeatureEngine
from fitr.modeling.position_sizer import PositionSizer
from fitr.modeling.predictor import LivePredictor, format_predictions
from fitr.modeling.trainer import TrainedModel
from fitr.scanning.scanner import BreakoutScanner


def write_predictions_csv(predictions, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["rank", "ticker", "date", "predicted_probability", "matched_setups", "recommended_fraction", "recommended_shares"]
        )
        for p in predictions:
            ps = p.position_size
            writer.writerow(
                [
                    p.rank, p.ticker, p.as_of_date.date(), f"{p.predicted_probability:.6f}", "|".join(p.matched_setups),
                    f"{ps.recommended_fraction:.6f}" if ps else "", ps.recommended_shares if ps else "",
                ]
            )


def _check_sizing_matches_labeling(sizing_config: PositionSizingConfig, labeling_config_path: str) -> None:
    """Warns (doesn't fail) if position_sizing.yaml's target/stop don't
    match labeling.yaml's -- an easy, easy-to-miss mismatch to introduce
    when editing one file and not the other, and one that silently changes
    what the sizing math actually assumes versus what the model was
    trained to predict. A mismatch isn't necessarily wrong (see
    position_sizing_schema.py's own docs -- you might deliberately want to
    size for a different planned exit), but it should be a deliberate
    choice, not an accident."""
    try:
        labeling_config = load_config(labeling_config_path, LabelingConfig)
    except ConfigError:
        return  # no labeling.yaml to compare against -- nothing to warn about
    if (
        sizing_config.target_return != labeling_config.target_return
        or sizing_config.stop_loss_return != labeling_config.stop_loss_return
    ):
        print(
            f"WARNING: position sizing config's target/stop ({sizing_config.target_return:+.1%}/"
            f"{sizing_config.stop_loss_return:+.1%}) doesn't match {labeling_config_path}'s "
            f"({labeling_config.target_return:+.1%}/{labeling_config.stop_loss_return:+.1%}). "
            f"If this isn't deliberate, update one to match the other.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a saved model .joblib (from scripts/train_model.py)")
    parser.add_argument("--date", default=None, help="Date to scan as-of (default: today)")
    parser.add_argument("--universe", default="config/universe.txt")
    parser.add_argument("--scanning-config", default="config/scanning_criteria.yaml")
    parser.add_argument("--features-config", default="config/features.yaml")
    parser.add_argument(
        "--position-sizing-config", default=None,
        help="Optional -- if given, each candidate also gets a suggested equity allocation.",
    )
    parser.add_argument("--labeling-config", default="config/labeling.yaml", help="Only used to sanity-check against --position-sizing-config")
    parser.add_argument("--equity", type=float, default=100_000.0, help="Account equity, for position sizing (default 100,000)")
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
        sizing_config = load_config(args.position_sizing_config, PositionSizingConfig) if args.position_sizing_config else None
    except ConfigError as exc:
        print(f"Config error: {exc}")
        raise SystemExit(1)

    if sizing_config is not None:
        _check_sizing_matches_labeling(sizing_config, args.labeling_config)

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
    if sizing_config:
        print(f"Position sizing: {sizing_config.method}, equity={args.equity:,.2f}")
    print(f"Scanning as of {as_of_date}...\n")

    predictions = predictor.predict(universe, as_of_date)

    if args.threshold is not None:
        predictions = [p for p in predictions if p.predicted_probability >= args.threshold]

    if sizing_config is not None:
        sizer = PositionSizer(sizing_config)
        for p in predictions:
            try:
                price_row = proxy.get_price_on(p.ticker, p.as_of_date)
                entry_price = float(price_row["Close"])
                p.position_size = sizer.size(p.ticker, p.predicted_probability, entry_price, args.equity)
            except (TickerNotFoundError, NoDataInRangeError) as exc:
                print(f"  (could not size {p.ticker}: {exc})")

    for i, p in enumerate(predictions, start=1):
        p.rank = i

    print(format_predictions(predictions, trained_model, top_n=args.top, show_scan_metrics=False, show_model_features=False))

    if args.output:
        write_predictions_csv(predictions, args.output)
        print(f"\nWrote {len(predictions)} rows to {args.output}")
    print()


if __name__ == "__main__":
    main()
