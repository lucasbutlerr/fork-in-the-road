"""Production pipeline script: runs a return-aware backtest of a saved
model against a test set -- simulating actual equity growth using
PositionSizer + real forward_return, not just classification accuracy.
This is the final piece of Phase 2, tying calibration (meaningful
probabilities) and PositionSizer (probability -> allocation) together into
an actual P&L simulation.

Requires a test.csv built with a version of DatasetBuilder that writes
forward_return (Phase 2 Step 1 onward) -- older datasets built before that
won't have the column this needs.

Usage:
    python -m scripts.backtest --model data/models/model_20240601.joblib --test data/processed/test.csv

    python -m scripts.backtest \\
        --model data/models/model_20240601.joblib --test data/processed/test.csv \\
        --position-sizing-config config/position_sizing.yaml \\
        --threshold 0.33 --starting-equity 100000 \\
        --trades-output data/backtests/trades.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from fitr.config_schemas.loader import ConfigError, load_config
from fitr.config_schemas.position_sizing_schema import PositionSizingConfig
from fitr.modeling.backtest import BacktestSimulator, format_backtest_report
from fitr.modeling.position_sizer import PositionSizer
from fitr.modeling.trainer import TrainedModel


def write_trades_csv(trades, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["date", "ticker", "predicted_probability", "recommended_fraction", "forward_return", "equity_before", "equity_after", "pnl"]
        )
        for t in trades:
            writer.writerow(
                [t.date.date(), t.ticker, f"{t.predicted_probability:.6f}", f"{t.recommended_fraction:.6f}",
                 f"{t.forward_return:.6f}", f"{t.equity_before:.2f}", f"{t.equity_after:.2f}", f"{t.pnl:.2f}"]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to a saved model .joblib (from scripts/train_model.py)")
    parser.add_argument("--test", required=True, help="Path to test.csv (must include a forward_return column)")
    parser.add_argument("--position-sizing-config", default="config/position_sizing.yaml")
    parser.add_argument("--threshold", type=float, default=0.5, help="Only trade candidates at or above this P(success)")
    parser.add_argument("--starting-equity", type=float, default=100_000.0)
    parser.add_argument("--trades-output", default="data/backtests/trades.csv", help="Where to write the full trade log. Pass '' to skip.")
    parser.add_argument("--max-trades-shown", type=int, default=20, help="How many trades to print in full (the CSV always has all of them)")
    args = parser.parse_args()

    try:
        trained = TrainedModel.load(args.model)
    except FileNotFoundError as exc:
        print(f"Model file not found: {exc}")
        raise SystemExit(1)

    try:
        sizing_config = load_config(args.position_sizing_config, PositionSizingConfig)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        raise SystemExit(1)

    try:
        test_df = pd.read_csv(args.test, parse_dates=["date"])
    except FileNotFoundError:
        print(f"Test file not found: {args.test}")
        raise SystemExit(1)

    if test_df.empty:
        print(f"{args.test} has no rows -- nothing to backtest.")
        raise SystemExit(1)

    if "forward_return" not in test_df.columns:
        print(
            f"{args.test} has no 'forward_return' column -- it was built with a version of "
            "DatasetBuilder from before Phase 2 Step 1. Rebuild the dataset to backtest it."
        )
        raise SystemExit(1)

    print(f"Model: {args.model} (trained {trained.trained_at})")
    print(f"Position sizing: {sizing_config.method}")
    print(f"Threshold: {args.threshold}   Starting equity: {args.starting_equity:,.2f}\n")

    simulator = BacktestSimulator(PositionSizer(sizing_config))
    report = simulator.run(trained, test_df, threshold=args.threshold, starting_equity=args.starting_equity)

    print(format_backtest_report(report, max_trades_shown=args.max_trades_shown))

    if args.trades_output:
        out_path = Path(args.trades_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_trades_csv(report.trades, str(out_path))
        print(f"Wrote {len(report.trades)} trades to {out_path}\n")


if __name__ == "__main__":
    main()
