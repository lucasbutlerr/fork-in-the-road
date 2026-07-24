"""Production pipeline script: runs walk-forward validation -- the same
model.yaml config trained and evaluated across several sequential
train/test splits, instead of trusting one fixed split.

Takes full.csv (the UNSPLIT dataset written by scripts/build_dataset.py
alongside train.csv/test.csv) as input, since walk-forward does its own
date-based slicing into folds -- a pre-split train/test pair isn't the
right shape for this.

Usage:
    python -m scripts.walk_forward --data data/processed/full.csv

    python -m scripts.walk_forward \\
        --data data/processed/full.csv \\
        --walk-forward-config config/walk_forward.yaml \\
        --model-config config/model.yaml \\
        --position-sizing-config config/position_sizing.yaml \\
        --threshold 0.33
"""
from __future__ import annotations

import argparse

import pandas as pd

from fitr.config_schemas.loader import ConfigError, load_config
from fitr.config_schemas.model_schema import ModelConfig
from fitr.config_schemas.position_sizing_schema import PositionSizingConfig
from fitr.config_schemas.walk_forward_schema import WalkForwardConfig
from fitr.modeling.position_sizer import PositionSizer
from fitr.modeling.walk_forward import WalkForwardValidator, format_walk_forward_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Path to full.csv (the unsplit dataset from scripts/build_dataset.py)")
    parser.add_argument("--walk-forward-config", default="config/walk_forward.yaml")
    parser.add_argument("--model-config", default="config/model.yaml")
    parser.add_argument(
        "--position-sizing-config", default=None,
        help="Optional -- if given, each fold also gets a return-aware backtest, not just classification metrics.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--brief", action="store_true",
        help="Hide the per-fold calibration curve and setup breakdown -- useful once a run looks fine "
        "and you just want the headline numbers, not the full diagnostic detail.",
    )
    args = parser.parse_args()

    try:
        wf_config = load_config(args.walk_forward_config, WalkForwardConfig)
        model_config = load_config(args.model_config, ModelConfig)
        sizing_config = load_config(args.position_sizing_config, PositionSizingConfig) if args.position_sizing_config else None
    except ConfigError as exc:
        print(f"Config error: {exc}")
        raise SystemExit(1)

    try:
        full_df = pd.read_csv(args.data, parse_dates=["date"])
    except FileNotFoundError:
        print(f"Data file not found: {args.data}")
        raise SystemExit(1)

    if full_df.empty:
        print(f"{args.data} has no rows -- nothing to validate.")
        raise SystemExit(1)

    print(f"Data: {len(full_df)} rows, {full_df['date'].min().date()} to {full_df['date'].max().date()}")
    print(f"Folds: {wf_config.n_folds} ({wf_config.window_type})")
    print(f"Threshold: {args.threshold}")
    if sizing_config:
        print(f"Position sizing: {sizing_config.method} (backtest included per fold)")
    print()

    position_sizer = PositionSizer(sizing_config) if sizing_config else None
    validator = WalkForwardValidator(model_config, position_sizer=position_sizer)
    report = validator.run(full_df, wf_config, threshold=args.threshold)

    print(format_walk_forward_report(report, show_calibration=not args.brief, show_setups=not args.brief))


if __name__ == "__main__":
    main()
