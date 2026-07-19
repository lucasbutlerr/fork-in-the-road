"""Production pipeline script: scans a historical date range for breakout
candidates, computes their full ML feature vectors, labels their forward
outcomes, and writes a time-split train/test dataset.

Only the config files should ever need editing to change what this script
does: config/universe.txt, config/scanning_criteria.yaml,
config/features.yaml, config/labeling.yaml.

Usage:
    python -m scripts.build_dataset --start 2022-01-01 --end 2024-12-31 --split-date 2024-06-01

    python -m scripts.build_dataset \\
        --start 2022-01-01 --end 2024-12-31 --split-date 2024-06-01 \\
        --universe config/universe.txt --cooldown-days 5 \\
        --output-dir data/processed -v

Requires an editable install (`pip install -e ".[dev]"`) and a live network
connection for yfinance. Expect a full run over the ~1,800-ticker universe
across several years to take a while -- try a small date range and/or a
trimmed universe file first to sanity-check before a full run.
"""
from __future__ import annotations

import argparse
import logging

from fitr.config_schemas.features_schema import FeaturesConfig
from fitr.config_schemas.labeling_schema import LabelingConfig
from fitr.config_schemas.loader import ConfigError, load_config
from fitr.config_schemas.scanning_schema import ScanningConfig
from fitr.data.yf_proxy import YFProxy
from fitr.dataset.builder import DatasetBuilder, split_by_date
from fitr.dataset.universe import UniverseError, load_universe
from fitr.features.feature_engine import FeatureEngine
from fitr.labeling.labeler import OutcomeLabeler
from fitr.scanning.scanner import BreakoutScanner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="Start of the historical scan range, e.g. 2022-01-01")
    parser.add_argument("--end", required=True, help="End of the historical scan range, e.g. 2024-12-31")
    parser.add_argument("--split-date", required=True, help="Rows before this date -> train.csv, on/after -> test.csv")
    parser.add_argument("--universe", default="config/universe.txt")
    parser.add_argument("--scanning-config", default="config/scanning_criteria.yaml")
    parser.add_argument("--features-config", default="config/features.yaml")
    parser.add_argument("--labeling-config", default="config/labeling.yaml")
    parser.add_argument(
        "--cooldown-days", type=int, default=5, help="Per-ticker cooldown in trading days (default 5, ~1 week)"
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log progress every 20 trading days")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    try:
        scanning_config = load_config(args.scanning_config, ScanningConfig)
        features_config = load_config(args.features_config, FeaturesConfig)
        labeling_config = load_config(args.labeling_config, LabelingConfig)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        raise SystemExit(1)

    try:
        universe = load_universe(args.universe)
    except UniverseError as exc:
        print(f"Universe error: {exc}")
        raise SystemExit(1)

    print(f"Universe: {len(universe)} tickers")
    print(f"Scanning setups: {[s.name for s in scanning_config.setups]}")
    print(f"Features: {len(features_config.features)}")
    print(
        f"Labeling: {labeling_config.horizon_trading_days}-day horizon, "
        f"target {labeling_config.target_return:+.1%}, stop {labeling_config.stop_loss_return:+.1%}, "
        f"{labeling_config.evaluation_method}"
    )
    print(f"Cooldown: {args.cooldown_days} trading days\n")

    proxy = YFProxy()
    scanner = BreakoutScanner(proxy, scanning_config)
    feature_engine = FeatureEngine(proxy, features_config)
    labeler = OutcomeLabeler(proxy, labeling_config)

    builder = DatasetBuilder(
        proxy, scanner, feature_engine, labeler, universe, cooldown_trading_days=args.cooldown_days
    )

    print(f"Building dataset from {args.start} to {args.end}...\n")
    df = builder.build(args.start, args.end)
    stats = builder.last_build_stats

    print(f"Trading days scanned:     {stats.trading_days_scanned}")
    print(f"Total scan matches:       {stats.total_scan_matches}")
    print(f"Excluded (cooldown):      {stats.excluded_by_cooldown}")
    print(f"Excluded (inconclusive):  {stats.excluded_inconclusive}")
    print(f"Rows written:             {stats.rows_written}")
    if stats.rows_written:
        print(
            f"  SUCCESS: {stats.success_count} ({stats.success_count / stats.rows_written:.1%})   "
            f"FAILURE: {stats.failure_count} ({stats.failure_count / stats.rows_written:.1%})"
        )

    if df.empty:
        print("\nNo rows produced -- nothing to split or save.")
        return

    train_df, test_df = split_by_date(df, args.split_date)
    print(f"\nTrain: {len(train_df)} rows (before {args.split_date})")
    print(f"Test:  {len(test_df)} rows (on/after {args.split_date})")

    train_path = f"{args.output_dir}/train.csv"
    test_path = f"{args.output_dir}/test.csv"
    builder.save(train_df, train_path)
    builder.save(test_df, test_path)
    print(f"\nWrote {train_path}")
    print(f"Wrote {test_path}\n")


if __name__ == "__main__":
    main()
