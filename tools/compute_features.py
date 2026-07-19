"""Standalone tool: given a ticker, a date, and a features.yaml path,
compute and print every configured feature's value.

This is FeatureEngine's counterpart to tools/inspect_ticker.py: that script
hardcodes every indicator directly to sanity-check the underlying math
against a real chart; this one runs the actual config-driven path a real
pipeline run would take, so it doubles as a sanity check on features.yaml
itself -- does this config actually produce the columns and values you
think it does, with no NaNs where you don't expect any?

Usage:
    python -m tools.compute_features AAPL 2024-06-03
    python -m tools.compute_features AAPL 2024-06-03 --config config/features.yaml
    python -m tools.compute_features AAPL 2024-06-03 --lookback-days 700

Requires an editable install of this project (`pip install -e ".[dev]"`) so
`fitr` resolves as a package, and a live network connection for yfinance.
"""
from __future__ import annotations

import argparse
import math

import pandas as pd

from fitr.config_schemas.loader import ConfigError, load_config
from fitr.config_schemas.features_schema import FeaturesConfig
from fitr.data.yf_proxy import YFProxy
from fitr.features.feature_engine import FeatureEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker", help="Ticker to inspect, e.g. AAPL")
    parser.add_argument("as_of_date", help="Date to evaluate as-of, e.g. 2024-06-03")
    parser.add_argument("--config", default="config/features.yaml", help="Path to features.yaml (default: config/features.yaml)")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=500,
        help="Calendar days of history fetched before as_of_date (default 500). Matches "
        "FeatureEngine's own default -- raise this if you add a feature with a longer "
        "window and start seeing unexpected NaNs.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config, FeaturesConfig)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        raise SystemExit(1)

    proxy = YFProxy()
    engine = FeatureEngine(proxy, config, history_buffer_days=args.lookback_days)

    as_of = pd.Timestamp(args.as_of_date).normalize()
    resolved = proxy.previous_trading_day(as_of)
    if resolved != as_of:
        print(f"Note: {as_of.date()} is not a trading day; computing features as of {resolved.date()} instead.")

    try:
        features = engine.compute(args.ticker, args.as_of_date)
    except Exception as exc:
        print(f"\nFailed to compute features for {args.ticker}: {exc}")
        raise SystemExit(1)

    if config.benchmarks:
        bench_str = ", ".join(f"{key}={ticker}" for key, ticker in config.benchmarks.items())
        print(f"Benchmarks: {bench_str}")

    print(f"\n{args.ticker} as of {resolved.date()} -- {len(features)} features from {args.config}\n")

    name_width = max(len(name) for name in features) + 2
    nan_names = []
    for name, value in features.items():
        if isinstance(value, float) and math.isnan(value):
            nan_names.append(name)
            value_str = "NaN"
        else:
            value_str = f"{value:.6f}"
        print(f"  {name:<{name_width}} {value_str}")

    print(f"\n{len(features) - len(nan_names)}/{len(features)} features computed; {len(nan_names)} NaN.")
    if nan_names:
        print(
            f"NaN features: {nan_names}\n"
            "Usually means insufficient history for that window (a recently-listed ticker, "
            "or --lookback-days set too low for one of the configured windows) -- not "
            "necessarily a bug. Cross-check against tools/inspect_ticker.py if a specific "
            "value looks wrong rather than missing."
        )
    print()


if __name__ == "__main__":
    main()
