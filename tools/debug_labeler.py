"""Standalone tool to sanity-check OutcomeLabeler and labeling.yaml.

Given a ticker and a candidate date, prints the entry price, the
target/stop price levels derived from labeling.yaml, a day-by-day walk
through the forward horizon (so it's clear exactly which day, if any,
triggered the label and why), and the final outcome. Plays the same role
for labeling.yaml that tools/debug_scanner.py plays for
scanning_criteria.yaml.

Usage:
    python -m tools.debug_labeler AAPL 2024-06-03
    python -m tools.debug_labeler AAPL 2024-06-03 --config config/labeling.yaml

Requires an editable install (`pip install -e ".[dev]"`) and a live network
connection for yfinance.
"""
from __future__ import annotations

import argparse

import pandas as pd

from fitr.config_schemas.labeling_schema import LabelingConfig
from fitr.config_schemas.loader import ConfigError, load_config
from fitr.data.yf_proxy import NoDataInRangeError, TickerNotFoundError, YFProxy
from fitr.labeling.labeler import INCONCLUSIVE, OutcomeLabeler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker")
    parser.add_argument("candidate_date", help="e.g. 2024-06-03")
    parser.add_argument("--config", default="config/labeling.yaml")
    args = parser.parse_args()

    try:
        config = load_config(args.config, LabelingConfig)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        raise SystemExit(1)

    proxy = YFProxy()
    labeler = OutcomeLabeler(proxy, config)

    result = labeler.label(args.ticker, args.candidate_date)

    print(f"\n{args.ticker} candidate on {result.candidate_date.date()}")
    print(f"  Evaluation method: {config.evaluation_method}")
    print(f"  Horizon: {config.horizon_trading_days} trading days")
    print(f"  Target return: {config.target_return:+.1%}   Stop loss: {config.stop_loss_return:+.1%}")

    if result.outcome == INCONCLUSIVE:
        print(f"\n  Outcome: INCONCLUSIVE ({result.trigger_reason})")
        if result.trading_days_available is not None:
            print(
                f"  Only {result.trading_days_available} forward trading days available "
                f"(need {config.horizon_trading_days}) -- this candidate is too recent to judge yet."
            )
        print()
        return

    print(f"  Entry price (candidate day's close): {result.entry_price:.2f}")
    target_price = result.entry_price * (1 + config.target_return)
    stop_price = result.entry_price * (1 + config.stop_loss_return)
    print(f"  Target price: {target_price:.2f}   Stop price: {stop_price:.2f}")

    # Re-walk the horizon day by day purely for display -- OutcomeLabeler
    # already did the real evaluation; this just makes it visible.
    resolved = proxy.previous_trading_day(pd.Timestamp(args.candidate_date))
    horizon_end = resolved + pd.Timedelta(days=config.horizon_trading_days * 2 + 30)
    try:
        window_df = proxy.get_history(args.ticker, resolved, horizon_end)
    except (TickerNotFoundError, NoDataInRangeError):
        window_df = None

    if window_df is not None:
        forward_df = window_df.loc[window_df.index > resolved].iloc[: config.horizon_trading_days]
        print("\n  Day-by-day walk through the horizon:")
        for date, row in forward_df.iterrows():
            flags = []
            if row["High"] >= target_price:
                flags.append("TARGET HIT")
            if row["Low"] <= stop_price:
                flags.append("STOP HIT")
            flag_str = f"  <-- {', '.join(flags)}" if flags else ""
            print(
                f"    {date.date()}  O={row['Open']:.2f} H={row['High']:.2f} "
                f"L={row['Low']:.2f} C={row['Close']:.2f}{flag_str}"
            )

    print(f"\n  Outcome: {result.outcome}  ({result.trigger_reason})")
    print(f"  Trigger date: {result.trigger_date.date()}   Trigger price: {result.trigger_price:.2f}")
    print(f"  Forward return at trigger: {result.forward_return:+.2%}\n")


if __name__ == "__main__":
    main()
