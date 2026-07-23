"""Standalone tool to sanity-check PositionSizer and position_sizing.yaml.

Given a predicted probability, an entry price, and account equity, prints
the full sizing computation -- the raw Kelly fraction and payoff ratio (if
using fractional_kelly), the risk-fraction math (if using
fixed_fractional), whether any max/min guardrail capped the result, and
the resulting share count and dollar value -- so the math can be
hand-verified before trusting it with real capital.

No network required -- unlike most of the other debug tools in this
project, PositionSizer is pure calculation.

Usage:
    python -m tools.debug_position_sizer --probability 0.45 --entry-price 23.50 --equity 100000
    python -m tools.debug_position_sizer --entry-price 23.50 --equity 100000 --sweep
    python -m tools.debug_position_sizer --probability 0.45 --entry-price 23.50 --equity 100000 --config config/position_sizing.yaml
"""
from __future__ import annotations

import argparse

from fitr.config_schemas.loader import ConfigError, load_config
from fitr.config_schemas.position_sizing_schema import PositionSizingConfig
from fitr.modeling.position_sizer import PositionSizer


def print_single(sizer: PositionSizer, config: PositionSizingConfig, probability: float, entry_price: float, equity: float) -> None:
    result = sizer.size("DEBUG", probability, entry_price, equity)

    print(f"\nMethod: {config.method}")
    print(f"Predicted probability: {probability:.1%}")
    print(f"Entry price: {entry_price:.2f}   Equity: {equity:,.2f}\n")

    if config.method == "fractional_kelly":
        b = config.target_return / abs(config.stop_loss_return)
        breakeven_p = abs(config.stop_loss_return) / (config.target_return + abs(config.stop_loss_return))
        print(f"Payoff ratio (target / |stop|): {config.target_return:.1%} / {abs(config.stop_loss_return):.1%} = {b:.2f}")
        print(f"Breakeven probability: {breakeven_p:.1%}")
        print(
            f"Full Kelly fraction: f* = p/S - q/T = {probability:.1%}/{abs(config.stop_loss_return):.1%} "
            f"- {1 - probability:.1%}/{config.target_return:.1%} = {result.kelly_fraction:+.2f}"
        )
        print("  (proportional-payoff formula, not the textbook all-or-nothing (bp-q)/b -- see position_sizer.py)")
        print(f"Kelly multiplier: {config.kelly_multiplier:.0%}")
        print(f"Raw fraction = full Kelly * multiplier = {result.raw_fraction:+.1%}")
        if result.kelly_fraction is not None and result.kelly_fraction < 0:
            print("  (negative Kelly -- this probability is below breakeven for this payoff ratio; floored at 0)")
    else:
        print(f"Risk fraction (of equity, per trade): {config.risk_fraction:.1%}")
        print(f"Stop loss: {config.stop_loss_return:+.1%}")
        print(f"Raw fraction = risk_fraction / |stop_loss_return| = {config.risk_fraction:.1%} / {abs(config.stop_loss_return):.1%} = {result.raw_fraction:.1%}")
        print("  (predicted probability is not used by fixed_fractional)")

    print(f"\nGuardrails: max={config.max_position_fraction:.1%}  min={config.min_position_fraction:.1%}")
    print(f"Capping applied: {result.capped}")
    print(f"\nRecommended fraction of equity: {result.recommended_fraction:.1%}")
    print(f"Recommended shares: {result.recommended_shares}")
    print(f"Position value: {result.position_value:,.2f}\n")


def print_sweep(sizer: PositionSizer, entry_price: float, equity: float) -> None:
    print(f"\n{'probability':>12}{'raw fraction':>15}{'capped':>9}{'final fraction':>16}{'shares':>9}{'value':>14}")
    for i in range(1, 20):
        p = i / 20
        result = sizer.size("DEBUG", p, entry_price, equity)
        print(
            f"{p:>11.0%} {result.raw_fraction:>14.1%} {str(result.capped):>8} "
            f"{result.recommended_fraction:>15.1%} {result.recommended_shares:>8} {result.position_value:>13,.2f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probability", type=float, default=None, help="Predicted P(success) for a single sizing calculation.")
    parser.add_argument("--entry-price", type=float, required=True)
    parser.add_argument("--equity", type=float, required=True)
    parser.add_argument("--config", default="config/position_sizing.yaml")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Show a table across probabilities 5%%-95%% instead of a single calculation.",
    )
    args = parser.parse_args()

    if not args.sweep and args.probability is None:
        parser.error("Provide --probability for a single calculation, or --sweep for a table across probabilities.")

    try:
        config = load_config(args.config, PositionSizingConfig)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        raise SystemExit(1)

    sizer = PositionSizer(config)

    if args.sweep:
        print_sweep(sizer, args.entry_price, args.equity)
    else:
        print_single(sizer, config, args.probability, args.entry_price, args.equity)


if __name__ == "__main__":
    main()
