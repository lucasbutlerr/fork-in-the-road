"""Standalone tool to sanity-check BreakoutScanner and scanning_criteria.yaml.

Two modes:

  Single-ticker breakdown (give a date + ticker): prints every configured
  metric's value, then every setup's condition-by-condition pass/fail and
  final verdict -- exactly what's needed to hand-verify a scanner
  yes/no makes sense, the same role tools/inspect_ticker.py plays for raw
  indicator values and tools/compute_features.py plays for features.yaml.

  Full-universe scan (give a date + --universe instead): runs a real scan
  over a ticker list and prints which tickers matched which setups -- a
  quick "does today's watchlist look reasonable" check, without touching
  the eventual production scripts/scan_candidates.py.

Usage:
    python -m tools.debug_scanner 2024-06-03 AAPL
    python -m tools.debug_scanner 2024-06-03 AAPL --config config/scanning_criteria.yaml
    python -m tools.debug_scanner 2024-06-03 --universe config/universe.txt

Requires an editable install (`pip install -e ".[dev]"`) and a live network
connection for yfinance.
"""
from __future__ import annotations

import argparse

from fitr.config_schemas.loader import ConfigError, load_config
from fitr.config_schemas.scanning_schema import ScanningConfig
from fitr.data.yf_proxy import NoDataInRangeError, TickerNotFoundError, YFProxy
from fitr.scanning.scanner import BreakoutScanner


def _fmt(value: float) -> str:
    if value != value:  # NaN
        return "NaN"
    return f"{value:.6f}"


def print_single_ticker_breakdown(scanner: BreakoutScanner, ticker: str, as_of_date: str) -> None:
    try:
        explanation = scanner.explain(ticker, as_of_date)
    except (TickerNotFoundError, NoDataInRangeError) as exc:
        print(f"Could not evaluate {ticker} on {as_of_date}: {exc}")
        raise SystemExit(1)

    print(f"\n{ticker} as of {explanation.as_of_date.date()}\n")

    print("Metrics:")
    name_width = max(len(n) for n in explanation.metrics) + 2
    for name, value in explanation.metrics.items():
        print(f"  {name:<{name_width}} {_fmt(value)}")

    print("\nSetups:")
    for setup in explanation.setups:
        verdict = "MATCH" if setup.passed else "no match"
        print(f"\n  [{verdict}] {setup.name}  (logic: {setup.logic})")
        for cond in setup.conditions:
            mark = "PASS" if cond.passed else "fail"
            print(f"      {mark}  {cond.metric} {cond.operator} {cond.threshold}   (actual: {_fmt(cond.actual_value)})")

    matched = explanation.matched_setup_names
    verdict_line = f"MATCHED: {', '.join(matched)}" if matched else "no setups matched"
    print(f"\nOverall: {verdict_line}\n")


def print_universe_scan(scanner: BreakoutScanner, universe_path: str, as_of_date: str) -> None:
    with open(universe_path) as f:
        tickers = [line.strip() for line in f if line.strip()]

    print(f"\nScanning {len(tickers)} tickers as of {as_of_date}...\n")
    results = scanner.scan(tickers, as_of_date)

    if not results:
        print("No candidates matched any setup.\n")
        return

    name_width = max(len(r.ticker) for r in results) + 2
    for r in sorted(results, key=lambda r: r.ticker):
        print(f"  {r.ticker:<{name_width}} {', '.join(r.matched_setups)}")

    print(f"\n{len(results)}/{len(tickers)} tickers matched at least one setup.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("as_of_date", help="Date to evaluate as-of, e.g. 2024-06-03")
    parser.add_argument("ticker", nargs="?", help="Single ticker for a detailed breakdown.")
    parser.add_argument(
        "--universe", help="Path to a universe file (one ticker per line). Runs a full scan instead of a single-ticker breakdown."
    )
    parser.add_argument("--config", default="config/scanning_criteria.yaml", help="Path to scanning_criteria.yaml")
    args = parser.parse_args()

    if not args.ticker and not args.universe:
        parser.error("Provide either a ticker for a detailed breakdown, or --universe for a full scan.")
    if args.ticker and args.universe:
        parser.error("Provide a ticker OR --universe, not both.")

    try:
        config = load_config(args.config, ScanningConfig)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        raise SystemExit(1)

    proxy = YFProxy()
    scanner = BreakoutScanner(proxy, config)

    if args.ticker:
        print_single_ticker_breakdown(scanner, args.ticker, args.as_of_date)
    else:
        print_universe_scan(scanner, args.universe, args.as_of_date)


if __name__ == "__main__":
    main()
