"""Standalone diagnostic tool: pull real data for one ticker on one date and
print every indicator from fitr.features.indicators, each with a plain-
English description and instructions for verifying it against a real
charting platform (written with ThinkOrSwim in mind, since that's the
reference platform this was built to be checked against).

This is NOT part of the automated pipeline. indicators.py is unit-tested
against synthetic, hand-built data (tests/test_indicators.py) -- that
proves the *math* is internally consistent, but never checks the
calculations against what an actual chart shows for a real stock. This
script closes that gap and is meant to be kept around and rerun any time a
formula changes or a new indicator is added.

Usage:
    python -m tools.inspect_ticker AAPL 2024-06-03
    python -m tools.inspect_ticker AAPL 2024-06-03 --benchmark SPY --sector XLK
    python -m tools.inspect_ticker AAPL 2024-06-03 --lookback-days 600

Requires an editable install of this project (`pip install -e ".[dev]"`)
so `fitr` resolves as a package, and a live network connection for yfinance.
"""
from __future__ import annotations

import argparse
import math

import pandas as pd

from fitr.data.yf_proxy import YFProxy
from fitr.features import indicators as ind

# Indicator parameters used throughout this script. These match the
# defaults in indicators.py and the values discussed in the architecture
# plan; change them here if you want to sanity-check a different
# configuration; nothing else in the script needs to change.
PARAMS = {
    "short_return_window": 20,
    "medium_return_window": 50,
    "long_return_window": 200,
    "sma_short": 20,
    "sma_medium": 50,
    "sma_long": 200,
    "short_high_low_window": 20,
    "long_high_low_window": 252,
    "volume_window": 20,
    "volume_short_window": 5,
    "rsi_window": 14,
    "rsi_avg_window": 5,
    "rsi_slope_window": 5,
    "rsi_threshold": 60.0,
    "rsi_std_window": 10,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "macd_hist_avg_window": 5,
    "macd_hist_slope_window": 5,
    "macd_std_window": 10,
    "atr_window": 14,
    "return_vol_window": 20,
    "bollinger_window": 20,
    "bollinger_num_std": 2.0,
    "days_since_high_window": 252,
}


def fmt_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN (insufficient history for this window)"
    return f"{value * 100:.3f}%"


def fmt_num(value: float, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN (insufficient history for this window)"
    return f"{value:.{decimals}f}"


def fmt_count(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN (insufficient history for this window)"
    return f"{value:.0f}"


class Section:
    def __init__(self, title: str):
        self.title = title

    def __enter__(self):
        print(f"\n{'=' * 78}\n{self.title}\n{'=' * 78}")
        return self

    def __exit__(self, *exc):
        return False


def show(label: str, value_str: str, description: str, verify: str) -> None:
    print(f"\n{label}: {value_str}")
    print(f"  What it is: {description}")
    print(f"  Verify on ToS: {verify}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker", help="Ticker to inspect, e.g. AAPL")
    parser.add_argument("as_of_date", help="Date to evaluate as-of, e.g. 2024-06-03")
    parser.add_argument("--benchmark", default="SPY", help="Market benchmark ticker (default: SPY). Pass '' to skip.")
    parser.add_argument("--sector", default=None, help="Sector ETF ticker, e.g. XLK. Omit to skip.")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=500,
        help="Calendar days of history to pull before as_of_date (default 500, comfortably covers "
        "the 252-trading-day windows used below).",
    )
    args = parser.parse_args()

    proxy = YFProxy()
    as_of = pd.Timestamp(args.as_of_date).normalize()
    resolved = proxy.previous_trading_day(as_of)
    if resolved != as_of:
        print(f"Note: {as_of.date()} is not a trading day; using {resolved.date()} instead.")

    start = resolved - pd.Timedelta(days=args.lookback_days)
    df = proxy.get_history(args.ticker, start, resolved)

    print(f"\n{args.ticker} as of {resolved.date()}")
    print(f"  Close: {df['Close'].iloc[-1]:.2f}   Open: {df['Open'].iloc[-1]:.2f}   "
          f"High: {df['High'].iloc[-1]:.2f}   Low: {df['Low'].iloc[-1]:.2f}   "
          f"Volume: {df['Volume'].iloc[-1]:,.0f}")
    print(f"  ({len(df)} trading days of history loaded, {df.index.min().date()} to {df.index.max().date()})")
    print("\nCross-check this OHLCV row first: open ToS, go to the daily chart for "
          f"{args.ticker}, and hover/right-click on the {resolved.date()} candle -> "
          "'Price Info' (or similar). Every indicator below is derived from this row plus "
          "history before it, so if this doesn't match, nothing downstream will either.")

    benchmark_df = None
    if args.benchmark:
        benchmark_df = proxy.get_history(args.benchmark, start, resolved)
    sector_df = None
    if args.sector:
        sector_df = proxy.get_history(args.sector, start, resolved)

    p = PARAMS

    # ---- Price trend ----------------------------------------------------
    with Section("PRICE TREND"):
        for window, label in [
            (p["short_return_window"], "20-day return"),
            (p["medium_return_window"], "50-day return"),
            (p["long_return_window"], "200-day return"),
        ]:
            value = ind.n_day_return(df, window)
            show(
                label,
                fmt_pct(value),
                f"(Close today / Close {window} trading days ago) - 1.",
                f"Open the daily chart, hover over the candle exactly {window} trading days before "
                f"{resolved.date()} to read its close, then compute (today's close / that close - 1). "
                "ToS's 'Percent Change' study over a custom date range should also match if you set "
                f"it to a {window}-bar lookback.",
            )

        for window, label in [
            (p["sma_short"], "Relative distance above SMA20"),
            (p["sma_medium"], "Relative distance above SMA50"),
            (p["sma_long"], "Relative distance above SMA200"),
        ]:
            value = ind.relative_distance_from_ma(df, window, ma_type="sma")
            show(
                label,
                fmt_pct(value),
                f"(Close - SMA{window}) / SMA{window}. Positive means price is above its own "
                f"{window}-day trend.",
                f"Add a 'SimpleMovingAvg' study (length={window}) to the daily chart, read its "
                f"value on {resolved.date()}, then compute (Close - SMA)/SMA by hand.",
            )

        for window, label in [
            (p["short_high_low_window"], "20-day relative high"),
            (p["long_high_low_window"], "52-week relative high"),
        ]:
            value = ind.relative_n_day_high(df, window)
            show(
                label,
                fmt_pct(value),
                f"(Close - highest High in the trailing {window} days) / that high. Always <= 0; "
                "closer to 0 means price is sitting right at its recent high.",
                f"On the daily chart, visually scan the trailing {window} sessions (or add a "
                f"'Highest' study with length={window}) to find the highest High, then compute "
                "(today's Close - that High)/that High.",
            )

        value = ind.relative_n_day_low(df, p["short_high_low_window"])
        show(
            "20-day relative low",
            fmt_pct(value),
            f"(Close - lowest Low in the trailing {p['short_high_low_window']} days) / that low. "
            "Always >= 0.",
            f"Same approach as relative high, but with a 'Lowest' study (length={p['short_high_low_window']}) "
            "or by eye, using the Low of each candle.",
        )

    # ---- Volume -----------------------------------------------------------
    with Section("VOLUME"):
        value = ind.relative_volume(df, p["volume_window"])
        show(
            "Relative volume",
            fmt_num(value, 3) + "x",
            f"Today's volume / average volume over the *preceding* {p['volume_window']} days "
            "(today excluded from its own baseline).",
            f"Add a Simple Moving Average study to the Volume sub-panel (length={p['volume_window']}), "
            f"read its value on the trading day *before* {resolved.date()} (not on {resolved.date()} "
            "itself, since this script excludes the current day from the baseline), then divide "
            f"{resolved.date()}'s raw volume (right-click the volume bar -> Price Info) by that average.",
        )

        value = ind.volume_ratio_short_long(df, p["volume_short_window"], p["volume_window"])
        show(
            "5-day volume vs 20-day volume",
            fmt_num(value, 3) + "x",
            f"Average volume over the trailing {p['volume_short_window']} days divided by average "
            f"volume over the trailing {p['volume_window']} days.",
            f"Add two Volume Moving Average studies (lengths {p['volume_short_window']} and "
            f"{p['volume_window']}) to the Volume panel; read both values on {resolved.date()} "
            "and divide the short one by the long one.",
        )

        value = ind.volume_zscore(df, p["volume_window"])
        show(
            "Volume z-score",
            fmt_num(value, 3),
            f"How many standard deviations today's volume is from its trailing {p['volume_window']}-day "
            "mean (using the preceding window, today excluded).",
            "No direct ToS study for this one. Right-click the Volume panel -> 'Price Info' (or "
            f"export data) for each of the {p['volume_window']} trading days before {resolved.date()}, "
            "compute the mean and standard deviation by hand or in a spreadsheet, then "
            "(today's volume - mean) / std.",
        )

        value = ind.obv_slope(df, p["volume_window"])
        show(
            "OBV slope",
            fmt_num(value, 6),
            f"Trend of On-Balance-Volume over the trailing {p['volume_window']} days, normalized by "
            "average daily volume so it's comparable across tickers. Positive = OBV rising.",
            "Apply the 'OnBalanceVolume' study to the chart. The exact number is impractical to "
            "hand-verify (it's a normalized regression slope), but the *sign* and general steepness "
            f"should match what you see visually in the OBV line over the last {p['volume_window']} bars: "
            "clearly rising -> positive, clearly falling -> negative, choppy/flat -> near zero.",
        )

    # ---- RSI ----------------------------------------------------------------
    with Section("RSI"):
        value = ind.rsi(df, p["rsi_window"])
        show(
            f"RSI({p['rsi_window']})",
            fmt_num(value, 2),
            "Wilder's Relative Strength Index -- the easiest one to check, since ToS uses the same "
            "Wilder smoothing by default.",
            f"Add the 'RSI' study (length={p['rsi_window']}) to the chart; the printed value on "
            f"{resolved.date()} should match this script's output almost exactly (small differences "
            "can come from how far back your chart's data goes, since Wilder's smoothing has "
            "infinite memory -- a longer chart history converges to the same value).",
        )

        value = ind.rsi_moving_average(df, p["rsi_window"], p["rsi_avg_window"])
        show(
            f"{p['rsi_avg_window']}-day average RSI",
            fmt_num(value, 2),
            f"Simple average of the last {p['rsi_avg_window']} days of RSI({p['rsi_window']}) values.",
            f"On the RSI study, read its last {p['rsi_avg_window']} daily printed values (scroll back "
            "day by day, or add a moving average of the study output directly if your ToS version "
            "supports 'study over study') and average them by hand.",
        )

        value = ind.rsi_slope(df, p["rsi_window"], p["rsi_slope_window"])
        show(
            "RSI slope",
            fmt_num(value, 3) + " RSI points/day",
            f"Linear trend of RSI({p['rsi_window']}) over the trailing {p['rsi_slope_window']} days.",
            "No direct ToS study. Verify directionally: is the RSI line visibly sloping up or down "
            f"over the last {p['rsi_slope_window']} bars? For a rough magnitude check, read the RSI "
            f"study's printed values on the first and last of those {p['rsi_slope_window']} bars and "
            f"divide the difference by {p['rsi_slope_window'] - 1}.",
        )

        value = ind.days_rsi_above_threshold(df, p["rsi_window"], p["rsi_threshold"])
        show(
            f"Days RSI above {p['rsi_threshold']:.0f}",
            fmt_count(value),
            f"Length of the CONSECUTIVE streak, ending today, where RSI({p['rsi_window']}) has stayed "
            f"strictly above {p['rsi_threshold']:.0f}. 0 if today's RSI is at or below the threshold. "
            "This is a streak, not a count of how many days out of some window -- e.g. if RSI was "
            "above the threshold 10 days ago, dipped below, then rose above again 2 days ago, this "
            "reports 2 (or 3, including today), not 10+.",
            f"On the RSI study, scroll back day by day from {resolved.date()}, reading the printed "
            f"RSI value on each prior bar. Count consecutive days where RSI > {p['rsi_threshold']:.0f}, "
            "stopping counting at the first day it wasn't (even if it goes back above later).",
        )

        value = ind.rsi_std(df, p["rsi_window"], p["rsi_std_window"])
        show(
            f"RSI {p['rsi_std_window']}-day standard deviation",
            fmt_num(value, 3),
            f"Standard deviation of RSI({p['rsi_window']}) over the trailing {p['rsi_std_window']} days "
            "-- how choppy vs. steady recent momentum has been.",
            f"Read the RSI study's last {p['rsi_std_window']} daily values and compute std dev in a "
            "calculator or spreadsheet. No direct ToS display for this.",
        )

    # ---- MACD -----------------------------------------------------------------
    with Section("MACD"):
        macd_desc = (
            f"MACD(fast={p['macd_fast']}, slow={p['macd_slow']}, signal={p['macd_signal']}) -- these "
            "are ToS's own defaults, so no config mismatch to worry about."
        )

        value = ind.macd_value(df, p["macd_fast"], p["macd_slow"], p["macd_signal"])
        show(
            "MACD value",
            fmt_pct(value),
            f"{macd_desc} IMPORTANT: this script reports the MACD line as a PERCENTAGE OF PRICE "
            "(MACD line / Close), not the raw dollar value ToS displays, because raw MACD scales "
            "with share price and isn't comparable across tickers. Divide ToS's raw number by the "
            "close price before comparing.",
            f"Add the 'MACD' study to the chart, read the MACD line's raw value on {resolved.date()}, "
            "then divide it by that day's closing price. That result should match this script's "
            "output, not the raw MACD study value.",
        )

        value = ind.macd_histogram_moving_average(df, p["macd_fast"], p["macd_slow"], p["macd_signal"], p["macd_hist_avg_window"])
        show(
            f"{p['macd_hist_avg_window']}-day average histogram",
            fmt_pct(value),
            "Average of the (price-normalized) MACD histogram over the trailing window. Same "
            "normalization caveat as MACD value above applies to each day before averaging.",
            f"Read the histogram (the bar plot within the MACD study) for the last "
            f"{p['macd_hist_avg_window']} days, divide each day's value by that day's close, then "
            "average.",
        )

        value = ind.macd_histogram_slope(df, p["macd_fast"], p["macd_slow"], p["macd_signal"], p["macd_hist_slope_window"])
        show(
            "Histogram slope",
            fmt_num(value, 6),
            f"Linear trend of the (price-normalized) MACD histogram over the trailing "
            f"{p['macd_hist_slope_window']} days.",
            f"Visually: is the histogram growing or shrinking over the last {p['macd_hist_slope_window']} "
            "bars? Sign should match; exact magnitude is impractical to hand-verify.",
        )

        value = ind.days_macd_above_signal(df, p["macd_fast"], p["macd_slow"], p["macd_signal"])
        show(
            "Days MACD above signal line",
            fmt_count(value),
            "Length of the CONSECUTIVE streak, ending today, where the MACD line has stayed above "
            "its signal line -- same streak convention as the RSI streak feature above.",
            "On the MACD study, count consecutive days (scrolling back) where the MACD line "
            "(typically blue) sits above the signal line (typically red/orange), stopping at the "
            "first day it didn't.",
        )

        value = ind.macd_std(df, p["macd_fast"], p["macd_slow"], p["macd_signal"], p["macd_std_window"])
        show(
            f"MACD {p['macd_std_window']}-day standard deviation",
            fmt_num(value, 5),
            "Standard deviation of the (price-normalized) MACD line over the trailing window.",
            f"Read the MACD line's raw values for the last {p['macd_std_window']} days, divide each "
            "by that day's close, then compute std dev by hand.",
        )

    # ---- Volatility and breakout quality --------------------------------------
    with Section("VOLATILITY AND BREAKOUT QUALITY"):
        value = ind.atr_normalized(df, p["atr_window"])
        show(
            f"ATR({p['atr_window']}) normalized",
            fmt_pct(value),
            f"Average True Range(length={p['atr_window']}) divided by closing price, so it reads as "
            "a percentage rather than a raw dollar figure.",
            f"Add the 'ATR' study (length={p['atr_window']}) to the chart, read its raw value on "
            f"{resolved.date()}, then divide by that day's close.",
        )

        value = ind.return_volatility(df, p["return_vol_window"])
        show(
            f"Return volatility ({p['return_vol_window']}-day)",
            fmt_pct(value),
            f"Standard deviation of daily percentage returns over the trailing {p['return_vol_window']} days.",
            "No single matching ToS study (ToS's 'HistoricalVolatility' uses a different, "
            f"annualized formula -- don't use it as a direct check). Read the last "
            f"{p['return_vol_window'] + 1} daily closes off the chart, compute day-over-day percent "
            "changes, then the standard deviation of those percent changes by hand or in a spreadsheet.",
        )

        value = ind.bollinger_width(df, p["bollinger_window"], p["bollinger_num_std"])
        show(
            "Bollinger width",
            fmt_pct(value),
            f"(Upper - Lower) / Middle from Bollinger Bands(length={p['bollinger_window']}, "
            f"num_std={p['bollinger_num_std']}) -- relative to the middle band so it's comparable "
            "across tickers.",
            f"Add the 'BollingerBands' study (length={p['bollinger_window']}, num dev="
            f"{p['bollinger_num_std']:.0f}) to the chart, read the Upper/Middle/Lower band values on "
            f"{resolved.date()}, then compute (Upper - Lower) / Middle.",
        )

        value = ind.days_since_high(df, p["days_since_high_window"])
        show(
            f"Days since previous high ({p['days_since_high_window']}-day window)",
            fmt_count(value),
            f"Trading days between {resolved.date()} and the day the trailing "
            f"{p['days_since_high_window']}-day High was set. 0 means today IS that high.",
            f"Scroll back {p['days_since_high_window']} sessions on the daily chart and visually "
            "identify the tallest High in that stretch (or use a 'Highest' study to mark it), then "
            f"count the trading days between that bar and {resolved.date()}.",
        )

        value = ind.close_location_value(df)
        show(
            "Close location value",
            fmt_num(value, 3),
            "Where today's close sits within today's High-Low range, scaled to [-1, 1]. +1 = closed "
            "at the high, -1 = closed at the low, 0 = closed exactly mid-range.",
            f"Read {resolved.date()}'s High, Low, and Close directly off the daily candle "
            "(right-click -> Price Info), then compute ((Close-Low) - (High-Close)) / (High-Low) by hand.",
        )

        value = ind.high_low_range_pct(df)
        show(
            "High-low range %",
            fmt_pct(value),
            "Today's (High - Low) / Close -- intraday range as a percentage of price.",
            f"Same source data as close location value: (High - Low) / Close for {resolved.date()}.",
        )

        value = ind.close_to_open_gap_pct(df)
        show(
            "Close-to-open gap %",
            fmt_pct(value),
            "Today's Open relative to yesterday's Close.",
            f"Read {resolved.date()}'s Open and the prior trading day's Close off the chart (this is "
            "visually the size of the gap between the two candles), then compute "
            "(Open - prior Close) / prior Close.",
        )

        value = ind.consecutive_up_days(df)
        show(
            "Consecutive up days",
            fmt_count(value),
            f"Length of the CONSECUTIVE streak, ending {resolved.date()}, of positive daily returns.",
            "Scroll back day by day on the daily chart, counting consecutive green/up candles "
            "(Close > prior Close), stopping at the first red/down candle.",
        )

        value = ind.volume_trend_slope(df, p["volume_window"])
        show(
            f"Volume trend slope ({p['volume_window']}-day)",
            fmt_num(value, 6),
            f"Linear trend of raw volume over the trailing {p['volume_window']} days, normalized by "
            "average volume in that window.",
            f"Visually inspect whether the Volume panel's bars have been trending taller or shorter "
            f"over the last {p['volume_window']} sessions. Exact magnitude is impractical to "
            "hand-verify (same caveat as OBV slope above) -- check the sign and general shape.",
        )

    # ---- Market and sector context ----------------------------------------
    with Section("MARKET AND SECTOR CONTEXT"):
        if benchmark_df is not None:
            bench_return = ind.n_day_return(benchmark_df, p["short_return_window"])
            show(
                f"{args.benchmark} 20-day return",
                fmt_pct(bench_return),
                f"Same n_day_return calculation as above, applied to {args.benchmark} instead of "
                f"{args.ticker}.",
                f"Open {args.benchmark}'s own daily chart in ToS and apply the same verification as "
                "the ticker's 20-day return, above.",
            )

            rel_value = ind.relative_return_vs_benchmark(df, benchmark_df, p["short_return_window"])
            show(
                f"20-day return vs {args.benchmark}",
                fmt_pct(rel_value),
                f"{args.ticker}'s 20-day return minus {args.benchmark}'s 20-day return (a difference, "
                "not a ratio -- see the docstring on relative_return_vs_benchmark for why).",
                f"Compute {args.ticker}'s and {args.benchmark}'s 20-day returns independently "
                "(each verified as described above), then subtract.",
            )
        else:
            print("\n(Benchmark comparison skipped -- no --benchmark ticker provided.)")

        if sector_df is not None:
            sector_return = ind.n_day_return(sector_df, p["short_return_window"])
            show(
                f"{args.sector} 20-day return",
                fmt_pct(sector_return),
                f"Same n_day_return calculation, applied to sector ETF {args.sector}.",
                f"Open {args.sector}'s own daily chart in ToS and apply the same verification as "
                "the ticker's 20-day return, above.",
            )

            rel_sector_value = ind.relative_return_vs_benchmark(df, sector_df, p["short_return_window"])
            show(
                f"20-day return vs {args.sector}",
                fmt_pct(rel_sector_value),
                f"{args.ticker}'s 20-day return minus {args.sector}'s 20-day return.",
                f"Compute {args.ticker}'s and {args.sector}'s 20-day returns independently, then "
                "subtract.",
            )
        else:
            print("\n(Sector comparison skipped -- pass --sector TICKER, e.g. --sector XLK, to include it.)")

    print()


if __name__ == "__main__":
    main()
