"""Pure indicator/feature calculation functions.

Every public function here takes an OHLCV DataFrame (columns: Open, High,
Low, Close, Volume; ascending DatetimeIndex) and returns a single float --
the value of that indicator "as of" the DataFrame's last row. Callers
(FeatureEngine, BreakoutScanner) are responsible for truncating the frame to
the correct as-of date before calling in; that truncation is the single
enforcement point for point-in-time correctness across the whole project.

Design rules followed throughout:

1. No side effects, no I/O, no yfinance import -- these are testable with
   nothing but a hand-built DataFrame.
2. Every value is a ratio, percentage, z-score, or otherwise scale-free
   number, per the project's requirement that features be comparable
   across tickers with wildly different share prices. A couple of the
   requested indicators (raw MACD, raw ATR, raw OBV) are scale-dependent
   in their textbook form, so those are normalized here -- see the
   docstrings on macd_value, atr_normalized, and obv_slope for specifics.
3. Insufficient history returns NaN rather than raising. With ~1,800
   tickers of varying listing age, some candidates simply won't have
   enough lookback for a given window -- NaN lets that propagate cleanly
   into the training set instead of crashing a batch job. XGBoost handles
   NaN features natively, so this doesn't require imputation later.
4. Where a "streak" concept is requested (e.g. "days RSI above 60"), it's
   implemented as the *consecutive trailing streak ending today*, not a
   count within a lookback window -- that's the more useful persistence
   signal for a breakout-detection use case, and it's called out in each
   such function's docstring.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Private series builders -- shared building blocks for the public functions
# below. These return full pd.Series aligned to the input's index, not
# scalars; the public functions slice off whatever point/window they need.
# ============================================================================

def _sma_series(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _ema_series(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def _rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    # avg_gain == 0 and avg_loss == 0 means price hasn't moved at all over
    # the window -- rs is 0/0 (NaN) in that case; treat it as neutral (50)
    # rather than letting it propagate as NaN.
    flat = (avg_gain == 0) & (avg_loss == 0)
    rsi_values = rsi_values.where(~flat, 50.0)
    return rsi_values


def _macd_series(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram), all in raw price units
    (i.e. not yet normalized -- callers normalize by price where needed)."""
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema_series(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _true_range_series(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    ranges = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _atr_series(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tr = _true_range_series(df)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _bollinger_bands_series(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = _sma_series(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def _rolling_high_series(high: pd.Series, window: int) -> pd.Series:
    return high.rolling(window=window, min_periods=window).max()


def _rolling_low_series(low: pd.Series, window: int) -> pd.Series:
    return low.rolling(window=window, min_periods=window).min()


def _linreg_slope(values: pd.Series) -> float:
    """Ordinary least squares slope of `values` against 0..n-1 (i.e. "per
    trading day"). NaNs are dropped before fitting; returns NaN if fewer
    than 2 clean points remain."""
    clean = values.dropna()
    if len(clean) < 2:
        return float("nan")
    x = np.arange(len(clean))
    slope, _intercept = np.polyfit(x, clean.to_numpy(dtype=float), 1)
    return float(slope)


def _trailing_streak(values: list[float], predicate) -> float:
    """Length of the consecutive run at the *end* of `values` for which
    predicate(value) is True. Stops at the first False or NaN encountered
    walking backward from the end."""
    streak = 0
    for value in reversed(values):
        if value is None or (isinstance(value, float) and np.isnan(value)) or not predicate(value):
            break
        streak += 1
    return float(streak)


# ============================================================================
# Price trend
# ============================================================================

def n_day_return(df: pd.DataFrame, window: int, price_col: str = "Close") -> float:
    """Percentage return over the trailing `window` trading days:
    (price_today / price_n_days_ago) - 1. A ratio by construction, so
    directly comparable across tickers regardless of share price. Use with
    window=20/50/200 for the 20/50/200-day return features, and reuse
    directly against a benchmark ETF's DataFrame for market/sector context
    (e.g. n_day_return(spy_df, 20))."""
    if len(df) < window + 1:
        return float("nan")
    close = df[price_col]
    prior = close.iloc[-1 - window]
    if prior == 0 or pd.isna(prior):
        return float("nan")
    return float(close.iloc[-1] / prior - 1)


def relative_distance_from_ma(
    df: pd.DataFrame, window: int, ma_type: str = "sma", price_col: str = "Close"
) -> float:
    """(price - moving_average) / moving_average -- how far above or below
    its own trend the price is sitting, as a percentage. Positive means
    price is above the average. Covers both the SMA20/50/200 distance
    features (ma_type="sma") and an EMA variant if you want it
    (ma_type="ema") without needing separate functions."""
    if len(df) < window:
        return float("nan")
    close = df[price_col]
    ma_series = _sma_series(close, window) if ma_type == "sma" else _ema_series(close, window)
    ma_value = ma_series.iloc[-1]
    if pd.isna(ma_value) or ma_value == 0:
        return float("nan")
    return float((close.iloc[-1] - ma_value) / ma_value)


def relative_n_day_high(
    df: pd.DataFrame, window: int, price_col: str = "Close", high_col: str = "High"
) -> float:
    """(price - rolling_high) / rolling_high over the trailing `window`
    days. Always <= 0 by construction; closer to 0 means price is sitting
    right at (or near) its recent high -- precisely the breakout setup this
    project is trying to detect. Use window=20 and window=252 for the
    20-day and 52-week relative-high features."""
    if len(df) < window:
        return float("nan")
    rolling_high = _rolling_high_series(df[high_col], window).iloc[-1]
    if pd.isna(rolling_high) or rolling_high == 0:
        return float("nan")
    return float((df[price_col].iloc[-1] - rolling_high) / rolling_high)


def relative_n_day_low(
    df: pd.DataFrame, window: int, price_col: str = "Close", low_col: str = "Low"
) -> float:
    """(price - rolling_low) / rolling_low over the trailing `window` days.
    Always >= 0 by construction; larger means price has run further above
    its recent low."""
    if len(df) < window:
        return float("nan")
    rolling_low = _rolling_low_series(df[low_col], window).iloc[-1]
    if pd.isna(rolling_low) or rolling_low == 0:
        return float("nan")
    return float((df[price_col].iloc[-1] - rolling_low) / rolling_low)


# ============================================================================
# Volume
# ============================================================================

def relative_volume(
    df: pd.DataFrame, window: int, volume_col: str = "Volume", exclude_current: bool = True
) -> float:
    """Today's volume divided by the average volume over the trailing
    `window` days. exclude_current=True (default) compares today against
    the *preceding* window rather than a window that includes today --
    including today in its own baseline mutes exactly the spike you're
    trying to detect."""
    volume = df[volume_col]
    if exclude_current:
        if len(volume) < window + 1:
            return float("nan")
        baseline = volume.iloc[-1 - window : -1].mean()
    else:
        if len(volume) < window:
            return float("nan")
        baseline = volume.iloc[-window:].mean()
    if baseline == 0 or pd.isna(baseline):
        return float("nan")
    return float(volume.iloc[-1] / baseline)


def volume_ratio_short_long(
    df: pd.DataFrame, short_window: int = 5, long_window: int = 20, volume_col: str = "Volume"
) -> float:
    """Ratio of average volume over a short window to average volume over a
    longer window -- a simple volume-trend indicator. > 1 means volume has
    picked up recently relative to its longer-term baseline."""
    if len(df) < long_window:
        return float("nan")
    volume = df[volume_col]
    short_avg = volume.iloc[-short_window:].mean()
    long_avg = volume.iloc[-long_window:].mean()
    if long_avg == 0 or pd.isna(long_avg):
        return float("nan")
    return float(short_avg / long_avg)


def volume_zscore(
    df: pd.DataFrame, window: int, volume_col: str = "Volume", exclude_current: bool = True
) -> float:
    """How many standard deviations today's volume is from its trailing
    `window`-day mean. More statistically robust than a raw ratio for
    tickers whose day-to-day volume is naturally more variable."""
    volume = df[volume_col]
    if exclude_current:
        if len(volume) < window + 1:
            return float("nan")
        baseline = volume.iloc[-1 - window : -1]
        current = volume.iloc[-1]
    else:
        if len(volume) < window:
            return float("nan")
        baseline = volume.iloc[-window:]
        current = volume.iloc[-1]
    std = baseline.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return float((current - baseline.mean()) / std)


def obv_slope(df: pd.DataFrame, window: int, price_col: str = "Close", volume_col: str = "Volume") -> float:
    """Trend of On-Balance-Volume over the trailing `window` days,
    expressed as the average daily OBV change divided by the window's
    average daily volume. Raw OBV is a running cumulative sum of volume,
    so its magnitude scales with a ticker's typical trading volume --
    without this normalization it would not be comparable across tickers.
    """
    if len(df) < window + 1:
        return float("nan")
    obv = _obv_series(df[price_col], df[volume_col])
    windowed = obv.iloc[-window:]
    slope = _linreg_slope(windowed)
    avg_volume = df[volume_col].iloc[-window:].mean()
    if avg_volume == 0 or pd.isna(avg_volume) or pd.isna(slope):
        return float("nan")
    return float(slope / avg_volume)


# ============================================================================
# RSI
# ============================================================================

def rsi(df: pd.DataFrame, window: int = 14, price_col: str = "Close") -> float:
    """Latest RSI(window) value, 0-100. Already scale-free by construction."""
    if len(df) < window + 1:
        return float("nan")
    return float(_rsi_series(df[price_col], window).iloc[-1])


def rsi_moving_average(
    df: pd.DataFrame, rsi_window: int = 14, avg_window: int = 5, price_col: str = "Close"
) -> float:
    """Simple moving average of RSI(rsi_window) over the trailing
    avg_window days -- a smoothed read on momentum, less noisy than the
    single-day RSI value."""
    if len(df) < rsi_window + avg_window:
        return float("nan")
    series = _rsi_series(df[price_col], rsi_window)
    return float(series.iloc[-avg_window:].mean())


def rsi_slope(
    df: pd.DataFrame, rsi_window: int = 14, slope_window: int = 5, price_col: str = "Close"
) -> float:
    """Linear trend of RSI over the trailing slope_window days, in RSI
    points per day. RSI is already bounded 0-100, so no further
    normalization is needed for cross-ticker comparability."""
    if len(df) < rsi_window + slope_window:
        return float("nan")
    series = _rsi_series(df[price_col], rsi_window)
    return _linreg_slope(series.iloc[-slope_window:])


def days_rsi_above_threshold(
    df: pd.DataFrame,
    rsi_window: int = 14,
    threshold: float = 60.0,
    price_col: str = "Close",
    max_lookback: int = 60,
) -> float:
    """Length of the current consecutive streak (ending today) where
    RSI(rsi_window) has stayed strictly above `threshold`. 0 if today's RSI
    isn't above the threshold. Capped at max_lookback so a ticker that's
    been overbought for months doesn't produce an unbounded value."""
    if len(df) < rsi_window + 1:
        return float("nan")
    series = _rsi_series(df[price_col], rsi_window).iloc[-max_lookback:]
    return _trailing_streak(series.tolist(), lambda v: v > threshold)


def rsi_std(df: pd.DataFrame, rsi_window: int = 14, std_window: int = 10, price_col: str = "Close") -> float:
    """Standard deviation of RSI over the trailing std_window days -- how
    choppy vs. steady recent momentum has been."""
    if len(df) < rsi_window + std_window:
        return float("nan")
    series = _rsi_series(df[price_col], rsi_window)
    return float(series.iloc[-std_window:].std())


# ============================================================================
# MACD
# ============================================================================

def macd_value(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, price_col: str = "Close") -> float:
    """Latest MACD line value, expressed as a percentage of price rather
    than raw price units. Raw MACD (EMA_fast - EMA_slow) scales with the
    stock's price level -- a $500 stock and a $20 stock aren't comparable
    without this normalization, so it's applied here rather than left to
    the caller."""
    if len(df) < slow + signal:
        return float("nan")
    close = df[price_col]
    macd_line, _signal_line, _hist = _macd_series(close, fast, slow, signal)
    price = close.iloc[-1]
    if price == 0 or pd.isna(price) or pd.isna(macd_line.iloc[-1]):
        return float("nan")
    return float(macd_line.iloc[-1] / price)


def macd_histogram_moving_average(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    avg_window: int = 5,
    price_col: str = "Close",
) -> float:
    """Average of the (price-normalized) MACD histogram over the trailing
    avg_window days -- smoothed momentum-of-momentum."""
    if len(df) < slow + signal + avg_window:
        return float("nan")
    close = df[price_col]
    _macd_line, _signal_line, hist = _macd_series(close, fast, slow, signal)
    normalized_hist = hist / close
    return float(normalized_hist.iloc[-avg_window:].mean())


def macd_histogram_slope(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    slope_window: int = 5,
    price_col: str = "Close",
) -> float:
    """Linear trend of the (price-normalized) MACD histogram over the
    trailing slope_window days."""
    if len(df) < slow + signal + slope_window:
        return float("nan")
    close = df[price_col]
    _macd_line, _signal_line, hist = _macd_series(close, fast, slow, signal)
    normalized_hist = hist / close
    return _linreg_slope(normalized_hist.iloc[-slope_window:])


def days_macd_above_signal(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = "Close",
    max_lookback: int = 60,
) -> float:
    """Length of the current consecutive streak (ending today) where the
    MACD line has stayed above its signal line -- a persistence measure,
    distinct from the histogram's magnitude or slope."""
    if len(df) < slow + signal + 1:
        return float("nan")
    close = df[price_col]
    macd_line, signal_line, _hist = _macd_series(close, fast, slow, signal)
    diff = (macd_line - signal_line).iloc[-max_lookback:]
    return _trailing_streak(diff.tolist(), lambda v: v > 0)


def macd_std(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    std_window: int = 10,
    price_col: str = "Close",
) -> float:
    """Standard deviation of the (price-normalized) MACD line over the
    trailing std_window days."""
    if len(df) < slow + signal + std_window:
        return float("nan")
    close = df[price_col]
    macd_line, _signal_line, _hist = _macd_series(close, fast, slow, signal)
    normalized = macd_line / close
    return float(normalized.iloc[-std_window:].std())


# ============================================================================
# Volatility and breakout quality
# ============================================================================

def atr_normalized(df: pd.DataFrame, window: int = 14) -> float:
    """Average True Range divided by closing price. Raw ATR scales with
    share price, so this expresses volatility as a percentage of price,
    comparable across tickers."""
    if len(df) < window + 1:
        return float("nan")
    atr_series = _atr_series(df, window)
    price = df["Close"].iloc[-1]
    atr_value = atr_series.iloc[-1]
    if price == 0 or pd.isna(price) or pd.isna(atr_value):
        return float("nan")
    return float(atr_value / price)


def return_volatility(df: pd.DataFrame, window: int = 20, price_col: str = "Close") -> float:
    """Standard deviation of daily percentage returns over the trailing
    `window` days. Already a ratio-based measure, no further normalization
    needed."""
    if len(df) < window + 1:
        return float("nan")
    returns = df[price_col].pct_change()
    return float(returns.iloc[-window:].std())


def bollinger_width(df: pd.DataFrame, window: int = 20, num_std: float = 2.0, price_col: str = "Close") -> float:
    """Bollinger Band width relative to the middle band: (upper - lower) /
    middle. A ratio, directly comparable across tickers; a narrow width
    signals a volatility squeeze that often precedes a breakout."""
    if len(df) < window:
        return float("nan")
    mid, upper, lower = _bollinger_bands_series(df[price_col], window, num_std)
    mid_value = mid.iloc[-1]
    if pd.isna(mid_value) or mid_value == 0:
        return float("nan")
    return float((upper.iloc[-1] - lower.iloc[-1]) / mid_value)


def days_since_high(df: pd.DataFrame, window: int, high_col: str = "High") -> float:
    """Number of trading days between today and the day the trailing
    `window`-day high was set. 0 means today IS the high (a fresh
    breakout); larger values mean price is extending away from an older
    high (ties broken toward the earlier occurrence, i.e. the larger
    days-ago value)."""
    if len(df) < window:
        return float("nan")
    windowed = df[high_col].iloc[-window:]
    high_position = int(np.argmax(windowed.to_numpy()))
    days_ago = (len(windowed) - 1) - high_position
    return float(days_ago)


# ---- additions beyond the original list -----------------------------------
# These weren't in the original request but fill gaps a GBT would likely
# benefit from, particularly for judging "breakout quality" specifically
# (as opposed to just detecting that a breakout-shaped move happened).

def close_location_value(df: pd.DataFrame) -> float:
    """Where today's close falls within today's high-low range, scaled to
    [-1, 1]: +1 means it closed at the high, -1 means it closed at the low.
    A classic breakout-quality signal -- a breakout that closes near the
    day's high says something different than one that closes mid-range,
    even if the day's return is identical."""
    if len(df) < 1:
        return float("nan")
    high = df["High"].iloc[-1]
    low = df["Low"].iloc[-1]
    close = df["Close"].iloc[-1]
    day_range = high - low
    if day_range == 0 or pd.isna(day_range):
        return 0.0
    return float(((close - low) - (high - close)) / day_range)


def high_low_range_pct(df: pd.DataFrame) -> float:
    """Today's (high - low) / close -- intraday range as a percentage of
    price. Distinguishes a wide-ranging breakout day from a quiet one."""
    if len(df) < 1:
        return float("nan")
    high = df["High"].iloc[-1]
    low = df["Low"].iloc[-1]
    close = df["Close"].iloc[-1]
    if close == 0 or pd.isna(close):
        return float("nan")
    return float((high - low) / close)


def close_to_open_gap_pct(df: pd.DataFrame) -> float:
    """Today's open relative to yesterday's close, as a percentage --
    captures overnight gap behavior, which is common around real,
    news-driven breakouts as opposed to slow intraday grinds."""
    if len(df) < 2:
        return float("nan")
    prev_close = df["Close"].iloc[-2]
    today_open = df["Open"].iloc[-1]
    if prev_close == 0 or pd.isna(prev_close):
        return float("nan")
    return float((today_open - prev_close) / prev_close)


def consecutive_up_days(df: pd.DataFrame, price_col: str = "Close", max_lookback: int = 60) -> float:
    """Length of the current consecutive streak (ending today) of positive
    daily returns -- straightforward momentum persistence, complementary to
    the RSI/MACD streak features."""
    if len(df) < 2:
        return float("nan")
    returns = df[price_col].pct_change().iloc[-max_lookback:]
    return _trailing_streak(returns.tolist(), lambda v: v > 0)


def volume_trend_slope(df: pd.DataFrame, window: int = 20, volume_col: str = "Volume") -> float:
    """Linear trend of raw volume over the trailing `window` days,
    normalized by the window's average volume so it's comparable across
    tickers with very different typical volume levels. Distinct from
    volume_ratio_short_long: this captures the trend's steadiness/shape
    over the whole window rather than a two-point comparison."""
    if len(df) < window:
        return float("nan")
    volume = df[volume_col].iloc[-window:]
    avg_volume = volume.mean()
    slope = _linreg_slope(volume)
    if avg_volume == 0 or pd.isna(avg_volume) or pd.isna(slope):
        return float("nan")
    return float(slope / avg_volume)


# ============================================================================
# Market and sector context
# ============================================================================
#
# Note: there's no separate "spy_20_day_return" function -- n_day_return()
# above already handles this by simply being called against the
# benchmark's own DataFrame (n_day_return(spy_df, 20)). FeatureEngine is
# responsible for fetching the configured benchmark ticker(s) via YFProxy
# and passing their DataFrames in; this module only knows about DataFrames,
# never tickers.

def relative_return_vs_benchmark(
    df: pd.DataFrame, benchmark_df: pd.DataFrame, window: int, price_col: str = "Close"
) -> float:
    """Stock's n-day return minus the benchmark's n-day return over the
    same window -- excess return vs. the market or sector.

    Uses a difference rather than a ratio deliberately: with returns that
    can be negative or near zero, stock_return / benchmark_return is
    unstable (sign flips when either return crosses zero, blows up as the
    benchmark return approaches zero) and doesn't correspond to a
    meaningful "excess return" concept the way the difference does.
    """
    stock_return = n_day_return(df, window, price_col)
    benchmark_return = n_day_return(benchmark_df, window, price_col)
    if pd.isna(stock_return) or pd.isna(benchmark_return):
        return float("nan")
    return float(stock_return - benchmark_return)
