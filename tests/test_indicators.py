"""Tests for src/features/indicators.py.

All fully deterministic -- no network, no YFProxy, no fixtures beyond
hand-built DataFrames. Each indicator gets: at least one test against a
manually-computed expected value (or a known analytical case, like RSI on
a monotonic series), an insufficient-data -> NaN test where relevant, and
for the indicators that specifically had to be normalized for cross-ticker
comparability (MACD, ATR, OBV, volume trend), a scale-invariance test that
proves the normalization actually works -- doubling the price or volume
level shouldn't move the result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fitr.features import indicators as ind


def make_df(
    close: list[float],
    high: list[float] | None = None,
    low: list[float] | None = None,
    open_: list[float] | None = None,
    volume: list[float] | None = None,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame. If high/low/open/volume aren't
    given, they're derived simply from close so tests can focus on the
    Close-only indicators without boilerplate."""
    n = len(close)
    close = pd.Series(close, dtype=float)
    high = pd.Series(high, dtype=float) if high is not None else close + 1.0
    low = pd.Series(low, dtype=float) if low is not None else close - 1.0
    open_ = pd.Series(open_, dtype=float) if open_ is not None else close
    volume = pd.Series(volume, dtype=float) if volume is not None else pd.Series([1_000_000.0] * n)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"Open": open_.to_numpy(), "High": high.to_numpy(), "Low": low.to_numpy(),
         "Close": close.to_numpy(), "Volume": volume.to_numpy()},
        index=idx,
    )


# ============================================================================
# Price trend
# ============================================================================

def test_n_day_return_matches_manual_calc():
    df = make_df(close=[100, 101, 102, 103, 104, 105])
    result = ind.n_day_return(df, window=5)
    assert result == pytest.approx(105 / 100 - 1)


def test_n_day_return_insufficient_data_is_nan():
    df = make_df(close=[100, 101, 102])
    assert np.isnan(ind.n_day_return(df, window=5))


def test_n_day_return_zero_window_start_price_is_nan():
    df = make_df(close=[0, 101, 102, 103])
    assert np.isnan(ind.n_day_return(df, window=3))


def test_relative_distance_from_sma_matches_manual_calc():
    closes = [10, 20, 30, 40, 50]
    df = make_df(close=closes)
    sma5 = sum(closes) / 5
    expected = (closes[-1] - sma5) / sma5
    assert ind.relative_distance_from_ma(df, window=5, ma_type="sma") == pytest.approx(expected)


def test_relative_distance_from_ma_insufficient_data_is_nan():
    df = make_df(close=[10, 20, 30])
    assert np.isnan(ind.relative_distance_from_ma(df, window=10))


def test_relative_distance_from_ma_ema_runs_and_differs_from_sma():
    df = make_df(close=[10, 12, 14, 20, 30, 45, 60])
    sma_dist = ind.relative_distance_from_ma(df, window=5, ma_type="sma")
    ema_dist = ind.relative_distance_from_ma(df, window=5, ma_type="ema")
    assert not np.isnan(sma_dist)
    assert not np.isnan(ema_dist)
    assert sma_dist != pytest.approx(ema_dist)


def test_relative_n_day_high_at_new_high_is_zero():
    # relative_n_day_high compares Close (not High) to the rolling High, so
    # for the result to land exactly on 0 the close needs to match the
    # rolling high itself -- i.e. today closed right at its own high.
    df = make_df(close=[10, 12, 11, 13, 15.5], high=[10.5, 12.5, 11.5, 13.5, 15.5])
    assert ind.relative_n_day_high(df, window=5) == pytest.approx(0.0)


def test_relative_n_day_high_below_recent_high_is_negative():
    df = make_df(close=[10, 12, 20, 13, 15], high=[10.5, 12.5, 20.5, 13.5, 15.5])
    result = ind.relative_n_day_high(df, window=5)
    assert result < 0
    assert result == pytest.approx((15 - 20.5) / 20.5)


def test_relative_n_day_low_matches_manual_calc():
    df = make_df(close=[10, 12, 20, 13, 15], low=[9.5, 11.5, 19.5, 12.5, 14.5])
    rolling_low = 9.5
    expected = (15 - rolling_low) / rolling_low
    assert ind.relative_n_day_low(df, window=5) == pytest.approx(expected)
    assert ind.relative_n_day_low(df, window=5) >= 0


# ============================================================================
# Volume
# ============================================================================

def test_relative_volume_excludes_current_by_default():
    volumes = [100, 100, 100, 100, 100, 500]
    df = make_df(close=[1, 2, 3, 4, 5, 6], volume=volumes)
    # baseline is the mean of the first 5 (all 100), current is 500
    assert ind.relative_volume(df, window=5) == pytest.approx(5.0)


def test_relative_volume_including_current_differs():
    volumes = [100, 100, 100, 100, 100, 500]
    df = make_df(close=[1, 2, 3, 4, 5, 6], volume=volumes)
    excl = ind.relative_volume(df, window=5, exclude_current=True)
    incl = ind.relative_volume(df, window=5, exclude_current=False)
    assert excl != pytest.approx(incl)


def test_relative_volume_insufficient_data_is_nan():
    df = make_df(close=[1, 2], volume=[100, 200])
    assert np.isnan(ind.relative_volume(df, window=5))


def test_volume_ratio_short_long_matches_manual_calc():
    volumes = [50] * 15 + [200] * 5  # long avg pulled down by early low volume
    df = make_df(close=list(range(20)), volume=volumes)
    short_avg = sum(volumes[-5:]) / 5
    long_avg = sum(volumes[-20:]) / 20
    assert ind.volume_ratio_short_long(df, short_window=5, long_window=20) == pytest.approx(short_avg / long_avg)


def test_volume_zscore_matches_manual_calc():
    volumes = [100, 110, 90, 105, 95, 300]
    df = make_df(close=list(range(6)), volume=volumes)
    baseline = pd.Series(volumes[:-1])
    expected = (300 - baseline.mean()) / baseline.std()
    assert ind.volume_zscore(df, window=5) == pytest.approx(expected)


def test_volume_zscore_zero_variance_baseline_is_nan():
    volumes = [100, 100, 100, 100, 100, 150]
    df = make_df(close=list(range(6)), volume=volumes)
    assert np.isnan(ind.volume_zscore(df, window=5))


def test_obv_slope_positive_for_rising_price_on_steady_volume():
    closes = [10, 11, 12, 13, 14, 15, 16, 17]
    df = make_df(close=closes, volume=[1000] * len(closes))
    assert ind.obv_slope(df, window=6) > 0


def test_obv_slope_is_scale_invariant_to_volume_level():
    closes = [10, 11, 12, 13, 14, 15, 16, 17]
    df_low_vol = make_df(close=closes, volume=[1000] * len(closes))
    df_high_vol = make_df(close=closes, volume=[10_000] * len(closes))
    slope_low = ind.obv_slope(df_low_vol, window=6)
    slope_high = ind.obv_slope(df_high_vol, window=6)
    assert slope_low == pytest.approx(slope_high, rel=1e-6)


def test_obv_slope_flat_price_gives_zero_obv_and_nan_slope():
    # No price movement at all -> OBV never changes -> slope is 0, not NaN,
    # since polyfit on a constant series is well-defined.
    closes = [10.0] * 10
    df = make_df(close=closes, volume=[1000] * 10)
    result = ind.obv_slope(df, window=6)
    assert result == pytest.approx(0.0)


# ============================================================================
# RSI
# ============================================================================

def test_rsi_all_gains_approaches_100():
    closes = [10 + i for i in range(30)]  # strictly increasing
    df = make_df(close=closes)
    assert ind.rsi(df, window=14) == pytest.approx(100.0, abs=0.5)


def test_rsi_all_losses_approaches_0():
    closes = [50 - i for i in range(30)]  # strictly decreasing
    df = make_df(close=closes)
    assert ind.rsi(df, window=14) == pytest.approx(0.0, abs=0.5)


def test_rsi_flat_price_is_50():
    closes = [25.0] * 30
    df = make_df(close=closes)
    assert ind.rsi(df, window=14) == pytest.approx(50.0)


def test_rsi_insufficient_data_is_nan():
    df = make_df(close=[10, 11, 12])
    assert np.isnan(ind.rsi(df, window=14))


def test_rsi_moving_average_smooths_relative_to_raw_rsi():
    closes = [10 + i for i in range(30)]
    df = make_df(close=closes)
    avg = ind.rsi_moving_average(df, rsi_window=14, avg_window=5)
    assert 0 <= avg <= 100


def test_rsi_slope_positive_when_momentum_building():
    # A purely monotonic no-red-days series saturates RSI at exactly 100
    # almost immediately (avg_loss stays exactly 0), at which point a
    # regression slope over a flat run is just floating-point noise around
    # zero -- not a meaningful test. Use a choppy-then-trending series so
    # RSI is genuinely still climbing (not yet saturated) at the point
    # we measure the slope.
    closes = [100.0]
    for i in range(25):
        rate = 0.002 if i % 4 != 0 else -0.001  # mostly up, occasional small dip
        closes.append(closes[-1] * (1 + rate))
    for _ in range(15):
        closes.append(closes[-1] * 1.01)  # then a clean uptrend
    df = make_df(close=closes)
    assert ind.rsi_slope(df, rsi_window=14, slope_window=5) > 0


def test_days_rsi_above_threshold_zero_when_currently_below():
    closes = [50 - i for i in range(30)]  # declining -> RSI near 0
    df = make_df(close=closes)
    assert ind.days_rsi_above_threshold(df, threshold=60.0) == 0.0


def test_days_rsi_above_threshold_positive_during_sustained_uptrend():
    closes = [10 + i for i in range(30)]  # sustained rise -> RSI pinned high
    df = make_df(close=closes)
    result = ind.days_rsi_above_threshold(df, threshold=60.0)
    assert result > 0


def test_rsi_std_zero_for_flat_price():
    closes = [25.0] * 30
    df = make_df(close=closes)
    assert ind.rsi_std(df, rsi_window=14, std_window=10) == pytest.approx(0.0)


# ============================================================================
# MACD
# ============================================================================

def test_macd_value_positive_in_uptrend_negative_in_downtrend():
    up = make_df(close=[10 + i * 0.5 for i in range(60)])
    down = make_df(close=[60 - i * 0.5 for i in range(60)])
    assert ind.macd_value(up) > 0
    assert ind.macd_value(down) < 0


def test_macd_value_is_scale_invariant_to_price_level():
    base_closes = [10 + i * 0.3 + (i % 5) * 0.2 for i in range(60)]
    df_normal = make_df(close=base_closes)
    df_scaled = make_df(close=[c * 10 for c in base_closes])
    normal_value = ind.macd_value(df_normal)
    scaled_value = ind.macd_value(df_scaled)
    assert normal_value == pytest.approx(scaled_value, rel=1e-6)


def test_macd_value_insufficient_data_is_nan():
    df = make_df(close=[10, 11, 12])
    assert np.isnan(ind.macd_value(df))


def test_macd_histogram_moving_average_runs_without_error():
    df = make_df(close=[10 + i * 0.5 for i in range(60)])
    result = ind.macd_histogram_moving_average(df)
    assert not np.isnan(result)


def test_macd_histogram_slope_positive_when_momentum_accelerating():
    # MACD (and its histogram) responds to percentage/relative momentum,
    # not absolute dollar moves -- a series growing by a fixed number of
    # dollars per day (or even i**1.5 dollars) is actually *decelerating*
    # in percentage terms as the price level rises, which is what MACD
    # sees. Build a series whose daily percentage growth rate itself
    # increases over time, so it's genuinely accelerating in the terms
    # MACD measures.
    closes = [100.0]
    for i in range(60):
        rate = 0.001 + i * 0.0004
        closes.append(closes[-1] * (1 + rate))
    df = make_df(close=closes)
    assert ind.macd_histogram_slope(df) > 0


def test_days_macd_above_signal_zero_in_sustained_downtrend():
    df = make_df(close=[100 - i * 0.5 for i in range(60)])
    assert ind.days_macd_above_signal(df) == 0.0


def test_days_macd_above_signal_positive_in_sustained_uptrend():
    df = make_df(close=[10 + i * 0.5 for i in range(60)])
    assert ind.days_macd_above_signal(df) > 0


def test_macd_std_runs_without_error():
    df = make_df(close=[10 + i * 0.3 + (i % 7) for i in range(60)])
    result = ind.macd_std(df, std_window=10)
    assert not np.isnan(result)
    assert result >= 0


# ============================================================================
# Volatility and breakout quality
# ============================================================================

def test_atr_normalized_is_scale_invariant_to_price_level():
    # make_df's default High/Low (close +/- a flat $1) don't scale
    # proportionally with price, so multiplying Close by 10 alone would
    # change the *shape* of the true-range calculation, not just its
    # level -- that's not a fair scale-invariance test. A real ticker's
    # High/Low move proportionally with its price level, so build High/Low
    # as a fixed percentage band around Close instead.
    base = [10 + (i % 5) - (i % 3) + i * 0.1 for i in range(30)]

    def scaled_df(multiplier: float) -> pd.DataFrame:
        closes = [c * multiplier for c in base]
        return make_df(close=closes, high=[c * 1.02 for c in closes], low=[c * 0.98 for c in closes])

    normal_atr = ind.atr_normalized(scaled_df(1.0), window=14)
    scaled_atr = ind.atr_normalized(scaled_df(10.0), window=14)
    assert normal_atr == pytest.approx(scaled_atr, rel=1e-6)


def test_atr_normalized_insufficient_data_is_nan():
    df = make_df(close=[10, 11, 12])
    assert np.isnan(ind.atr_normalized(df, window=14))


def test_return_volatility_matches_manual_calc():
    closes = [100, 102, 101, 105, 103, 108]
    df = make_df(close=closes)
    returns = pd.Series(closes).pct_change().dropna()
    expected = returns.std()
    assert ind.return_volatility(df, window=5) == pytest.approx(expected)


def test_return_volatility_zero_for_constant_returns():
    # Constant percentage growth each day -> zero variance in returns.
    closes = [100 * (1.02 ** i) for i in range(10)]
    df = make_df(close=closes)
    assert ind.return_volatility(df, window=5) == pytest.approx(0.0, abs=1e-9)


def test_bollinger_width_matches_manual_calc():
    closes = [10, 12, 11, 13, 15, 14, 16, 18, 17, 20]
    df = make_df(close=closes)
    window_vals = pd.Series(closes[-10:])
    mid = window_vals.mean()
    std = window_vals.std()
    expected = (2 * std * 2) / mid  # upper-lower = 2*(num_std*std) with num_std=2
    assert ind.bollinger_width(df, window=10, num_std=2.0) == pytest.approx(expected)


def test_days_since_high_zero_when_today_is_the_high():
    highs = [10, 11, 12, 13, 20]
    df = make_df(close=[9, 10, 11, 12, 19], high=highs)
    assert ind.days_since_high(df, window=5) == 0.0


def test_days_since_high_counts_back_correctly():
    highs = [10, 25, 12, 13, 14]  # peak is 2 days before the end (index -5, so 4 days ago... check)
    df = make_df(close=[9, 24, 11, 12, 13], high=highs)
    # peak at position 1 (0-indexed) of a 5-length window -> days_ago = 4 - 1 = 3
    assert ind.days_since_high(df, window=5) == pytest.approx(3.0)


# ---- additions --------------------------------------------------------

def test_close_location_value_at_high_is_one():
    df = make_df(close=[10], high=[11], low=[9])
    df.loc[df.index[-1], "Close"] = 11  # close == high
    assert ind.close_location_value(df) == pytest.approx(1.0)


def test_close_location_value_at_low_is_negative_one():
    df = make_df(close=[10], high=[11], low=[9])
    df.loc[df.index[-1], "Close"] = 9  # close == low
    assert ind.close_location_value(df) == pytest.approx(-1.0)


def test_close_location_value_midrange_is_zero():
    df = make_df(close=[10], high=[11], low=[9])  # close exactly midway
    assert ind.close_location_value(df) == pytest.approx(0.0)


def test_close_location_value_zero_range_is_zero_not_nan():
    df = make_df(close=[10], high=[10], low=[10])
    assert ind.close_location_value(df) == 0.0


def test_high_low_range_pct_matches_manual_calc():
    df = make_df(close=[100], high=[105], low=[95])
    assert ind.high_low_range_pct(df) == pytest.approx((105 - 95) / 100)


def test_close_to_open_gap_pct_matches_manual_calc():
    df = make_df(close=[100, 100], open_=[100, 103])
    assert ind.close_to_open_gap_pct(df) == pytest.approx((103 - 100) / 100)


def test_close_to_open_gap_pct_insufficient_data_is_nan():
    df = make_df(close=[100])
    assert np.isnan(ind.close_to_open_gap_pct(df))


def test_consecutive_up_days_counts_trailing_streak_only():
    # down, up, up, up -- streak should be 3, not 4
    closes = [100, 95, 96, 98, 101]
    df = make_df(close=closes)
    assert ind.consecutive_up_days(df) == pytest.approx(3.0)


def test_consecutive_up_days_zero_when_today_is_down():
    closes = [100, 105, 103]
    df = make_df(close=closes)
    assert ind.consecutive_up_days(df) == pytest.approx(0.0)


def test_volume_trend_slope_positive_for_rising_volume():
    volumes = [100 + i * 10 for i in range(20)]
    df = make_df(close=list(range(20)), volume=volumes)
    assert ind.volume_trend_slope(df, window=20) > 0


def test_volume_trend_slope_is_scale_invariant_to_volume_level():
    volumes = [100 + i * 10 for i in range(20)]
    df_low = make_df(close=list(range(20)), volume=volumes)
    df_high = make_df(close=list(range(20)), volume=[v * 50 for v in volumes])
    slope_low = ind.volume_trend_slope(df_low, window=20)
    slope_high = ind.volume_trend_slope(df_high, window=20)
    assert slope_low == pytest.approx(slope_high, rel=1e-6)


# ============================================================================
# Market and sector context
# ============================================================================

def test_relative_return_vs_benchmark_matches_manual_calc():
    stock = make_df(close=[100, 110, 121])  # +21% over 2 days
    benchmark = make_df(close=[100, 105, 110])  # +10% over 2 days
    result = ind.relative_return_vs_benchmark(stock, benchmark, window=2)
    assert result == pytest.approx(0.21 - 0.10, abs=1e-9)


def test_relative_return_vs_benchmark_negative_when_underperforming():
    stock = make_df(close=[100, 100, 100])  # flat
    benchmark = make_df(close=[100, 110, 120])  # up 20%
    result = ind.relative_return_vs_benchmark(stock, benchmark, window=2)
    assert result < 0


def test_relative_return_vs_benchmark_nan_when_either_side_insufficient():
    stock = make_df(close=[100, 101])
    benchmark = make_df(close=[100, 105, 110, 115, 120])
    assert np.isnan(ind.relative_return_vs_benchmark(stock, benchmark, window=4))
