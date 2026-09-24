"""Tests for BacktestSimulator. Hand-computed equity curves, same
philosophy as test_position_sizer.py -- every number here is exactly
derivable by hand, not just plausible-looking."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fitr.config_schemas.position_sizing_schema import PositionSizingConfig
from fitr.modeling.backtest import BacktestSimulator
from fitr.modeling.position_sizer import PositionSizer


class FakeTrainedModel:
    def __init__(self, probs):
        self._probs = np.array(probs, dtype=float)
        self.label_column = "label"
        self.feature_columns = []

    def predict_proba_success(self, df):
        return self._probs


def make_sizer(**overrides):
    base = {
        "method": "fixed_fractional",
        "risk_fraction": 0.01,
        "target_return": 0.08,
        "stop_loss_return": -0.04,
        "max_position_fraction": 1.0,
        "min_position_fraction": 0.0,
    }
    base.update(overrides)
    return PositionSizer(PositionSizingConfig.model_validate(base))


def test_equity_curve_matches_hand_calc():
    # fixed_fractional, risk 1% / 4% stop -> 25% of equity per trade, every trade.
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "forward_return": [0.08, -0.04, 0.10],
        }
    )
    model = FakeTrainedModel(probs=[0.9, 0.9, 0.9])
    report = BacktestSimulator(make_sizer()).run(model, df, threshold=0.5, starting_equity=100_000)

    # 100000 * 1.02 * 0.99 * 1.025 = 103504.5
    assert report.ending_equity == pytest.approx(103_504.5)
    assert report.total_return == pytest.approx(0.035045)
    assert report.equity_curve == pytest.approx([100_000, 102_000, 100_980, 103_504.5])
    # test_df has no trading_days_held column -- backward-compat path,
    # report field is present but NaN rather than raising.
    assert report.average_holding_days != report.average_holding_days  # NaN


def test_average_holding_days_computed_from_trading_days_held_column():
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "forward_return": [0.08, -0.04, 0.10],
            "trading_days_held": [3, 9, 6],
        }
    )
    model = FakeTrainedModel(probs=[0.9, 0.9, 0.9])
    report = BacktestSimulator(make_sizer()).run(model, df, threshold=0.5, starting_equity=100_000)

    assert report.average_holding_days == pytest.approx(6.0)
    assert [t.trading_days_held for t in report.trades] == [3, 9, 6]


def test_win_loss_counts_and_averages():
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "forward_return": [0.08, -0.04, 0.10],
        }
    )
    model = FakeTrainedModel(probs=[0.9, 0.9, 0.9])
    report = BacktestSimulator(make_sizer()).run(model, df, threshold=0.5, starting_equity=100_000)

    assert report.win_count == 2
    assert report.loss_count == 1
    assert report.win_rate == pytest.approx(2 / 3)
    assert report.average_win == pytest.approx(0.09)  # mean(0.08, 0.10)
    assert report.average_loss == pytest.approx(-0.04)


def test_max_drawdown_matches_hand_calc():
    # Equity path 100000 -> 102000 -> 100980 -> 103504.5: the only
    # drawdown is from the 102000 peak down to 100980, a 1.0% decline.
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "forward_return": [0.08, -0.04, 0.10],
        }
    )
    model = FakeTrainedModel(probs=[0.9, 0.9, 0.9])
    report = BacktestSimulator(make_sizer()).run(model, df, threshold=0.5, starting_equity=100_000)
    assert report.max_drawdown == pytest.approx(0.01)


def test_candidates_below_threshold_are_not_traded():
    df = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "forward_return": [0.08, -0.04],
        }
    )
    model = FakeTrainedModel(probs=[0.9, 0.1])  # B is below threshold
    report = BacktestSimulator(make_sizer()).run(model, df, threshold=0.5, starting_equity=100_000)

    assert report.n_candidates == 2
    assert report.n_trades == 1
    assert report.trades[0].ticker == "A"


def test_trades_processed_in_chronological_order_not_input_order():
    df = pd.DataFrame(
        {
            "ticker": ["LATER", "EARLIER"],
            "date": pd.to_datetime(["2024-02-01", "2024-01-01"]),  # deliberately out of order
            "forward_return": [0.08, -0.04],
        }
    )
    model = FakeTrainedModel(probs=[0.9, 0.9])
    report = BacktestSimulator(make_sizer()).run(model, df, threshold=0.5, starting_equity=100_000)

    assert [t.ticker for t in report.trades] == ["EARLIER", "LATER"]


def test_missing_forward_return_column_raises():
    df = pd.DataFrame({"ticker": ["A"], "date": pd.to_datetime(["2024-01-01"])})
    model = FakeTrainedModel(probs=[0.9])
    with pytest.raises(ValueError, match="forward_return"):
        BacktestSimulator(make_sizer()).run(model, df)


def test_empty_test_set_raises():
    df = pd.DataFrame({"ticker": [], "date": pd.to_datetime([]), "forward_return": []})
    model = FakeTrainedModel(probs=[])
    with pytest.raises(ValueError):
        BacktestSimulator(make_sizer()).run(model, df)


def test_no_trades_taken_still_produces_a_valid_report():
    df = pd.DataFrame(
        {
            "ticker": ["A"],
            "date": pd.to_datetime(["2024-01-01"]),
            "forward_return": [0.08],
        }
    )
    model = FakeTrainedModel(probs=[0.01])  # below any reasonable threshold
    report = BacktestSimulator(make_sizer()).run(model, df, threshold=0.5, starting_equity=100_000)

    assert report.n_trades == 0
    assert report.ending_equity == 100_000
    assert report.total_return == 0.0
    assert report.win_rate != report.win_rate  # NaN -- no trades to compute a rate over


def test_fractional_kelly_sizing_used_when_configured():
    # Confirms BacktestSimulator actually delegates sizing to PositionSizer
    # (not hardcoding fixed_fractional) -- fraction should vary with
    # predicted probability under fractional_kelly, unlike fixed_fractional.
    # kelly_multiplier is deliberately tiny here so neither probability's
    # raw fraction hits max_position_fraction's cap -- otherwise both
    # would be capped to the same ceiling and look identical despite
    # genuinely different underlying Kelly fractions.
    kelly_sizer = make_sizer(method="fractional_kelly", kelly_multiplier=0.01, max_position_fraction=1.0)
    df = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "forward_return": [0.08, 0.08],
        }
    )
    model = FakeTrainedModel(probs=[0.9, 0.6])  # different confidence -> different Kelly fraction
    report = BacktestSimulator(kelly_sizer).run(model, df, threshold=0.5, starting_equity=100_000)
    assert report.trades[0].recommended_fraction != report.trades[1].recommended_fraction
