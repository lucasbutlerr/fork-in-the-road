"""Tests for PositionSizer. Pure calculation, no I/O -- every value here is
hand-computed and asserted exactly, same philosophy as test_indicators.py."""
from __future__ import annotations

import pytest

from fitr.config_schemas.position_sizing_schema import PositionSizingConfig
from fitr.modeling.position_sizer import PositionSizer


def make_config(**overrides):
    base = {
        "method": "fixed_fractional",
        "risk_fraction": 0.01,
        "kelly_multiplier": 0.25,
        "target_return": 0.08,
        "stop_loss_return": -0.04,
        "max_position_fraction": 1.0,  # uncapped by default in these tests, unless a test says otherwise
        "min_position_fraction": 0.0,
    }
    base.update(overrides)
    return PositionSizingConfig.model_validate(base)


def test_fixed_fractional_matches_hand_calc():
    # risk 1% of equity, 4% stop -> position = 1%/4% = 25% of equity
    config = make_config(method="fixed_fractional", risk_fraction=0.01, stop_loss_return=-0.04)
    result = PositionSizer(config).size("AAPL", 0.5, entry_price=20.0, equity=100_000)
    assert result.raw_fraction == pytest.approx(0.25)
    assert result.recommended_shares == 1250  # 25,000 / 20.00
    assert result.position_value == pytest.approx(25_000.0)


def test_fixed_fractional_ignores_predicted_probability():
    config = make_config(method="fixed_fractional")
    low = PositionSizer(config).size("AAPL", 0.05, entry_price=20.0, equity=100_000)
    high = PositionSizer(config).size("AAPL", 0.95, entry_price=20.0, equity=100_000)
    assert low.raw_fraction == high.raw_fraction
    assert low.kelly_fraction is None


def test_kelly_fraction_matches_hand_calc():
    # f* = p/S - q/T where T=target_return, S=|stop_loss_return|.
    # p=0.5, q=0.5, T=0.08, S=0.04 -> f* = 0.5/0.04 - 0.5/0.08 = 12.5 - 6.25 = 6.25
    # (NOT the textbook all-or-nothing formula (bp-q)/b, which assumes a
    # loss costs the entire wager -- this project's trades lose only
    # |stop_loss_return| of the position, a materially different payoff
    # structure requiring a re-derived formula. See position_sizer.py's
    # _kelly_fraction docstring for the full derivation.)
    config = make_config(method="fractional_kelly", kelly_multiplier=0.25, target_return=0.08, stop_loss_return=-0.04, max_position_fraction=1.0)
    result = PositionSizer(config).size("AAPL", 0.5, entry_price=20.0, equity=100_000)
    assert result.kelly_fraction == pytest.approx(6.25)
    assert result.raw_fraction == pytest.approx(6.25 * 0.25)  # kelly * multiplier
    assert result.recommended_fraction == pytest.approx(1.0)  # capped -- 156% is still way over the 100% cap


def test_kelly_zero_edge_exactly_at_breakeven_probability():
    # This should land exactly on the same breakeven precision formula
    # used elsewhere in the project: p* = |stop| / (target + |stop|).
    target, stop = 0.08, -0.04
    breakeven_p = abs(stop) / (target + abs(stop))
    config = make_config(method="fractional_kelly", target_return=target, stop_loss_return=stop)
    result = PositionSizer(config).size("AAPL", breakeven_p, entry_price=20.0, equity=100_000)
    assert result.kelly_fraction == pytest.approx(0.0, abs=1e-9)


def test_kelly_below_breakeven_probability_floors_at_zero():
    config = make_config(method="fractional_kelly", target_return=0.08, stop_loss_return=-0.04)
    result = PositionSizer(config).size("AAPL", 0.2, entry_price=20.0, equity=100_000)
    assert result.kelly_fraction < 0  # the raw (unfloored) Kelly value is genuinely negative
    assert result.raw_fraction == 0.0  # but never recommends a short
    assert result.recommended_shares == 0


def test_kelly_above_breakeven_probability_is_positive():
    config = make_config(method="fractional_kelly", target_return=0.08, stop_loss_return=-0.04)
    result = PositionSizer(config).size("AAPL", 0.6, entry_price=20.0, equity=100_000)
    assert result.kelly_fraction > 0
    assert result.raw_fraction > 0


def test_max_position_fraction_caps_an_oversized_recommendation():
    config = make_config(method="fixed_fractional", risk_fraction=0.5, stop_loss_return=-0.01, max_position_fraction=0.1)
    result = PositionSizer(config).size("AAPL", 0.5, entry_price=20.0, equity=100_000)
    assert result.raw_fraction == pytest.approx(50.0)  # what the formula alone would say -- absurd, needs capping
    assert result.capped is True
    assert result.recommended_fraction == pytest.approx(0.1)


def test_min_position_fraction_rounds_small_recommendations_to_zero():
    config = make_config(method="fixed_fractional", risk_fraction=0.001, stop_loss_return=-0.04, min_position_fraction=0.05)
    result = PositionSizer(config).size("AAPL", 0.5, entry_price=20.0, equity=100_000)
    assert result.raw_fraction == pytest.approx(0.025)  # below the 0.05 floor
    assert result.capped is True
    assert result.recommended_fraction == 0.0
    assert result.recommended_shares == 0


def test_no_capping_when_recommendation_is_within_guardrails():
    config = make_config(method="fixed_fractional", risk_fraction=0.01, stop_loss_return=-0.04, max_position_fraction=0.5)
    result = PositionSizer(config).size("AAPL", 0.5, entry_price=20.0, equity=100_000)
    assert result.capped is False
    assert result.recommended_fraction == result.raw_fraction


def test_shares_are_floored_to_whole_units():
    config = make_config(method="fixed_fractional", risk_fraction=0.01, stop_loss_return=-0.04)
    # 25% of 100,000 = 25,000; at 23.37/share that's 1069.7 shares -> floors to 1069
    result = PositionSizer(config).size("AAPL", 0.5, entry_price=23.37, equity=100_000)
    assert result.recommended_shares == 1069
    assert result.position_value == pytest.approx(1069 * 23.37)


def test_zero_entry_price_does_not_crash():
    config = make_config(method="fixed_fractional")
    result = PositionSizer(config).size("AAPL", 0.5, entry_price=0.0, equity=100_000)
    assert result.recommended_shares == 0
