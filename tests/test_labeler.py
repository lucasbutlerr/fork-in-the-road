"""Tests for OutcomeLabeler. Uses hand-placed price paths (not random
synthetic data) so every trigger day/price is known exactly -- same
philosophy as test_indicators.py's hand-built DataFrames, just going
through YFProxy via monkeypatched yfinance.download instead of being
passed directly."""
from __future__ import annotations

import pandas as pd
import pytest

from fitr.config_schemas.labeling_schema import LabelingConfig
from fitr.data import yf_proxy as yfp
from fitr.data.yf_proxy import YFProxy
from fitr.labeling.labeler import FAILURE, INCONCLUSIVE, SUCCESS, OutcomeLabeler

BUSINESS_DAYS = pd.bdate_range("2020-01-01", "2024-12-31")
CANDIDATE = pd.Timestamp("2024-01-08")  # a Monday
_candidate_idx = BUSINESS_DAYS.get_loc(CANDIDATE)
HORIZON_DATES = BUSINESS_DAYS[_candidate_idx + 1 : _candidate_idx + 6]  # next 5 trading days


def _flat_row(price: float) -> tuple:
    return (price, price + 1, price - 1, price)


def _build_series(overrides: dict) -> pd.DataFrame:
    """overrides: {date: (Open, High, Low, Close)} for specific days. The
    candidate day is always a flat close=100 entry; every other day
    defaults to a boring flat 100 (calendar padding)."""
    rows = {}
    for d in BUSINESS_DAYS:
        if d == CANDIDATE:
            rows[d] = (100, 101, 99, 100)
        elif d in overrides:
            rows[d] = overrides[d]
        else:
            rows[d] = _flat_row(100)
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["Open", "High", "Low", "Close"])
    df.index.name = "Date"
    df["Volume"] = 1_000_000.0
    return df


@pytest.fixture
def price_paths():
    """Maps ticker -> full price DataFrame. Tests populate this per-case;
    any ticker not present (including AAPL, used internally by YFProxy for
    its trading calendar) falls back to a boring flat series."""
    return {}


@pytest.fixture
def mock_download(monkeypatch, price_paths):
    def fake_download(ticker, start=None, end=None, interval="1d", progress=False, auto_adjust=True):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        full = price_paths[ticker] if ticker in price_paths else _build_series({})
        return full.loc[(full.index >= start_ts) & (full.index < end_ts)]

    monkeypatch.setattr(yfp.yf, "download", fake_download)
    return fake_download


@pytest.fixture
def proxy(tmp_path, mock_download):
    return YFProxy(cache_dir=tmp_path / "cache", reference_ticker="AAPL", calendar_lookback_years=5)


FIRST_TOUCH = LabelingConfig.model_validate(
    {"horizon_trading_days": 5, "target_return": 0.08, "stop_loss_return": -0.04, "evaluation_method": "first_touch"}
)
END_OF_HORIZON = LabelingConfig.model_validate(
    {"horizon_trading_days": 5, "target_return": 0.08, "stop_loss_return": -0.04, "evaluation_method": "end_of_horizon"}
)


def test_target_hit_before_stop_is_success(proxy, price_paths):
    price_paths["TARGET_HIT"] = _build_series(
        {
            HORIZON_DATES[0]: (100, 102, 99, 101),
            HORIZON_DATES[1]: (101, 104, 100, 103),
            HORIZON_DATES[2]: (103, 109, 102, 108),  # High=109 >= target 108
            HORIZON_DATES[3]: (108, 110, 107, 109),
            HORIZON_DATES[4]: (109, 111, 108, 110),
        }
    )
    result = OutcomeLabeler(proxy, FIRST_TOUCH).label("TARGET_HIT", CANDIDATE)
    assert result.outcome == SUCCESS
    assert result.trigger_reason == "target_hit"
    assert result.trigger_date == HORIZON_DATES[2]
    assert result.trading_days_held == 3  # target hit on the 3rd trading day of the horizon


def test_stop_hit_before_target_is_failure(proxy, price_paths):
    price_paths["STOP_HIT"] = _build_series(
        {
            HORIZON_DATES[0]: (100, 101, 98, 99),
            HORIZON_DATES[1]: (99, 100, 95, 96),  # Low=95 <= stop 96
            HORIZON_DATES[2]: (96, 97, 94, 95),
        }
    )
    result = OutcomeLabeler(proxy, FIRST_TOUCH).label("STOP_HIT", CANDIDATE)
    assert result.outcome == FAILURE
    assert result.trigger_reason == "stop_hit"
    assert result.trigger_date == HORIZON_DATES[1]
    assert result.trading_days_held == 2  # stop hit on the 2nd trading day of the horizon


def test_neither_hit_within_horizon_is_failure(proxy, price_paths):
    price_paths["FLAT"] = _build_series(
        {
            HORIZON_DATES[0]: (100, 102, 99, 101),
            HORIZON_DATES[1]: (101, 103, 100, 102),
            HORIZON_DATES[2]: (102, 103, 100, 101),
            HORIZON_DATES[3]: (101, 102, 99, 100),
            HORIZON_DATES[4]: (100, 102, 99, 101),
        }
    )
    result = OutcomeLabeler(proxy, FIRST_TOUCH).label("FLAT", CANDIDATE)
    assert result.outcome == FAILURE
    assert result.trigger_reason == "horizon_expired_flat"
    assert result.trading_days_held == len(HORIZON_DATES)
    # Important, non-buggy quirk: a horizon-expired FAILURE can still carry
    # a positive forward_return (price drifted up without ever touching
    # target or stop) -- this is why backtest win-rate and model
    # success-precision can diverge.
    assert result.forward_return > 0


def test_same_day_ambiguous_touch_is_conservatively_a_failure(proxy, price_paths):
    price_paths["BOTH_SAME_DAY"] = _build_series(
        {HORIZON_DATES[0]: (100, 109, 95, 102)}  # High crosses target AND Low crosses stop, same day
    )
    result = OutcomeLabeler(proxy, FIRST_TOUCH).label("BOTH_SAME_DAY", CANDIDATE)
    assert result.outcome == FAILURE
    assert result.trigger_reason == "stop_hit_same_day_as_target_ambiguous"


def test_too_recent_candidate_is_inconclusive(proxy, price_paths):
    late_candidate = BUSINESS_DAYS[-3]  # only 2 trading days remain after this in the fixture's calendar
    result = OutcomeLabeler(proxy, FIRST_TOUCH).label("ANY", late_candidate)
    assert result.outcome == INCONCLUSIVE
    assert result.trigger_reason == "insufficient_forward_data"
    assert result.trading_days_available == 2


def test_no_data_for_ticker_is_inconclusive(proxy, monkeypatch):
    def empty_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(yfp.yf, "download", empty_download)
    result = OutcomeLabeler(proxy, FIRST_TOUCH).label("GHOST", CANDIDATE)
    assert result.outcome == INCONCLUSIVE
    assert result.trigger_reason == "no_data"


def test_end_of_horizon_ignores_intra_horizon_stop_touch(proxy, price_paths):
    price_paths["DIP_THEN_RECOVER"] = _build_series(
        {
            HORIZON_DATES[0]: (100, 101, 90, 91),  # Low=90 well past the stop -- end_of_horizon ignores this
            HORIZON_DATES[1]: (91, 100, 90, 95),
            HORIZON_DATES[2]: (95, 105, 94, 104),
            HORIZON_DATES[3]: (104, 109, 103, 108),
            HORIZON_DATES[4]: (108, 111, 107, 109),  # final close 109 -> +9%, clears the 8% target
        }
    )
    result = OutcomeLabeler(proxy, END_OF_HORIZON).label("DIP_THEN_RECOVER", CANDIDATE)
    assert result.outcome == SUCCESS
    assert result.trigger_reason == "horizon_end"


def test_first_touch_fails_the_same_path_end_of_horizon_calls_a_success(proxy, price_paths):
    # Same DIP_THEN_RECOVER path as above -- confirms the two evaluation
    # methods can genuinely disagree on the identical price history,
    # which is the whole point of end_of_horizon ignoring intra-horizon
    # touches.
    price_paths["DIP_THEN_RECOVER"] = _build_series(
        {
            HORIZON_DATES[0]: (100, 101, 90, 91),
            HORIZON_DATES[1]: (91, 100, 90, 95),
            HORIZON_DATES[2]: (95, 105, 94, 104),
            HORIZON_DATES[3]: (104, 109, 103, 108),
            HORIZON_DATES[4]: (108, 111, 107, 109),
        }
    )
    result = OutcomeLabeler(proxy, FIRST_TOUCH).label("DIP_THEN_RECOVER", CANDIDATE)
    assert result.outcome == FAILURE
    assert result.trigger_reason == "stop_hit"


def test_weekend_candidate_date_resolves_to_prior_trading_day(proxy, price_paths):
    price_paths["TARGET_HIT"] = _build_series(
        {
            HORIZON_DATES[0]: (100, 102, 99, 101),
            HORIZON_DATES[1]: (101, 104, 100, 103),
            HORIZON_DATES[2]: (103, 109, 102, 108),
            HORIZON_DATES[3]: (108, 110, 107, 109),
            HORIZON_DATES[4]: (109, 111, 108, 110),
        }
    )
    labeler = OutcomeLabeler(proxy, FIRST_TOUCH)
    result_weekday = labeler.label("TARGET_HIT", "2024-01-08")
    result_weekend = labeler.label("TARGET_HIT", "2024-01-07")  # the Sunday before
    assert result_weekday.outcome == result_weekend.outcome
    assert result_weekday.trigger_date == result_weekend.trigger_date
