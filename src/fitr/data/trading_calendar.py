"""Trading-calendar helpers.

Rather than pulling in a separate market-holiday-calendar dependency, the
calendar is derived from a reference ticker's own price history: a liquid,
always-trading stock (AAPL by default) only has rows on days the market was
actually open, so its returned date index *is* the trading calendar. YFProxy
builds this once at startup by querying the reference ticker and hands the
resulting DatetimeIndex to the functions below.

Kept as plain functions (not a class) operating on a passed-in calendar so
they're trivial to unit test with a small hand-built DatetimeIndex -- no
network, no YFProxy instance required.
"""
from __future__ import annotations

import pandas as pd

DateLike = str | pd.Timestamp


def is_trading_day(calendar: pd.DatetimeIndex, check_date: DateLike) -> bool:
    return pd.Timestamp(check_date).normalize() in calendar


def previous_trading_day(calendar: pd.DatetimeIndex, check_date: DateLike) -> pd.Timestamp:
    """Return check_date itself if it's a trading day, otherwise the most
    recent prior trading day present in the calendar.

    Raises ValueError if the calendar has no sessions on or before
    check_date at all (e.g. check_date is before the calendar's coverage).
    """
    ts = pd.Timestamp(check_date).normalize()
    prior = calendar[calendar <= ts]
    if len(prior) == 0:
        raise ValueError(
            f"No trading days on or before {ts.date()} in calendar "
            f"(calendar starts {calendar.min().date()})."
        )
    return prior.max()


def trading_days_between(calendar: pd.DatetimeIndex, start: DateLike, end: DateLike) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    mask = (calendar >= start_ts) & (calendar <= end_ts)
    return list(calendar[mask])
