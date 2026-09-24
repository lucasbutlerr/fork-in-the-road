"""OutcomeLabeler: given (ticker, candidate_date), looks forward over the
configured horizon and decides whether the breakout succeeded, failed, or
is too recent to judge yet.

Two evaluation methods, set via labeling.yaml:
  first_touch    -- whichever of target_return / stop_loss_return is hit
                     first within the horizon wins. If price touches BOTH
                     levels on the same trading day (only possible to see
                     from daily OHLC, not the true intraday sequence), the
                     stop is assumed to have been hit first -- a
                     deliberately conservative tie-break, since assuming
                     the better outcome would systematically overstate the
                     success rate.
  end_of_horizon -- ignores intra-horizon touches entirely; looks only at
                     the close on the horizon's last day vs. entry.

A candidate too close to "now" for a full horizon of forward data to exist
yet is never silently guessed at: it comes back INCONCLUSIVE, which
DatasetBuilder (a later step) is expected to exclude from training
entirely rather than treat as a FAILURE.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from fitr.config_schemas.labeling_schema import LabelingConfig
from fitr.data.yf_proxy import NoDataInRangeError, TickerNotFoundError, YFProxy

logger = logging.getLogger(__name__)

SUCCESS = "SUCCESS"
FAILURE = "FAILURE"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class LabelResult:
    ticker: str
    candidate_date: pd.Timestamp
    outcome: str  # SUCCESS | FAILURE | INCONCLUSIVE
    entry_price: float | None = None
    trigger_date: pd.Timestamp | None = None
    trigger_price: float | None = None
    trigger_reason: str | None = None
    forward_return: float | None = None
    trading_days_available: int | None = None
    trading_days_held: int | None = None


class OutcomeLabeler:
    def __init__(self, yf_proxy: YFProxy, config: LabelingConfig):
        self._proxy = yf_proxy
        self._config = config

    def label(self, ticker: str, candidate_date) -> LabelResult:
        candidate = self._proxy.previous_trading_day(pd.Timestamp(candidate_date).normalize())
        horizon = self._config.horizon_trading_days
        buffer_days = horizon * 2 + 30  # generous padding for weekends/holidays

        try:
            window_df = self._proxy.get_history(ticker, candidate, candidate + pd.Timedelta(days=buffer_days))
        except (TickerNotFoundError, NoDataInRangeError) as exc:
            logger.debug("No data for %s around %s: %s", ticker, candidate.date(), exc)
            return LabelResult(ticker=ticker, candidate_date=candidate, outcome=INCONCLUSIVE, trigger_reason="no_data")

        if candidate not in window_df.index:
            logger.debug("%s has no row on %s specifically.", ticker, candidate.date())
            return LabelResult(ticker=ticker, candidate_date=candidate, outcome=INCONCLUSIVE, trigger_reason="no_data")

        forward_df = window_df.loc[window_df.index > candidate]
        if len(forward_df) < horizon:
            return LabelResult(
                ticker=ticker,
                candidate_date=candidate,
                outcome=INCONCLUSIVE,
                trading_days_available=len(forward_df),
                trigger_reason="insufficient_forward_data",
            )

        horizon_df = forward_df.iloc[:horizon]
        entry_price = float(window_df.loc[candidate, "Close"])
        target_price = entry_price * (1 + self._config.target_return)
        stop_price = entry_price * (1 + self._config.stop_loss_return)

        if self._config.evaluation_method == "first_touch":
            return self._label_first_touch(ticker, candidate, entry_price, target_price, stop_price, horizon_df)
        return self._label_end_of_horizon(ticker, candidate, entry_price, horizon_df)

    def _label_first_touch(
        self, ticker: str, candidate: pd.Timestamp, entry_price: float,
        target_price: float, stop_price: float, horizon_df: pd.DataFrame,
    ) -> LabelResult:
        for i, (date, row) in enumerate(horizon_df.iterrows()):
            hit_target = row["High"] >= target_price
            hit_stop = row["Low"] <= stop_price
            if hit_target and hit_stop:
                # Can't tell which happened first intraday from daily OHLC
                # alone -- assume the worse outcome rather than risk
                # overstating the success rate.
                return LabelResult(
                    ticker=ticker, candidate_date=candidate, outcome=FAILURE,
                    entry_price=entry_price, trigger_date=date, trigger_price=stop_price,
                    trigger_reason="stop_hit_same_day_as_target_ambiguous",
                    forward_return=(stop_price - entry_price) / entry_price,
                    trading_days_held=i + 1,
                )
            if hit_stop:
                return LabelResult(
                    ticker=ticker, candidate_date=candidate, outcome=FAILURE,
                    entry_price=entry_price, trigger_date=date, trigger_price=stop_price,
                    trigger_reason="stop_hit",
                    forward_return=(stop_price - entry_price) / entry_price,
                    trading_days_held=i + 1,
                )
            if hit_target:
                return LabelResult(
                    ticker=ticker, candidate_date=candidate, outcome=SUCCESS,
                    entry_price=entry_price, trigger_date=date, trigger_price=target_price,
                    trigger_reason="target_hit",
                    forward_return=(target_price - entry_price) / entry_price,
                    trading_days_held=i + 1,
                )
        # Neither threshold hit within the horizon -- counts as a failure
        # to achieve the target, regardless of whether it also avoided
        # the stop. Note this can still carry a positive forward_return
        # (e.g. price drifted up without ever touching target or stop) --
        # that's intentional, not a bug, and is why backtest win-rate and
        # model success-precision can diverge.
        last_date = horizon_df.index[-1]
        last_close = float(horizon_df["Close"].iloc[-1])
        return LabelResult(
            ticker=ticker, candidate_date=candidate, outcome=FAILURE,
            entry_price=entry_price, trigger_date=last_date, trigger_price=last_close,
            trigger_reason="horizon_expired_flat",
            forward_return=(last_close - entry_price) / entry_price,
            trading_days_held=len(horizon_df),
        )

    def _label_end_of_horizon(
        self, ticker: str, candidate: pd.Timestamp, entry_price: float, horizon_df: pd.DataFrame,
    ) -> LabelResult:
        last_date = horizon_df.index[-1]
        last_close = float(horizon_df["Close"].iloc[-1])
        forward_return = (last_close - entry_price) / entry_price
        outcome = SUCCESS if forward_return >= self._config.target_return else FAILURE
        return LabelResult(
            ticker=ticker, candidate_date=candidate, outcome=outcome,
            entry_price=entry_price, trigger_date=last_date, trigger_price=last_close,
            trigger_reason="horizon_end",
            forward_return=forward_return,
            trading_days_held=len(horizon_df),
        )
