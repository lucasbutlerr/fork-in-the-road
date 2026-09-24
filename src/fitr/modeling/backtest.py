"""BacktestSimulator: walks a labeled test set chronologically, applying
PositionSizer + REAL forward_return (DatasetBuilder's trigger_reason/
forward_return columns, added in Step 1 of this phase) to simulate an
actual equity curve -- answering "how much money would this have made"
rather than just classification accuracy. This is the final piece tying
the whole Phase 2 arc together: calibration makes the probability
meaningful, PositionSizer turns that probability into an allocation, and
this turns a sequence of allocations + real outcomes into P&L.

KNOWN SIMPLIFICATION, stated plainly rather than hidden: every trade is
sized against the CURRENT total equity at its entry date, as if the full
account balance were always available. This does NOT model concurrent
capital lockup across overlapping open positions -- if the scanner flags
five different tickers within the same horizon window, a real account
would be splitting capital across five concurrent positions, not treating
each one as if the full account were free. For a large number of
simultaneous signals, this will overstate achievable returns relative to a
real portfolio with genuine position limits. Modeling true concurrent-
position capital constraints is a meaningfully bigger scope than this
step -- flagged here as a known gap, not solved.

Also note: PositionSizer.size() is called with a placeholder entry_price
of 1.0 here, not a real price -- DatasetBuilder's output never stored raw
entry prices (only percentage-based features/returns), so real share
counts can't be reconstructed for backtesting. Only PositionSizer's
FRACTION-of-equity output is used; its share-count/position_value fields
are meaningless in this context and deliberately ignored.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fitr.modeling.position_sizer import PositionSizer
from fitr.modeling.trainer import TrainedModel

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    ticker: str
    date: pd.Timestamp
    predicted_probability: float
    recommended_fraction: float
    forward_return: float
    equity_before: float
    equity_after: float
    pnl: float
    trading_days_held: int | None = None


@dataclass
class BacktestReport:
    starting_equity: float
    ending_equity: float
    total_return: float
    n_candidates: int  # every row in test_df, whether or not it cleared the threshold
    n_trades: int  # rows that cleared the threshold and were actually taken
    win_count: int
    loss_count: int
    win_rate: float
    average_win: float  # mean forward_return among winning trades
    average_loss: float  # mean forward_return among losing trades (a negative number)
    max_drawdown: float  # largest peak-to-trough decline in the equity curve, as a fraction
    sharpe_like_ratio: float  # see docstring below -- NOT a standard annualized Sharpe ratio
    average_holding_days: float = float("nan")  # mean trading_days_held across trades that have it
    equity_curve: list[float] = field(default_factory=list)  # starts with starting_equity, one value per trade after
    trades: list[TradeRecord] = field(default_factory=list)


class BacktestSimulator:
    def __init__(self, position_sizer: PositionSizer):
        self._sizer = position_sizer

    def run(
        self,
        trained: TrainedModel,
        test_df: pd.DataFrame,
        threshold: float = 0.5,
        starting_equity: float = 100_000.0,
        date_column: str = "date",
    ) -> BacktestReport:
        if test_df.empty:
            raise ValueError("Cannot backtest on an empty test set.")
        if "forward_return" not in test_df.columns:
            raise ValueError(
                "test_df has no 'forward_return' column -- rebuild the dataset with a version of "
                "DatasetBuilder that writes it (added in Phase 2 Step 1), or there's no real return "
                "data for this backtest to work with."
            )

        probs = trained.predict_proba_success(test_df)
        work = pd.DataFrame(
            {
                "ticker": test_df["ticker"].to_numpy() if "ticker" in test_df.columns else [""] * len(test_df),
                "date": pd.to_datetime(test_df[date_column]).to_numpy(),
                "predicted_probability": probs,
                "forward_return": test_df["forward_return"].to_numpy(dtype=float),
                "trading_days_held": (
                    test_df["trading_days_held"].to_numpy()
                    if "trading_days_held" in test_df.columns
                    else np.full(len(test_df), np.nan)
                ),
            }
        )
        # Chronological order is what makes this a real simulation of
        # "trading forward through time" rather than an arbitrary ordering
        # -- ticker is just a deterministic tiebreaker for same-day candidates.
        work = work.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)

        equity = starting_equity
        equity_curve = [equity]
        trades: list[TradeRecord] = []

        for row in work.itertuples(index=False):
            if row.predicted_probability < threshold:
                continue
            if pd.isna(row.forward_return):
                logger.warning("Skipping %s on %s: forward_return is missing/NaN.", row.ticker, row.date)
                continue

            sizing = self._sizer.size(row.ticker, row.predicted_probability, entry_price=1.0, equity=equity)
            fraction = sizing.recommended_fraction

            equity_before = equity
            equity = equity * (1 + fraction * row.forward_return)
            equity_curve.append(equity)

            trades.append(
                TradeRecord(
                    ticker=row.ticker,
                    date=pd.Timestamp(row.date),
                    predicted_probability=float(row.predicted_probability),
                    recommended_fraction=fraction,
                    forward_return=float(row.forward_return),
                    equity_before=equity_before,
                    equity_after=equity,
                    pnl=equity - equity_before,
                    trading_days_held=(
                        int(row.trading_days_held) if pd.notna(row.trading_days_held) else None
                    ),
                )
            )

        return self._build_report(starting_equity, equity, len(work), equity_curve, trades)

    @staticmethod
    def _build_report(
        starting_equity: float, ending_equity: float, n_candidates: int,
        equity_curve: list[float], trades: list[TradeRecord],
    ) -> BacktestReport:
        wins = [t for t in trades if t.forward_return > 0]
        losses = [t for t in trades if t.forward_return <= 0]
        # "per-trade portfolio return" -- the fraction of equity gained or
        # lost by that single trade, used for the win-rate/Sharpe-like
        # stats below (distinct from forward_return, which is the raw
        # underlying stock move -- a trade sized at 5% of equity moving
        # +8% only changes the PORTFOLIO by 0.4%, and that portfolio-level
        # number is what these stats describe).
        per_trade_portfolio_returns = [t.pnl / t.equity_before for t in trades if t.equity_before > 0]

        sharpe_like = float("nan")
        if len(per_trade_portfolio_returns) > 1:
            std = float(np.std(per_trade_portfolio_returns, ddof=1))
            if std > 0:
                sharpe_like = float(np.mean(per_trade_portfolio_returns)) / std

        holding_days = [t.trading_days_held for t in trades if t.trading_days_held is not None]
        average_holding_days = float(np.mean(holding_days)) if holding_days else float("nan")

        return BacktestReport(
            starting_equity=starting_equity,
            ending_equity=ending_equity,
            total_return=(ending_equity - starting_equity) / starting_equity if starting_equity > 0 else float("nan"),
            n_candidates=n_candidates,
            n_trades=len(trades),
            win_count=len(wins),
            loss_count=len(losses),
            win_rate=(len(wins) / len(trades)) if trades else float("nan"),
            average_win=float(np.mean([t.forward_return for t in wins])) if wins else float("nan"),
            average_loss=float(np.mean([t.forward_return for t in losses])) if losses else float("nan"),
            max_drawdown=BacktestSimulator._max_drawdown(equity_curve),
            sharpe_like_ratio=sharpe_like,
            average_holding_days=average_holding_days,
            equity_curve=equity_curve,
            trades=trades,
        )

    @staticmethod
    def _max_drawdown(equity_curve: list[float]) -> float:
        peak = equity_curve[0]
        max_dd = 0.0
        for e in equity_curve:
            peak = max(peak, e)
            if peak > 0:
                max_dd = max(max_dd, (peak - e) / peak)
        return max_dd


def format_backtest_report(report: BacktestReport, max_trades_shown: int = 20) -> str:
    lines = [
        f"Candidates considered: {report.n_candidates}   Trades taken: {report.n_trades} "
        f"({report.n_trades / report.n_candidates:.1%} of candidates cleared the threshold)"
        if report.n_candidates else "No candidates.",
        "",
        f"Starting equity: {report.starting_equity:,.2f}",
        f"Ending equity:   {report.ending_equity:,.2f}",
        f"Total return:    {report.total_return:+.2%}",
        f"Max drawdown:    {report.max_drawdown:.2%}",
        f"Sharpe-like ratio (mean/std of per-trade portfolio return, NOT annualized -- trades aren't "
        f"evenly spaced in time, so this is a rough consistency signal, not a standard Sharpe ratio): "
        f"{report.sharpe_like_ratio:.2f}" if report.sharpe_like_ratio == report.sharpe_like_ratio else
        "Sharpe-like ratio: N/A (fewer than 2 trades, or zero variance)",
        "",
        f"Win rate: {report.win_rate:.1%}   ({report.win_count} wins, {report.loss_count} losses)"
        if report.n_trades else "No trades taken.",
    ]
    if report.average_holding_days == report.average_holding_days:  # not NaN
        lines.append(f"Average holding period: {report.average_holding_days:.1f} trading days")
    if report.win_count:
        lines.append(f"Average win (underlying stock return, not portfolio %):  {report.average_win:+.2%}")
    if report.loss_count:
        lines.append(f"Average loss (underlying stock return, not portfolio %): {report.average_loss:+.2%}")
    lines.append("")

    if report.trades:
        shown = report.trades[:max_trades_shown]
        lines.append(f"First {len(shown)} of {len(report.trades)} trades (chronological):")
        lines.append(f"  {'date':<12}{'ticker':<10}{'P(success)':>12}{'fraction':>10}{'return':>9}{'equity':>14}")
        for t in shown:
            lines.append(
                f"  {t.date.date()!s:<12}{t.ticker:<10}{t.predicted_probability:>11.1%} "
                f"{t.recommended_fraction:>9.1%} {t.forward_return:>+8.1%} {t.equity_after:>14,.2f}"
            )
        lines.append("")

    return "\n".join(lines)


def write_trades_csv(trades: list[TradeRecord], path: str) -> None:
    """Shared by scripts/backtest.py and scripts/walk_forward.py -- one
    definition of the trade-log CSV format, rather than each script
    keeping its own copy that could quietly drift apart."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["date", "ticker", "predicted_probability", "recommended_fraction", "forward_return", "equity_before", "equity_after", "pnl"]
        )
        for t in trades:
            writer.writerow(
                [
                    t.date.date(), t.ticker, f"{t.predicted_probability:.6f}", f"{t.recommended_fraction:.6f}",
                    f"{t.forward_return:.6f}", f"{t.equity_before:.2f}", f"{t.equity_after:.2f}", f"{t.pnl:.2f}",
                ]
            )
