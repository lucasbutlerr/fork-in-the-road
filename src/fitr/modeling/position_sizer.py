"""PositionSizer: pure calculation, no I/O -- given a predicted probability,
an entry price, and account equity, recommends how much to allocate to a
candidate. Two methods, config-driven:

  fixed_fractional -- risks a fixed fraction of equity per trade, sized so
                       that a full stop-out (per the configured
                       stop_loss_return) loses exactly that fraction. Does
                       NOT use the predicted probability at all -- every
                       candidate gets the same risk regardless of
                       conviction. Works fine with raw (uncalibrated)
                       probabilities since it never touches them.

  fractional_kelly  -- sizes proportional to the model's edge via Kelly's
                        criterion, adapted for this project's actual
                        payoff structure (a win multiplies the position by
                        1+target_return, a loss by 1-|stop_loss_return| --
                        proportional gain/loss on committed capital, not
                        the textbook "lose the entire wager" case), scaled
                        down by kelly_multiplier. This is the method that
                        actually needs a genuinely calibrated probability
                        (see model.yaml's calibration_method) -- feed it an
                        overconfident raw probability and it will
                        recommend an oversized position, since Kelly sizing
                        is a direct function of the probability value, not
                        just its rank. See _kelly_fraction below for the
                        full derivation.

Kelly fractions can come out negative (the setup has no edge, or a
negative one, at the configured target/stop) -- size() floors this at 0
rather than recommending a short position, since this project's exit model
doesn't support shorting.
"""
from __future__ import annotations

from dataclasses import dataclass

from fitr.config_schemas.position_sizing_schema import PositionSizingConfig


@dataclass
class PositionSizeResult:
    ticker: str
    predicted_probability: float
    method: str
    raw_fraction: float  # recommended fraction of equity BEFORE min/max guardrails
    recommended_fraction: float  # AFTER guardrails -- what to actually use
    recommended_shares: int
    position_value: float
    entry_price: float
    equity: float
    capped: bool  # whether a guardrail changed raw_fraction -> recommended_fraction
    kelly_fraction: float | None = None  # full (unmultiplied) Kelly fraction -- fractional_kelly only


class PositionSizer:
    def __init__(self, config: PositionSizingConfig):
        self._config = config

    def size(self, ticker: str, predicted_probability: float, entry_price: float, equity: float) -> PositionSizeResult:
        kelly_fraction = None
        if self._config.method == "fixed_fractional":
            raw_fraction = self._fixed_fractional_fraction()
        else:
            kelly_fraction = self._kelly_fraction(predicted_probability)
            raw_fraction = kelly_fraction * self._config.kelly_multiplier

        raw_fraction = max(raw_fraction, 0.0)  # never recommend a short

        recommended_fraction = min(raw_fraction, self._config.max_position_fraction)
        if recommended_fraction < self._config.min_position_fraction:
            recommended_fraction = 0.0
        capped = recommended_fraction != raw_fraction

        position_value = equity * recommended_fraction
        shares = int(position_value // entry_price) if entry_price > 0 else 0
        actual_position_value = shares * entry_price

        return PositionSizeResult(
            ticker=ticker,
            predicted_probability=predicted_probability,
            method=self._config.method,
            raw_fraction=raw_fraction,
            recommended_fraction=recommended_fraction,
            recommended_shares=shares,
            position_value=actual_position_value,
            entry_price=entry_price,
            equity=equity,
            capped=capped,
            kelly_fraction=kelly_fraction,
        )

    def _fixed_fractional_fraction(self) -> float:
        """Risking risk_fraction of equity on a move that fails at
        stop_loss_return means the position itself should be
        risk_fraction / |stop_loss_return| of equity -- e.g. risking 1% of
        equity on a trade with a 4% stop means a 25%-of-equity position
        (since a 4% adverse move on a 25% position is exactly 1% of total
        equity)."""
        return self._config.risk_fraction / abs(self._config.stop_loss_return)

    def _kelly_fraction(self, p: float) -> float:
        """Full Kelly fraction for THIS project's actual payoff structure:
        a win multiplies the position by (1+target_return), a loss
        multiplies it by (1-|stop_loss_return|) -- proportional gain/loss
        on the capital committed to the trade, not "lose the entire
        wager" the way the textbook gambling formula f*=(bp-q)/b assumes.
        That distinction matters: naively reusing the textbook formula
        with b=target/|stop| gives a DIFFERENT (wrong) answer for this
        payoff structure, even though both formulas happen to agree on
        where the fraction crosses zero (the breakeven probability).

        Maximizing E[log wealth] = p*ln(1+f*T) + q*ln(1-f*S) over f, where
        T=target_return, S=|stop_loss_return|, gives:

            f* = (p*T - q*S) / (T*S) = p/S - q/T

        This can come out very large for a tight stop (small S) even with
        a modest edge, since the true per-trade downside is small relative
        to capital -- max_position_fraction exists precisely to guardrail
        against taking this raw number literally, the same way
        "fractional Kelly" (kelly_multiplier below 1.0) exists to guard
        against estimation error in p. Don't rely on the raw f* alone."""
        target = self._config.target_return
        stop = abs(self._config.stop_loss_return)
        q = 1 - p
        return p / stop - q / target
