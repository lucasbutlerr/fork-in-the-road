"""Scoring: the single definition of "how good is this config" that every
future tuning objective function (model hyperparameters, feature
selection, labeling parameters, scanning criteria -- whichever comes
first) will share, so results stay comparable across different tuning
phases even though what each one varies is completely different.

DISCIPLINE THIS EXISTS TO ENFORCE: every tuning trial scores against
train.csv (via WalkForwardValidator, for a robust multi-fold read rather
than one noisy split) -- NEVER test.csv. test.csv is checked exactly
once, by hand, after tuning concludes and a winning config has been
chosen. Search enough configs against a fixed target and some will look
good by pure chance; scoring hundreds of trials against test.csv would
quietly destroy its value as an honest final check.

Guards against the classic hyperparameter-search failure mode: pure
precision maximization can be "won" by a config that only ever predicts
SUCCESS for a handful of candidates -- technically excellent precision,
statistically meaningless. min_trades below prevents a search from
treating that as a good outcome, the same way format_walk_forward_report
flags (rather than hides) a fold with too few test rows.

When aggregating across walk-forward folds, an empty or gated-out fold
pulls the average down HARD rather than being averaged away gently --
deliberately. A config that goes silent during some folds (like the
regime-shift fold seen earlier in this project) is a genuinely less
robust config, and the score should reflect that instability, not hide it
behind folds that happened to look fine.
"""
from __future__ import annotations

import numpy as np

from fitr.modeling.evaluator import EvaluationReport
from fitr.modeling.walk_forward import WalkForwardReport

# Any real precision value is in [0, 1] -- this is deliberately far below
# that range so a gated-out report can never look competitive with a real
# score, no matter how the caller combines/averages scores downstream.
GATED_SCORE = -1.0


def score_from_report(report: EvaluationReport, min_trades: int = 30) -> float:
    """Returns success_precision, unless fewer than min_trades candidates
    were predicted SUCCESS (including the case of exactly zero, where
    success_precision is NaN) -- in which case returns GATED_SCORE
    instead, so a search can't "win" by shrinking volume to nothing."""
    cm = report.confusion_matrix
    predicted_success = cm["true_success_pred_success"] + cm["true_failure_pred_success"]
    if predicted_success < min_trades:
        return GATED_SCORE
    if report.success_precision != report.success_precision:  # NaN check
        return GATED_SCORE
    return report.success_precision


def score_from_walk_forward(report: WalkForwardReport, min_trades: int = 30) -> float:
    """Mean of score_from_report across every fold -- the single number a
    tuning objective function should return. Uses the plain mean, not a
    median or other outlier-robust aggregate, deliberately: a config that
    only performs well in some regimes is genuinely less robust than one
    that performs consistently, and the score should penalize that rather
    than average it away. Returns GATED_SCORE if there are no folds to
    score at all."""
    if not report.folds:
        return GATED_SCORE
    per_fold_scores = [score_from_report(fr.report, min_trades=min_trades) for fr in report.folds]
    return float(np.mean(per_fold_scores))


def expected_value_score_from_report(
    report: EvaluationReport, target_return: float, stop_loss_magnitude: float, min_trades: int = 30
) -> float:
    """Expected value per trade, in return terms: EV = p*T - (1-p)*S,
    where p is success_precision, T is target_return, S is
    |stop_loss_return|. This is the fix for a real scoring bug: raw
    precision alone lets a tuning search find configs with a tiny target
    and/or huge stop, which trivially inflates precision (SUCCESS becomes
    easy to hit) without the result being economically better -- a config
    needing 71% precision to break even can score "well" on raw precision
    while being deeply unprofitable. EV directly weighs each outcome by
    its own payout size, so it can't be gamed the same way: the same raw
    precision number scores differently depending on whether it clears
    that trial's OWN breakeven bar, exactly as it should.

    Deliberately does NOT run a backtest or involve position sizing at
    all -- this is the average return per $1 of capital allocated to ONE
    trade, no compounding, no equity curve. That's intentional, not a
    simplification: scoring via a real backtest would require a FIXED
    position-sizing scheme for every trial, and since fractional_kelly's
    raw fraction scales with 1/stop, a tiny stop would inflate the
    ALLOCATED SIZE the same way it inflates precision -- trading one
    payout-ratio confound for another, just routed through the sizing
    math instead. EV is genuinely sizing-independent by construction.

    Same min_trades gating as score_from_report, and for the same reason
    -- a lucky handful of trades shouldn't be able to produce a
    competitive-looking score."""
    cm = report.confusion_matrix
    predicted_success = cm["true_success_pred_success"] + cm["true_failure_pred_success"]
    if predicted_success < min_trades:
        return GATED_SCORE
    p = report.success_precision
    if p != p:  # NaN check
        return GATED_SCORE
    return p * target_return - (1 - p) * stop_loss_magnitude


def expected_value_score_from_walk_forward(
    report: WalkForwardReport, target_return: float, stop_loss_magnitude: float, min_trades: int = 30
) -> float:
    """Mean of expected_value_score_from_report across every fold -- same
    aggregation philosophy as score_from_walk_forward (plain mean, an
    unstable fold pulls the score down hard rather than being averaged
    away). This is the score JointTuningObjective should return once
    target_return/stop_loss_return are themselves part of the search
    space, since score_from_walk_forward's raw precision is exactly what
    a payout-ratio-varying search can exploit."""
    if not report.folds:
        return GATED_SCORE
    per_fold_scores = [
        expected_value_score_from_report(fr.report, target_return, stop_loss_magnitude, min_trades=min_trades)
        for fr in report.folds
    ]
    return float(np.mean(per_fold_scores))
