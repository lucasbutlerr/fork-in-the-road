"""JointTuningObjective: the Optuna objective function for the merged
model + feature-selection + labeling study (Phase 3 Step 5).

Per trial: suggests labeling.yaml parameters, a cooldown, model.yaml
hyperparameters, and which feature groups to include; rebuilds the
dataset with those labeling/cooldown values (reusing the cached scan --
scanning_criteria.yaml, the universe, and the date range are FIXED across
every trial in this study, so after the first trial this is fast); runs
WalkForwardValidator over the tuning-visible portion only (strictly
before tuning_end_date -- everything at or after it is never touched);
and returns expected_value_score_from_walk_forward's single number for
Optuna to maximize.

SCORING: uses expected value per trade (fitr.tuning.scoring's
expected_value_score_from_*), not raw success_precision. This matters
specifically because target_return/stop_loss_return are THEMSELVES part
of this study's search space -- raw precision is not comparable across
different payout ratios (a config with a tiny target and/or huge stop
trivially inflates precision, since SUCCESS becomes an easier bar to
clear, without the result being economically better; a real example
found during tuning: target=0.04/stop=0.10 needs 71.4% precision just to
break even, so even a 70%-precision trial there is still net-negative).
EV directly weighs each outcome by its own payout size, so it can't be
gamed the same way. See expected_value_score_from_report's docstring for
why this is computed directly rather than by running a real backtest --
short version: a backtest needs a fixed position-sizing scheme, and
fractional_kelly's fraction scales with 1/stop, which would just move the
same confound into the sizing math instead of removing it.

KNOWN, DELIBERATELY UNADDRESSED INEFFICIENCY: features.yaml itself never
varies in this study, only which already-computed columns get used --
but DatasetBuilder currently recomputes every feature on every trial
regardless, since labeling and featurizing happen in the same internal
pass. That's real wasted work (labeling is cheap; feature computation
isn't free). Worth splitting DatasetBuilder into a separate
"featurize once, re-label many times" cache if per-trial time becomes a
real bottleneck in practice -- not built now, on the same reasoning as
the metric-value-cache idea for scanning_criteria.yaml: don't build a
second layer of caching before confirming the first one wasn't already
enough.
"""
from __future__ import annotations

import logging

from fitr.config_schemas.labeling_schema import LabelingConfig
from fitr.config_schemas.model_schema import ModelConfig
from fitr.config_schemas.tuning_schema import JointTuningConfig, ParamRange
from fitr.config_schemas.walk_forward_schema import WalkForwardConfig
from fitr.data.yf_proxy import YFProxy
from fitr.dataset.builder import DatasetBuilder
from fitr.features.feature_engine import FeatureEngine
from fitr.labeling.labeler import OutcomeLabeler
from fitr.modeling.walk_forward import WalkForwardValidator
from fitr.scanning.scanner import BreakoutScanner
from fitr.tuning.scoring import GATED_SCORE, expected_value_score_from_walk_forward

logger = logging.getLogger(__name__)


def _suggest(trial, name: str, param_range: ParamRange, as_int: bool = False):
    if as_int:
        return trial.suggest_int(name, int(param_range.low), int(param_range.high))
    return trial.suggest_float(name, param_range.low, param_range.high, log=param_range.log)


class JointTuningObjective:
    def __init__(
        self,
        config: JointTuningConfig,
        universe: list[str],
        scanner: BreakoutScanner,
        feature_engine: FeatureEngine,
        proxy: YFProxy,
        walk_forward_config: WalkForwardConfig,
        labeler_factory=OutcomeLabeler,
    ):
        self._config = config
        self._universe = universe
        self._scanner = scanner
        self._feature_engine = feature_engine
        self._proxy = proxy
        self._wf_config = walk_forward_config
        self._labeler_factory = labeler_factory  # swappable for tests -- defaults to the real OutcomeLabeler
        self.trial_log: list[dict] = []  # every trial's params + score, for post-hoc inspection

    def __call__(self, trial) -> float:
        c = self._config

        horizon = _suggest(trial, "horizon_trading_days", c.horizon_trading_days, as_int=True)
        target = _suggest(trial, "target_return", c.target_return)
        stop_magnitude = _suggest(trial, "stop_loss_magnitude", c.stop_loss_magnitude)
        cooldown = _suggest(trial, "cooldown_trading_days", c.cooldown_trading_days, as_int=True)

        labeling_config = LabelingConfig.model_validate(
            {
                "horizon_trading_days": horizon,
                "target_return": target,
                "stop_loss_return": -stop_magnitude,
                "evaluation_method": "first_touch",
            }
        )

        labeler = self._labeler_factory(self._proxy, labeling_config)
        builder = DatasetBuilder(
            self._proxy, self._scanner, self._feature_engine, labeler, self._universe,
            cooldown_trading_days=cooldown, cache_dir=c.scan_cache_dir,
        )
        full_df = builder.build(c.scan_start_date, c.scan_end_date)

        if full_df.empty:
            return self._log_and_return(trial, GATED_SCORE, note="empty dataset for this trial's labeling params")

        tuning_df = full_df[full_df["date"] < c.tuning_end_date]
        if tuning_df.empty:
            return self._log_and_return(trial, GATED_SCORE, note="no rows before tuning_end_date")

        n_estimators = _suggest(trial, "n_estimators", c.n_estimators, as_int=True)
        max_depth = _suggest(trial, "max_depth", c.max_depth, as_int=True)
        learning_rate = _suggest(trial, "learning_rate", c.learning_rate)

        feature_columns = self._suggest_feature_columns(trial, tuning_df)

        model_config = ModelConfig.model_validate(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "feature_columns": feature_columns,
            }
        )

        try:
            validator = WalkForwardValidator(model_config)
            wf_report = validator.run(tuning_df, self._wf_config)
        except ValueError as exc:
            return self._log_and_return(trial, GATED_SCORE, note=f"walk-forward failed: {exc}")

        score = expected_value_score_from_walk_forward(
            wf_report, target_return=target, stop_loss_magnitude=stop_magnitude, min_trades=c.min_trades_for_scoring
        )
        return self._log_and_return(trial, score)

    def _suggest_feature_columns(self, trial, tuning_df) -> str | list[str]:
        """"auto" (every available feature column) unless feature_groups
        is configured -- in which case each group is toggled on/off as one
        search dimension, and excluded groups' columns are dropped from
        the returned list."""
        if not self._config.feature_groups:
            return "auto"

        grouped_columns: set[str] = set()
        for names in self._config.feature_groups.values():
            grouped_columns.update(names)

        excluded: set[str] = set()
        for group_name, names in self._config.feature_groups.items():
            include = trial.suggest_categorical(f"include_{group_name}", [True, False])
            if not include:
                excluded.update(names)

        non_feature_columns = {"ticker", "date", "label", "trigger_reason", "forward_return"}
        all_columns = [c for c in tuning_df.columns if c not in non_feature_columns]
        return [c for c in all_columns if c not in excluded]

    def _log_and_return(self, trial, score: float, note: str = "") -> float:
        entry = {"trial_number": trial.number, "params": dict(trial.params), "score": score}
        if note:
            entry["note"] = note
        self.trial_log.append(entry)
        if logger.isEnabledFor(logging.INFO):
            logger.info("Trial %d: score=%.4f%s", trial.number, score, f" ({note})" if note else "")
        return score
