"""WalkForwardValidator: runs the SAME ModelConfig through multiple
sequential train/test splits across time, instead of trusting one fixed
split. This is what tells you whether a config's performance is genuinely
stable, or was just a lucky (or unlucky) read on one particular period --
directly answering the noise concern raised repeatedly earlier in this
project (non-monotonic threshold sweeps, a single bad fold skewing the
picture).

Fold generation: the full dataset's date range is split into an initial
training period (initial_train_fraction of the total span -- reserved
purely as a training floor, never itself a test window, since fold 1
needs real data to train on) followed by n_folds equal-sized,
non-overlapping test windows carved out of what remains. Each fold's
training window is either:

  expanding -- always starts at the very beginning of the data and grows
               with each fold. Maximizes training data per fold, which
               matters a lot for a still-data-starved project -- this is
               the default for exactly that reason.
  rolling   -- a fixed-size window (rolling_train_window_days) that
               slides forward with each fold, so old data eventually
               falls out of the training set entirely. Worth switching to
               once there's enough total history that old data being
               stale (regime change) is a bigger risk than having less of
               it.

Reuses ModelTrainer and ModelEvaluator completely unchanged -- a
WalkForwardReport is just n_folds worth of ordinary EvaluationReports,
plus the aggregate stats across them. If a PositionSizer is supplied, each
fold ALSO gets a real return-aware backtest (BacktestSimulator), so you
can see whether realized P&L holds up across folds too, not just
classification metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fitr.config_schemas.model_schema import ModelConfig
from fitr.config_schemas.walk_forward_schema import WalkForwardConfig
from fitr.modeling.backtest import BacktestReport, BacktestSimulator
from fitr.modeling.evaluator import EvaluationReport, ModelEvaluator
from fitr.modeling.position_sizer import PositionSizer
from fitr.modeling.trainer import ModelTrainer


@dataclass
class Fold:
    fold_number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive on all but the last fold


@dataclass
class FoldResult:
    fold: Fold
    n_train_rows: int
    n_test_rows: int
    low_confidence: bool  # True if n_test_rows < min_test_rows
    report: EvaluationReport
    backtest: BacktestReport | None = None


@dataclass
class WalkForwardReport:
    folds: list[FoldResult] = field(default_factory=list)
    mean_success_precision: float = float("nan")
    std_success_precision: float = float("nan")
    mean_success_accuracy: float = float("nan")
    std_success_accuracy: float = float("nan")


class WalkForwardValidator:
    def __init__(self, model_config: ModelConfig, position_sizer: PositionSizer | None = None):
        self._model_config = model_config
        self._position_sizer = position_sizer

    def run(
        self, full_df: pd.DataFrame, config: WalkForwardConfig, date_column: str = "date", threshold: float = 0.5
    ) -> WalkForwardReport:
        if full_df.empty:
            raise ValueError("Cannot walk-forward validate an empty dataset.")

        folds = self._generate_folds(full_df, config, date_column)
        fold_results: list[FoldResult] = []

        for fold in folds:
            dates = pd.to_datetime(full_df[date_column])
            train_df = full_df.loc[(dates >= fold.train_start) & (dates < fold.train_end)]
            test_df = full_df.loc[(dates >= fold.test_start) & (dates < fold.test_end)]

            if train_df.empty or test_df.empty:
                continue  # not enough data at the edges of the range to form this fold at all

            trained = ModelTrainer(self._model_config).fit(train_df, date_column=date_column)
            report = ModelEvaluator().evaluate(trained, test_df, threshold=threshold)

            backtest_report = None
            if self._position_sizer is not None and "forward_return" in test_df.columns:
                backtest_report = BacktestSimulator(self._position_sizer).run(
                    trained, test_df, threshold=threshold, date_column=date_column
                )

            fold_results.append(
                FoldResult(
                    fold=fold,
                    n_train_rows=len(train_df),
                    n_test_rows=len(test_df),
                    low_confidence=len(test_df) < config.min_test_rows,
                    report=report,
                    backtest=backtest_report,
                )
            )

        return self._aggregate(fold_results)

    @staticmethod
    def _generate_folds(full_df: pd.DataFrame, config: WalkForwardConfig, date_column: str) -> list[Fold]:
        dates = pd.to_datetime(full_df[date_column])
        overall_start, overall_end = dates.min(), dates.max()
        total_span = overall_end - overall_start

        initial_train_end = overall_start + total_span * config.initial_train_fraction
        test_breakpoints = pd.date_range(initial_train_end, overall_end, periods=config.n_folds + 1)

        folds = []
        for i in range(config.n_folds):
            test_start, test_end = test_breakpoints[i], test_breakpoints[i + 1]
            if config.window_type == "expanding":
                train_start = overall_start
            else:
                train_start = max(overall_start, test_start - pd.Timedelta(days=config.rolling_train_window_days))
            folds.append(
                Fold(fold_number=i + 1, train_start=train_start, train_end=test_start, test_start=test_start, test_end=test_end)
            )
        return folds

    @staticmethod
    def _aggregate(fold_results: list[FoldResult]) -> WalkForwardReport:
        if not fold_results:
            return WalkForwardReport(folds=[])

        precisions = [fr.report.success_precision for fr in fold_results if fr.report.success_precision == fr.report.success_precision]
        accuracies = [fr.report.success_accuracy for fr in fold_results if fr.report.success_accuracy == fr.report.success_accuracy]

        return WalkForwardReport(
            folds=fold_results,
            mean_success_precision=float(np.mean(precisions)) if precisions else float("nan"),
            std_success_precision=float(np.std(precisions, ddof=1)) if len(precisions) > 1 else float("nan"),
            mean_success_accuracy=float(np.mean(accuracies)) if accuracies else float("nan"),
            std_success_accuracy=float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else float("nan"),
        )


def _fmt(value: float) -> str:
    return "N/A" if value != value else f"{value:.1%}"


def format_walk_forward_report(report: WalkForwardReport, show_calibration: bool = True, show_setups: bool = True) -> str:
    if not report.folds:
        return "No folds produced -- date range too short for the configured n_folds/initial_train_fraction."

    lines = [
        f"{len(report.folds)} folds",
        f"Success precision across folds: mean={_fmt(report.mean_success_precision)}  "
        f"std={_fmt(report.std_success_precision)}",
        f"Success accuracy across folds:  mean={_fmt(report.mean_success_accuracy)}  "
        f"std={_fmt(report.std_success_accuracy)}",
        "",
    ]
    for fr in report.folds:
        f_ = fr.fold
        flag = "  [LOW CONFIDENCE -- few test rows]" if fr.low_confidence else ""
        lines.append(
            f"Fold {f_.fold_number}: train [{f_.train_start.date()} -> {f_.train_end.date()}) "
            f"({fr.n_train_rows} rows)  test [{f_.test_start.date()} -> {f_.test_end.date()}) "
            f"({fr.n_test_rows} rows){flag}"
        )
        cm = fr.report.confusion_matrix
        predicted_success = cm["true_success_pred_success"] + cm["true_failure_pred_success"]
        lines.append(
            f"    accuracy={fr.report.overall_accuracy:.1%}  "
            f"success_precision={_fmt(fr.report.success_precision)}  "
            f"success_recall={_fmt(fr.report.success_accuracy)}  "
            f"predicted_success_count={predicted_success} (of {fr.n_test_rows} test rows)"
        )
        if fr.backtest is not None:
            lines.append(
                f"    backtest: {fr.backtest.n_trades} trades, total_return={fr.backtest.total_return:+.2%}, "
                f"max_drawdown={fr.backtest.max_drawdown:.2%}"
            )

        if show_calibration and fr.report.calibration_curve:
            lines.append("    calibration (mean predicted -> actual rate, by bucket):")
            for b in fr.report.calibration_curve:
                lines.append(
                    f"      [{b.bucket_low:.0%}-{b.bucket_high:.0%})  n={b.n_rows:<5}  "
                    f"predicted={b.mean_predicted_probability:.1%}  actual={b.actual_success_rate:.1%}"
                )
            highest_bucket = max(fr.report.calibration_curve, key=lambda b: b.bucket_low)
            lines.append(f"    highest bucket reached: [{highest_bucket.bucket_low:.0%}-{highest_bucket.bucket_high:.0%})")

        if show_setups and fr.report.setup_breakdown:
            lines.append("    per-setup (base_rate = actual SUCCESS fraction within this setup, this fold):")
            for name, sb in sorted(fr.report.setup_breakdown.items()):
                base_rate = sb.success_count / sb.n_rows if sb.n_rows > 0 else float("nan")
                lines.append(
                    f"      {name:<28} n={sb.n_rows:<5} base_rate={base_rate:.1%}  "
                    f"success_precision={_fmt(sb.success_precision)}"
                )

        lines.append("")

    return "\n".join(lines)
