"""ModelEvaluator: given a TrainedModel and a labeled test set, computes
accuracy metrics.

"success_accuracy" / "failure_accuracy" here specifically mean per-class
RECALL: of the rows that were actually SUCCESS, what fraction did the
model correctly call SUCCESS (and likewise for FAILURE). That's a
deliberate choice matching the original project goal word-for-word --
"% accuracy of successful breakouts and % accuracy of failed breakouts" --
rather than precision, which answers a different question (of the rows
predicted SUCCESS, how many actually were). Both are reported, since
they're both useful, but the two headline numbers are the recall pair.

Also computes a per-setup breakdown using the setup_* one-hot columns
DatasetBuilder writes, so setups with genuinely different behavior --
like the pullback_reclaim vs. classic_high_breakout difference surfaced
earlier in this project -- show up as separate numbers instead of being
averaged into one aggregate accuracy figure. This is what makes the
earlier "should each setup get its own model" question answerable with
real data instead of a guess: if one setup's numbers are wildly
different here, that's the signal to consider splitting it out, not an
assumption to build in ahead of time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fitr.labeling.labeler import FAILURE, SUCCESS
from fitr.modeling.trainer import TrainedModel


@dataclass
class SetupBreakdown:
    setup_name: str
    n_rows: int
    success_count: int
    failure_count: int
    accuracy: float
    success_accuracy: float | None  # None if this setup had zero actual SUCCESS rows in the test set
    failure_accuracy: float | None  # None if this setup had zero actual FAILURE rows in the test set


@dataclass
class EvaluationReport:
    n_rows: int
    threshold: float
    overall_accuracy: float
    success_accuracy: float
    failure_accuracy: float
    success_precision: float
    failure_precision: float
    confusion_matrix: dict[str, int]
    feature_importances: dict[str, float]
    setup_breakdown: dict[str, SetupBreakdown] = field(default_factory=dict)


class ModelEvaluator:
    def evaluate(self, trained: TrainedModel, test_df: pd.DataFrame, threshold: float = 0.5) -> EvaluationReport:
        if test_df.empty:
            raise ValueError("Cannot evaluate on an empty test set.")

        y_true = test_df[trained.label_column].map({SUCCESS: 1, FAILURE: 0})
        if y_true.isna().any():
            bad = sorted(test_df.loc[y_true.isna(), trained.label_column].unique())
            raise ValueError(f"Unexpected label value(s) {bad} in column '{trained.label_column}'.")
        y_true = y_true.astype(int)

        probs = trained.predict_proba_success(test_df)
        y_pred = (probs >= threshold).astype(int)

        cm = self._confusion_matrix(y_true.to_numpy(), y_pred)
        overall_accuracy = float((y_true.to_numpy() == y_pred).mean())
        success_accuracy = self._safe_divide(
            cm["true_success_pred_success"], cm["true_success_pred_success"] + cm["true_success_pred_failure"]
        )
        failure_accuracy = self._safe_divide(
            cm["true_failure_pred_failure"], cm["true_failure_pred_failure"] + cm["true_failure_pred_success"]
        )
        success_precision = self._safe_divide(
            cm["true_success_pred_success"], cm["true_success_pred_success"] + cm["true_failure_pred_success"]
        )
        failure_precision = self._safe_divide(
            cm["true_failure_pred_failure"], cm["true_failure_pred_failure"] + cm["true_success_pred_failure"]
        )

        return EvaluationReport(
            n_rows=len(test_df),
            threshold=threshold,
            overall_accuracy=overall_accuracy,
            success_accuracy=success_accuracy,
            failure_accuracy=failure_accuracy,
            success_precision=success_precision,
            failure_precision=failure_precision,
            confusion_matrix=cm,
            feature_importances=self._feature_importances(trained),
            setup_breakdown=self._setup_breakdown(test_df, y_true, y_pred),
        )

    @staticmethod
    def _safe_divide(numerator: int, denominator: int) -> float:
        """NaN (not a ZeroDivisionError, not a silent 0.0) when a class is
        entirely absent -- 0.0 would misleadingly claim "the model got
        every SUCCESS wrong" when really there were no SUCCESS rows to
        get right or wrong at all."""
        return float(numerator / denominator) if denominator > 0 else float("nan")

    @staticmethod
    def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
        return {
            "true_success_pred_success": int(((y_true == 1) & (y_pred == 1)).sum()),
            "true_success_pred_failure": int(((y_true == 1) & (y_pred == 0)).sum()),
            "true_failure_pred_success": int(((y_true == 0) & (y_pred == 1)).sum()),
            "true_failure_pred_failure": int(((y_true == 0) & (y_pred == 0)).sum()),
        }

    @staticmethod
    def _feature_importances(trained: TrainedModel) -> dict[str, float]:
        raw = trained.model.feature_importances_
        if raw is None:
            return {}
        pairs = sorted(zip(trained.feature_columns, raw), key=lambda p: p[1], reverse=True)
        return {name: float(value) for name, value in pairs}

    @staticmethod
    def _setup_breakdown(test_df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, SetupBreakdown]:
        setup_columns = [c for c in test_df.columns if c.startswith("setup_")]
        y_true_arr = y_true.to_numpy()
        breakdown: dict[str, SetupBreakdown] = {}
        for col in setup_columns:
            mask = test_df[col].astype(bool).to_numpy()
            if not mask.any():
                continue  # no rows matched this setup in the test set -- skip rather than report empty/NaN stats
            name = col[len("setup_"):]
            sub_true, sub_pred = y_true_arr[mask], y_pred[mask]
            n_success, n_failure = int((sub_true == 1).sum()), int((sub_true == 0).sum())
            breakdown[name] = SetupBreakdown(
                setup_name=name,
                n_rows=int(mask.sum()),
                success_count=n_success,
                failure_count=n_failure,
                accuracy=float((sub_true == sub_pred).mean()),
                success_accuracy=float(((sub_true == 1) & (sub_pred == 1)).sum() / n_success) if n_success > 0 else None,
                failure_accuracy=float(((sub_true == 0) & (sub_pred == 0)).sum() / n_failure) if n_failure > 0 else None,
            )
        return breakdown


def _fmt_pct(value: float | None) -> str:
    if value is None or value != value:  # None or NaN
        return "N/A"
    return f"{value:.1%}"


def format_evaluation_report(report: EvaluationReport, top_n_features: int = 15) -> str:
    """Plain-text rendering of an EvaluationReport, shared by
    scripts/train_model.py (when --test is given) and
    scripts/evaluate_model.py, so both print an identical report rather
    than two scripts slowly drifting into slightly different formats."""
    lines = [
        f"Test set: {report.n_rows} rows, decision threshold: {report.threshold}",
        "",
        f"Overall accuracy:   {report.overall_accuracy:.1%}",
        f"Success accuracy (recall of actual SUCCESS rows): {_fmt_pct(report.success_accuracy)}",
        f"Failure accuracy (recall of actual FAILURE rows): {_fmt_pct(report.failure_accuracy)}",
        f"Success precision (of predicted SUCCESS, how many really were): {_fmt_pct(report.success_precision)}",
        f"Failure precision (of predicted FAILURE, how many really were): {_fmt_pct(report.failure_precision)}",
        "",
        "Confusion matrix:",
    ]
    cm = report.confusion_matrix
    lines.append(f"{'':22}{'predicted SUCCESS':>18}{'predicted FAILURE':>19}")
    lines.append(f"{'actual SUCCESS':22}{cm['true_success_pred_success']:>18}{cm['true_success_pred_failure']:>19}")
    lines.append(f"{'actual FAILURE':22}{cm['true_failure_pred_success']:>18}{cm['true_failure_pred_failure']:>19}")
    lines.append("")

    if report.setup_breakdown:
        lines.append("Per-setup breakdown:")
        for name, sb in sorted(report.setup_breakdown.items()):
            lines.append(
                f"  {name:<28} n={sb.n_rows:<5} accuracy={sb.accuracy:.1%}  "
                f"success_acc={_fmt_pct(sb.success_accuracy)}  failure_acc={_fmt_pct(sb.failure_accuracy)}"
            )
        lines.append("")

    if report.feature_importances:
        shown = min(top_n_features, len(report.feature_importances))
        lines.append(f"Top {shown} feature importances:")
        for name, value in list(report.feature_importances.items())[:top_n_features]:
            lines.append(f"  {name:<32} {value:.4f}")

    return "\n".join(lines)
