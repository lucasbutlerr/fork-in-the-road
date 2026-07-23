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
    success_accuracy: float | None  # recall: of actual SUCCESS rows, fraction correctly caught. None if n_success==0.
    failure_accuracy: float | None  # recall: of actual FAILURE rows, fraction correctly caught. None if n_failure==0.
    success_precision: float  # of rows PREDICTED success, fraction that really were -- the number that
    # actually matters for "are this setup's signals profitable", not success_accuracy/recall. NaN if the
    # model never predicted SUCCESS for this setup at all (0 predicted positives).
    failure_precision: float  # same idea, for predicted FAILURE. NaN if never predicted.


@dataclass
class CalibrationBucket:
    bucket_low: float
    bucket_high: float
    n_rows: int
    mean_predicted_probability: float
    actual_success_rate: float


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
    calibration_curve: list[CalibrationBucket] = field(default_factory=list)


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
            calibration_curve=self._calibration_curve(probs, y_true.to_numpy()),
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
        return trained.feature_importances()

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
            tp = int(((sub_true == 1) & (sub_pred == 1)).sum())
            fn = int(((sub_true == 1) & (sub_pred == 0)).sum())
            fp = int(((sub_true == 0) & (sub_pred == 1)).sum())
            tn = int(((sub_true == 0) & (sub_pred == 0)).sum())
            breakdown[name] = SetupBreakdown(
                setup_name=name,
                n_rows=int(mask.sum()),
                success_count=n_success,
                failure_count=n_failure,
                accuracy=float((sub_true == sub_pred).mean()),
                success_accuracy=float(tp / n_success) if n_success > 0 else None,
                failure_accuracy=float(tn / n_failure) if n_failure > 0 else None,
                success_precision=ModelEvaluator._safe_divide(tp, tp + fp),
                failure_precision=ModelEvaluator._safe_divide(tn, tn + fn),
            )
        return breakdown

    @staticmethod
    def _calibration_curve(probs: np.ndarray, y_true: np.ndarray, n_buckets: int = 10) -> list[CalibrationBucket]:
        """Buckets predictions into n_buckets equal-width probability
        ranges (deciles by default) and compares the mean predicted
        probability in each bucket to the actual observed success rate --
        a well-calibrated model should show these two numbers close
        together in every bucket. This is what confirms calibration
        actually worked, rather than assuming it did just because
        calibration_method was set. Empty buckets are skipped rather than
        reported with misleading NaN/zero stats."""
        edges = np.linspace(0, 1, n_buckets + 1)
        buckets = []
        for i in range(n_buckets):
            lo, hi = edges[i], edges[i + 1]
            mask = (probs >= lo) & (probs <= hi) if i == n_buckets - 1 else (probs >= lo) & (probs < hi)
            if not mask.any():
                continue
            buckets.append(
                CalibrationBucket(
                    bucket_low=float(lo),
                    bucket_high=float(hi),
                    n_rows=int(mask.sum()),
                    mean_predicted_probability=float(probs[mask].mean()),
                    actual_success_rate=float(y_true[mask].mean()),
                )
            )
        return buckets

    def predictions_detail(self, trained: TrainedModel, test_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Row-level predictions: every test row's true label, predicted
        probability, predicted class at the given threshold, and whether
        the prediction was correct -- sorted by predicted probability,
        highest first. This is what a trade-tracking tool or spot-check of
        specific misses would want; evaluate() above only summarizes."""
        if test_df.empty:
            raise ValueError("Cannot produce prediction detail on an empty test set.")

        y_true_labels = test_df[trained.label_column]
        y_true = y_true_labels.map({SUCCESS: 1, FAILURE: 0})
        if y_true.isna().any():
            bad = sorted(test_df.loc[y_true.isna(), trained.label_column].unique())
            raise ValueError(f"Unexpected label value(s) {bad} in column '{trained.label_column}'.")

        probs = trained.predict_proba_success(test_df)
        predicted_labels = [SUCCESS if p >= threshold else FAILURE for p in probs]
        correct = [pl == tl for pl, tl in zip(predicted_labels, y_true_labels)]

        detail = pd.DataFrame(
            {
                "predicted_probability": probs,
                "predicted_label": predicted_labels,
                "true_label": y_true_labels.to_numpy(),
                "correct": correct,
            }
        )
        id_cols = [c for c in ("ticker", "date") if c in test_df.columns]
        if id_cols:
            detail = pd.concat([test_df[id_cols].reset_index(drop=True), detail], axis=1)
        return detail.sort_values("predicted_probability", ascending=False).reset_index(drop=True)


def _fmt_pct(value: float | None) -> str:
    if value is None or value != value:  # None or NaN
        return "N/A"
    return f"{value:.1%}"


def format_evaluation_report(report: EvaluationReport, top_n_features: int = 15, bottom_n_features: int = 10) -> str:
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
        lines.append(
            "Per-setup breakdown (base_rate = actual SUCCESS fraction within this setup, independent of the "
            "model -- a low base_rate is a setup-definition problem; low precision/accuracy despite a "
            "healthy base_rate is a modeling problem specific to that setup):"
        )
        for name, sb in sorted(report.setup_breakdown.items()):
            base_rate = sb.success_count / sb.n_rows if sb.n_rows > 0 else float("nan")
            lines.append(
                f"  {name:<28} n={sb.n_rows:<5} base_rate={base_rate:.1%}  "
                f"(success={sb.success_count}, failure={sb.failure_count})"
            )
            lines.append(
                f"  {'':<28} accuracy={sb.accuracy:.1%}  "
                f"success_recall={_fmt_pct(sb.success_accuracy)}  failure_recall={_fmt_pct(sb.failure_accuracy)}  "
                f"success_precision={_fmt_pct(sb.success_precision)}  failure_precision={_fmt_pct(sb.failure_precision)}"
            )
        lines.append("")

    if report.feature_importances:
        items = list(report.feature_importances.items())
        shown = min(top_n_features, len(items))
        lines.append(f"Top {shown} feature importances:")
        for name, value in items[:top_n_features]:
            lines.append(f"  {name:<32} {value:.4f}")
        lines.append("")

        # Least-important features -- useful when deciding what to prune
        # from features.yaml. Only shown as a distinct bottom section (not
        # just "scroll to the end of the full list") when there's enough
        # daylight between top and bottom that they wouldn't already
        # overlap in a single top-N listing.
        remaining = items[top_n_features:]
        if remaining:
            shown_bottom = min(bottom_n_features, len(remaining))
            lines.append(f"Bottom {shown_bottom} feature importances (candidates for pruning from features.yaml):")
            for name, value in remaining[-bottom_n_features:]:
                lines.append(f"  {name:<32} {value:.4f}")

    if report.calibration_curve:
        lines.append(
            "Calibration (mean predicted probability vs. actual success rate per bucket -- a "
            "well-calibrated model shows these close together; a large gap means the raw probability "
            "value shouldn't be trusted directly, e.g. for position sizing, even if ranking/threshold "
            "decisions using it are still fine):"
        )
        lines.append(f"  {'bucket':<12}{'n':>6}{'mean predicted':>17}{'actual rate':>15}{'gap':>9}")
        for b in report.calibration_curve:
            gap = b.mean_predicted_probability - b.actual_success_rate
            bucket_label = f"[{b.bucket_low:.0%}-{b.bucket_high:.0%})"
            lines.append(
                f"  {bucket_label:<12}{b.n_rows:>6}{b.mean_predicted_probability:>17.1%}"
                f"{b.actual_success_rate:>15.1%}{gap:>+9.1%}"
            )
        lines.append("")

    return "\n".join(lines)
