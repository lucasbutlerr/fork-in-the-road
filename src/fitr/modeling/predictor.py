"""LivePredictor: the capstone of the pipeline. Given a ticker universe and
a trained model, scans for today's (or any given date's) breakout
candidates, computes their features, and returns a ranked list with
predicted success probability -- the "only config files should need
editing" goal, fully realized end to end.

Constructor-time validation: checks that the currently loaded
features.yaml + scanning_criteria.yaml actually produce every column the
model expects, and raises immediately with a clear message if not. This
specifically catches the common footgun of retraining on a changed
features.yaml but forgetting to point LivePredictor at the new model, or
loading an old model against a since-changed config -- without this check,
that mismatch would otherwise surface as a cryptic KeyError deep inside
the first prediction instead of a clear error before anything runs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from fitr.data.yf_proxy import NoDataInRangeError, TickerNotFoundError
from fitr.features.feature_engine import FeatureEngine
from fitr.modeling.trainer import TrainedModel
from fitr.scanning.scanner import BreakoutScanner

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    ticker: str
    as_of_date: pd.Timestamp
    rank: int
    predicted_probability: float  # P(SUCCESS)
    matched_setups: list[str]
    scan_metrics: dict[str, float]  # the scanning_criteria.yaml metrics that flagged this candidate
    features: dict[str, float]  # the full features.yaml feature vector fed to the model


class LivePredictor:
    def __init__(self, scanner: BreakoutScanner, feature_engine: FeatureEngine, trained_model: TrainedModel):
        self._scanner = scanner
        self._feature_engine = feature_engine
        self._model = trained_model

        available = set(feature_engine.feature_names) | {f"setup_{n}" for n in scanner.setup_names}
        missing = sorted(set(trained_model.feature_columns) - available)
        if missing:
            raise ValueError(
                f"Loaded model expects feature(s) not produced by the current features.yaml / "
                f"scanning_criteria.yaml: {missing}. This usually means the model was trained "
                f"against a different config than what's loaded now -- retrain against the "
                f"current configs, or point --model at the matching one."
            )

    def predict(self, universe: list[str], as_of_date=None) -> list[PredictionResult]:
        """Scans `universe` for candidates as of as_of_date (default:
        today), computes each candidate's full feature vector, and returns
        every candidate ranked by predicted P(SUCCESS), highest first.
        Candidates the scanner flags but whose full feature set can't be
        computed (rare -- e.g. features.yaml needs a longer lookback than
        scanning_criteria.yaml's metrics do) are skipped with a warning
        rather than failing the whole run."""
        if as_of_date is None:
            as_of_date = pd.Timestamp.today().normalize()
        display_date = pd.Timestamp(as_of_date).normalize()

        scan_results = self._scanner.scan(universe, as_of_date)
        predictions: list[PredictionResult] = []

        for result in scan_results:
            try:
                features = self._feature_engine.compute(result.ticker, as_of_date)
            except (TickerNotFoundError, NoDataInRangeError) as exc:
                logger.warning("Skipping %s: could not compute full feature set (%s)", result.ticker, exc)
                continue

            row = dict(features)
            for setup_name in self._scanner.setup_names:
                row[f"setup_{setup_name}"] = setup_name in result.matched_setups

            probability = float(self._model.predict_proba_success(pd.DataFrame([row]))[0])

            predictions.append(
                PredictionResult(
                    ticker=result.ticker,
                    as_of_date=display_date,
                    rank=0,  # filled in below, after sorting
                    predicted_probability=probability,
                    matched_setups=result.matched_setups,
                    scan_metrics=result.metrics,
                    features=features,
                )
            )

        predictions.sort(key=lambda p: p.predicted_probability, reverse=True)
        for i, p in enumerate(predictions, start=1):
            p.rank = i

        return predictions


def format_predictions(
    predictions: list[PredictionResult], trained_model: TrainedModel, top_n: int = 20, top_features: int = 5
) -> str:
    """Plain-text rendering of a ranked prediction list -- shown in full
    (rank, probability, matched setups, scan metrics that drove the flag,
    and this candidate's own values for the model's most important
    features) for the top `top_n`, with a compact summary line for the
    rest. Shared by scripts/scan_live.py so the script stays thin."""
    if not predictions:
        return "No candidates matched any setup."

    global_importances = trained_model.feature_importances()
    top_feature_names = sorted(global_importances, key=global_importances.get, reverse=True)[:top_features]

    lines = [f"{len(predictions)} candidate(s), ranked by predicted P(success):\n"]

    for p in predictions[:top_n]:
        lines.append(f"#{p.rank}  {p.ticker}  --  P(success) = {p.predicted_probability:.1%}")
        lines.append(f"     Matched setups: {', '.join(p.matched_setups)}")
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in p.scan_metrics.items())
        lines.append(f"     Scan metrics: {metrics_str}")
        if top_feature_names:
            feat_str = ", ".join(f"{name}={p.features.get(name, float('nan')):.4f}" for name in top_feature_names)
            lines.append(f"     Top model features (globally most important): {feat_str}")
        lines.append("")

    if len(predictions) > top_n:
        remainder = predictions[top_n:]
        tickers = ", ".join(f"{p.ticker} ({p.predicted_probability:.1%})" for p in remainder)
        lines.append(f"...and {len(remainder)} more: {tickers}")

    return "\n".join(lines)
