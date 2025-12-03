"""Model training and inference utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import shap


@dataclass
class ModelConfig:
    contamination: float = 0.05
    random_state: int = 42
    n_estimators: int = 200


class ModelTrainer:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

    def train(self, features: pd.DataFrame, feature_columns: Iterable[str]) -> dict:
        X = features[list(feature_columns)].fillna(0)
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    IsolationForest(
                        contamination=self.config.contamination,
                        n_estimators=self.config.n_estimators,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipeline.fit(X)
        background = shap.sample(X, nsamples=min(50, len(X))).astype(float)
        return {"pipeline": pipeline, "feature_columns": list(feature_columns), "background": background}

    def save(self, model_bundle: dict, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_bundle, path)

    def load(self, path: str | Path) -> dict:
        return joblib.load(path)


class AnomalyDetector:
    def __init__(self, model_bundle: dict) -> None:
        self.pipeline: Pipeline = model_bundle["pipeline"]
        self.feature_columns: List[str] = model_bundle["feature_columns"]
        self.background: pd.DataFrame | None = model_bundle.get("background")
        model = self.pipeline
        if self.background is not None:
            self.explainer = shap.Explainer(lambda x: -model.decision_function(x), self.background)
        else:
            self.explainer = None

    def score(self, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        X = features[self.feature_columns].fillna(0).astype(float)
        anomaly_raw = -self.pipeline.decision_function(X)
        raw_range = np.ptp(anomaly_raw)
        score = (anomaly_raw - anomaly_raw.min()) / (raw_range + 1e-9)
        explanations: list[str] = ["" for _ in range(len(features))]
        if self.explainer is not None:
            explain_count = min(50, len(X))
            shap_values = self.explainer(X.iloc[:explain_count])
            global_importance = np.abs(shap_values.values).mean(axis=0)
            global_top_idx = global_importance.argsort()[-3:][::-1]
            global_top = ", ".join(
                f"{self.feature_columns[j]} ({global_importance[j]:.3f})" for j in global_top_idx
            )

            for i, sample_values in enumerate(shap_values.values):
                feature_importance = np.abs(sample_values)
                top_idx = feature_importance.argsort()[-3:][::-1]
                top_features = [
                    f"{self.feature_columns[j]} ({sample_values[j]:.3f})" for j in top_idx
                ]
                explanations[i] = ", ".join(top_features)

            for i in range(explain_count, len(explanations)):
                explanations[i] = global_top
        return score, explanations
