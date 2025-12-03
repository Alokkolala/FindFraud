"""Score transactions with combined ML and rule-based logic."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data_loader import TransactionLoader
from .features import FeatureEngineer, FeatureConfig
from .model import AnomalyDetector, ModelTrainer, ModelConfig
from .rules import RuleEngine, RuleConfig


class ScoringPipeline:
    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        model_config: ModelConfig | None = None,
        rule_config: RuleConfig | None = None,
    ) -> None:
        self.loader = TransactionLoader()
        self.feature_engineer = FeatureEngineer(config=feature_config)
        self.model_trainer = ModelTrainer(config=model_config)
        self.rule_engine = RuleEngine(config=rule_config)

    def train(self, csv_path: str, model_output: str) -> dict:
        raw = self.loader.load(csv_path)
        self.feature_engineer.fit(raw)
        enriched, feature_cols = self.feature_engineer.transform(raw)
        model_bundle = self.model_trainer.train(enriched, feature_cols)
        model_bundle["category_levels"] = self.feature_engineer.category_levels_
        model_bundle["category_frequency"] = self.feature_engineer.category_frequency_
        self.model_trainer.save(model_bundle, model_output)
        return model_bundle

    def _restore_feature_engineer(self, model_bundle: dict) -> None:
        category_levels = model_bundle.get("category_levels")
        category_freq = model_bundle.get("category_frequency")
        if category_levels and category_freq:
            self.feature_engineer.category_levels_ = category_levels
            self.feature_engineer.category_frequency_ = category_freq

    def score(
        self,
        csv_path: str,
        model_path: str,
        output_csv: str,
        html_report: Optional[str] = None,
        pdf_report: Optional[str] = None,
    ) -> pd.DataFrame:
        model_bundle = self.model_trainer.load(model_path)
        self._restore_feature_engineer(model_bundle)
        raw = self.loader.load(csv_path)
        enriched, feature_cols = self.feature_engineer.transform(raw)
        detector = AnomalyDetector(model_bundle)
        ml_scores, shap_text = detector.score(enriched)

        rule_results = self.rule_engine.evaluate(enriched)
        rule_text = self.rule_engine.summarize(rule_results)
        rule_scores = np.array([1.0 if r else 0.0 for r in rule_results])

        combined_score = 0.7 * ml_scores + 0.3 * rule_scores
        is_suspicious = combined_score >= 0.6
        explanation = []
        for shap_desc, rule_desc in zip(shap_text, rule_text):
            parts = []
            if shap_desc:
                parts.append(f"Top drivers: {shap_desc}")
            if rule_desc:
                parts.append(f"Rules: {rule_desc}")
            explanation.append(" | ".join(parts))

        output = pd.DataFrame(
            {
                "transaction_id": raw["transaction_id"],
                "fraud_score": combined_score,
                "is_suspicious": is_suspicious,
                "explanation": explanation,
            }
        )
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_csv, index=False)

        from .report import ReportBuilder

        if html_report:
            builder = ReportBuilder()
            builder.build_html(raw, output, html_report)
            if pdf_report:
                builder.html_to_pdf(html_report, pdf_report)

        return output
