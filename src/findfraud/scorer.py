"""Score transactions with combined ML and rule-based logic."""
from __future__ import annotations

from dataclasses import asdict
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
        model_choice: str = "tabular",
        graph_config: "GraphBuilderConfig | None" = None,
        graph_model_config: "GraphModelConfig | None" = None,
    ) -> None:
        self.loader = TransactionLoader()
        self.feature_engineer = FeatureEngineer(config=feature_config)
        self.model_trainer = ModelTrainer(config=model_config)
        self.rule_engine = RuleEngine(config=rule_config)
        self.model_choice = model_choice

        if model_choice == "gnn":
            from .graph_builder import GraphBuilder, GraphBuilderConfig
            from .graph_model import GraphModelConfig, GraphModelTrainer

            self.graph_builder = GraphBuilder(config=graph_config or GraphBuilderConfig())
            self.graph_model_trainer = GraphModelTrainer(config=graph_model_config or GraphModelConfig())
        else:
            self.graph_builder = None
            self.graph_model_trainer = None

    def train(self, csv_path: str, model_output: str, graph_output: str | None = None) -> dict:
        raw = self.loader.load(csv_path)

        if self.model_choice == "gnn":
            if self.graph_builder is None or self.graph_model_trainer is None:
                raise ValueError("Graph components are not initialized.")
            artifacts = self.graph_builder.build(raw)
            model_bundle = self.graph_model_trainer.train(artifacts)
            model_bundle["builder_config"] = asdict(self.graph_builder.config)
            model_bundle["node_mapping"] = artifacts.node_mapping
            self.graph_model_trainer.save(model_bundle, model_output)
            if graph_output:
                self.graph_model_trainer.save_artifacts(artifacts, graph_output)
            return model_bundle

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

    def _augment_rule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        augmented = df.copy()
        augmented["origin_balance_delta"] = augmented["oldbalanceOrg"] - augmented["newbalanceOrig"] - augmented["amount"]
        augmented["dest_is_new"] = ((augmented["oldbalanceDest"] == 0) & (augmented["newbalanceDest"] > 0)).astype(int)
        return augmented

    def score_frame(
        self,
        raw: pd.DataFrame,
        model_path: str,
        graph_output: Optional[str] = None,
        profile_output: bool = False,
        model_bundle: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict[str, object], dict[str, object] | None, pd.DataFrame | None]:
        training_details: dict[str, object] = {"model_type": self.model_choice, "model_path": Path(model_path).resolve()}
        graph_details: dict[str, object] | None = None
        profiles: pd.DataFrame | None = None

        if self.model_choice == "gnn":
            if self.graph_builder is None or self.graph_model_trainer is None:
                raise ValueError("Graph components are not initialized.")
            from .graph_builder import GraphBuilderConfig
            from .graph_model import GraphAnomalyDetector

            model_bundle = model_bundle or self.graph_model_trainer.load(model_path)
            builder_cfg = model_bundle.get("builder_config")
            if builder_cfg:
                self.graph_builder.config = GraphBuilderConfig(**builder_cfg)
            artifacts = self.graph_builder.build(raw)
            detector = GraphAnomalyDetector(model_bundle)
            ml_scores = np.array(detector.score(artifacts))
            shap_text = [f"Graph risk={score:.3f}" for score in ml_scores]
            if graph_output:
                Path(graph_output).parent.mkdir(parents=True, exist_ok=True)
                self.graph_model_trainer.save_artifacts(artifacts, graph_output)
            graph_details = {
                "nodes": len(artifacts.node_mapping),
                "edges": int(artifacts.edge_index.size(1)),
                "min_edge_count_used": artifacts.min_edge_count_used,
                "graph_path": Path(graph_output).resolve() if graph_output else "<not saved>",
            }
            edge_table = (
                artifacts.transactions.groupby(["nameOrig", "nameDest"]).agg(
                    txn_count=("amount", "count"), total_amount=("amount", "sum")
                )
            )
            edge_table = edge_table.sort_values(["txn_count", "total_amount"], ascending=False).reset_index()
            graph_details["top_edges"] = edge_table.head(20)
            enriched = artifacts.transactions
            rule_frame = self._augment_rule_features(raw)
            training_details["gnn_config"] = model_bundle.get("config")
            training_details["graph_features"] = model_bundle.get("feature_names")
            training_details["graph_builder"] = asdict(self.graph_builder.config)
        else:
            model_bundle = model_bundle or self.model_trainer.load(model_path)
            self._restore_feature_engineer(model_bundle)
            enriched, feature_cols = self.feature_engineer.transform(raw)
            detector = AnomalyDetector(model_bundle)
            ml_scores, shap_text = detector.score(enriched)
            rule_frame = enriched
            training_details["feature_columns"] = feature_cols
            training_details["contamination"] = getattr(self.model_trainer.config, "contamination", None)

        rule_results = self.rule_engine.evaluate(rule_frame)
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

        if profile_output:
            profiles = self._classify_profiles(raw, combined_score, is_suspicious)

        return output, training_details, graph_details, profiles

    def score(
        self,
        csv_path: str,
        model_path: str,
        output_csv: str,
        html_report: Optional[str] = None,
        pdf_report: Optional[str] = None,
        graph_output: Optional[str] = None,
        profile_output: Optional[str] = None,
    ) -> pd.DataFrame:
        raw = self.loader.load(csv_path)

        output, training_details, graph_details, profiles = self.score_frame(
            raw,
            model_path,
            graph_output=graph_output,
            profile_output=bool(profile_output),
        )

        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_csv, index=False)

        if profile_output and profiles is not None:
            Path(profile_output).parent.mkdir(parents=True, exist_ok=True)
            profiles.to_csv(profile_output, index=False)

        from .report import ReportBuilder

        if html_report:
            builder = ReportBuilder()
            builder.build_html(
                raw,
                output,
                html_report,
                training_info=training_details,
                graph_info=graph_details,
                profiles=profiles,
            )
            if pdf_report:
                builder.html_to_pdf(html_report, pdf_report)

        return output

    def _classify_profiles(
        self,
        df: pd.DataFrame,
        fraud_scores: np.ndarray,
        is_suspicious: np.ndarray,
    ) -> pd.DataFrame:
        """Aggregate transaction scores into account-level risk profiles."""

        role_rows = [
            pd.DataFrame(
                {
                    "account": df["nameOrig"],
                    "role": "origin",
                    "amount": df["amount"],
                    "fraud_score": fraud_scores,
                    "is_suspicious": is_suspicious,
                }
            ),
            pd.DataFrame(
                {
                    "account": df["nameDest"],
                    "role": "destination",
                    "amount": df["amount"],
                    "fraud_score": fraud_scores,
                    "is_suspicious": is_suspicious,
                }
            ),
        ]
        merged = pd.concat(role_rows, ignore_index=True)

        grouped = merged.groupby("account")
        summary = grouped.agg(
            total_transactions=("fraud_score", "size"),
            suspicious_transactions=("is_suspicious", "sum"),
            total_amount=("amount", "sum"),
            max_score=("fraud_score", "max"),
            mean_score=("fraud_score", "mean"),
        ).reset_index()

        def label_row(row: pd.Series) -> str:
            if row["max_score"] >= 0.8 or row["suspicious_transactions"] > 0:
                return "high_risk"
            if row["mean_score"] >= 0.5:
                return "elevated"
            return "low_risk"

        summary["risk_level"] = summary.apply(label_row, axis=1)
        risk_rank = {"high_risk": 0, "elevated": 1, "low_risk": 2}
        summary["risk_rank"] = summary["risk_level"].map(risk_rank).fillna(3)
        summary = summary.sort_values(["risk_rank", "max_score"], ascending=[True, False])
        return summary[
            [
                "account",
                "risk_level",
                "max_score",
                "mean_score",
                "total_transactions",
                "suspicious_transactions",
                "total_amount",
            ]
        ]
