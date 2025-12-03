"""Feature engineering utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    rolling_window_steps: int = 5
    category_columns: Optional[List[str]] = None


class FeatureEngineer:
    """Generate numeric and categorical features for fraud detection."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.category_levels_: dict[str, pd.Index] | None = None
        self.category_frequency_: dict[str, pd.Series] | None = None

    def fit(self, df: pd.DataFrame) -> None:
        categories = self.config.category_columns or ["type"]
        self.category_levels_ = {}
        self.category_frequency_ = {}
        for col in categories:
            freq = df[col].fillna("UNK").value_counts(normalize=True)
            self.category_levels_[col] = freq.index
            self.category_frequency_[col] = freq

    def _rolling_counts(self, df: pd.DataFrame) -> pd.Series:
        window = self.config.rolling_window_steps
        df_sorted = df.sort_values("step")
        counts = (
            df_sorted.groupby("nameOrig", group_keys=False)
            .apply(lambda g: g.set_index("step")["amount"].rolling(window).count())
            .reset_index(level=0, drop=True)
        )
        return counts.reindex(df_sorted.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.category_levels_ is None or self.category_frequency_ is None:
            raise ValueError("FeatureEngineer must be fit before calling transform.")

        df = df.copy()
        df["step"] = df["step"].astype(int)

        df["origin_balance_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
        df["dest_balance_delta"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
        df["dest_is_new"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] > 0)).astype(int)

        df["step_gap"] = df.groupby("nameOrig")["step"].diff().fillna(0)
        df["rolling_txn_count"] = self._rolling_counts(df).reindex(df.index).fillna(0)

        for col in self.category_levels_:
            freq = self.category_frequency_[col]
            df[f"{col}_frequency"] = df[col].fillna("UNK").map(freq).fillna(0)

        feature_cols = [
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "origin_balance_delta",
            "dest_balance_delta",
            "dest_is_new",
            "step_gap",
            "rolling_txn_count",
        ] + [f"{col}_frequency" for col in self.category_levels_.keys()]

        return df, feature_cols
