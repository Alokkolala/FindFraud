"""Utilities for loading and validating transaction data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class TransactionSchema:
    """Schema description for transaction CSV files."""

    dtypes: Dict[str, str]
    required_columns: List[str]


DEFAULT_SCHEMA = TransactionSchema(
    dtypes={
        "step": "int64",
        "type": "string",
        "amount": "float",
        "nameOrig": "string",
        "oldbalanceOrg": "float",
        "newbalanceOrig": "float",
        "nameDest": "string",
        "oldbalanceDest": "float",
        "newbalanceDest": "float",
        "isFraud": "float",
        "isFlaggedFraud": "float",
    },
    required_columns=[
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
    ],
)


class TransactionLoader:
    """Load and preprocess transaction CSV files."""

    def __init__(self, schema: TransactionSchema = DEFAULT_SCHEMA) -> None:
        self.schema = schema

    def load(self, path: str) -> pd.DataFrame:
        """Load a transaction CSV and enforce schema."""
        df = pd.read_csv(path, dtype=self.schema.dtypes)
        return self.validate_frame(df)

    def validate_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and coerce an in-memory transaction frame."""

        missing = [c for c in self.schema.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.dropna(subset=self.schema.required_columns).copy()

        for col, dtype in self.schema.dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        df = df.sort_values(["step", "nameOrig"]).reset_index(drop=True)

        if "transaction_id" not in df.columns:
            df.insert(0, "transaction_id", df.apply(lambda r: f"{r['nameOrig']}_{int(r['step'])}", axis=1))

        return df
