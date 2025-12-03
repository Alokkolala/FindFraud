"""Configurable rule engine for transaction risk flags."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class RuleConfig:
    high_amount_threshold: float = 200000.0
    rapid_steps: int = 3
    rapid_count: int = 5
    balance_mismatch_tolerance: float = 1e-3
    new_destination_amount: float = 50000.0


@dataclass
class RuleResult:
    name: str
    triggered: bool
    detail: str


class RuleEngine:
    def __init__(self, config: RuleConfig | None = None) -> None:
        self.config = config or RuleConfig()

    def evaluate(self, df: pd.DataFrame) -> List[List[RuleResult]]:
        results: List[List[RuleResult]] = []
        for idx, row in df.iterrows():
            triggered: List[RuleResult] = []

            if row.get("amount", 0) >= self.config.high_amount_threshold:
                triggered.append(
                    RuleResult(
                        name="high_amount",
                        triggered=True,
                        detail=f"Amount {row['amount']:.2f} >= {self.config.high_amount_threshold}",
                    )
                )

            window_mask = (df["nameOrig"] == row["nameOrig"]) & (df["step"] >= row["step"] - self.config.rapid_steps)
            recent_count = window_mask.sum()
            if recent_count >= self.config.rapid_count:
                triggered.append(
                    RuleResult(
                        name="rapid_sequence",
                        triggered=True,
                        detail=f"{recent_count} txns within {self.config.rapid_steps} steps",
                    )
                )

            origin_delta = abs(row.get("origin_balance_delta", 0))
            if origin_delta > self.config.balance_mismatch_tolerance:
                triggered.append(
                    RuleResult(
                        name="origin_balance_mismatch",
                        triggered=True,
                        detail=f"Origin balance off by {origin_delta:.2f}",
                    )
                )

            if row.get("dest_is_new", 0) == 1 and row.get("amount", 0) >= self.config.new_destination_amount:
                triggered.append(
                    RuleResult(
                        name="new_destination_large_amount",
                        triggered=True,
                        detail=f"New destination received {row['amount']:.2f}",
                    )
                )

            results.append(triggered)
        return results

    def summarize(self, rule_results: List[List[RuleResult]]) -> List[str]:
        summaries: List[str] = []
        for result in rule_results:
            if not result:
                summaries.append("")
                continue
            parts = [f"{r.name}: {r.detail}" for r in result if r.triggered]
            summaries.append("; ".join(parts))
        return summaries
