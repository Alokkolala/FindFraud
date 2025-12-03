"""Graph construction utilities for transaction networks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import warnings

import importlib.util
import pandas as pd


def _require_torch() -> None:
    if importlib.util.find_spec("torch") is None:
        raise ImportError("PyTorch is required for graph utilities. Install the optional 'graph' dependencies.")
    if importlib.util.find_spec("torch_geometric") is None:
        raise ImportError(
            "PyTorch Geometric is required for graph utilities. Install the optional 'graph' dependencies."
        )


@dataclass
class GraphBuilderConfig:
    """Configuration for aggregating transactions into a graph."""

    window_size: int = 24
    min_edge_count: int = 1


@dataclass
class GraphArtifacts:
    """Serialized components that describe the constructed graph."""

    node_mapping: Dict[str, int]
    node_features: pd.DataFrame
    edge_index: "torch.Tensor"
    edge_attr: "torch.Tensor"
    feature_names: List[str]
    transactions: pd.DataFrame
    node_labels: Optional["torch.Tensor"] = None
    min_edge_count_used: int | None = None

    def as_data(self) -> "torch_geometric.data.Data":
        _require_torch()
        import torch
        from torch_geometric.data import Data

        x = torch.tensor(self.node_features[self.feature_names].values, dtype=torch.float)
        data_kwargs = {"x": x, "edge_index": self.edge_index}
        if self.edge_attr is not None:
            data_kwargs["edge_attr"] = self.edge_attr
        if self.node_labels is not None:
            data_kwargs["y"] = self.node_labels
        return Data(**data_kwargs)

    def to_networkx(
        self,
        min_txn_count: int = 1,
        min_total_amount: float | None = None,
        top_n_nodes: int | None = None,
    ) -> "networkx.Graph":
        """Convert the graph to a NetworkX graph for visualization or exploration.

        Parameters
        ----------
        min_txn_count:
            Drop edges with fewer transactions than this threshold before plotting.
        min_total_amount:
            Drop edges whose aggregated amount is below this value.
        top_n_nodes:
            Keep only the highest-degree nodes to reduce clutter (helpful for dense stars).
        """

        _require_torch()
        import torch
        from torch_geometric.utils import to_networkx

        data = self.as_data()
        edge_attr = getattr(data, "edge_attr", None)

        if edge_attr is not None:
            mask = torch.ones(edge_attr.size(0), dtype=torch.bool)
            if min_txn_count > 1:
                mask &= edge_attr[:, 1] >= float(min_txn_count)
            if min_total_amount is not None:
                mask &= edge_attr[:, 0] >= float(min_total_amount)

            if mask.any():
                data.edge_index = data.edge_index[:, mask]
                data.edge_attr = edge_attr[mask]

        graph = to_networkx(data, node_attrs=[], edge_attrs=["edge_attr"], to_undirected=True)

        if top_n_nodes is not None and graph.number_of_nodes() > 0:
            top_n_nodes = min(top_n_nodes, graph.number_of_nodes())
            ranked = sorted(graph.degree, key=lambda item: item[1], reverse=True)
            keep_nodes = {node for node, _ in ranked[:top_n_nodes]}
            graph = graph.subgraph(keep_nodes).copy()

        reverse_mapping = {idx: name for name, idx in self.node_mapping.items()}
        for node_id in graph.nodes:
            graph.nodes[node_id]["account_name"] = reverse_mapping.get(node_id, str(node_id))

        return graph

    def to_metadata(self) -> dict:
        return {
            "node_mapping": self.node_mapping,
            "feature_names": self.feature_names,
            "min_edge_count_used": self.min_edge_count_used,
        }


class GraphBuilder:
    """Construct account-to-account graphs from transactions with rich edge/node features."""

    def __init__(self, config: GraphBuilderConfig | None = None) -> None:
        self.config = config or GraphBuilderConfig()

    def build(self, df: pd.DataFrame) -> GraphArtifacts:
        _require_torch()
        import torch

        transactions = df.copy()
        transactions["step"] = transactions["step"].astype(int)
        transactions = transactions.sort_values("step").reset_index(drop=True)

        accounts = pd.Index(pd.unique(transactions[["nameOrig", "nameDest"]].values.ravel()))
        node_mapping = {name: idx for idx, name in enumerate(accounts)}

        outgoing_sum = transactions.groupby("nameOrig")["amount"].sum()
        incoming_sum = transactions.groupby("nameDest")["amount"].sum()
        outgoing_count = transactions.groupby("nameOrig")["amount"].count()
        incoming_count = transactions.groupby("nameDest")["amount"].count()

        recent_outgoing = self._window_aggregate(transactions, "nameOrig")
        recent_incoming = self._window_aggregate(transactions, "nameDest")

        outgoing_gap = self._mean_gap(transactions, "nameOrig")
        incoming_gap = self._mean_gap(transactions, "nameDest")

        node_features = pd.DataFrame({"node": accounts})
        node_features["outgoing_amount_sum"] = node_features["node"].map(outgoing_sum).fillna(0)
        node_features["incoming_amount_sum"] = node_features["node"].map(incoming_sum).fillna(0)
        node_features["outgoing_txn_count"] = node_features["node"].map(outgoing_count).fillna(0)
        node_features["incoming_txn_count"] = node_features["node"].map(incoming_count).fillna(0)
        node_features["recent_outgoing_amount"] = node_features["node"].map(recent_outgoing).fillna(0)
        node_features["recent_incoming_amount"] = node_features["node"].map(recent_incoming).fillna(0)
        node_features["mean_outgoing_step_delta"] = node_features["node"].map(outgoing_gap).fillna(0)
        node_features["mean_incoming_step_delta"] = node_features["node"].map(incoming_gap).fillna(0)

        feature_cols = [
            "outgoing_amount_sum",
            "incoming_amount_sum",
            "outgoing_txn_count",
            "incoming_txn_count",
            "recent_outgoing_amount",
            "recent_incoming_amount",
            "mean_outgoing_step_delta",
            "mean_incoming_step_delta",
        ]

        edge_df = self._edge_features(transactions)
        edge_df = edge_df[edge_df["txn_count"] >= self.config.min_edge_count]
        effective_min_count = self.config.min_edge_count
        if edge_df.empty and self.config.min_edge_count > 1:
            warnings.warn(
                "All edges were filtered out by min_edge_count; falling back to 1 to keep sparse graphs visible.",
                RuntimeWarning,
            )
            edge_df = self._edge_features(transactions)
            effective_min_count = 1
        src = torch.tensor(edge_df["src_id"].values, dtype=torch.long)
        dst = torch.tensor(edge_df["dst_id"].values, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.tensor(
            edge_df[["total_amount", "txn_count", "mean_step_delta", "last_seen_step"]].values, dtype=torch.float
        )

        labels = None
        if "isFraud" in transactions.columns:
            fraud_by_orig = transactions.groupby("nameOrig")["isFraud"].max()
            labels = torch.tensor(node_features["node"].map(fraud_by_orig).fillna(0).values, dtype=torch.float)

        return GraphArtifacts(
            node_mapping=node_mapping,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            feature_names=feature_cols,
            node_labels=labels,
            transactions=transactions,
            min_edge_count_used=effective_min_count,
        )

    def _window_aggregate(self, df: pd.DataFrame, column: str) -> pd.Series:
        df_sorted = df.sort_values("step")
        window = self.config.window_size
        rolled = (
            df_sorted.set_index("step")
            .groupby(column)["amount"]
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        last_seen = rolled.groupby(df_sorted[column]).last()
        return last_seen

    def _mean_gap(self, df: pd.DataFrame, column: str) -> pd.Series:
        def gap(series: pd.Series) -> float:
            if len(series) < 2:
                return float(self.config.window_size)
            return float(series.sort_values().diff().dropna().mean())

        return df.groupby(column)["step"].apply(gap)

    def _edge_features(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = df.groupby(["nameOrig", "nameDest"])
        edge = grouped.agg(total_amount=("amount", "sum"), txn_count=("amount", "count"), last_seen_step=("step", "max"))
        edge["mean_step_delta"] = grouped["step"].apply(self._edge_gap)
        edge = edge.reset_index()

        node_lookup = {name: idx for idx, name in enumerate(pd.unique(df[["nameOrig", "nameDest"]].values.ravel()))}
        edge["src_id"] = edge["nameOrig"].map(node_lookup)
        edge["dst_id"] = edge["nameDest"].map(node_lookup)
        return edge

    def _edge_gap(self, series: pd.Series) -> float:
        ordered = series.sort_values().diff().dropna()
        if ordered.empty:
            return 0.0
        return float(ordered.mean())


__all__ = ["GraphBuilder", "GraphBuilderConfig", "GraphArtifacts"]
