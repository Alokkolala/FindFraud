"""Graph neural network model definitions and training utilities."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import inspect
import importlib.util


def _require_torch() -> None:
    if importlib.util.find_spec("torch") is None:
        raise ImportError("PyTorch is required for graph models. Install the optional 'graph' dependencies.")
    if importlib.util.find_spec("torch_geometric") is None:
        raise ImportError(
            "PyTorch Geometric is required for graph models. Install the optional 'graph' dependencies."
        )


_require_torch()
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


@dataclass
class GraphModelConfig:
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 0.005
    weight_decay: float = 1e-4
    epochs: int = 30
    use_cuda: bool = False


class FraudGNN(nn.Module):
    """Graph neural network for node-level fraud scoring."""

    def __init__(self, input_dim: int, config: GraphModelConfig) -> None:
        super().__init__()
        self.config = config
        layers = []
        in_channels = input_dim
        for _ in range(config.num_layers - 1):
            layers.append(SAGEConv(in_channels, config.hidden_channels))
            in_channels = config.hidden_channels
        layers.append(SAGEConv(in_channels, config.hidden_channels))
        self.layers = layers
        self.classifier = None
        self.dropout = None
        self._build_modules()

    def _build_modules(self) -> None:
        self.layers = nn.ModuleList(self.layers)
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(self.config.hidden_channels, 1)

    def to(self, device: torch.device) -> "FraudGNN":
        self.layers = self.layers.to(device)
        self.classifier = self.classifier.to(device)
        self.dropout = self.dropout.to(device)
        return self

    def parameters(self):  # type: ignore[override]
        params = list(self.layers.parameters())
        params.extend(self.classifier.parameters())
        return params

    def train_step(self, data: "torch_geometric.data.Data", optimizer: "torch.optim.Optimizer") -> float:
        import torch.nn.functional as F

        self.train()
        optimizer.zero_grad()
        logits = self.forward(data)
        labels = data.y if getattr(data, "y", None) is not None else logits.new_zeros(logits.size())
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1))
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def eval_step(self, data: "torch_geometric.data.Data") -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            return torch.sigmoid(logits.view(-1))

    def forward(self, data: "torch_geometric.data.Data") -> torch.Tensor:
        import torch.nn.functional as F

        x = data.x
        edge_index = data.edge_index
        edge_weight = None
        if getattr(data, "edge_attr", None) is not None:
            edge_weight = data.edge_attr[:, 1]  # transaction count as edge weight

        for layer in self.layers:
            kwargs = {}
            if edge_weight is not None and "edge_weight" in inspect.signature(layer.forward).parameters:
                kwargs["edge_weight"] = edge_weight
            x = layer(x, edge_index, **kwargs)
            x = F.relu(x)
            x = self.dropout(x)
        logits = self.classifier(x)
        return logits.squeeze(-1)


class GraphModelTrainer:
    """Train, save, and load graph neural networks."""

    def __init__(self, config: GraphModelConfig | None = None) -> None:
        self.config = config or GraphModelConfig()

    def train(self, artifacts: "GraphArtifacts") -> dict:
        from .graph_builder import GraphArtifacts

        if not isinstance(artifacts, GraphArtifacts):
            raise TypeError("artifacts must be a GraphArtifacts instance")

        device = torch.device("cuda" if self.config.use_cuda and torch.cuda.is_available() else "cpu")
        data = artifacts.as_data().to(device)
        model = FraudGNN(input_dim=data.num_node_features, config=self.config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        for _ in range(self.config.epochs):
            model.train_step(data, optimizer)

        state_dict = {
            "layers": model.layers.state_dict(),
            "classifier": model.classifier.state_dict(),
        }

        return {
            "state_dict": state_dict,
            "config": asdict(self.config),
            "input_dim": data.num_node_features,
            "feature_names": artifacts.feature_names,
        }

    def save(self, model_bundle: dict, path: str) -> None:
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_bundle, path)

    def load(self, path: str) -> dict:
        return torch.load(path, map_location="cpu")

    def save_artifacts(self, artifacts: "GraphArtifacts", path: str) -> None:
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifacts, path)

    def load_artifacts(self, path: str) -> "GraphArtifacts":
        return torch.load(path, map_location="cpu")


class GraphAnomalyDetector:
    """Inference helper for fraud probability estimation with a trained GNN."""

    def __init__(self, model_bundle: dict) -> None:
        config = GraphModelConfig(**model_bundle["config"])
        self.device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
        self.model = FraudGNN(model_bundle["input_dim"], config).to(self.device)

        layers_state = model_bundle["state_dict"].get("layers", {})
        classifier_state = model_bundle["state_dict"].get("classifier", {})
        self.model.layers.load_state_dict(layers_state, strict=False)
        if classifier_state:
            self.model.classifier.load_state_dict(classifier_state, strict=False)

    def score(self, artifacts: "GraphArtifacts") -> list[float]:
        from .graph_builder import GraphArtifacts

        if not isinstance(artifacts, GraphArtifacts):
            raise TypeError("artifacts must be a GraphArtifacts instance")

        data = artifacts.as_data().to(self.device)
        scores = self.model.eval_step(data).cpu()
        node_scores = {
            node: score.item()
            for node, score in zip(artifacts.node_features["node"].tolist(), scores)
        }
        txn_scores: list[float] = []
        for _, row in artifacts.transactions.iterrows():
            origin_score = node_scores.get(row["nameOrig"], 0.0)
            dest_score = node_scores.get(row["nameDest"], 0.0)
            txn_scores.append(max(origin_score, dest_score))
        return txn_scores


__all__ = ["GraphModelConfig", "GraphModelTrainer", "GraphAnomalyDetector"]
