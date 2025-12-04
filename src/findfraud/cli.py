"""Command-line interface for FindFraud."""
from __future__ import annotations

import argparse
from pathlib import Path

from .scorer import ScoringPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FindFraud anomaly detection")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Train model from CSV")
    train_parser.add_argument("csv", help="Path to training CSV")
    train_parser.add_argument("model", help="Path to save trained model", nargs="?", default="models/anomaly.joblib")
    train_parser.add_argument("--model-type", choices=["tabular", "gnn"], default="tabular")
    train_parser.add_argument("--graph-artifacts", help="Optional path to persist built graph features")
    train_parser.add_argument("--window-size", type=int, default=24, help="Step window for rolling graph aggregates")
    train_parser.add_argument("--min-edge-count", type=int, default=1, help="Minimum transactions required to keep an edge")
    train_parser.add_argument("--gnn-hidden", type=int, default=64, help="Hidden size for GNN layers")
    train_parser.add_argument("--gnn-layers", type=int, default=2, help="Number of GNN layers")
    train_parser.add_argument("--gnn-dropout", type=float, default=0.2, help="Dropout applied after each GNN layer")
    train_parser.add_argument("--gnn-epochs", type=int, default=30, help="Training epochs for the GNN")
    train_parser.add_argument("--gnn-lr", type=float, default=0.005, help="Learning rate for the GNN optimizer")
    train_parser.add_argument(
        "--gnn-weight-decay", type=float, default=1e-4, help="Weight decay regularization for the GNN optimizer"
    )
    train_parser.add_argument("--use-cuda", action="store_true", help="Use CUDA when available for GNN training")

    score_parser = sub.add_parser("score", help="Score new CSV")
    score_parser.add_argument("csv", help="Path to CSV for scoring")
    score_parser.add_argument("model", help="Path to trained model")
    score_parser.add_argument("output", help="Output CSV path")
    score_parser.add_argument("--html-report", dest="html_report", help="Optional HTML report path")
    score_parser.add_argument("--pdf-report", dest="pdf_report", help="Optional PDF report path")
    score_parser.add_argument("--model-type", choices=["tabular", "gnn"], default="tabular")
    score_parser.add_argument("--graph-artifacts", dest="graph_artifacts", help="Optional path to persist graph snapshot")
    score_parser.add_argument(
        "--profiles-csv",
        dest="profiles_csv",
        help="Optional path to write account-level risk classifications",
    )
    score_parser.add_argument("--window-size", type=int, default=24, help="Step window for graph reconstruction")
    score_parser.add_argument("--min-edge-count", type=int, default=1, help="Minimum transactions required to keep an edge")

    return parser


def _build_pipeline(args: argparse.Namespace) -> ScoringPipeline:
    if getattr(args, "model_type", "tabular") == "gnn":
        from .graph_builder import GraphBuilderConfig
        from .graph_model import GraphModelConfig

        graph_config = GraphBuilderConfig(window_size=args.window_size, min_edge_count=args.min_edge_count)
        graph_model_config = None
        if hasattr(args, "gnn_hidden"):
            graph_model_config = GraphModelConfig(
                hidden_channels=args.gnn_hidden,
                num_layers=args.gnn_layers,
                dropout=args.gnn_dropout,
                lr=args.gnn_lr,
                weight_decay=args.gnn_weight_decay,
                epochs=args.gnn_epochs,
                use_cuda=args.use_cuda,
            )

        return ScoringPipeline(
            model_choice="gnn",
            graph_config=graph_config,
            graph_model_config=graph_model_config,
        )

    return ScoringPipeline()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        pipeline = _build_pipeline(args)
        pipeline.train(args.csv, args.model, graph_output=getattr(args, "graph_artifacts", None))
        print(f"Model saved to {Path(args.model).resolve()}")
        if getattr(args, "graph_artifacts", None):
            print(f"Graph artifacts written to {Path(args.graph_artifacts).resolve()}")
    elif args.command == "score":
        pipeline = _build_pipeline(args)
        pipeline.score(
            args.csv,
            args.model,
            args.output,
            html_report=args.html_report,
            pdf_report=args.pdf_report,
            graph_output=getattr(args, "graph_artifacts", None),
            profile_output=getattr(args, "profiles_csv", None),
        )
        print(f"Scores written to {Path(args.output).resolve()}")
        if args.html_report:
            print(f"HTML report written to {Path(args.html_report).resolve()}")
        if args.pdf_report:
            print(f"PDF report written to {Path(args.pdf_report).resolve()}")
        if getattr(args, "graph_artifacts", None):
            print(f"Graph snapshot written to {Path(args.graph_artifacts).resolve()}")
        if getattr(args, "profiles_csv", None):
            print(f"Account profiles written to {Path(args.profiles_csv).resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
