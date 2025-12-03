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

    score_parser = sub.add_parser("score", help="Score new CSV")
    score_parser.add_argument("csv", help="Path to CSV for scoring")
    score_parser.add_argument("model", help="Path to trained model")
    score_parser.add_argument("output", help="Output CSV path")
    score_parser.add_argument("--html-report", dest="html_report", help="Optional HTML report path")
    score_parser.add_argument("--pdf-report", dest="pdf_report", help="Optional PDF report path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline = ScoringPipeline()

    if args.command == "train":
        pipeline.train(args.csv, args.model)
        print(f"Model saved to {Path(args.model).resolve()}")
    elif args.command == "score":
        pipeline.score(args.csv, args.model, args.output, html_report=args.html_report, pdf_report=args.pdf_report)
        print(f"Scores written to {Path(args.output).resolve()}")
        if args.html_report:
            print(f"HTML report written to {Path(args.html_report).resolve()}")
        if args.pdf_report:
            print(f"PDF report written to {Path(args.pdf_report).resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
