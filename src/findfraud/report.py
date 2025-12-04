"""Generate HTML/PDF reports summarizing scoring results."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class ReportBuilder:
    def build_html(
        self,
        raw: pd.DataFrame,
        scores: pd.DataFrame,
        output_path: str,
        training_info: dict | None = None,
        graph_info: dict | None = None,
        profiles: pd.DataFrame | None = None,
    ) -> str:
        """Render a single HTML report that includes model, scoring, graph, and profile views."""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        suspicious = scores[scores["is_suspicious"]]
        summary = {
            "total_transactions": len(scores),
            "suspicious_count": len(suspicious),
            "suspicious_rate": round(len(suspicious) / max(len(scores), 1), 4),
            "avg_amount": round(float(raw["amount"].mean()), 2),
        }

        def _render_training(info: dict | None) -> str:
            if not info:
                return "<p>No training metadata was provided.</p>"
            items = []
            for key, value in info.items():
                if isinstance(value, (list, tuple)):
                    items.append(f"<li><strong>{key}</strong>: {', '.join(map(str, value))}</li>")
                else:
                    items.append(f"<li><strong>{key}</strong>: {value}</li>")
            return "<ul>" + "".join(items) + "</ul>"

        def _render_graph(info: dict | None) -> str:
            if not info:
                return "<p>No graph artifacts were generated for this run.</p>"
            lines = ["<ul>"]
            for key in ["nodes", "edges", "min_edge_count_used", "graph_path"]:
                if key in info:
                    lines.append(f"<li><strong>{key}</strong>: {info[key]}</li>")
            lines.append("</ul>")
            table_html = ""
            top_edges = info.get("top_edges")
            if top_edges is not None and not top_edges.empty:
                table_html = "<h4>Top connections by transaction count</h4>" + top_edges.to_html(
                    index=False
                )
            return "".join(lines) + table_html

        def _render_profiles(df: pd.DataFrame | None) -> str:
            if df is None:
                return "<p>No account profiles were requested.</p>"
            if df.empty:
                return "<p>No accounts were profiled for this run.</p>"
            return df.to_html(index=False)

        html_lines = [
            "<html><head><title>FindFraud Scoring Report</title></head><body>",
            "<h1>FindFraud Scoring Report</h1>",
            "<h2>Summary</h2>",
            "<ul>",
            *(f"<li>{k}: {v}</li>" for k, v in summary.items()),
            "</ul>",
            "<h2>Training & Model Details</h2>",
            _render_training(training_info),
            "<h2>Top Suspicious Transactions</h2>",
            suspicious.head(20).to_html(index=False),
            "<h2>All Scored Transactions</h2>",
            scores.to_html(index=False),
            "<h2>Graph Snapshot</h2>",
            _render_graph(graph_info),
            "<h2>Account Risk Profiles</h2>",
            _render_profiles(profiles),
            "</body></html>",
        ]
        html = "\n".join(html_lines)
        Path(output_path).write_text(html, encoding="utf-8")
        return html

    def html_to_pdf(self, html_path: str, pdf_path: str) -> Optional[str]:
        try:
            import weasyprint

            pdf_bytes = weasyprint.HTML(html_path).write_pdf()
            Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
            Path(pdf_path).write_bytes(pdf_bytes)
            return pdf_path
        except ImportError:
            return None
