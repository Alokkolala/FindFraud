"""Generate HTML/PDF reports summarizing scoring results."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class ReportBuilder:
    def build_html(self, raw: pd.DataFrame, scores: pd.DataFrame, output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        suspicious = scores[scores["is_suspicious"]]
        summary = {
            "total_transactions": len(scores),
            "suspicious_count": len(suspicious),
            "suspicious_rate": round(len(suspicious) / max(len(scores), 1), 4),
            "avg_amount": raw["amount"].mean(),
        }

        html_lines = [
            "<html><head><title>FindFraud Scoring Report</title></head><body>",
            "<h1>FindFraud Scoring Report</h1>",
            "<h2>Summary</h2>",
            "<ul>",
            *(f"<li>{k}: {v}</li>" for k, v in summary.items()),
            "</ul>",
            "<h2>Top Suspicious Transactions</h2>",
            suspicious.head(20).to_html(index=False),
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
