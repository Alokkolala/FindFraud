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

        base_css = """
        <style>
            :root {
                --bg: #0f172a;
                --card: #111827;
                --text: #e5e7eb;
                --muted: #9ca3af;
                --accent: #22c55e;
                --danger: #f59e0b;
                --border: #1f2937;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                padding: 24px;
                font-family: "Inter", "Segoe UI", system-ui, sans-serif;
                background: radial-gradient(circle at 20% 20%, #111827 0, #0b1223 40%, #050816 100%);
                color: var(--text);
                line-height: 1.6;
            }
            h1 { letter-spacing: 0.5px; }
            a { color: var(--accent); text-decoration: none; }
            a:hover { text-decoration: underline; }
            .grid { display: grid; gap: 16px; }
            .cards { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
            .card {
                background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 16px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.35);
            }
            .pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px; font-weight: 600; font-size: 12px; }
            .pill.ok { background: rgba(34,197,94,0.1); color: var(--accent); border: 1px solid rgba(34,197,94,0.3); }
            .pill.warn { background: rgba(245,158,11,0.12); color: var(--danger); border: 1px solid rgba(245,158,11,0.35); }
            .section { margin-top: 28px; }
            .section h2 { margin-bottom: 8px; }
            .muted { color: var(--muted); font-size: 14px; }
            table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 14px; }
            table thead { background: #111827; }
            table th, table td { padding: 10px 12px; border: 1px solid var(--border); }
            table tr:nth-child(even) { background: rgba(255,255,255,0.03); }
            table tr:hover { background: rgba(34,197,94,0.05); }
            .table-title { display: flex; align-items: center; justify-content: space-between; }
            .note { background: rgba(255,255,255,0.04); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
            .danger { color: var(--danger); font-weight: 600; }
            footer { margin-top: 32px; color: var(--muted); font-size: 13px; }
        </style>
        """

        def _format_card(label: str, value: str, pill: str | None = None) -> str:
            pill_html = f"<span class='pill {pill}'>{pill.upper()}</span>" if pill else ""
            return f"<div class='card'><div class='muted'>{label}</div><div style='font-size:24px;font-weight:700;margin-top:4px;'>{value}</div>{pill_html}</div>"

        def _render_training(info: dict | None) -> str:
            if not info:
                return "<div class='note'>No training metadata was provided.</div>"
            items = []
            for key, value in info.items():
                if isinstance(value, (list, tuple)):
                    items.append(f"<li><strong>{key}</strong>: {', '.join(map(str, value))}</li>")
                else:
                    items.append(f"<li><strong>{key}</strong>: {value}</li>")
            return "<ul>" + "".join(items) + "</ul>"

        def _render_graph(info: dict | None) -> str:
            if not info:
                return "<div class='note'>No graph artifacts were generated for this run.</div>"
            lines = ["<ul>"]
            for key in ["nodes", "edges", "min_edge_count_used", "graph_path"]:
                if key in info:
                    lines.append(f"<li><strong>{key}</strong>: {info[key]}</li>")
            lines.append("</ul>")
            table_html = ""
            top_edges = info.get("top_edges")
            if top_edges is not None and not top_edges.empty:
                table_html = "<h4>Top connections by transaction count</h4>" + top_edges.to_html(
                    index=False, classes="table"
                )
            return "".join(lines) + table_html

        def _render_profiles(df: pd.DataFrame | None) -> str:
            if df is None:
                return "<div class='note'>No account profiles were requested.</div>"
            if df.empty:
                return "<div class='note'>No accounts were profiled for this run.</div>"
            return df.to_html(index=False, classes="table")

        def _style_table(df: pd.DataFrame, highlight: bool = False) -> str:
            if df.empty:
                return "<div class='note'>No rows available.</div>"

            def _highlight(row: pd.Series) -> list[str]:
                if not highlight:
                    return ["" for _ in row]
                return [
                    "background-color: rgba(245,158,11,0.18); font-weight: 600;"
                    if row.get("is_suspicious")
                    else ""
                    for _ in row
                ]

            styler = (
                df.style.hide(axis="index")
                .set_table_attributes('class="table"')
                .format(precision=4)
                .apply(_highlight, axis=1)
            )
            return styler.to_html()

        html_lines = [
            "<html><head><title>FindFraud Scoring Report</title>",
            base_css,
            "</head><body>",
            "<h1>FindFraud Scoring Report</h1>",
            "<div class='muted'>Training results, scoring outputs, graph context, and profiles in one place.</div>",
            "<div class='grid cards' style='margin-top:18px;'>",
            _format_card("Total transactions", f"{summary['total_transactions']:,}"),
            _format_card("Suspicious flagged", f"{summary['suspicious_count']:,}", pill="warn" if summary["suspicious_count"] else None),
            _format_card("Suspicious rate", f"{summary['suspicious_rate']*100:.2f}%"),
            _format_card("Average amount", f"{summary['avg_amount']:,}", pill="ok"),
            "</div>",

            "<div class='section' id='training'>",
            "<div class='table-title'><h2>Training & Model Details</h2></div>",
            _render_training(training_info),
            "</div>",

            "<div class='section' id='suspicious'>",
            "<div class='table-title'><h2>Top Suspicious Transactions</h2><span class='muted'>Highest fraud scores (max 20)</span></div>",
            _style_table(suspicious.head(20), highlight=True),
            "</div>",

            "<div class='section' id='all'>",
            "<div class='table-title'><h2>All Scored Transactions</h2><span class='muted'>Scroll to inspect every record</span></div>",
            _style_table(scores, highlight=True),
            "</div>",

            "<div class='section' id='graph'>",
            "<div class='table-title'><h2>Graph Snapshot</h2><span class='muted'>Aggregated structure from nameOrig/nameDest</span></div>",
            _render_graph(graph_info),
            "</div>",

            "<div class='section' id='profiles'>",
            "<div class='table-title'><h2>Account Risk Profiles</h2><span class='muted'>Aggregated by account or merchant</span></div>",
            _render_profiles(profiles),
            "</div>",

            "<footer>Generated by FindFraud.</footer>",
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
