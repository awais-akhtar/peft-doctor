"""Risk scoring and human explanations for reports."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .report import DiagnosisReport


def risk_summary(report: DiagnosisReport) -> dict[str, Any]:
    errors = [issue for issue in report.issues if issue.severity == "error"]
    warnings = [issue for issue in report.issues if issue.severity == "warning"]
    score = min(100, len(errors) * 35 + len(warnings) * 15)
    if score >= 70 or errors:
        level = "HIGH"
    elif score >= 30:
        level = "MEDIUM"
    else:
        level = "LOW"
    reasons = [issue.title for issue in errors[:3] + warnings[:5]]
    return {"level": level, "score": score, "reasons": reasons}


def explanation_text(report: DiagnosisReport) -> str:
    risk = risk_summary(report)
    lines = [f"Run risk: {risk['level']} ({risk['score']}/100)"]
    if risk["reasons"]:
        lines.append("Reasons:")
        for reason in risk["reasons"]:
            lines.append(f"- {reason}")
    fixes = [issue.fix for issue in report.sorted_issues() if issue.fix]
    if fixes:
        lines.append("Copy-paste fixes:")
        for fix in fixes[:8]:
            lines.append(f"- {fix}")
    return "\n".join(lines)


def report_to_html(report: DiagnosisReport) -> str:
    risk = risk_summary(report)
    rows = []
    for issue in report.sorted_issues():
        rows.append(
            "<tr>"
            f"<td>{html.escape(issue.severity.upper())}</td>"
            f"<td>{html.escape(issue.code)}</td>"
            f"<td><strong>{html.escape(issue.title)}</strong><br>{html.escape(issue.message)}</td>"
            f"<td>{html.escape(issue.fix or '')}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PEFT Doctor Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; vertical-align: top; }}
    th {{ background: #f3f4f6; text-align: left; }}
    .risk {{ padding: 12px; background: #fff7ed; border: 1px solid #fed7aa; margin-bottom: 20px; }}
  </style>
</head>
<body>
  <h1>PEFT Doctor Report</h1>
  <div class="risk"><strong>Run risk:</strong> {html.escape(risk["level"])} ({risk["score"]}/100)</div>
  <h2>Metadata</h2>
  <pre>{html.escape(str(report.metadata))}</pre>
  <h2>Findings</h2>
  <table>
    <thead><tr><th>Severity</th><th>Code</th><th>Finding</th><th>Fix</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""


def write_html_report(report: DiagnosisReport, path: Path) -> None:
    path.write_text(report_to_html(report), encoding="utf-8")


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def report_to_pdf_bytes(report: DiagnosisReport) -> bytes:
    """Create a small dependency-free PDF summary."""

    risk = risk_summary(report)
    lines = [
        "PEFT Doctor Report",
        f"Run risk: {risk['level']} ({risk['score']}/100)",
        "",
    ]
    for issue in report.sorted_issues()[:18]:
        lines.append(f"{issue.severity.upper()}: {issue.title}")
        if issue.fix:
            lines.append(f"Fix: {issue.fix}")
        lines.append("")

    content_lines = ["BT", "/F1 11 Tf", "50 780 Td", "14 TL"]
    for line in lines[:52]:
        content_lines.append(f"({_pdf_escape(line[:100])}) Tj")
        content_lines.append("T*")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_at = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(output)


def write_pdf_report(report: DiagnosisReport, path: Path) -> None:
    path.write_bytes(report_to_pdf_bytes(report))
