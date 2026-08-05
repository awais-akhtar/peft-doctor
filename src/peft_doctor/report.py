"""Report objects used by the diagnostics and CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

Severity = str

SEVERITY_ORDER = {
    "error": 0,
    "warning": 1,
    "ok": 2,
    "info": 3,
}


@dataclass
class DiagnosticIssue:
    """One finding from a PEFT Doctor check."""

    code: str
    title: str
    severity: Severity
    message: str
    fix: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "title": self.title,
            "severity": self.severity,
            "message": self.message,
        }
        if self.fix:
            data["fix"] = self.fix
        if self.details:
            data["details"] = self.details
        return data

    def to_markdown(self) -> str:
        parts = [
            f"### {self.severity.upper()}: {self.title}",
            "",
            self.message,
        ]
        if self.fix:
            parts.extend(["", f"Fix: {self.fix}"])
        if self.details:
            parts.extend(["", "Details:", ""])
            for key, value in self.details.items():
                parts.append(f"- `{key}`: `{value}`")
        return "\n".join(parts)


@dataclass
class DiagnosisReport:
    """A collection of findings with helper renderers."""

    issues: list[DiagnosticIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        code: str,
        title: str,
        severity: Severity,
        message: str,
        fix: Optional[str] = None,
        **details: Any,
    ) -> DiagnosticIssue:
        issue = DiagnosticIssue(
            code=code,
            title=title,
            severity=severity,
            message=message,
            fix=fix,
            details={k: v for k, v in details.items() if v is not None},
        )
        self.issues.append(issue)
        return issue

    def extend(self, issues: Iterable[DiagnosticIssue]) -> None:
        self.issues.extend(issues)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.severity == "warning" for issue in self.issues)

    @property
    def summary(self) -> dict[str, int]:
        counts = {"error": 0, "warning": 0, "ok": 0, "info": 0}
        for issue in self.issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts

    def sorted_issues(self) -> list[DiagnosticIssue]:
        return sorted(self.issues, key=lambda issue: SEVERITY_ORDER.get(issue.severity, 99))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "summary": self.summary,
            "issues": [issue.to_dict() for issue in self.sorted_issues()],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def to_markdown(self) -> str:
        lines = ["# PEFT Doctor Report", ""]
        if self.metadata:
            lines.extend(["## Metadata", ""])
            for key, value in self.metadata.items():
                lines.append(f"- `{key}`: `{value}`")
            lines.append("")

        summary = self.summary
        lines.extend(
            [
                "## Summary",
                "",
                (
                    f"- errors: {summary.get('error', 0)}\n"
                    f"- warnings: {summary.get('warning', 0)}\n"
                    f"- ok: {summary.get('ok', 0)}\n"
                    f"- info: {summary.get('info', 0)}"
                ),
                "",
                "## Findings",
                "",
            ]
        )

        if not self.issues:
            lines.append("No findings were produced.")
        else:
            for issue in self.sorted_issues():
                lines.append(issue.to_markdown())
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"
