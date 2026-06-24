"""Training log scanning and NaN-loss guard."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from .report import DiagnosticIssue

LOSS_RE = re.compile(r"(?:loss|train_loss)\s*[=:]\s*([-+a-zA-Z0-9.]+)")


def _parse_loss_token(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"nan", "+nan", "-nan"}:
        return math.nan
    if text in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if text in {"-inf", "-infinity"}:
        return -math.inf
    try:
        return float(text)
    except ValueError:
        return None


def _record_from_line(line: str) -> dict[str, Any]:
    stripped = line.strip()
    if not stripped:
        return {}
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    record: dict[str, Any] = {"message": stripped}
    match = LOSS_RE.search(stripped)
    if match:
        record["loss"] = match.group(1)
    return record


def scan_training_log(
    log: Union[str, Path, Iterable[str], Iterable[dict[str, Any]]],
) -> list[DiagnosticIssue]:
    """Scan a trainer log for NaN, infinity, CUDA OOM, overflow, and loss jumps."""

    if isinstance(log, (str, Path)):
        lines_or_records: Iterable[Any] = Path(log).read_text(encoding="utf-8").splitlines()
    else:
        lines_or_records = log

    guard = NanLossGuard()
    issues: list[DiagnosticIssue] = []
    for item in lines_or_records:
        record = _record_from_line(item) if isinstance(item, str) else dict(item)
        message = str(record.get("message", ""))
        lowered = message.lower()

        if "cuda out of memory" in lowered or "outofmemoryerror" in lowered:
            issues.append(
                DiagnosticIssue(
                    code="log.cuda_oom",
                    title="CUDA out of memory found in logs",
                    severity="error",
                    message="The log contains a CUDA OOM error.",
                    fix="Use load_in_4bit=True, batch size 1, gradient checkpointing, and a shorter sequence length.",
                )
            )

        if "overflow" in lowered and "loss" in lowered:
            issues.append(
                DiagnosticIssue(
                    code="log.overflow",
                    title="Loss overflow found in logs",
                    severity="warning",
                    message="The log mentions overflow around the loss.",
                    fix="Prefer bf16 over fp16 when supported and lower the learning rate.",
                )
            )

        issues.extend(guard.update(record))

    if not issues:
        issues.append(
            DiagnosticIssue(
                code="log.clean",
                title="No obvious loss failures found",
                severity="ok",
                message="The scanned log did not contain NaN, infinity, CUDA OOM, or obvious loss jumps.",
            )
        )
    return issues


class NanLossGuard:
    """Small stateful guard for Trainer log dictionaries."""

    def __init__(self, jump_ratio: float = 4.0) -> None:
        self.jump_ratio = jump_ratio
        self.last_loss: Optional[float] = None

    def update(self, logs: dict[str, Any]) -> list[DiagnosticIssue]:
        issues: list[DiagnosticIssue] = []
        raw_loss = logs.get("loss", logs.get("train_loss"))
        loss = _parse_loss_token(raw_loss)
        if loss is None:
            return issues

        if math.isnan(loss):
            issues.append(
                DiagnosticIssue(
                    code="loss.nan",
                    title="NaN loss detected",
                    severity="error",
                    message="A training log entry reported loss=nan.",
                    fix="Lower the learning rate, prefer bf16 over fp16, remove empty/bad samples, and check that labels are valid.",
                    details={"loss": str(raw_loss)},
                )
            )
            return issues

        if math.isinf(loss):
            issues.append(
                DiagnosticIssue(
                    code="loss.inf",
                    title="Infinite loss detected",
                    severity="error",
                    message="A training log entry reported an infinite loss.",
                    fix="Lower the learning rate, enable gradient clipping, and inspect samples with very long or corrupted labels.",
                    details={"loss": str(raw_loss)},
                )
            )
            return issues

        if self.last_loss is not None and self.last_loss > 0 and loss / self.last_loss >= self.jump_ratio:
            issues.append(
                DiagnosticIssue(
                    code="loss.jump",
                    title="Loss jumped sharply",
                    severity="warning",
                    message=f"Loss jumped from {self.last_loss:.4g} to {loss:.4g}.",
                    fix="Watch the next steps. If it keeps climbing, lower LR and inspect recent samples.",
                    details={"previous_loss": self.last_loss, "loss": loss},
                )
            )

        self.last_loss = loss
        return issues
