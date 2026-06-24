"""Training log scanning and NaN-loss guard."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from .report import DiagnosticIssue

LOSS_RE = re.compile(r"(?:loss|train_loss)\s*[=:]\s*([-+a-zA-Z0-9.]+)")
GRAD_NORM_RE = re.compile(r"(?:grad_norm|gradient_norm)\s*[=:]\s*([-+a-zA-Z0-9.]+)")


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
    grad_match = GRAD_NORM_RE.search(stripped)
    if grad_match:
        record["grad_norm"] = grad_match.group(1)
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

        if "no space left on device" in lowered or "disk quota exceeded" in lowered:
            issues.append(
                DiagnosticIssue(
                    code="log.disk_full",
                    title="Disk is full during training",
                    severity="error",
                    message="The log shows a disk-full or quota error.",
                    fix="Reduce checkpoint frequency, set save_total_limit=2, or write outputs to a larger disk.",
                )
            )

        if "illegal memory access" in lowered:
            issues.append(
                DiagnosticIssue(
                    code="log.cuda_illegal_memory",
                    title="CUDA illegal memory access found",
                    severity="error",
                    message="The log contains a CUDA illegal memory access error.",
                    fix="Restart the runtime, disable torch_compile first, then reduce batch size or sequence length.",
                )
            )

        if "expected all tensors to be on the same device" in lowered:
            issues.append(
                DiagnosticIssue(
                    code="log.device_mismatch",
                    title="Tensor device mismatch found",
                    severity="error",
                    message="The log shows tensors split across different devices.",
                    fix="Check device_map, DDP launch settings, and make sure labels/batches move to the model device.",
                )
            )

        if "mat1 and mat2 shapes cannot be multiplied" in lowered or "size mismatch" in lowered:
            issues.append(
                DiagnosticIssue(
                    code="log.shape_mismatch",
                    title="Shape mismatch found",
                    severity="error",
                    message="The log contains a tensor shape mismatch.",
                    fix="Check tokenizer/model embedding resize, LoRA target modules, and label/input length alignment.",
                )
            )

        if "token indices sequence length is longer than" in lowered:
            issues.append(
                DiagnosticIssue(
                    code="log.sequence_too_long",
                    title="Sequence length exceeds model limit",
                    severity="warning",
                    message="The log indicates tokenized inputs are longer than the model limit.",
                    fix="Set truncation=True, lower max_seq_length, or use a model-supported context extension.",
                )
            )

        issues.extend(guard.update(record))

    if not issues:
        issues.append(
            DiagnosticIssue(
                code="log.clean",
                title="No obvious loss failures found",
                severity="ok",
                message="The scanned log did not contain NaN, infinity, CUDA OOM, device, disk, shape, or obvious loss-jump failures.",
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
        raw_grad_norm = logs.get("grad_norm", logs.get("gradient_norm"))
        grad_norm = _parse_loss_token(raw_grad_norm)
        if grad_norm is not None:
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                issues.append(
                    DiagnosticIssue(
                        code="grad_norm.invalid",
                        title="Invalid gradient norm detected",
                        severity="error",
                        message="A training log reported NaN or infinite gradient norm.",
                        fix="Lower the learning rate, enable max_grad_norm=1.0, and inspect recent batches.",
                        details={"grad_norm": str(raw_grad_norm)},
                    )
                )
            elif grad_norm > 100:
                issues.append(
                    DiagnosticIssue(
                        code="grad_norm.spike",
                        title="Gradient norm is very high",
                        severity="warning",
                        message=f"A training log reported grad_norm={grad_norm:.4g}.",
                        fix="Use gradient clipping, lower LR, and check for corrupted or extremely long samples.",
                        details={"grad_norm": grad_norm},
                    )
                )

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
