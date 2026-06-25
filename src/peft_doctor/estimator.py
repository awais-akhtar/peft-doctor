"""Approximate VRAM estimator for PEFT, LoRA, and QLoRA plans."""

from __future__ import annotations

import re
from typing import Optional

from .profiles import profile_for
from .report import DiagnosisReport


PARAM_RE = re.compile(r"(?P<count>\d+(?:\.\d+)?)\s*[bB](?:\b|[-_])")


def infer_params_billion(model: str, fallback: float = 7.0) -> float:
    match = PARAM_RE.search(model)
    if not match:
        return fallback
    return float(match.group("count"))


def estimate_vram_gb(
    model: str,
    *,
    seq_len: int = 2048,
    batch_size: int = 1,
    qlora: bool = False,
    lora: bool = True,
    gradient_checkpointing: bool = True,
    target_vram_gb: Optional[float] = None,
) -> DiagnosisReport:
    """Return an approximate memory report before training."""

    profile = profile_for(model)
    params_b = infer_params_billion(model)
    hidden = profile.hidden_size if profile else 4096
    layers = profile.layers if profile else 32

    weight_bytes_per_param = 0.58 if qlora else 2.0
    weight_gb = params_b * 1_000_000_000 * weight_bytes_per_param / (1024**3)
    adapter_gb = max(0.25, params_b * 0.055) if lora else 0.0
    optimizer_gb = 0.35 if qlora else (adapter_gb * 2.0 if lora else weight_gb * 2.0)
    activation_factor = 2.0 if gradient_checkpointing else 4.0
    activation_gb = (seq_len * batch_size * hidden * layers * activation_factor * 2) / (1024**3)
    fragmentation_gb = max(1.0, (weight_gb + adapter_gb + optimizer_gb + activation_gb) * 0.12)
    total_gb = weight_gb + adapter_gb + optimizer_gb + activation_gb + fragmentation_gb

    report = DiagnosisReport(
        metadata={
            "model": model,
            "profile": profile.name if profile else "unknown",
            "params_billion": round(params_b, 2),
            "seq_len": seq_len,
            "batch_size": batch_size,
            "qlora": qlora,
            "gradient_checkpointing": gradient_checkpointing,
            "estimated_total_gb": round(total_gb, 2),
        }
    )
    report.add(
        "estimate.total_vram",
        "Estimated training VRAM",
        "ok" if target_vram_gb is None or total_gb <= target_vram_gb else "warning",
        f"Estimated VRAM is about {total_gb:.1f} GB before dataloader and runtime variance.",
        "Lower batch size, lower sequence length, enable QLoRA, or enable gradient checkpointing if this is above your GPU.",
        weight_gb=round(weight_gb, 2),
        adapter_gb=round(adapter_gb, 2),
        optimizer_gb=round(optimizer_gb, 2),
        activation_gb=round(activation_gb, 2),
        overhead_gb=round(fragmentation_gb, 2),
        target_vram_gb=target_vram_gb,
    )
    if not qlora and params_b >= 7:
        report.add(
            "estimate.qlora_recommended",
            "QLoRA is recommended for common single-GPU 7B runs",
            "warning",
            "The plan does not use QLoRA for a 7B-class or larger model.",
            "Use --qlora or load_in_4bit=True for low-VRAM fine-tuning.",
        )
    if qlora and not lora:
        report.add(
            "estimate.qlora_without_lora",
            "QLoRA estimate has LoRA disabled",
            "warning",
            "4-bit base weights are normally trained through PEFT adapters, not full-weight updates.",
            "Remove --no-lora for QLoRA, or remove --qlora for a full fine-tuning estimate.",
        )
    if seq_len > 4096:
        report.add(
            "estimate.sequence_high",
            "Sequence length is high",
            "warning",
            "Long sequence lengths grow activation memory quickly.",
            "Start at 1024 or 2048, then increase after a stable run.",
        )
    if batch_size > 1 and target_vram_gb and target_vram_gb <= 24:
        report.add(
            "estimate.batch_risky",
            "Batch size may be risky on limited VRAM",
            "warning",
            "Batch size above 1 is often the first cause of QLoRA OOM on small GPUs.",
            "Use batch size 1 and raise gradient_accumulation_steps.",
        )
    return report
