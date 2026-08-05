"""Local planning, dataset review, and troubleshooting helpers.

This module is intentionally local-first. It does not call remote APIs, does not
send datasets anywhere, and does not require model-provider tokens.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .datasets import check_dataset, load_dataset_records
from .environment import CORE_PACKAGES, OPTIONAL_FINETUNING_PACKAGES, collect_environment
from .estimator import estimate_vram_gb, infer_params_billion
from .explain import risk_summary, write_html_report
from .fixer import repair_dataset_file, repair_python_file
from .logs import scan_training_log
from .report import DiagnosisReport
from .targets import infer_model_family, recommend_target_modules


PROMPT_INJECTION_PATTERNS = [
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "developer message",
    "jailbreak",
    "do anything now",
    "disregard the above",
]

BOILERPLATE_ANSWER_PATTERNS = [
    "as an ai language model",
    "i do not have access to real-time",
    "i cannot browse",
    "i cannot access external",
    "citation needed",
]

UNSUPPORTED_CLAIM_PATTERNS = [
    "studies show",
    "research proves",
    "experts agree",
    "according to wikipedia",
    "according to sources",
    "it is widely known",
    "citation needed",
]

KNOWN_ISSUES = {
    "cuda illegal memory access": {
        "fix": "Restart the runtime, disable torch_compile first, then reduce sequence length or batch size.",
    },
    "cuda out of memory": {
        "fix": "Use QLoRA 4-bit loading, batch size 1, gradient checkpointing, and shorter sequences.",
    },
    "nan loss": {
        "fix": "Lower learning rate, prefer bf16 over fp16, clip gradients, and remove empty labels.",
    },
    "wrong target modules": {
        "fix": "Use family targets such as q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.",
    },
    "adapter merge": {
        "fix": "Load base model in fp16/bf16/fp32, then PeftModel.from_pretrained(...).merge_and_unload().",
    },
    "empty assistant": {
        "fix": "Remove rows where assistant messages are blank before tokenization.",
    },
}

GPU_PROFILES = {
    "t4": {"vram": 16.0},
    "l4": {"vram": 24.0},
    "a10g": {"vram": 24.0},
    "a100": {"vram": 40.0},
    "h100": {"vram": 80.0},
    "rtx 3060": {"vram": 12.0},
    "rtx 3090": {"vram": 24.0},
    "rtx 4090": {"vram": 24.0},
}


@dataclass
class DatasetIntelligence:
    rows: int
    malformed_rows: int
    duplicates: int
    empty_assistant: int
    assistant_only: int
    prompt_injections: int
    boilerplate_answers: int
    unsupported_claim_markers: int
    avg_response_chars: float
    longest_chars: int
    flagged_rows: int
    clean_row_percent: float
    role_counts: dict[str, int]
    length_histogram: dict[str, int]
    token_histogram: dict[str, int]
    character_sets: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "malformed_rows": self.malformed_rows,
            "duplicates": self.duplicates,
            "empty_assistant": self.empty_assistant,
            "assistant_only": self.assistant_only,
            "prompt_injections": self.prompt_injections,
            "possible_boilerplate_answers": self.boilerplate_answers,
            "unsupported_claim_markers": self.unsupported_claim_markers,
            "avg_response_chars": round(self.avg_response_chars, 2),
            "longest_chars": self.longest_chars,
            "flagged_rows": self.flagged_rows,
            "rows_without_flagged_issues_percent": round(self.clean_row_percent, 2),
            "role_counts": self.role_counts,
            "length_histogram": self.length_histogram,
            "token_histogram": self.token_histogram,
            "character_set_distribution": self.character_sets,
        }


def _text_bar(value: float, maximum: float, *, width: int = 24) -> str:
    if maximum <= 0:
        return "-" * width
    filled = max(0, min(width, int(round((value / maximum) * width))))
    return "#" * filled + "-" * (width - filled)


def _row_text(row: Any) -> str:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return ""
    if isinstance(row.get("messages"), list):
        return "\n".join(str(msg.get("content", "")) for msg in row["messages"] if isinstance(msg, dict))
    pieces = []
    for key in ["text", "prompt", "instruction", "input", "response", "completion", "output", "answer"]:
        value = row.get(key)
        if isinstance(value, str):
            pieces.append(value)
    return "\n".join(pieces)


def _assistant_texts(row: Any) -> list[str]:
    if not isinstance(row, dict):
        return []
    messages = row.get("messages")
    if not isinstance(messages, list):
        return []
    return [
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant"
    ]


def _response_text(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ["response", "completion", "output", "answer"]:
        value = row.get(key)
        if isinstance(value, str):
            return value
    assistants = _assistant_texts(row)
    return assistants[-1] if assistants else ""


def _rough_tokens(text: str) -> int:
    return max(len(text.split()), len(text) // 4)


def _bucket(value: int, buckets: list[int]) -> str:
    for limit in buckets:
        if value <= limit:
            return f"<= {limit}"
    return f"> {buckets[-1]}"


def _character_bucket(text: str) -> str:
    if not text.strip():
        return "empty"
    ascii_ratio = sum(1 for char in text if ord(char) < 128) / max(1, len(text))
    if ascii_ratio > 0.92:
        return "ascii-dominant"
    if ascii_ratio > 0.60:
        return "mixed"
    return "non-ascii-dominant"


def _load_rows_lenient(path: Path, limit: int = 1000) -> tuple[list[dict[str, Any]], int]:
    suffix = path.suffix.lower()
    malformed = 0
    rows: list[dict[str, Any]] = []
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(rows) >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if isinstance(value, dict):
                    rows.append(value)
        return rows, malformed
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for _, row in zip(range(limit), reader):
                rows.append(dict(row))
        return rows, malformed
    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(rows) >= limit:
                    break
                text = line.strip()
                if text:
                    rows.append({"text": text})
        return rows, malformed
    try:
        return load_dataset_records(path, limit=limit), malformed
    except Exception:
        return [], 1


def analyze_dataset_intelligence(dataset: Path, *, limit: int = 1000) -> DatasetIntelligence:
    rows, malformed = _load_rows_lenient(dataset, limit=limit)
    normalized = [" ".join(_row_text(row).lower().split()) for row in rows]
    normalized_counts = Counter(text for text in normalized if text)
    duplicates = sum(count - 1 for count in normalized_counts.values() if count > 1)
    role_counts: Counter[str] = Counter()
    assistant_only = 0
    empty_assistant = 0
    prompt_injections = 0
    boilerplate = 0
    unsupported_claim_markers = 0
    response_lengths = []
    token_hist: Counter[str] = Counter()
    length_hist: Counter[str] = Counter()
    character_sets: Counter[str] = Counter()
    flagged_rows = 0

    for row in rows:
        text = _row_text(row)
        response = _response_text(row)
        response_lengths.append(len(response))
        length_hist[_bucket(len(text), [250, 750, 1500, 3000, 6000])] += 1
        token_hist[_bucket(_rough_tokens(text), [64, 256, 512, 1024, 2048])] += 1
        character_sets[_character_bucket(text)] += 1
        lowered = text.lower()
        row_flagged = bool(normalized_counts.get(" ".join(lowered.split()), 0) > 1)
        if any(pattern in lowered for pattern in PROMPT_INJECTION_PATTERNS):
            prompt_injections += 1
            row_flagged = True
        if any(pattern in response.lower() for pattern in BOILERPLATE_ANSWER_PATTERNS):
            boilerplate += 1
            row_flagged = True
        if any(pattern in response.lower() for pattern in UNSUPPORTED_CLAIM_PATTERNS):
            unsupported_claim_markers += 1
            row_flagged = True

        messages = row.get("messages") if isinstance(row, dict) else None
        if isinstance(messages, list):
            roles = [
                str(message.get("role", "")).lower()
                for message in messages
                if isinstance(message, dict)
            ]
            role_counts.update(role or "missing" for role in roles)
            if roles and "assistant" in roles and "user" not in roles:
                assistant_only += 1
                row_flagged = True
            if any(
                str(message.get("role", "")).lower() == "assistant"
                and not str(message.get("content", "")).strip()
                for message in messages
                if isinstance(message, dict)
            ):
                empty_assistant += 1
                row_flagged = True
        if row_flagged:
            flagged_rows += 1

    total_observed = len(rows) + malformed
    clean_rows = max(0, len(rows) - flagged_rows)
    clean_percent = (clean_rows / total_observed * 100) if total_observed else 0.0
    return DatasetIntelligence(
        rows=len(rows),
        malformed_rows=malformed,
        duplicates=duplicates,
        empty_assistant=empty_assistant,
        assistant_only=assistant_only,
        prompt_injections=prompt_injections,
        boilerplate_answers=boilerplate,
        unsupported_claim_markers=unsupported_claim_markers,
        avg_response_chars=sum(response_lengths) / max(1, len(response_lengths)),
        longest_chars=max((len(_row_text(row)) for row in rows), default=0),
        flagged_rows=flagged_rows,
        clean_row_percent=clean_percent,
        role_counts=dict(role_counts),
        length_histogram=dict(length_hist),
        token_histogram=dict(token_hist),
        character_sets=dict(character_sets),
    )


def dataset_intelligence_report(dataset: Path, *, limit: int = 1000) -> DiagnosisReport:
    intel = analyze_dataset_intelligence(dataset, limit=limit)
    report = DiagnosisReport(metadata={"dataset": str(dataset), **intel.to_dict()})
    severity = "ok" if intel.flagged_rows == 0 and intel.malformed_rows == 0 else "warning"
    report.add(
        "dataset_intel.coverage",
        f"Rows without flagged issues: {intel.clean_row_percent:.1f}%",
        severity,
        f"{intel.flagged_rows} loaded row(s) matched at least one rule; {intel.malformed_rows} row(s) could not be parsed.",
        "Review the reported rule matches, repair confirmed issues, then rerun dataset-doctor.",
    )
    checks = [
        ("dataset_intel.duplicates", "Duplicated conversations", intel.duplicates, "Deduplicate repeated examples."),
        ("dataset_intel.empty_assistant", "Empty assistant responses", intel.empty_assistant, "Remove or repair blank assistant messages."),
        ("dataset_intel.assistant_only", "Assistant-only messages", intel.assistant_only, "Keep the user prompt with each assistant answer."),
        ("dataset_intel.prompt_injection", "Possible prompt injections", intel.prompt_injections, "Review and remove hostile instruction text."),
        ("dataset_intel.malformed", "Malformed rows", intel.malformed_rows, "Fix invalid JSON/JSONL rows before training."),
        ("dataset_intel.boilerplate", "Possible boilerplate answers", intel.boilerplate_answers, "Remove canned refusal or browsing-disclaimer answers when they are not desired."),
        ("dataset_intel.unsupported_claim_markers", "Unsupported-claim phrase matches", intel.unsupported_claim_markers, "Review answers that use vague attribution; a phrase match is not proof that an answer is false."),
    ]
    for code, title, count, fix in checks:
        report.add(
            code,
            title,
            "warning" if count else "ok",
            f"{count} row(s) matched this dataset-intelligence check.",
            fix if count else None,
            count=count,
        )
    return report


def dataset_report_html(dataset: Path, *, limit: int = 1000) -> str:
    intel = analyze_dataset_intelligence(dataset, limit=limit)
    duplicate_rows = _duplicate_cluster_rows(dataset, limit=limit)

    def rows(mapping: dict[str, int]) -> str:
        maximum = max(mapping.values(), default=1)
        return "".join(
            "<tr>"
            f"<td>{html.escape(str(key))}</td>"
            f"<td>{value}</td>"
            f"<td><code>{_text_bar(value, maximum)}</code></td>"
            "</tr>"
            for key, value in mapping.items()
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PEFT Doctor Dataset Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    td, th {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    .score {{ font-size: 24px; font-weight: 700; }}
    code {{ white-space: pre; }}
  </style>
</head>
<body>
  <h1>PEFT Doctor Dataset Report</h1>
  <p class="score">Rows without flagged issues: {intel.clean_row_percent:.1f}%</p>
  <h2>Summary</h2>
  <table>
    <tr><th>Rows scanned</th><td>{intel.rows}</td></tr>
    <tr><th>Malformed rows</th><td>{intel.malformed_rows}</td></tr>
    <tr><th>Duplicates</th><td>{intel.duplicates}</td></tr>
    <tr><th>Empty assistant responses</th><td>{intel.empty_assistant}</td></tr>
    <tr><th>Assistant-only messages</th><td>{intel.assistant_only}</td></tr>
    <tr><th>Possible prompt injections</th><td>{intel.prompt_injections}</td></tr>
    <tr><th>Unsupported-claim phrase matches</th><td>{intel.unsupported_claim_markers}</td></tr>
    <tr><th>Average response length</th><td>{intel.avg_response_chars:.1f} chars</td></tr>
    <tr><th>Longest conversation</th><td>{intel.longest_chars} chars</td></tr>
  </table>
  <h2>Conversation Length Histogram</h2>
  <table><tr><th>Bucket</th><th>Rows</th><th>Bar</th></tr>{rows(intel.length_histogram)}</table>
  <h2>Token Histogram</h2>
  <table><tr><th>Bucket</th><th>Rows</th><th>Bar</th></tr>{rows(intel.token_histogram)}</table>
  <h2>Role Distribution</h2>
  <table><tr><th>Role</th><th>Count</th><th>Bar</th></tr>{rows(intel.role_counts)}</table>
  <h2>Duplicate Clusters</h2>
  <table><tr><th>Repeated text preview</th><th>Count</th></tr>{duplicate_rows}</table>
  <h2>Character-Set Distribution</h2>
  <table><tr><th>Character bucket</th><th>Rows</th><th>Bar</th></tr>{rows(intel.character_sets)}</table>
</body>
</html>
"""


def _duplicate_cluster_rows(dataset: Path, *, limit: int = 1000) -> str:
    rows, _ = _load_rows_lenient(dataset, limit=limit)
    clusters: Counter[str] = Counter()
    for row in rows:
        normalized = " ".join(_row_text(row).split())
        if normalized:
            clusters[normalized] += 1
    repeated = [(text, count) for text, count in clusters.most_common(10) if count > 1]
    if not repeated:
        return "<tr><td colspan=\"2\">No duplicate clusters found.</td></tr>"
    return "".join(
        "<tr>"
        f"<td>{html.escape(text[:220])}</td>"
        f"<td>{count}</td>"
        "</tr>"
        for text, count in repeated
    )


def write_dataset_report_html(dataset: Path, output: Path, *, limit: int = 1000) -> None:
    output.write_text(dataset_report_html(dataset, limit=limit), encoding="utf-8")


def diagnosis_report(
    script: Optional[Path] = None,
    *,
    dataset: Optional[Path] = None,
    model: Optional[str] = None,
    gpu: Optional[str] = None,
    sequence_length: int = 2048,
    batch_size: int = 1,
    qlora: bool = True,
) -> DiagnosisReport:
    report = DiagnosisReport(
        metadata={
            "mode": "local rule-based diagnosis",
            "script": str(script) if script else None,
            "dataset": str(dataset) if dataset else None,
            "model": model,
            "gpu": gpu,
        }
    )
    if script:
        report.extend(repair_python_file(script, dry_run=True).issues)
    if dataset:
        check_dataset(report, dataset_path=dataset, sequence_length=sequence_length)
        intel = analyze_dataset_intelligence(dataset)
        report.metadata["dataset_rows_without_flagged_issues_percent"] = round(intel.clean_row_percent, 2)
        report.metadata["empty_assistant_responses"] = intel.empty_assistant
        report.metadata["prompt_injections"] = intel.prompt_injections
    if model:
        report.extend(
            estimate_vram_gb(
                model,
                seq_len=sequence_length,
                batch_size=batch_size,
                qlora=qlora,
                target_vram_gb=_gpu_vram(gpu),
            ).issues
        )

    risk = risk_summary(report)
    report.metadata["risk"] = risk
    report.add(
        "doctor.local_reasoning",
        "Pre-flight diagnosis complete",
        "ok" if risk["level"] == "LOW" else "warning",
        f"Run risk is {risk['level']} under the documented rule set.",
        "Review the findings, apply confirmed fixes, then run peft-doctor simulate before training.",
    )
    return report


def diagnosis_text(report: DiagnosisReport) -> str:
    risk = report.metadata.get("risk") or risk_summary(report)
    lines = [
        "Diagnosis:",
        "",
        "High-risk findings:" if risk["level"] == "HIGH" else "Pre-flight findings:",
    ]
    for issue in report.sorted_issues()[:10]:
        marker = {
            "ok": "OK",
            "warning": "WARN",
            "error": "FAIL",
            "info": "INFO",
        }.get(issue.severity, issue.severity.upper())
        lines.append(f"{marker}: {issue.title}")
    lines.extend(
        [
            "",
            "Recommended fixes:",
        ]
    )
    fixes = [
        issue.fix
        for issue in report.sorted_issues()
        if issue.fix and issue.severity in {"error", "warning"}
    ]
    if not fixes:
        fixes = [
            issue.fix
            for issue in report.sorted_issues()
            if issue.fix and issue.severity == "info"
        ]
    for index, fix in enumerate(fixes[:8], start=1):
        lines.append(f"{index}. {fix}")
    if not fixes:
        lines.append("1. No urgent fix was found. Keep the run small and monitor loss.")
    lines.extend(["", "Risk level is rule-based; it is not a probability of training success."])
    return "\n".join(lines)


def _gpu_vram(gpu: Optional[str]) -> Optional[float]:
    if not gpu:
        return None
    lowered = gpu.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*gb", lowered)
    if match:
        return float(match.group(1))
    for name, profile in GPU_PROFILES.items():
        if name in lowered:
            return float(profile["vram"])
    return None


def memory_timeline(
    model: str,
    *,
    seq_len: int = 2048,
    batch_size: int = 1,
    qlora: bool = True,
    gradient_checkpointing: bool = True,
) -> DiagnosisReport:
    estimate = estimate_vram_gb(
        model,
        seq_len=seq_len,
        batch_size=batch_size,
        qlora=qlora,
        gradient_checkpointing=gradient_checkpointing,
    )
    detail = next(issue.details for issue in estimate.issues if issue.code == "estimate.total_vram")
    weights = float(detail.get("weight_gb", 0.0)) + float(detail.get("adapter_gb", 0.0))
    activation = float(detail.get("activation_gb", 0.0))
    optimizer = float(detail.get("optimizer_gb", 0.0))
    overhead = float(detail.get("overhead_gb", 0.0))
    phases = [
        {"phase": "Step 0 load", "gb": round(weights + overhead * 0.25, 2)},
        {"phase": "Forward", "gb": round(weights + activation * 0.55 + overhead * 0.5, 2)},
        {"phase": "Backward", "gb": round(weights + activation + overhead, 2)},
        {"phase": "Optimizer", "gb": round(weights + optimizer + activation * 0.35 + overhead, 2)},
    ]
    peak = max(phase["gb"] for phase in phases)
    report = DiagnosisReport(metadata={"model": model, "phases": phases, "peak_gb": peak, "method": "formula-based phase estimate"})
    for phase in phases:
        report.add(
            f"memory_timeline.{phase['phase'].lower().replace(' ', '_')}",
            phase["phase"],
            "ok",
            f"{phase['gb']} GB {_text_bar(float(phase['gb']), peak)}",
        )
    report.add(
        "memory_timeline.peak",
        f"Peak {peak} GB",
        "ok",
        "The highest estimated memory point is shown by phase.",
        "Lower sequence length or batch size if peak is above GPU VRAM.",
    )
    return report


def simulate_training(
    *,
    model: str,
    dataset: Optional[Path] = None,
    gpu: Optional[str] = None,
    seq_len: int = 2048,
    batch_size: int = 1,
    qlora: bool = True,
    eval_batch_size: int = 1,
    save_steps: int = 500,
    total_steps: int = 1000,
    disk_free_gb: Optional[float] = None,
    checkpoint_gb: Optional[float] = None,
    save_total_limit: Optional[int] = None,
    tokens_per_second: Optional[float] = None,
    epochs: float = 1.0,
) -> DiagnosisReport:
    target_vram = _gpu_vram(gpu)
    estimate = estimate_vram_gb(
        model,
        seq_len=seq_len,
        batch_size=batch_size,
        qlora=qlora,
        target_vram_gb=target_vram,
    )
    peak = float(estimate.metadata["estimated_total_gb"])
    rows: Optional[int] = None
    if dataset and dataset.exists():
        rows = max(1, analyze_dataset_intelligence(dataset).rows)
    eta_hours = None
    if rows is not None and tokens_per_second and tokens_per_second > 0:
        eta_hours = round((rows * seq_len * epochs) / tokens_per_second / 3600, 2)
    report = DiagnosisReport(
        metadata={
            "model": model,
            "dataset": str(dataset) if dataset else None,
            "gpu": gpu,
            "peak_vram_gb": peak,
            "eta_hours": eta_hours,
            "tokens_per_second": tokens_per_second,
            "steps": ["Reading plan", "Sampling dataset", "Estimating VRAM", "Checking checkpoint schedule"],
        }
    )
    report.extend(estimate.issues)
    report.add(
        "simulate.vram_plan",
        "VRAM planning result",
        "ok" if not target_vram or peak <= target_vram else "warning",
        f"The formula-based peak estimate is {peak:.1f} GB.",
        "Reduce batch size or sequence length if peak exceeds available VRAM.",
    )
    if eta_hours is not None:
        report.add(
            "simulate.eta",
            f"Runtime estimate {eta_hours}h",
            "info",
            "Runtime uses the supplied measured tokens-per-second value and assumes every sampled row reaches seq_len.",
            "Use throughput from a short run on the same model, hardware, and settings.",
        )
    else:
        report.add(
            "simulate.eta_unavailable",
            "Runtime was not estimated",
            "info",
            "No measured throughput was supplied, or no dataset row count was available.",
            "Pass --tokens-per-second from a short benchmark and --dataset to estimate runtime.",
        )
    if target_vram and peak * (1.0 + max(0, eval_batch_size - batch_size) * 0.25) > target_vram:
        report.add(
            "simulate.eval_oom",
            "CUDA OOM risk at evaluation",
            "warning",
            "Evaluation can peak higher than training when eval batch size is larger or cache is enabled.",
            "Use eval batch size 1, eval_strategy='no' for first run, or evaluate after training.",
        )
    checkpoint_count = math.ceil(total_steps / max(1, save_steps))
    retained = min(checkpoint_count, save_total_limit) if save_total_limit is not None else checkpoint_count
    needed_disk = checkpoint_gb * retained if checkpoint_gb is not None else None
    report.metadata.update({"scheduled_checkpoints": checkpoint_count, "retained_checkpoints": retained, "checkpoint_disk_gb": needed_disk})
    if disk_free_gb is not None and needed_disk is not None and needed_disk > disk_free_gb:
        report.add(
            "simulate.disk_full",
            "Checkpoint schedule may fill disk",
            "warning",
            f"The explicit checkpoint budget needs {needed_disk:.1f} GB, above the given free disk.",
            "Increase save_steps, set save_total_limit=2, or save to a larger disk.",
        )
    elif save_steps <= 100:
        report.add(
            "simulate.frequent_saves",
            "Frequent checkpoint saves",
            "warning",
            "Saving every 100 steps or less can fill disks on long runs.",
            "Use save_total_limit=2 or a larger save_steps value.",
        )
    return report


def estimate_cost_report(
    *,
    model: str,
    dataset_size: int = 8000,
    seq_len: int = 2048,
    batch_size: int = 1,
    qlora: bool = True,
    gpus: Optional[list[str]] = None,
    offers: Optional[list[str]] = None,
    epochs: float = 1.0,
) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"model": model, "dataset_size": dataset_size, "offers": offers or []})
    if not offers:
        detail = f" GPU names without rates: {', '.join(gpus)}." if gpus else ""
        report.add(
            "cost.inputs_required",
            "Price and throughput inputs are required",
            "error",
            "Cloud prices and training throughput vary by provider, region, model, and software stack." + detail,
            "Pass --offer NAME:HOURLY_RATE:TOKENS_PER_SECOND, for example --offer 'L4:0.80:950'.",
        )
        return report
    if dataset_size <= 0 or seq_len <= 0 or batch_size <= 0 or epochs <= 0:
        report.add("cost.invalid_plan", "Cost plan contains non-positive values", "error", "Dataset size, sequence length, batch size, and epochs must be positive.")
        return report
    total_tokens = dataset_size * seq_len * epochs
    rows = []
    for raw in offers:
        parts = raw.rsplit(":", 2)
        if len(parts) != 3:
            report.add("cost.offer_format", "Cost offer has the wrong format", "error", f"Could not parse {raw!r}.", "Use NAME:HOURLY_RATE:TOKENS_PER_SECOND.")
            continue
        gpu, rate_text, throughput_text = parts
        try:
            rate = float(rate_text)
            throughput = float(throughput_text)
        except ValueError:
            report.add("cost.offer_number", "Cost offer contains an invalid number", "error", f"Could not parse {raw!r}.", "Use numeric hourly rate and tokens per second.")
            continue
        if rate < 0 or throughput <= 0:
            report.add("cost.offer_range", "Cost offer contains an invalid value", "error", f"Could not use {raw!r}.", "Hourly rate must be non-negative and throughput must be positive.")
            continue
        hours = round(total_tokens / throughput / 3600, 3)
        cost = round(hours * rate, 2)
        rows.append({"gpu": gpu, "hours": hours, "cost": cost, "hourly_rate": rate, "tokens_per_second": throughput})
        report.add(
            f"cost.{gpu.lower().replace(' ', '_')}",
            f"{gpu}: {hours}h, ${cost}",
            "info",
            "Calculated from the supplied hourly rate and measured token throughput.",
            vram_estimate_gb=estimate_vram_gb(model, seq_len=seq_len, batch_size=batch_size, qlora=qlora).metadata["estimated_total_gb"],
        )
    if not rows:
        return report
    cheapest = min(rows, key=lambda row: row["cost"])
    report.metadata["estimates"] = rows
    report.metadata["cheapest"] = cheapest
    report.add("cost.cheapest", f"Lowest supplied estimate: {cheapest['gpu']}", "ok", f"Calculated cost ${cheapest['cost']} from the values provided.")
    return report


def advise_hyperparameters(
    *,
    model: str,
    dataset_size: int,
    gpu_vram_gb: Optional[float] = None,
    task: str = "chat",
) -> DiagnosisReport:
    if dataset_size <= 0 or (gpu_vram_gb is not None and gpu_vram_gb <= 0):
        report = DiagnosisReport(metadata={"model": model, "dataset_size": dataset_size, "gpu_vram_gb": gpu_vram_gb})
        report.add("hparams.invalid_input", "Hyperparameter inputs are invalid", "error", "Dataset size and supplied GPU VRAM must be positive.")
        return report
    params = infer_params_billion(model)
    if dataset_size < 1000 or (gpu_vram_gb and gpu_vram_gb <= 12):
        rank = 8
    elif dataset_size < 5000 or params <= 3:
        rank = 16
    elif dataset_size < 25000 and (not gpu_vram_gb or gpu_vram_gb >= 20):
        rank = 32
    else:
        rank = 64
    dropout = 0.08 if dataset_size < 1000 else 0.05 if dataset_size < 50000 else 0.03
    alpha = rank * 2
    report = DiagnosisReport(
        metadata={
            "model": model,
            "dataset_size": dataset_size,
            "gpu_vram_gb": gpu_vram_gb,
            "task": task,
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "basis": "rule-based starting point",
        }
    )
    report.add(
        "hparams.starting_point",
        f"rank={rank}, alpha={alpha}, dropout={dropout}",
        "ok",
        f"Rule-based starting point for {model} with {dataset_size} samples on {gpu_vram_gb or 'unknown'} GB VRAM.",
        "Compare at least two ranks on a held-out evaluation set; this rule does not predict final quality.",
    )
    return report


def monitor_report(log_file: Optional[Path] = None) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"log_file": str(log_file) if log_file else None})
    losses = []
    if log_file and log_file.exists():
        report.extend(scan_training_log(log_file))
        for line in log_file.read_text(encoding="utf-8").splitlines():
            match = re.search(r"(?:loss|train_loss)\s*[=:]\s*([-+0-9.]+)", line)
            if match:
                try:
                    losses.append(float(match.group(1)))
                except ValueError:
                    pass
    else:
        report.add("monitor.no_log", "No log file provided", "info", "Live GPU monitoring is optional; pass a Trainer log for health analysis.")
    gpu_snapshot = _gpu_runtime_snapshot()
    trend = _loss_trend(losses)
    report.metadata.update(
        {
            "loss_trend": trend,
            "loss_points": len(losses),
            "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
            "gpu": gpu_snapshot,
        }
    )
    report.add(
        "monitor.gpu_snapshot",
        "GPU runtime snapshot",
        "ok" if gpu_snapshot.get("available") else "info",
        gpu_snapshot["message"],
        "Compare this snapshot with the trainer process and framework metrics when debugging memory pressure.",
    )
    report.add(
        "monitor.summary",
        "Training log summary",
        "error" if report.has_errors else "warning" if report.has_warnings else "ok",
        f"Loss trend: {trend}; parsed {len(losses)} finite loss value(s).",
        "Review reported log signatures and inspect raw metrics before deciding whether to stop a run.",
    )
    return report


def _gpu_runtime_snapshot() -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return {"available": False, "message": "PyTorch is not available, so live GPU usage could not be read."}
    if not torch.cuda.is_available():
        return {"available": False, "message": "CUDA is not available in this runtime."}
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    total = props.total_memory / (1024**3)
    return {
        "available": True,
        "name": props.name,
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "total_gb": round(total, 2),
        "usage_percent": round((reserved / total) * 100, 1) if total else 0.0,
        "message": (
            f"{props.name}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, "
            f"{total:.1f} GB total."
        ),
    }


def _loss_trend(losses: list[float]) -> str:
    if len(losses) < 2:
        return "not enough data"
    first = losses[0]
    last = losses[-1]
    tolerance = max(0.01, abs(first) * 0.02)
    if last < first - tolerance:
        return "falling"
    if last > first + tolerance:
        return "rising"
    return "roughly flat"


def auto_tune_report(
    *,
    model: str,
    batch_size: int,
    grad_accum: int,
    seq_len: int,
    target_vram_gb: float,
    qlora: bool = True,
) -> DiagnosisReport:
    effective = batch_size * grad_accum
    tuned_batch = max(1, min(batch_size, 2))
    tuned_seq = min(seq_len, 2048 if target_vram_gb <= 16 else 4096)
    tuned_accum = max(1, math.ceil(effective / tuned_batch))
    before = float(estimate_vram_gb(model, seq_len=seq_len, batch_size=batch_size, qlora=qlora).metadata["estimated_total_gb"])
    after = float(estimate_vram_gb(model, seq_len=tuned_seq, batch_size=tuned_batch, qlora=qlora).metadata["estimated_total_gb"])
    reduction = max(0, round((1 - after / max(before, 0.01)) * 100))
    report = DiagnosisReport(
        metadata={
            "model": model,
            "before_batch_size": batch_size,
            "before_grad_accum": grad_accum,
            "after_batch_size": tuned_batch,
            "after_grad_accum": tuned_accum,
            "after_seq_len": tuned_seq,
            "effective_batch": effective,
            "memory_reduced_percent": reduction,
        }
    )
    report.add(
        "autotune.batch",
        "Memory-oriented adjustment",
        "ok" if after <= target_vram_gb else "warning",
        f"batch_size={tuned_batch}, gradient_accumulation_steps={tuned_accum}, seq_len={tuned_seq}.",
        f"Same effective batch target with about {reduction}% lower estimated memory.",
    )
    return report


def score_report(
    *,
    model: Optional[str] = None,
    dataset: Optional[Path] = None,
    gpu: Optional[str] = None,
    script: Optional[Path] = None,
) -> DiagnosisReport:
    findings = DiagnosisReport()
    if dataset:
        findings.extend(dataset_intelligence_report(dataset).issues)
    if script:
        findings.extend(repair_python_file(script, dry_run=True).issues)
    if not dataset and not script:
        report = DiagnosisReport(
            metadata={
                "project_score": None,
                "label": "Insufficient input",
                "model": model,
                "gpu": gpu,
            }
        )
        report.add(
            "score.input_missing",
            "No scorable project input was supplied",
            "warning",
            "A model or GPU name alone is not enough to score a project.",
            "Pass a training script, a dataset, or both.",
        )
        return report
    errors = findings.summary.get("error", 0)
    warnings = findings.summary.get("warning", 0)
    overall = max(0, 100 - errors * 25 - warnings * 7)
    label = "No blockers in checked rules" if not errors and not warnings else "Review warnings" if not errors else "Blocked by errors"
    report = DiagnosisReport(
        metadata={
            "project_score": overall,
            "errors": errors,
            "warnings": warnings,
            "scoring_rule": "100 - 25 per error - 7 per warning, minimum 0",
            "label": label,
            "model": model,
            "gpu": gpu,
        }
    )
    report.extend(findings.issues)
    report.add(
        "score.project",
        f"Project Score {overall}/100",
        "error" if errors else "warning" if warnings else "ok",
        f"Status: {label}. This score summarizes only rules that had enough local input to run.",
        "Fix errors first, then review warnings and rerun the same command.",
    )
    return report


def lora_efficiency_report(
    *,
    model: str,
    rank: int = 16,
    target_modules: Optional[list[str]] = None,
    dataset_size: int = 8000,
) -> DiagnosisReport:
    modules = target_modules or recommend_target_modules(model_family=infer_model_family(model_name=model))
    report = DiagnosisReport(
        metadata={
            "model": model,
            "rank": rank,
            "target_modules": modules,
            "dataset_size": dataset_size,
            "target_module_count": len(modules),
        }
    )
    if rank <= 0 or dataset_size <= 0:
        report.add("lora_efficiency.invalid_input", "LoRA planning inputs are invalid", "error", "Rank and dataset size must be positive.")
        return report
    report.add(
        "lora_efficiency.plan",
        f"Rank {rank} across {len(modules)} target module name(s)",
        "info",
        "Adapter bytes, trainable parameters, runtime overhead, and quality cannot be derived from rank and model name alone.",
        "Run adapter-tensors on a saved adapter and compare held-out evaluation metrics across experiments.",
    )
    if rank > 64:
        report.add("lora_efficiency.rank_high", "LoRA rank is high", "warning", f"Configured rank is {rank}.", "Verify that the extra trainable capacity improves held-out metrics enough to justify memory and storage.")
    return report


def compare_adapters_report(adapter_a: Path, adapter_b: Path) -> DiagnosisReport:
    info_a = _adapter_info(adapter_a)
    info_b = _adapter_info(adapter_b)
    report = DiagnosisReport(metadata={"adapter_a": info_a, "adapter_b": info_b})
    rank_a = int(info_a.get("r") or 0)
    rank_b = int(info_b.get("r") or 0)
    size_a = float(info_a.get("size_mb") or 0.0)
    size_b = float(info_b.get("size_mb") or 0.0)
    params_a = int(info_a.get("parameter_count") or 0)
    params_b = int(info_b.get("parameter_count") or 0)
    report.add(
        "adapter_compare.summary",
        f"Rank {rank_a} vs {rank_b}",
        "ok",
        (
            f"Files: {size_a:.2f} MiB vs {size_b:.2f} MiB. Tensor parameters: "
            f"{params_a:,} vs {params_b:,}. Targets: {len(info_a.get('target_modules') or [])} vs "
            f"{len(info_b.get('target_modules') or [])}."
        ),
        "Use held-out metrics to choose between adapters; file metadata does not establish output quality.",
    )
    return report


def _adapter_info(path: Path) -> dict[str, Any]:
    config_path = path / "adapter_config.json"
    data: dict[str, Any] = {}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            data = {}
    size = sum(file.stat().st_size for file in path.glob("adapter_model*") if file.is_file()) / (1024 * 1024)
    target_modules = data.get("target_modules") or []
    tensor_count = 0
    parameter_count = 0
    dtypes: Counter[str] = Counter()
    invalid_files = []
    from .operations import read_safetensors_header

    for weight in path.glob("adapter_model*.safetensors"):
        try:
            header = read_safetensors_header(weight)
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                tensor_count += 1
                parameter_count += math.prod(info["shape"])
                dtypes[str(info.get("dtype", "unknown"))] += 1
        except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            invalid_files.append(weight.name)
    return {
        "path": str(path),
        "r": data.get("r"),
        "lora_alpha": data.get("lora_alpha"),
        "target_modules": target_modules,
        "size_mb": round(size, 2),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "dtype_tensor_counts": dict(dtypes),
        "invalid_safetensors": invalid_files,
        "peft_type": data.get("peft_type"),
        "base_model_name_or_path": data.get("base_model_name_or_path"),
    }


def upgrade_suggestions_report() -> DiagnosisReport:
    env = collect_environment()
    report = DiagnosisReport(metadata={"packages": env["packages"]})
    minimums = {**CORE_PACKAGES, **OPTIONAL_FINETUNING_PACKAGES}
    for name in ("transformers", "peft", "trl", "bitsandbytes"):
        target = minimums[name]
        pkg = env["packages"].get(name, {})
        version = pkg.get("version")
        if not version:
            report.add(f"upgrade.{name}.missing", f"{name} is not installed", "info", "This optional fine-tuning package is not available in the current environment.", f"Install with python -m pip install -U {name}>={target} when the project needs it.")
            continue
        old = bool(pkg.get("is_old"))
        report.add(
            f"upgrade.{name}",
            f"{name} {version}",
            "warning" if old else "ok",
            f"Installed version {version}; PEFT Doctor's declared minimum is {target}.",
            f"Upgrade with python -m pip install -U {name}>={target}." if old else None,
            package_minimum=target,
        )
    return report


def gpu_fingerprint_report(gpu: Optional[str] = None) -> DiagnosisReport:
    if not gpu:
        env = collect_environment()
        devices = env.get("cuda", {}).get("devices", [])
        gpu = str(devices[0]["name"]) if devices else "unknown"
    profile = _gpu_profile(gpu)
    report = DiagnosisReport(metadata={"gpu": gpu, "profile": profile})
    if profile:
        report.add(
            "gpu.profile",
            f"{gpu} detected",
            "ok",
            f"Built-in reference VRAM: {profile['vram']} GB.",
            "Run peft-doctor env and confirm the exact card variant, available memory, driver, Torch build, precision support, and attention backend.",
        )
    else:
        report.add("gpu.unknown", "GPU profile unknown", "info", "No built-in profile matched this GPU.", "Run peft-doctor env and use conservative settings.")
    lowered = gpu.lower()
    if "3060" in lowered:
        report.add("gpu.rtx3060.fp16", "RTX 3060 fp16 caution", "warning", "Consumer GPUs can be sensitive to fp16 overflow.", "Use fp16 carefully, lower LR, and prefer QLoRA.")
    if "4090" in lowered:
        report.add("gpu.rtx4090.bf16", "RTX 4090 precision advice", "info", "Ada GPUs are usually better with bf16 when the stack supports it.", "Prefer bf16=True and fp16=False.")
    return report


def _gpu_profile(gpu: str) -> Optional[dict[str, Any]]:
    lowered = gpu.lower()
    for name, profile in GPU_PROFILES.items():
        if name in lowered:
            return dict(profile)
    return None


def history_report(
    root: Path,
    *,
    add_status: Optional[str] = None,
    metric: Optional[str] = None,
    note: Optional[str] = None,
) -> DiagnosisReport:
    history_dir = root / ".peft-doctor"
    history_file = history_dir / "history.jsonl"
    if add_status:
        history_dir.mkdir(parents=True, exist_ok=True)
        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "status": add_status,
            "metric": metric,
            "note": note,
        }
        with history_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    events = []
    if history_file.exists():
        for line in history_file.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    report = DiagnosisReport(metadata={"root": str(root), "runs": events})
    if not events:
        report.add("history.empty", "No PEFT Doctor history found", "info", "No run history exists yet.", "Add one with peft-doctor history --add-status completed --metric 'BLEU +3.1'.")
    for index, event in enumerate(events, start=1):
        status = str(event.get("status", "unknown"))
        report.add(
            f"history.run_{index}",
            f"Run #{index}: {status}",
            "ok" if status.lower() in {"completed", "best", "success"} else "warning",
            str(event.get("note") or event.get("metric") or "Recorded run."),
        )
    return report


def knowledge_base_report(query: str) -> DiagnosisReport:
    lowered = query.lower()
    report = DiagnosisReport(metadata={"query": query})
    matches = [
        (name, data)
        for name, data in KNOWN_ISSUES.items()
        if name in lowered or any(word in name for word in lowered.split())
    ]
    if not matches:
        matches = list(KNOWN_ISSUES.items())[:3]
    for name, data in matches:
        report.add(
            f"kb.{name.replace(' ', '_')}",
            name,
            "info",
            "Matched a bundled troubleshooting rule.",
            str(data["fix"]),
        )
    return report


def _first_problem_row(dataset: Path) -> Optional[dict[str, Any]]:
    rows, malformed = _load_rows_lenient(dataset, limit=200)
    if malformed:
        return {"reason": "Malformed JSON/JSONL row detected.", "preview": "Run dataset-doctor for the exact bad row location."}
    for index, row in enumerate(rows, start=1):
        text = _row_text(row)
        response = _response_text(row)
        lowered = text.lower()
        if any(pattern in lowered for pattern in PROMPT_INJECTION_PATTERNS):
            return {"row": index, "reason": "Possible prompt injection text.", "preview": text[:240]}
        if not response.strip():
            return {"row": index, "reason": "Empty assistant or response text.", "preview": text[:240]}
        if any(pattern in response.lower() for pattern in BOILERPLATE_ANSWER_PATTERNS):
            return {"row": index, "reason": "Possible boilerplate answer.", "preview": response[:240]}
        if any(pattern in response.lower() for pattern in UNSUPPORTED_CLAIM_PATTERNS):
            return {"row": index, "reason": "Possible unsupported answer marker.", "preview": response[:240]}
    return None


def chat_answer_report(
    question: str,
    *,
    dataset: Optional[Path] = None,
    log_file: Optional[Path] = None,
) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"question": question})
    report.extend(knowledge_base_report(question).issues)
    if dataset:
        report.extend(dataset_intelligence_report(dataset).issues[:4])
        problem_row = _first_problem_row(dataset)
        if problem_row:
            report.metadata["problem_row"] = problem_row
            row_label = f"row {problem_row.get('row')}" if problem_row.get("row") else "dataset"
            report.add(
                "chat.problem_row",
                f"Problem sample found in {row_label}",
                "warning",
                str(problem_row["reason"]),
                f"Preview: {problem_row['preview']}",
            )
    if log_file:
        report.extend(scan_training_log(log_file))
    report.add(
        "chat.local_answer",
        "Matched local troubleshooting guidance",
        "ok",
        "The findings combine local file checks with bundled troubleshooting rules.",
        "Run the referenced command for a full report before changing a production training job.",
    )
    return report


def optimize_project_report(
    path: Path,
    *,
    write: bool = False,
    html_report: Optional[Path] = None,
) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"project": str(path), "write": write})
    train_py = path / "train.py"
    if train_py.exists():
        report.extend(repair_python_file(train_py, write=write, dry_run=not write).issues)
        report.add("optimize.config", "Training script reviewed", "ok", "Training script checks and repair candidates were evaluated.")
        report.add("optimize.trainer", "Trainer settings reviewed", "ok", "Warmup, logging, saving, precision, and memory settings were checked.")
        report.add("optimize.lora", "LoRA settings reviewed", "ok", "LoRA target-module and risky-head checks were included.")
        report.add("optimize.tokenizer", "Tokenizer setup reviewed", "ok", "Tokenizer padding and cache/checkpointing compatibility were checked.")
        report.add("optimize.memory", "Memory settings reviewed", "ok", "Batch size, sequence length, QLoRA, and checkpointing risks were reviewed.")
    else:
        report.add("optimize.train_missing", "train.py not found", "warning", "No train.py file was found in the project root.", "Pass the project root that contains train.py.")
    dataset = next(iter(path.glob("*.jsonl")), None)
    if dataset:
        report.extend(dataset_intelligence_report(dataset).issues)
        report.extend(repair_dataset_file(dataset, write=False, dry_run=True).issues)
        report.add("optimize.dataset", "Dataset reviewed", "ok", "Dataset structure, labels, duplicates, and repair opportunities were checked. Dry-run does not modify data.")
    score = score_report(dataset=dataset, script=train_py if train_py.exists() else None)
    report.extend(score.issues)
    report.metadata["project_score"] = score.metadata.get("project_score")
    if html_report:
        write_html_report(report, html_report)
        report.metadata["html_report"] = str(html_report)
        report.add("optimize.html_report", "Generated HTML report", "ok", f"Report written to {html_report}.")
    report.add(
        "optimize.complete",
        "Project optimizer finished",
        "ok" if not report.has_errors else "warning",
        "Config, dataset, trainer, tokenizer, and memory checks were combined into one optimization report.",
        "Run with --write only after reviewing dry-run changes.",
    )
    return report


def parse_policy(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    policy: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line == "policy:":
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            policy[key.strip()] = value.lower() == "true"
        else:
            try:
                policy[key.strip()] = int(value)
            except ValueError:
                policy[key.strip()] = value.strip("'\"")
    return policy


def audit_policy_report(project: Path, policy_path: Path) -> DiagnosisReport:
    policy = parse_policy(policy_path)
    report = DiagnosisReport(metadata={"project": str(project), "policy": policy})
    train_text = ""
    train_py = project / "train.py"
    if train_py.exists():
        train_text = train_py.read_text(encoding="utf-8")
    max_seq = policy.get("max_seq_len")
    if isinstance(max_seq, int):
        found = _find_int_setting(train_text, ["max_seq_length", "sequence_length", "block_size"])
        if found and found > max_seq:
            report.add("audit.max_seq_len", "Sequence length violates policy", "error", f"Configured {found}, policy maximum is {max_seq}.", f"Set max_seq_length <= {max_seq}.")
        else:
            report.add("audit.max_seq_len", "Sequence length policy passed", "ok", f"Policy maximum is {max_seq}.")
    if policy.get("require_bf16"):
        if "bf16=True" in train_text or '"bf16": true' in train_text.lower():
            report.add("audit.bf16", "bf16 policy passed", "ok", "bf16 appears enabled.")
        else:
            report.add("audit.bf16", "bf16 is required", "error", "Policy requires bf16=True.", "Set bf16=True on supported GPUs.")
    if policy.get("forbid_fp16") and ("fp16=True" in train_text or '"fp16": true' in train_text.lower()):
        report.add("audit.fp16", "fp16 is forbidden", "error", "Policy forbids fp16=True.", "Set fp16=False.")
    elif policy.get("forbid_fp16"):
        report.add("audit.fp16", "fp16 policy passed", "ok", "No fp16=True setting was found.")
    if policy.get("require_dataset_validation"):
        has_dataset = any(project.glob("*.jsonl")) or any(project.glob("*.json"))
        report.add(
            "audit.dataset_validation",
            "Dataset validation policy",
            "ok" if has_dataset else "warning",
            "Dataset file found." if has_dataset else "No local dataset file was found for validation.",
            "Run peft-doctor dataset-doctor data.jsonl before training." if not has_dataset else None,
        )
    return report


def _find_int_setting(text: str, names: list[str]) -> Optional[int]:
    for name in names:
        match = re.search(rf"{re.escape(name)}\s*=\s*(\d+)", text)
        if match:
            return int(match.group(1))
    return None


def cloud_plan_report() -> DiagnosisReport:
    report = DiagnosisReport(metadata={"cloud": "roadmap", "privacy": "no upload in local CLI"})
    report.add(
        "cloud.local_first",
        "Cloud mode is not enabled in the local CLI",
        "ok",
        "PEFT Doctor does not upload logs, adapters, configs, or datasets from this command.",
        "For teams, share generated markdown/HTML reports manually until a hosted service exists.",
    )
    report.add(
        "cloud.roadmap",
        "PEFT Doctor Cloud roadmap",
        "info",
        "A future hosted service could provide diagnostics, optimization suggestions, benchmark comparisons, reproducibility reports, and downloadable fixes.",
        "Keep sensitive datasets local; upload metadata or redacted reports only when a future cloud product supports it.",
    )
    return report
