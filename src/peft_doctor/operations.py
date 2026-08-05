"""Operational safety checks for PEFT training projects.

These checks stay local, avoid loading pickle checkpoints, and keep large-file
work bounded or streaming so they remain useful on Colab and small machines.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from .datasets import load_dataset_records
from .environment import collect_environment
from .report import DiagnosisReport


MANIFEST_SUFFIXES = {".py", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".txt"}
MANIFEST_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "venv",
}
RESUME_COMPARE_KEYS = {
    "base_model_name_or_path": "error",
    "model_name_or_path": "error",
    "target_modules": "error",
    "r": "error",
    "lora_alpha": "warning",
    "max_seq_length": "warning",
    "max_length": "warning",
    "per_device_train_batch_size": "warning",
    "gradient_accumulation_steps": "warning",
    "learning_rate": "warning",
    "seed": "warning",
    "bf16": "warning",
    "fp16": "warning",
    "load_in_4bit": "warning",
    "load_in_8bit": "warning",
}


def calculate_training_plan(
    *,
    dataset_size: int,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    world_size: int = 1,
    num_train_epochs: float = 3.0,
    max_steps: Optional[int] = None,
    warmup_ratio: float = 0.03,
    save_steps: int = 500,
    save_total_limit: Optional[int] = 2,
) -> DiagnosisReport:
    """Calculate effective batch size, optimizer steps, warmup, and saves."""

    report = DiagnosisReport(metadata={"dataset_size": dataset_size})
    values = {
        "dataset_size": dataset_size,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "world_size": world_size,
        "save_steps": save_steps,
    }
    invalid = [name for name, value in values.items() if value <= 0]
    if num_train_epochs <= 0 and not max_steps:
        invalid.append("num_train_epochs")
    if not 0 <= warmup_ratio < 1:
        invalid.append("warmup_ratio")
    if save_total_limit is not None and save_total_limit < 1:
        invalid.append("save_total_limit")
    if invalid:
        report.add(
            "plan.invalid_input",
            "Training plan inputs are invalid",
            "error",
            f"Positive values are required; invalid fields: {', '.join(sorted(set(invalid)))}.",
            "Correct the values and run the plan again.",
        )
        return report

    effective_batch = per_device_batch_size * gradient_accumulation_steps * world_size
    steps_per_epoch = max(1, math.ceil(dataset_size / effective_batch))
    total_steps = int(max_steps) if max_steps and max_steps > 0 else math.ceil(steps_per_epoch * num_train_epochs)
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    scheduled_saves = total_steps // save_steps
    retained_saves = min(scheduled_saves, save_total_limit) if save_total_limit else scheduled_saves
    examples_seen = min(dataset_size * num_train_epochs, total_steps * effective_batch)
    report.metadata.update(
        {
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "world_size": world_size,
            "effective_batch_size": effective_batch,
            "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_steps,
            "warmup_steps": warmup_steps,
            "scheduled_checkpoints": scheduled_saves,
            "retained_checkpoints": retained_saves,
            "estimated_examples_seen": round(examples_seen),
        }
    )
    report.add(
        "plan.summary",
        "Training plan calculated",
        "ok",
        (
            f"Effective batch {effective_batch}; {steps_per_epoch} optimizer steps per epoch; "
            f"{total_steps} total optimizer steps; {warmup_steps} warmup steps."
        ),
        "Use these values to verify Trainer logs and learning-rate scheduling before a long run.",
    )
    if scheduled_saves > 0 and save_total_limit is None:
        report.add(
            "plan.unbounded_checkpoints",
            "Checkpoint retention is unbounded",
            "warning",
            f"The plan schedules about {scheduled_saves} checkpoints without a retention limit.",
            "Set save_total_limit=2 or another deliberate limit to control disk usage.",
        )
    return report


def check_disk_budget(
    path: Union[str, Path],
    *,
    expected_checkpoint_gb: float = 1.0,
    save_total_limit: int = 2,
    dataset_gb: float = 0.0,
    reserve_gb: float = 5.0,
) -> DiagnosisReport:
    """Compare real free disk space with an explicit checkpoint budget."""

    requested = Path(path).expanduser()
    probe = requested.resolve(strict=False)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    report = DiagnosisReport(metadata={"requested_path": str(requested), "checked_path": str(probe)})
    if not probe.exists():
        report.add("disk.path_missing", "Disk path does not exist", "error", f"Could not find an existing parent for {requested}.", "Choose an existing output path.")
        return report
    if min(expected_checkpoint_gb, dataset_gb, reserve_gb) < 0 or save_total_limit < 0:
        report.add("disk.invalid_budget", "Disk budget is invalid", "error", "Disk estimates and retention limits cannot be negative.", "Use zero or positive values.")
        return report

    usage = shutil.disk_usage(probe)
    free_gb = usage.free / (1024**3)
    required_gb = expected_checkpoint_gb * save_total_limit + dataset_gb + reserve_gb
    margin_gb = free_gb - required_gb
    report.metadata.update(
        {
            "free_gb": round(free_gb, 2),
            "total_gb": round(usage.total / (1024**3), 2),
            "required_budget_gb": round(required_gb, 2),
            "margin_gb": round(margin_gb, 2),
            "expected_checkpoint_gb": expected_checkpoint_gb,
            "save_total_limit": save_total_limit,
        }
    )
    severity = "error" if margin_gb < 0 else "warning" if margin_gb < max(2.0, required_gb * 0.2) else "ok"
    report.add(
        "disk.budget",
        "Training disk budget",
        severity,
        f"{free_gb:.2f} GB free; explicit training budget is {required_gb:.2f} GB; margin is {margin_gb:.2f} GB.",
        "Reduce checkpoint retention or choose a larger output disk." if severity != "ok" else None,
    )
    return report


def _checkpoint_number(path: Path) -> int:
    name = path.name
    for prefix in ("checkpoint-", "checkpoint_"):
        if name.startswith(prefix):
            try:
                return int(name[len(prefix) :])
            except ValueError:
                return -1
    return -1


def _resolve_checkpoint(path: Path) -> Optional[Path]:
    if not path.exists() or not path.is_dir():
        return None
    if _checkpoint_number(path) >= 0:
        return path
    candidates = [item for item in path.iterdir() if item.is_dir() and _checkpoint_number(item) >= 0]
    return max(candidates, key=_checkpoint_number) if candidates else path


def inspect_trainer_checkpoint(path: Union[str, Path]) -> DiagnosisReport:
    """Inspect a Trainer or Accelerate checkpoint without loading pickle files."""

    requested = Path(path)
    checkpoint = _resolve_checkpoint(requested)
    report = DiagnosisReport(metadata={"requested_path": str(requested)})
    if checkpoint is None:
        report.add("checkpoint.missing", "Checkpoint path is missing", "error", f"No checkpoint directory exists at {requested}.", "Pass a checkpoint directory or an output directory containing checkpoint-* folders.")
        return report
    report.metadata["checkpoint"] = str(checkpoint)
    report.metadata["checkpoint_step"] = _checkpoint_number(checkpoint)

    weight_patterns = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    weights = [checkpoint / name for name in weight_patterns if (checkpoint / name).exists()]
    weights.extend(checkpoint.glob("model-*.safetensors"))
    weights.extend(checkpoint.glob("pytorch_model-*.bin"))
    if not weights:
        report.add("checkpoint.weights_missing", "Checkpoint weights are missing", "error", "No adapter or model weight file was found.", "Save the model/adapter again before relying on this checkpoint.")
    else:
        empty = [file.name for file in weights if file.stat().st_size == 0]
        if empty:
            report.add("checkpoint.weights_empty", "Checkpoint contains empty weight files", "error", f"Empty files: {', '.join(empty)}.", "Delete the incomplete checkpoint and resume from an earlier valid one.")
        else:
            report.add("checkpoint.weights_found", "Checkpoint weights found", "ok", f"Found {len(weights)} non-empty weight file(s).")

    state_path = checkpoint / "trainer_state.json"
    if state_path.exists():
        try:
            state = _load_json(state_path)
            report.metadata["global_step"] = state.get("global_step")
            report.metadata["best_model_checkpoint"] = state.get("best_model_checkpoint")
            report.add("checkpoint.trainer_state", "Trainer state is readable", "ok", f"trainer_state.json reports global_step={state.get('global_step', 'unknown')}.")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report.add("checkpoint.trainer_state_invalid", "Trainer state is invalid", "error", f"trainer_state.json could not be parsed: {exc}", "Use an earlier checkpoint or regenerate the state.")
    else:
        report.add("checkpoint.trainer_state_missing", "Trainer progress state is missing", "warning", "trainer_state.json was not found, so Trainer progress and best-checkpoint metadata cannot be restored.", "Resume only if you intentionally saved model weights without full Trainer state.")

    state_groups = {
        "optimizer": list(checkpoint.glob("optimizer*")),
        "scheduler": list(checkpoint.glob("scheduler*")),
        "rng": list(checkpoint.glob("rng_state*")) + list(checkpoint.glob("random_states*")),
    }
    for name, files in state_groups.items():
        report.add(
            f"checkpoint.{name}_state",
            f"{name.capitalize()} state {'found' if files else 'missing'}",
            "ok" if files else "warning",
            f"Found {len(files)} {name} state file(s)." if files else f"No {name} state file was found; an exact training resume may not be possible.",
            None if files else "Use a checkpoint saved with full Trainer/Accelerate state.",
        )

    adapter_config = checkpoint / "adapter_config.json"
    if any(file.name.startswith("adapter_model") for file in weights) and not adapter_config.exists():
        report.add("checkpoint.adapter_config_missing", "Adapter config is missing", "error", "Adapter weights exist without adapter_config.json.", "Save the adapter with model.save_pretrained().")
    return report


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _load_structured(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return _load_json(path)
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as exc:
        raise ValueError("YAML input requires PyYAML; install peft-doctor again or use JSON.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        else:
            flat[name] = value
    return flat


def _lookup(flat: dict[str, Any], key: str) -> Any:
    if key in flat:
        return flat[key]
    matches = [value for name, value in flat.items() if name.endswith(f".{key}")]
    return matches[-1] if matches else None


def _normalized(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, float):
        return round(value, 12)
    return value


def check_resume_compatibility(
    checkpoint_path: Union[str, Path],
    *,
    current_config: Optional[Union[str, Path]] = None,
) -> DiagnosisReport:
    """Check whether a checkpoint and current JSON/YAML config agree."""

    checkpoint_report = inspect_trainer_checkpoint(checkpoint_path)
    report = DiagnosisReport(metadata=dict(checkpoint_report.metadata))
    report.extend(checkpoint_report.issues)
    checkpoint = Path(str(checkpoint_report.metadata.get("checkpoint", checkpoint_path)))

    baseline: dict[str, Any] = {}
    for name in ("training_config.json", "peft_doctor_config.json", "adapter_config.json"):
        candidate = checkpoint / name
        if candidate.exists():
            try:
                baseline.update(_flatten(_load_json(candidate)))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                report.add("resume.saved_config_invalid", "Saved checkpoint config is invalid", "error", f"Could not parse {candidate.name}: {exc}", "Repair or replace the checkpoint config.")

    if current_config is None:
        report.add("resume.current_config_missing", "Current config was not supplied", "info", "Checkpoint files were inspected, but configuration compatibility was not compared.", "Pass --config config.json or config.yaml for a full resume check.")
        return report
    config_path = Path(current_config)
    try:
        current = _flatten(_load_structured(config_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.add("resume.current_config_invalid", "Current config is invalid", "error", f"Could not parse {config_path}: {exc}", "Use a valid JSON or YAML configuration file.")
        return report
    report.metadata["current_config"] = str(config_path)
    if not baseline:
        report.add("resume.baseline_missing", "No saved training config to compare", "warning", "The checkpoint has no training_config.json, peft_doctor_config.json, or adapter_config.json.", "Keep a run manifest or training config beside future checkpoints.")
        return report

    compared = 0
    for key, severity in RESUME_COMPARE_KEYS.items():
        before = _lookup(baseline, key)
        after = _lookup(current, key)
        if before is None or after is None:
            continue
        compared += 1
        if _normalized(before) != _normalized(after):
            report.add(
                f"resume.mismatch.{key}",
                f"Resume setting changed: {key}",
                severity,
                f"Checkpoint value is {before!r}; current value is {after!r}.",
                "Use the original value or start a new run instead of resuming this checkpoint.",
            )
    if compared == 0:
        report.add("resume.no_shared_settings", "No shared resume settings found", "info", "The saved and current configs did not contain comparable keys.", "Store training_config.json with future checkpoints.")
    elif not any(issue.code.startswith("resume.mismatch") for issue in report.issues):
        report.add("resume.config_compatible", "Resume configuration matches", "ok", f"Compared {compared} saved setting(s) without a mismatch.")
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _manifest_candidates(root: Path, output: Path, files: Optional[Iterable[Union[str, Path]]]) -> list[Path]:
    if files is not None:
        candidates = [(root / Path(file)).resolve() if not Path(file).is_absolute() else Path(file).resolve() for file in files]
    else:
        candidates = []
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MANIFEST_SUFFIXES:
                continue
            if MANIFEST_IGNORED_DIRS.intersection(path.relative_to(root).parts):
                continue
            candidates.append(path.resolve())
    return sorted({path for path in candidates if path != output.resolve()})


def create_run_manifest(
    root: Union[str, Path],
    output: Union[str, Path],
    *,
    files: Optional[Iterable[Union[str, Path]]] = None,
    max_file_mb: float = 256.0,
) -> DiagnosisReport:
    """Write a reproducibility manifest with streamed SHA-256 file hashes."""

    root_path = Path(root).resolve()
    output_path = Path(output).resolve()
    report = DiagnosisReport(metadata={"root": str(root_path), "manifest": str(output_path)})
    if not root_path.is_dir():
        report.add("manifest.root_missing", "Project root is missing", "error", f"{root_path} is not a directory.", "Pass the training project directory.")
        return report
    if max_file_mb <= 0:
        report.add("manifest.max_size_invalid", "Manifest size limit is invalid", "error", "max_file_mb must be positive.", "Use a positive limit such as 256.")
        return report

    entries = []
    skipped = 0
    for path in _manifest_candidates(root_path, output_path, files):
        if not path.is_file():
            report.add(
                "manifest.file_missing",
                "Manifest input file is missing",
                "warning",
                f"Skipped {path} because it is not a file.",
                "Correct the explicit file list and create the manifest again.",
            )
            skipped += 1
            continue
        try:
            relative = path.relative_to(root_path)
        except ValueError:
            report.add("manifest.outside_root", "Manifest file is outside project root", "warning", f"Skipped {path}.", "Keep tracked files inside the project root.")
            skipped += 1
            continue
        size = path.stat().st_size
        entry: dict[str, Any] = {"path": relative.as_posix(), "size": size}
        if size > max_file_mb * 1024 * 1024:
            entry["sha256"] = None
            entry["skipped_reason"] = f"larger than {max_file_mb:g} MB"
            skipped += 1
        else:
            entry["sha256"] = _sha256_file(path)
        entries.append(entry)

    environment = collect_environment()
    packages = {
        name: info.get("version")
        for name, info in environment.get("packages", {}).items()
        if isinstance(info, dict) and info.get("version")
    }
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_name": root_path.name,
        "root_relative_to_manifest": os.path.relpath(root_path, output_path.parent),
        "git_commit": _git_commit(root_path),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "packages": packages,
        "files": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.metadata.update({"files": len(entries), "hashed": len(entries) - skipped, "skipped": skipped})
    report.add("manifest.written", "Run manifest written", "ok", f"Recorded {len(entries)} file(s); {skipped} large/outside file(s) were not hashed.", "Commit the manifest with experiment notes, but review paths before sharing it.")
    if skipped:
        report.add("manifest.files_skipped", "Some files were not hashed", "info", "Large or outside-root files were recorded without a SHA-256 digest.", "Pass explicit --include files or increase --max-file-mb when full hashing is required.")
    return report


def verify_run_manifest(
    manifest_path: Union[str, Path],
    *,
    max_file_mb: float = 1024.0,
) -> DiagnosisReport:
    """Verify files and important package versions recorded in a run manifest."""

    path = Path(manifest_path).resolve()
    report = DiagnosisReport(metadata={"manifest": str(path)})
    try:
        manifest = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.add("manifest.invalid", "Run manifest is invalid", "error", f"Could not parse {path}: {exc}", "Create a new manifest with peft-doctor manifest.")
        return report
    if manifest.get("schema_version") != 1:
        report.add("manifest.schema_unsupported", "Run manifest schema is unsupported", "error", f"Expected schema_version=1, found {manifest.get('schema_version')!r}.", "Create a new manifest with the installed PEFT Doctor version.")
        return report
    entries = manifest.get("files")
    if not isinstance(entries, list):
        report.add("manifest.files_invalid", "Run manifest file list is invalid", "error", "The files field must be a JSON array.", "Create a new manifest instead of editing this one by hand.")
        return report
    if max_file_mb <= 0:
        report.add("manifest.verify_limit_invalid", "Manifest verification limit is invalid", "error", "max_file_mb must be positive.", "Use a positive limit such as 1024.")
        return report
    root_value = manifest.get("root_relative_to_manifest", ".")
    if not isinstance(root_value, str):
        report.add("manifest.root_invalid", "Run manifest root is invalid", "error", "root_relative_to_manifest must be a string.", "Create a new manifest with PEFT Doctor.")
        return report
    root = (path.parent / root_value).resolve()
    report.metadata["root"] = str(root)
    changed = missing = verified = skipped = 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        candidate = (root / entry["path"]).resolve()
        try:
            candidate.relative_to(root)
            inside = True
        except ValueError:
            inside = False
        if not inside:
            report.add("manifest.path_escape", "Manifest path escapes project root", "error", f"Rejected path: {entry['path']}.", "Remove unsafe paths and recreate the manifest.")
            continue
        if not candidate.exists():
            missing += 1
            report.add("manifest.file_missing", "Manifest file is missing", "error", f"{entry['path']} no longer exists.", "Restore the file or create a new manifest for this run.")
            continue
        expected = entry.get("sha256")
        if expected:
            actual_size = candidate.stat().st_size
            expected_size = entry.get("size")
            if isinstance(expected_size, int) and actual_size != expected_size:
                changed += 1
                report.add("manifest.file_size_changed", "Manifest file size changed", "warning", f"Size changed for {entry['path']}: recorded {expected_size}, current {actual_size} bytes.", "Start a new run identity or restore the recorded file.")
                continue
            if actual_size > max_file_mb * 1024 * 1024:
                skipped += 1
                report.add("manifest.file_too_large", "Manifest file exceeds verification limit", "warning", f"Skipped hashing {entry['path']} because it is larger than {max_file_mb:g} MB.", "Increase --max-file-mb only when the file and manifest are trusted.")
                continue
            actual = _sha256_file(candidate)
            if actual != expected:
                changed += 1
                report.add("manifest.file_changed", "Manifest file changed", "warning", f"SHA-256 changed for {entry['path']}.", "Start a new run identity or restore the recorded file.")
            else:
                verified += 1

    recorded_packages = manifest.get("packages", {})
    if not isinstance(recorded_packages, dict):
        recorded_packages = {}
        report.add("manifest.packages_invalid", "Recorded package versions are invalid", "warning", "The packages field is not a JSON object.", "Create a new manifest with PEFT Doctor.")
    current = collect_environment().get("packages", {})
    for name in ("torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes"):
        expected_version = recorded_packages.get(name)
        current_info = current.get(name, {}) if isinstance(current, dict) else {}
        current_version = current_info.get("version") if isinstance(current_info, dict) else None
        if expected_version and current_version and expected_version != current_version:
            report.add("manifest.package_changed", f"Package version changed: {name}", "warning", f"Recorded {expected_version}; current {current_version}.", "Reproduce in the recorded environment or create a new manifest.")
    report.metadata.update({"verified": verified, "changed": changed, "missing": missing, "skipped": skipped})
    if not changed and not missing and not skipped and not report.has_errors:
        report.add("manifest.verified", "Run manifest verified", "ok", f"Verified {verified} hashed file(s) without content changes.")
    return report


def _row_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, list):
        return "\n".join(
            f"{message.get('role', '')}: {message.get('content', '')}"
            for message in messages
            if isinstance(message, dict)
        )
    return "\n".join(
        str(row[key])
        for key in ("text", "prompt", "instruction", "input", "completion", "response", "output", "answer")
        if isinstance(row.get(key), str)
    )


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _token_length(row: dict[str, Any], tokenizer: Any) -> int:
    input_ids = row.get("input_ids")
    if isinstance(input_ids, list):
        return len(input_ids)
    messages = row.get("messages") or row.get("conversations")
    if tokenizer is not None and isinstance(messages, list) and hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        return len(ids["input_ids"] if isinstance(ids, dict) else ids)
    text = _row_text(row)
    if tokenizer is not None:
        encoded = tokenizer(text, add_special_tokens=True, truncation=False)
        ids = encoded.get("input_ids", []) if isinstance(encoded, dict) else getattr(encoded, "input_ids", [])
        return len(ids)
    return max(len(text.split()), len(text) // 4)


def analyze_token_lengths(
    dataset: Union[str, Path],
    *,
    tokenizer: Any = None,
    limit: int = 1000,
    max_length: int = 2048,
) -> DiagnosisReport:
    """Profile dataset token lengths using a tokenizer or a labeled estimate."""

    path = Path(dataset)
    report = DiagnosisReport(metadata={"dataset": str(path), "limit": limit, "max_length": max_length})
    if limit <= 0 or max_length <= 0:
        report.add("tokens.invalid_input", "Token profile inputs are invalid", "error", "limit and max_length must be positive.", "Use values such as --limit 1000 --max-length 2048.")
        return report
    try:
        rows = load_dataset_records(path, limit=limit)
    except Exception as exc:
        report.add("tokens.dataset_failed", "Dataset could not be read", "error", f"Could not load {path}: {exc}", "Use JSON, JSONL, CSV, or TXT data with valid rows.")
        return report
    if not rows:
        report.add("tokens.dataset_empty", "Dataset sample is empty", "error", "No rows were available for token profiling.", "Add valid training examples.")
        return report

    lengths = []
    failures = 0
    for row in rows:
        try:
            lengths.append(_token_length(row, tokenizer))
        except Exception:
            failures += 1
            lengths.append(max(len(_row_text(row).split()), len(_row_text(row)) // 4))
    over = sum(length > max_length for length in lengths)
    capped = [min(length, max_length) for length in lengths]
    static_padding_waste = 1 - (sum(capped) / (len(capped) * max_length))
    method = "tokenizer" if tokenizer is not None and failures == 0 else "mixed" if tokenizer is not None else "estimate"
    report.metadata.update(
        {
            "rows": len(lengths),
            "method": method,
            "tokenizer_failures": failures,
            "min": min(lengths),
            "p50": _percentile(lengths, 0.50),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": max(lengths),
            "over_limit": over,
            "over_limit_percent": round(over / len(lengths) * 100, 2),
            "static_padding_waste_percent": round(static_padding_waste * 100, 2),
        }
    )
    report.add(
        "tokens.profile",
        "Token-length profile calculated",
        "ok" if over == 0 else "warning",
        f"p50={report.metadata['p50']}, p95={report.metadata['p95']}, max={report.metadata['max']}; {over}/{len(lengths)} rows exceed {max_length} tokens.",
        "Lower max length, filter long outliers, or use packing/length grouping deliberately." if over else None,
        method=method,
    )
    if method != "tokenizer":
        report.add("tokens.estimated", "Token counts are estimated", "info", "At least some lengths used a tokenizer-free character/word estimate.", "Pass --model with an available tokenizer for exact model-specific counts.")
    return report


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def check_distributed_config(
    config_path: Union[str, Path],
    *,
    training_args_path: Optional[Union[str, Path]] = None,
) -> DiagnosisReport:
    """Validate Accelerate/DeepSpeed JSON or YAML configuration conflicts."""

    path = Path(config_path)
    report = DiagnosisReport(metadata={"config": str(path)})
    try:
        config = _load_structured(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.add("distributed.config_invalid", "Distributed config is invalid", "error", f"Could not parse {path}: {exc}", "Use valid Accelerate YAML or DeepSpeed JSON.")
        return report
    flat = _flatten(config)
    training: dict[str, Any] = {}
    if training_args_path is not None:
        try:
            training = _flatten(_load_structured(Path(training_args_path)))
            report.metadata["training_args"] = str(training_args_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report.add("distributed.training_args_invalid", "Training arguments are invalid", "error", f"Could not parse training arguments: {exc}", "Use valid JSON or YAML.")

    ds_file = _lookup(flat, "deepspeed_config_file")
    if isinstance(ds_file, str):
        candidate = (path.parent / ds_file).resolve()
        if not candidate.exists():
            report.add("distributed.deepspeed_file_missing", "DeepSpeed config file is missing", "error", f"Referenced file does not exist: {candidate}.", "Fix deepspeed_config_file or create the file.")
        else:
            try:
                ds_flat = _flatten(_load_structured(candidate))
                flat.update({f"deepspeed_file.{key}": value for key, value in ds_flat.items()})
                report.metadata["deepspeed_config_file"] = str(candidate)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                report.add("distributed.deepspeed_file_invalid", "DeepSpeed config file is invalid", "error", f"Could not parse {candidate}: {exc}", "Use valid DeepSpeed JSON.")

    precision = str(_lookup(flat, "mixed_precision") or "no").lower()
    if precision not in {"no", "fp16", "bf16", "fp8"}:
        report.add("distributed.precision_invalid", "Mixed precision value is invalid", "error", f"mixed_precision={precision!r} is not recognized.", "Use no, fp16, bf16, or fp8 when supported.")
    processes = _lookup(flat, "num_processes")
    if processes is not None:
        try:
            if int(processes) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            report.add("distributed.process_count_invalid", "Process count is invalid", "error", f"num_processes={processes!r} must be a positive integer.", "Set one process per intended worker/GPU.")

    fp16_enabled = _as_bool(_lookup(flat, "fp16.enabled"))
    bf16_enabled = _as_bool(_lookup(flat, "bf16.enabled"))
    if fp16_enabled and bf16_enabled:
        report.add("distributed.precision_conflict", "DeepSpeed enables fp16 and bf16 together", "error", "Both fp16.enabled and bf16.enabled are true.", "Enable only one mixed-precision mode.")

    zero_stage = _lookup(flat, "zero_optimization.stage")
    if zero_stage is None:
        zero_stage = _lookup(flat, "zero_stage")
    try:
        zero_stage_int = int(zero_stage) if zero_stage not in {None, "auto"} else None
    except (TypeError, ValueError):
        zero_stage_int = None
        report.add("distributed.zero_stage_invalid", "DeepSpeed ZeRO stage is invalid", "error", f"zero_stage={zero_stage!r} is invalid.", "Use 0, 1, 2, 3, or auto.")
    offload_optimizer = str(_lookup(flat, "offload_optimizer.device") or _lookup(flat, "offload_optimizer_device") or "none").lower()
    offload_param = str(_lookup(flat, "offload_param.device") or _lookup(flat, "offload_param_device") or "none").lower()
    if offload_optimizer not in {"none", "auto"} and zero_stage_int is not None and zero_stage_int < 2:
        report.add("distributed.optimizer_offload_stage", "Optimizer offload needs ZeRO stage 2 or 3", "error", f"Optimizer offload is {offload_optimizer} with ZeRO stage {zero_stage_int}.", "Use ZeRO stage 2/3 or disable optimizer offload.")
    if offload_param not in {"none", "auto"} and zero_stage_int is not None and zero_stage_int != 3:
        report.add("distributed.parameter_offload_stage", "Parameter offload needs ZeRO stage 3", "error", f"Parameter offload is {offload_param} with ZeRO stage {zero_stage_int}.", "Use ZeRO stage 3 or disable parameter offload.")

    if ds_file:
        ignored = [
            name
            for name in ("gradient_accumulation_steps", "gradient_clipping", "zero_stage", "offload_optimizer_device", "offload_param_device", "mixed_precision")
            if any(key.endswith(f"deepspeed_config.{name}") for key in flat)
        ]
        if ignored:
            report.add("distributed.deepspeed_duplicate_settings", "Accelerate settings may be ignored", "warning", f"deepspeed_config_file is set alongside: {', '.join(ignored)}.", "Keep those values in the DeepSpeed config file to avoid conflicting sources.")

    ds_accum = _lookup(flat, "gradient_accumulation_steps")
    train_accum = _lookup(training, "gradient_accumulation_steps")
    if ds_accum not in {None, "auto"} and train_accum is not None and str(ds_accum) != str(train_accum):
        report.add("distributed.accumulation_mismatch", "Gradient accumulation settings differ", "warning", f"Distributed config uses {ds_accum}; training args use {train_accum}.", "Use auto or keep the values identical.")
    device_map = str(_lookup(training, "device_map") or "").lower()
    try:
        process_count = int(processes or 1)
    except (TypeError, ValueError):
        process_count = 1
    if device_map == "auto" and process_count > 1:
        report.add("distributed.device_map_auto", "device_map=auto conflicts with multi-process training", "error", "Training args use device_map=auto while num_processes is greater than one.", "Remove device_map=auto and let Accelerate/DDP place each process.")

    report.metadata.update({"mixed_precision": precision, "num_processes": processes, "zero_stage": zero_stage, "offload_optimizer": offload_optimizer, "offload_param": offload_param})
    if not report.issues:
        report.add("distributed.config_ok", "Distributed config checks passed", "ok", "No obvious Accelerate/DeepSpeed precision, offload, or process conflicts were found.")
    return report


def _token_id(config: dict[str, Any], name: str) -> Any:
    value = config.get(name)
    if isinstance(value, dict):
        return value.get("id")
    return value


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def check_generation_settings(path: Union[str, Path]) -> DiagnosisReport:
    """Check local generation/tokenizer JSON for stopping and sampling errors."""

    source = Path(path)
    report = DiagnosisReport(metadata={"path": str(source)})
    generation: dict[str, Any] = {}
    tokenizer: dict[str, Any] = {}
    model: dict[str, Any] = {}
    try:
        if source.is_dir():
            for filename, target in (("generation_config.json", generation), ("tokenizer_config.json", tokenizer), ("config.json", model)):
                candidate = source / filename
                if candidate.exists():
                    target.update(_load_json(candidate))
        elif source.exists():
            generation = _load_json(source)
        else:
            raise OSError(f"{source} does not exist")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.add("generation.config_invalid", "Generation config is invalid", "error", f"Could not read generation settings: {exc}", "Pass generation_config.json or a local model directory.")
        return report
    if not generation:
        report.add("generation.config_missing", "Generation config is missing", "warning", "No generation_config.json values were found, so Transformers defaults will be used.", "Save deliberate generation settings after testing the fine-tuned model.")

    max_length = generation.get("max_length")
    max_new_tokens = generation.get("max_new_tokens")
    if max_length is not None and max_new_tokens is not None:
        report.add("generation.length_conflict", "Both max_length and max_new_tokens are set", "warning", "Both length controls are present; max_new_tokens takes precedence in current Transformers behavior.", "Keep max_new_tokens and remove max_length unless compatibility requires it.")
    if max_new_tokens is not None and (not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
        report.add("generation.max_new_tokens_invalid", "max_new_tokens is invalid", "error", f"max_new_tokens={max_new_tokens!r} must be a positive integer.", "Use a positive value such as 128 or 512.")
    min_new_tokens = generation.get("min_new_tokens")
    if min_new_tokens is not None and (not isinstance(min_new_tokens, int) or min_new_tokens < 0):
        report.add("generation.min_new_tokens_invalid", "min_new_tokens is invalid", "error", f"min_new_tokens={min_new_tokens!r} must be a non-negative integer.", "Use zero or a deliberate positive minimum.")
    elif isinstance(max_new_tokens, int) and isinstance(min_new_tokens, int) and min_new_tokens > max_new_tokens:
        report.add("generation.token_length_order", "Minimum generation length exceeds maximum", "error", f"min_new_tokens={min_new_tokens} is greater than max_new_tokens={max_new_tokens}.", "Lower min_new_tokens or raise max_new_tokens.")

    do_sample = _as_bool(generation.get("do_sample", False))
    temperature = generation.get("temperature")
    top_p = generation.get("top_p")
    temperature_number = _finite_number(temperature) if temperature is not None else None
    top_p_number = _finite_number(top_p) if top_p is not None else None
    if temperature is not None and temperature_number is None:
        report.add("generation.temperature_invalid", "Sampling temperature is invalid", "error", f"temperature={temperature!r} is not a finite number.", "Use a finite positive number or remove temperature.")
    elif do_sample and temperature_number is not None and temperature_number <= 0:
        report.add("generation.temperature_invalid", "Sampling temperature is invalid", "error", f"temperature={temperature!r} must be greater than zero when sampling.", "Use a positive temperature or disable sampling.")
    if top_p is not None and (top_p_number is None or not 0 < top_p_number <= 1):
        report.add("generation.top_p_invalid", "Top-p value is invalid", "error", f"top_p={top_p!r} must be greater than zero and no more than one.", "Use a value such as 0.9 or remove top_p.")
    top_k = generation.get("top_k")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 0):
        report.add("generation.top_k_invalid", "Top-k value is invalid", "error", f"top_k={top_k!r} must be a non-negative integer.", "Use zero to disable top-k filtering or a positive integer.")
    if not do_sample and ((temperature_number not in {None, 1.0}) or (top_p_number is not None and top_p_number < 1.0)):
        report.add("generation.sampling_ignored", "Sampling controls are set while sampling is disabled", "warning", "temperature/top_p will not shape greedy decoding when do_sample=False.", "Set do_sample=True or remove sampling-only values.")
    repetition = generation.get("repetition_penalty")
    repetition_number = _finite_number(repetition) if repetition is not None else None
    if repetition is not None and (repetition_number is None or repetition_number <= 0):
        report.add("generation.repetition_penalty_invalid", "Repetition penalty is invalid", "error", f"repetition_penalty={repetition!r} must be positive.", "Use 1.0 for no penalty or a tested positive value.")
    beams = generation.get("num_beams", 1)
    returns = generation.get("num_return_sequences", 1)
    if not isinstance(beams, int) or beams < 1:
        report.add("generation.num_beams_invalid", "num_beams is invalid", "error", f"num_beams={beams!r} must be at least one.", "Use num_beams=1 for greedy/sampling generation.")
    if not isinstance(returns, int) or returns < 1:
        report.add("generation.return_sequences_invalid", "num_return_sequences is invalid", "error", f"num_return_sequences={returns!r} must be at least one.", "Use a positive integer.")
    elif isinstance(beams, int) and not do_sample and returns > beams:
        report.add("generation.return_sequences_invalid", "Too many return sequences for beam search", "error", f"num_return_sequences={returns} is greater than num_beams={beams}.", "Increase num_beams or reduce num_return_sequences.")

    eos = generation.get("eos_token_id", model.get("eos_token_id"))
    pad = generation.get("pad_token_id", model.get("pad_token_id"))
    tokenizer_eos = _token_id(tokenizer, "eos_token_id")
    tokenizer_pad = _token_id(tokenizer, "pad_token_id")
    if eos is None and tokenizer_eos is None:
        report.add("generation.eos_missing", "EOS token id is missing", "warning", "No eos_token_id was found in generation, model, or tokenizer config.", "Set the model-family EOS token so generation can stop cleanly.")
    if pad is None and tokenizer_pad is None:
        report.add("generation.pad_missing", "Pad token id is missing", "warning", "No pad_token_id was found in generation, model, or tokenizer config.", "For causal LMs, set pad_token_id to a deliberate pad token or EOS when appropriate.")
    if eos is not None and tokenizer_eos is not None and _normalized(eos) != _normalized(tokenizer_eos):
        report.add("generation.eos_mismatch", "Generation and tokenizer EOS ids differ", "warning", f"Generation EOS is {eos!r}; tokenizer EOS is {tokenizer_eos!r}.", "Use the EOS id expected by the model's chat template.")
    if pad is not None and tokenizer_pad is not None and _normalized(pad) != _normalized(tokenizer_pad):
        report.add("generation.pad_mismatch", "Generation and tokenizer pad ids differ", "warning", f"Generation pad id is {pad!r}; tokenizer pad id is {tokenizer_pad!r}.", "Keep padding ids aligned across tokenizer, collator, and generation config.")

    report.metadata.update({"do_sample": do_sample, "max_length": max_length, "max_new_tokens": max_new_tokens, "eos_token_id": eos, "pad_token_id": pad})
    if not report.issues:
        report.add("generation.config_ok", "Generation settings checks passed", "ok", "No obvious stopping, padding, length, sampling, or beam-search conflicts were found.")
    return report


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read and validate safetensors metadata without loading tensor data."""

    size = path.stat().st_size
    if size < 8:
        raise ValueError("file is shorter than the safetensors header prefix")
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), byteorder="little", signed=False)
        if header_length <= 0 or header_length > size - 8 or header_length > 100 * 1024 * 1024:
            raise ValueError("invalid safetensors header length")
        raw_header = handle.read(header_length)
        if not raw_header.startswith(b"{"):
            raise ValueError("safetensors header must start with a JSON object")
        header = json.loads(raw_header.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    data_size = size - 8 - header_length
    tensors = 0
    ranges: list[tuple[int, int, str]] = []
    dtype_sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    for name, info in header.items():
        if name == "__metadata__":
            if not isinstance(info, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in info.items()
            ):
                raise ValueError("invalid safetensors metadata")
            continue
        if not isinstance(info, dict) or not isinstance(info.get("shape"), list):
            raise ValueError(f"invalid tensor metadata for {name}")
        shape = info["shape"]
        if not all(isinstance(dimension, int) and dimension >= 0 for dimension in shape):
            raise ValueError(f"invalid tensor shape for {name}")
        offsets = info.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"missing data offsets for {name}")
        start, end = offsets
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start or end > data_size:
            raise ValueError(f"invalid data offsets for {name}")
        dtype = info.get("dtype")
        if not isinstance(dtype, str):
            raise ValueError(f"missing tensor dtype for {name}")
        if dtype in dtype_sizes and end - start != math.prod(shape) * dtype_sizes[dtype]:
            raise ValueError(f"tensor byte length does not match shape and dtype for {name}")
        ranges.append((start, end, name))
        tensors += 1
    if tensors == 0:
        raise ValueError("no tensors found in safetensors header")
    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise ValueError(f"tensor data has a gap or overlap before {name}")
        cursor = end
    if cursor != data_size:
        raise ValueError("tensor offsets do not cover the complete data buffer")
    return header


def _inspect_safetensors_header(path: Path) -> int:
    header = read_safetensors_header(path)
    return sum(1 for name in header if name != "__metadata__")


def check_adapter_integrity(
    adapter_path: Union[str, Path],
    *,
    write_checksums: Optional[Union[str, Path]] = None,
) -> DiagnosisReport:
    """Validate adapter structure, safetensors metadata, and SHA-256 checksums."""

    adapter = Path(adapter_path).resolve()
    report = DiagnosisReport(metadata={"adapter": str(adapter)})
    if not adapter.is_dir():
        report.add("integrity.adapter_missing", "Adapter directory is missing", "error", f"{adapter} is not a directory.", "Pass a local saved adapter directory.")
        return report
    config_path = adapter / "adapter_config.json"
    if not config_path.exists():
        report.add("integrity.config_missing", "adapter_config.json is missing", "error", "The adapter cannot be loaded without its PEFT config.", "Save the adapter with model.save_pretrained().")
    else:
        try:
            config = _load_json(config_path)
            report.metadata["peft_type"] = config.get("peft_type")
            report.metadata["base_model_name_or_path"] = config.get("base_model_name_or_path")
            report.add("integrity.config_valid", "Adapter config is valid JSON", "ok", f"PEFT type: {config.get('peft_type', 'unknown')}.")
            if not config.get("base_model_name_or_path"):
                report.add("integrity.base_model_missing", "Base model id is missing", "warning", "adapter_config.json does not name the original base model.", "Record base_model_name_or_path before sharing the adapter.")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report.add("integrity.config_invalid", "Adapter config is invalid", "error", f"Could not parse adapter_config.json: {exc}", "Restore a valid config from the training output.")

    weights = sorted(adapter.glob("adapter_model*.safetensors")) + sorted(adapter.glob("adapter_model*.bin"))
    if not weights:
        report.add("integrity.weights_missing", "Adapter weights are missing", "error", "No adapter_model safetensors or bin file was found.", "Save the adapter again and verify the output directory.")
        return report
    checksums: dict[str, str] = {}
    tensor_count = 0
    for weight in weights:
        if weight.stat().st_size == 0:
            report.add("integrity.weights_empty", "Adapter weight file is empty", "error", f"{weight.name} is zero bytes.", "Delete the incomplete adapter and save it again.")
            continue
        checksums[weight.name] = _sha256_file(weight)
        if weight.suffix == ".safetensors":
            try:
                count = _inspect_safetensors_header(weight)
                tensor_count += count
                report.add("integrity.safetensors_valid", "Safetensors metadata is valid", "ok", f"{weight.name} contains metadata for {count} tensor(s).")
            except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                report.add("integrity.safetensors_invalid", "Safetensors file is invalid", "error", f"{weight.name}: {exc}", "Restore or re-save the adapter before loading or uploading it.")
        else:
            report.add("integrity.pickle_weights", "Adapter uses a pickle-based .bin file", "warning", f"{weight.name} was hashed but not deserialized.", "Prefer safe_serialization=True to save adapter_model.safetensors.")
    if config_path.exists():
        checksums[config_path.name] = _sha256_file(config_path)

    checksum_path = adapter / "adapter_checksums.json"
    if checksum_path.exists():
        try:
            expected_data = _load_json(checksum_path)
            algorithm = expected_data.get("algorithm", "sha256")
            if str(algorithm).lower() != "sha256":
                report.add("integrity.checksum_algorithm", "Checksum algorithm is unsupported", "error", f"Expected sha256, found {algorithm!r}.", "Regenerate checksums with PEFT Doctor.")
            expected = expected_data.get("files", expected_data)
            if not isinstance(expected, dict):
                raise ValueError("checksum files field must be an object")
            for name, digest in expected.items():
                if name not in checksums:
                    report.add("integrity.checksum_file_missing", "Checksummed adapter file is missing", "error", f"The checksum manifest names {name}, but that file is missing.", "Restore the complete adapter or regenerate checksums from a trusted source.")
                elif digest != checksums[name]:
                    report.add("integrity.checksum_mismatch", "Adapter checksum mismatch", "error", f"SHA-256 mismatch for {name}.", "Do not upload or load this copy until its origin is verified.")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report.add("integrity.checksum_file_invalid", "Checksum manifest is invalid", "warning", f"Could not parse adapter_checksums.json: {exc}", "Regenerate the checksum file.")
    if write_checksums is not None:
        output = Path(write_checksums)
        if not output.is_absolute():
            output = adapter / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"algorithm": "sha256", "files": checksums}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report.metadata["checksums_written"] = str(output)
        report.add("integrity.checksums_written", "Adapter checksums written", "ok", f"Wrote SHA-256 checksums to {output}.")
    report.metadata.update({"weight_files": len(weights), "tensor_count": tensor_count, "checksums": checksums})
    if not report.has_errors:
        report.add("integrity.adapter_ok", "Adapter integrity checks passed", "ok", f"Validated {len(weights)} weight file(s) and calculated SHA-256 checksums.")
    return report
