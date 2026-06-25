"""Conservative auto-repair helpers for common PEFT training files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from .report import DiagnosisReport
from .targets import recommend_target_modules


def _add_change(
    report: DiagnosisReport,
    code: str,
    title: str,
    message: str,
    fix: str,
    *,
    applied: bool = False,
    **details: Any,
) -> None:
    report.add(
        code,
        title,
        "ok" if applied else "warning",
        message,
        fix,
        applied=applied,
        **details,
    )


def _bool_true(source: str, key: str) -> bool:
    patterns = [
        rf"\b{re.escape(key)}\s*=\s*True\b",
        rf'"{re.escape(key)}"\s*:\s*true\b',
        rf'"{re.escape(key)}"\s*:\s*True\b',
    ]
    return any(re.search(pattern, source, flags=re.IGNORECASE) for pattern in patterns)


def _replace_bool(source: str, key: str, value: bool) -> tuple[str, bool]:
    replacement = "True" if value else "False"
    json_replacement = "true" if value else "false"
    updated = re.sub(
        rf"(\b{re.escape(key)}\s*=\s*)(True|False)\b",
        rf"\g<1>{replacement}",
        source,
    )
    updated = re.sub(
        rf'("{re.escape(key)}"\s*:\s*)(true|false|True|False)\b',
        rf"\g<1>{json_replacement}",
        updated,
        flags=re.IGNORECASE,
    )
    return updated, updated != source


def _replace_large_int(source: str, key: str, max_value: int, new_value: int) -> tuple[str, bool]:
    changed = False

    def replace_match(match: re.Match[str]) -> str:
        nonlocal changed
        value = int(match.group("value"))
        if value <= max_value:
            return match.group(0)
        changed = True
        return f"{match.group('prefix')}{new_value}"

    updated = re.sub(
        rf"(?P<prefix>\b{re.escape(key)}\s*=\s*)(?P<value>\d+)",
        replace_match,
        source,
    )
    updated = re.sub(
        rf'(?P<prefix>"{re.escape(key)}"\s*:\s*)(?P<value>\d+)',
        replace_match,
        updated,
    )
    return updated, changed


def _find_call_end(source: str, call_start: int) -> Optional[int]:
    open_index = source.find("(", call_start)
    if open_index < 0:
        return None
    depth = 0
    in_string: Optional[str] = None
    escaped = False
    for index in range(open_index, len(source)):
        char = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            continue
        if char in {'"', "'"}:
            in_string = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _insert_keyword_in_call(source: str, call_name: str, key: str, value: str) -> tuple[str, bool]:
    start = source.find(f"{call_name}(")
    if start < 0:
        return source, False
    end = _find_call_end(source, start)
    if end is None:
        return source, False
    block = source[start:end]
    if re.search(rf"\b{re.escape(key)}\s*=", block):
        return source, False
    line_start = source.rfind("\n", 0, start) + 1
    base_indent = re.match(r"\s*", source[line_start:start]).group(0)
    indent = base_indent + "    "
    insert = f"\n{indent}{key}={value},  # added by peft-doctor"
    return source[:end] + insert + source[end:], True


def _insert_after_line(source: str, pattern: str, line: str) -> tuple[str, bool]:
    if line.strip() in source:
        return source, False
    for match in re.finditer(pattern, source):
        line_end = source.find("\n", match.end())
        if line_end < 0:
            line_end = len(source)
        return source[:line_end] + "\n" + line + source[line_end:], True
    return source, False


def _target_modules_literal(model_family: Optional[str]) -> str:
    modules = recommend_target_modules(model_family=model_family)
    return "[" + ", ".join(repr(module) for module in modules) + "]"


def repair_python_source(
    source: str,
    *,
    model_family: Optional[str] = None,
    oom_safe: bool = True,
) -> tuple[str, DiagnosisReport]:
    """Return patched Python source and a report of safe repairs."""

    report = DiagnosisReport(metadata={"kind": "python"})
    fixed = source
    safe_fixes = 0
    issues = 0

    if "tokenizer" in fixed and "pad_token" not in fixed:
        fixed, changed = _insert_after_line(
            fixed,
            r"tokenizer\s*=.*AutoTokenizer\.from_pretrained\(.*\)",
            "tokenizer.pad_token = tokenizer.eos_token  # added by peft-doctor",
        )
        if changed:
            issues += 1
            safe_fixes += 1
            _add_change(
                report,
                "fix.python.pad_token",
                "Added tokenizer pad token fallback",
                "Tokenizer code was found without a pad_token assignment.",
                "Set tokenizer.pad_token = tokenizer.eos_token for causal LM batching.",
                applied=True,
            )

    if _bool_true(fixed, "gradient_checkpointing") and "model.config.use_cache = False" not in fixed:
        fixed, changed = _insert_after_line(
            fixed,
            r"model\s*=.*from_pretrained\(.*\)",
            "model.config.use_cache = False  # added by peft-doctor for gradient checkpointing",
        )
        if changed:
            issues += 1
            safe_fixes += 1
            _add_change(
                report,
                "fix.python.use_cache",
                "Disabled use_cache for checkpointing",
                "gradient_checkpointing=True was found without model.config.use_cache=False.",
                "Set model.config.use_cache = False before training.",
                applied=True,
            )

    if _bool_true(fixed, "bf16") and _bool_true(fixed, "fp16"):
        fixed, changed = _replace_bool(fixed, "fp16", False)
        if changed:
            issues += 1
            safe_fixes += 1
            _add_change(
                report,
                "fix.python.precision_conflict",
                "Blocked fp16/bf16 conflict",
                "Both bf16=True and fp16=True were configured.",
                "Keep bf16=True and set fp16=False on supported GPUs.",
                applied=True,
            )

    target_literal = _target_modules_literal(model_family)
    risky_target_pattern = r"target_modules\s*=\s*\[[^\]]*(?:lm_head|embed_tokens|wte|wpe)[^\]]*\]"
    if re.search(risky_target_pattern, fixed, flags=re.DOTALL):
        fixed = re.sub(
            r"target_modules\s*=\s*\[[^\]]*\]",
            f"target_modules={target_literal}",
            fixed,
            count=1,
            flags=re.DOTALL,
        )
        issues += 1
        safe_fixes += 1
        _add_change(
            report,
            "fix.python.target_modules",
            "Replaced risky LoRA target modules",
            "target_modules included output or embedding layers.",
            "Use attention and MLP projection targets unless you deliberately want embeddings/head adaptation.",
            applied=True,
            target_modules=target_literal,
        )
    elif "LoraConfig(" in fixed and "target_modules" not in fixed:
        fixed, changed = _insert_keyword_in_call(fixed, "LoraConfig", "target_modules", target_literal)
        if changed:
            issues += 1
            safe_fixes += 1
            _add_change(
                report,
                "fix.python.target_modules_added",
                "Added LoRA target modules",
                "A LoraConfig call was found without target_modules.",
                "Use model-family target modules as a practical starting point.",
                applied=True,
                target_modules=target_literal,
            )

    if oom_safe:
        for key in ["per_device_train_batch_size", "train_batch_size"]:
            fixed, changed = _replace_large_int(fixed, key, 2, 1)
            if changed:
                issues += 1
                safe_fixes += 1
                _add_change(
                    report,
                    f"fix.python.{key}",
                    "Reduced train batch size",
                    f"{key} was above a safe QLoRA starting point.",
                    f"Use {key}=1 and raise gradient_accumulation_steps for effective batch size.",
                    applied=True,
                    key=key,
                )
        for key in ["sequence_length", "max_seq_length", "block_size"]:
            fixed, changed = _replace_large_int(fixed, key, 4096, 2048)
            if changed:
                issues += 1
                safe_fixes += 1
                _add_change(
                    report,
                    f"fix.python.{key}",
                    "Reduced long sequence length",
                    f"{key} was above a conservative memory-safe starting point.",
                    f"Start with {key}=2048, then raise it after the run is stable.",
                    applied=True,
                    key=key,
                )

    for key, value in [
        ("warmup_ratio", "0.03"),
        ("logging_steps", "10"),
        ("save_strategy", '"steps"'),
    ]:
        fixed, changed = _insert_keyword_in_call(fixed, "TrainingArguments", key, value)
        if changed:
            issues += 1
            safe_fixes += 1
            _add_change(
                report,
                f"fix.python.{key}",
                f"Added {key}",
                f"TrainingArguments was missing {key}.",
                f"Use {key}={value} as a stable starter setting.",
                applied=True,
            )

    report.metadata.update(
        {
            "issues_found": issues,
            "safe_fixes": safe_fixes,
            "changed": fixed != source,
        }
    )
    if issues == 0:
        report.add(
            "fix.python.clean",
            "No safe Python fixes found",
            "ok",
            "The fixer did not find any obvious safe Python repairs.",
        )
    return fixed, report


def _repair_config_data(data: dict[str, Any], model_family: Optional[str]) -> tuple[dict[str, Any], DiagnosisReport]:
    report = DiagnosisReport(metadata={"kind": "config"})
    fixed = dict(data)
    issues = 0
    safe_fixes = 0

    def set_value(key: str, value: Any, title: str, message: str) -> None:
        nonlocal issues, safe_fixes
        if fixed.get(key) != value:
            fixed[key] = value
            issues += 1
            safe_fixes += 1
            _add_change(
                report,
                f"fix.config.{key}",
                title,
                message,
                f"Set {key}={value!r}.",
                applied=True,
            )

    if fixed.get("bf16") is True and fixed.get("fp16") is True:
        set_value("fp16", False, "Blocked fp16/bf16 conflict", "Both bf16 and fp16 were enabled.")

    if fixed.get("use_cache") is True and fixed.get("gradient_checkpointing") is True:
        set_value("use_cache", False, "Disabled use_cache", "use_cache conflicts with gradient checkpointing.")

    for key in ["per_device_train_batch_size", "train_batch_size"]:
        value = fixed.get(key)
        if isinstance(value, int) and value > 2:
            set_value(key, 1, "Reduced train batch size", f"{key} was above a safe starter value.")

    for key in ["sequence_length", "max_seq_length", "block_size"]:
        value = fixed.get(key)
        if isinstance(value, int) and value > 4096:
            set_value(key, 2048, "Reduced sequence length", f"{key} was high enough to increase OOM risk.")

    for key, value in [("warmup_ratio", 0.03), ("logging_steps", 10), ("save_strategy", "steps")]:
        if key not in fixed:
            set_value(key, value, f"Added {key}", f"{key} was missing from the config.")

    targets = fixed.get("target_modules")
    risky_targets = {"lm_head", "embed_tokens", "wte", "wpe"}
    if not targets or (isinstance(targets, list) and risky_targets.intersection(set(map(str, targets)))):
        fixed["target_modules"] = recommend_target_modules(model_family=model_family)
        issues += 1
        safe_fixes += 1
        _add_change(
            report,
            "fix.config.target_modules",
            "Set safe LoRA target modules",
            "target_modules was missing or included risky output/embedding layers.",
            "Use attention and MLP projection targets as a safe starting point.",
            applied=True,
            target_modules=", ".join(fixed["target_modules"]),
        )

    report.metadata.update({"issues_found": issues, "safe_fixes": safe_fixes, "changed": fixed != data})
    if issues == 0:
        report.add("fix.config.clean", "No safe config fixes found", "ok", "No config repairs were needed.")
    return fixed, report


def repair_config_file(
    path: Path,
    *,
    model_family: Optional[str] = None,
    output: Optional[Path] = None,
    write: bool = False,
    dry_run: bool = False,
) -> DiagnosisReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        report = DiagnosisReport(metadata={"path": str(path), "kind": "config"})
        report.add("fix.config.unsupported", "Config root is not an object", "error", "Only JSON object configs can be repaired.")
        return report
    fixed, report = _repair_config_data(data, model_family)
    destination = output or path
    if report.metadata.get("changed") and (write or output is not None) and not dry_run:
        destination.write_text(json.dumps(fixed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report.metadata["written_to"] = str(destination)
    return report


def repair_python_file(
    path: Path,
    *,
    model_family: Optional[str] = None,
    output: Optional[Path] = None,
    write: bool = False,
    dry_run: bool = False,
) -> DiagnosisReport:
    source = path.read_text(encoding="utf-8")
    fixed, report = repair_python_source(source, model_family=model_family)
    report.metadata["path"] = str(path)
    destination = output or path
    if report.metadata.get("changed") and (write or output is not None) and not dry_run:
        destination.write_text(fixed, encoding="utf-8")
        report.metadata["written_to"] = str(destination)
    return report


def _load_json_dataset(path: Path) -> tuple[list[dict[str, Any]], str]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows, "jsonl"
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)], "json"
    raise ValueError("Dataset auto-fix currently supports JSONL or JSON list files.")


def repair_dataset_file(
    path: Path,
    *,
    pad_token_id: int = 0,
    output: Optional[Path] = None,
    write: bool = False,
    dry_run: bool = False,
) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"path": str(path), "kind": "dataset", "pad_token_id": pad_token_id})
    try:
        rows, fmt = _load_json_dataset(path)
    except Exception as exc:
        report.add(
            "fix.dataset.read_failed",
            "Dataset could not be read",
            "error",
            f"Could not read dataset for repair: {exc}",
            "Use JSONL or a JSON list file for dataset auto-fix.",
        )
        return report

    changed_rows = 0
    masked_tokens = 0
    for row in rows:
        labels = row.get("labels")
        if not isinstance(labels, list):
            continue
        new_labels = [-100 if label == pad_token_id else label for label in labels]
        if new_labels != labels:
            changed_rows += 1
            masked_tokens += sum(1 for old, new in zip(labels, new_labels) if old != new)
            row["labels"] = new_labels

    if changed_rows:
        _add_change(
            report,
            "fix.dataset.mask_pad_labels",
            "Masked pad tokens in labels",
            "Some label arrays contained pad_token_id.",
            "Replace pad labels with -100 so the model does not learn to predict padding.",
            applied=True,
            changed_rows=changed_rows,
            masked_tokens=masked_tokens,
        )
    else:
        report.add(
            "fix.dataset.clean",
            "No pad labels found",
            "ok",
            "No labels containing the configured pad_token_id were found.",
        )

    report.metadata.update(
        {
            "issues_found": 1 if changed_rows else 0,
            "safe_fixes": 1 if changed_rows else 0,
            "changed": bool(changed_rows),
        }
    )
    destination = output or path
    if changed_rows and (write or output is not None) and not dry_run:
        if fmt == "jsonl":
            text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        else:
            text = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
        destination.write_text(text, encoding="utf-8")
        report.metadata["written_to"] = str(destination)
    return report
