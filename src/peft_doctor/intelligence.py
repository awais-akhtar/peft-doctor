"""Evidence-based local investigations for fine-tuning projects."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from .datasets import load_dataset_records
from .estimator import estimate_vram_gb
from .report import DiagnosisReport


def _read_object(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _rows(path: Path, limit: int = 10000) -> list[dict[str, Any]]:
    return [row for row in load_dataset_records(path, limit=limit) if isinstance(row, dict)]


def _text(row: dict[str, Any]) -> str:
    messages = row.get("messages", row.get("conversations"))
    if isinstance(messages, list):
        return "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict) and message.get("content")
        )
    values: list[str] = []
    for key in ("text", "prompt", "instruction", "input", "question", "completion", "response", "output", "answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return "\n".join(values)


def _has_answer_field(row: dict[str, Any]) -> bool:
    return any(key in row for key in ("completion", "response", "output", "answer", "messages", "conversations"))


def _answer(row: dict[str, Any]) -> str:
    for key in ("completion", "response", "output", "answer"):
        value = row.get(key)
        if isinstance(value, str):
            return value.strip()
    messages = row.get("messages", row.get("conversations"))
    if isinstance(messages, list):
        return "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant"
        ).strip()
    return ""


def _dataset_for(project: Path) -> Optional[Path]:
    for suffix in ("*.jsonl", "*.json", "*.csv"):
        match = next(iter(sorted(project.glob(suffix))), None)
        if match:
            return match
    return None


def investigate_project(project: Path) -> DiagnosisReport:
    """Join local configuration, dataset, and log evidence into causal findings."""
    project = project.expanduser()
    if project.is_file():
        project = project.parent
    report = DiagnosisReport(metadata={"project": str(project), "method": "local evidence rules"})
    dataset = _dataset_for(project)
    train = project / "train.py"
    source = train.read_text(encoding="utf-8", errors="replace") if train.exists() else ""
    rows: list[dict[str, Any]] = []
    if dataset:
        try:
            rows = _rows(dataset)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report.add("investigate.dataset_unreadable", "Dataset could not be inspected", "warning", str(exc))
    empty = [i for i, row in enumerate(rows, 1) if _has_answer_field(row) and not _answer(row)]
    fully_masked = [i for i, row in enumerate(rows, 1) if isinstance(row.get("labels"), list) and row["labels"] and all(x == -100 for x in row["labels"])]
    if empty:
        report.add("investigate.empty_to_loss", "Empty answers may leave rows without supervision", "warning", f"Found {len(empty)} rows without an answer in {dataset.name}; first rows: {empty[:10]}.", "Review or remove those rows, then inspect the resulting labels.", evidence_file=str(dataset), row_numbers=empty[:50], causal_chain=["empty assistant response", "no supervised answer tokens", "loss may have no useful targets"])
    if fully_masked:
        report.add("investigate.masked_to_loss", "Rows contain no supervised label tokens", "error", f"Found {len(fully_masked)} rows whose labels are entirely -100; first rows: {fully_masked[:10]}.", "Check the chat template, completion mask, and label construction.", evidence_file=str(dataset), row_numbers=fully_masked[:50], causal_chain=["all labels ignored", "zero supervised tokens", "invalid or uninformative loss"])
    if "gradient_checkpointing" in source and "use_cache = False" not in source and "use_cache=False" not in source:
        report.add("investigate.cache_checkpointing", "Cache setting conflicts with checkpointed training", "warning", "train.py enables gradient checkpointing but no `use_cache=False` assignment was found.", "Set `model.config.use_cache = False` before training.", evidence_file=str(train), causal_chain=["gradient checkpointing enabled", "model cache may remain enabled", "training can fail or retain unnecessary memory"])
    if "fp16=True" in source and "bf16=True" in source:
        report.add("investigate.precision_conflict", "Both fp16 and bf16 appear enabled", "error", "The training script contains both precision flags set to True.", "Enable only the precision supported by the active hardware.", evidence_file=str(train))
    logs = sorted(project.glob("*.log"))
    if logs:
        from .logs import scan_training_log

        for log in logs[:5]:
            report.extend(scan_training_log(log))
    report.metadata.update({"dataset": str(dataset) if dataset else None, "rows_inspected": len(rows), "training_script": str(train) if train.exists() else None, "logs_inspected": [str(path) for path in logs[:5]]})
    if not report.issues:
        report.add("investigate.no_cause_found", "No rule matched the available evidence", "info", "This result does not establish that the run is healthy. Provide its dataset, training script, and logs for a broader investigation.")
    return report


def inspect_objective(dataset: Path, row_number: int, tokenizer_path: Optional[str] = None) -> DiagnosisReport:
    """Show the selected row's source text and available supervision boundaries."""
    report = DiagnosisReport(metadata={"dataset": str(dataset), "row": row_number, "method": "stored row fields; no training collator executed"})
    if dataset.is_dir():
        candidates = [path for suffix in ("*.jsonl", "*.json", "*.csv") for path in sorted(dataset.glob(suffix))]
        if not candidates:
            report.add("objective.dataset_missing", "No supported dataset found", "error", f"No JSON, JSONL, or CSV dataset found in {dataset}.")
            return report
        dataset = candidates[0]
        report.metadata["dataset"] = str(dataset)
    try:
        rows = _rows(dataset, limit=min(100000, max(10000, row_number)))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.add("objective.dataset_unreadable", "Dataset could not be read", "error", str(exc))
        return report
    if row_number < 1 or row_number > len(rows):
        report.add("objective.row_missing", "Row number is outside the loaded dataset", "error", f"Requested row {row_number}; loaded {len(rows)} rows.")
        return report
    row = rows[row_number - 1]
    messages = row.get("messages", row.get("conversations"))
    if isinstance(messages, list):
        for idx, message in enumerate(messages):
            if isinstance(message, dict):
                role, content = str(message.get("role", "unknown")), str(message.get("content", ""))
                report.add("objective.message", f"Message {idx + 1}: {role}", "ok", content[:4000], message_index=idx, role=role, characters=len(content))
    else:
        for key in ("prompt", "instruction", "input", "question", "text", "completion", "response", "output", "answer"):
            value = row.get(key)
            if isinstance(value, str):
                report.add("objective.text_field", f"Stored field: {key}", "ok", value[:4000], field=key, characters=len(value))
    labels, ids = row.get("labels"), row.get("input_ids")
    if isinstance(labels, list):
        supervised = sum(value != -100 for value in labels)
        report.add("objective.label_mask", "Stored label mask", "ok" if supervised else "error", f"{supervised} of {len(labels)} label positions are supervised; -100 positions are ignored by standard causal language model loss.", supervised_tokens=supervised, ignored_tokens=len(labels) - supervised)
    if isinstance(ids, list):
        report.add("objective.token_ids", "Stored token IDs", "info", f"Found {len(ids)} input IDs. Token strings are decoded only when a local tokenizer is supplied.", token_count=len(ids))
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
            if isinstance(messages, list) and hasattr(tokenizer, "apply_chat_template"):
                rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
                token_ids = rendered.tolist() if hasattr(rendered, "tolist") else rendered
                if token_ids and isinstance(token_ids[0], list):
                    token_ids = token_ids[0]
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                report.add("objective.chat_template", "Rendered with the local chat template", "info", repr(tokens[:1000]), tokenizer=tokenizer_path, token_count=len(token_ids), note="This applies the tokenizer's chat template. Trainer truncation, packing, collator behavior, and loss masks are not applied here.")
            else:
                value = _text(row)
                encoded = tokenizer(value, truncation=False, return_tensors=None)
                tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
                report.add("objective.tokenizer_view", "Local tokenizer view", "info", repr(tokens[:1000]), tokenizer=tokenizer_path, token_count=len(encoded["input_ids"]), note="Tokenizer view of stored text; trainer truncation, packing, collator behavior, and loss masks are not applied here.")
        except Exception as exc:
            report.add("objective.tokenizer_unavailable", "Local tokenizer could not be loaded", "warning", str(exc), "Pass a local tokenizer directory with its required files.")
    report.metadata["row_keys"] = sorted(row)
    return report


def inspect_gradient_snapshot(snapshot: Path) -> DiagnosisReport:
    """Inspect a JSON gradient summary without loading executable checkpoint files."""
    report = DiagnosisReport(metadata={"snapshot": str(snapshot), "accepted_format": "JSON mapping with parameter name and gradient/update summary values"})
    try:
        data = _read_object(snapshot)
    except Exception as exc:
        report.add("gradients.snapshot_unreadable", "Gradient snapshot could not be read", "error", str(exc), "Use a JSON snapshot; this command does not load pickle-based .pt/.bin checkpoints.")
        return report
    parameters = data.get("parameters", data)
    if not isinstance(parameters, dict):
        report.add("gradients.invalid_schema", "Gradient snapshot schema is invalid", "error", "Expected a `parameters` mapping.")
        return report
    optimizer = set(data.get("optimizer_parameters", [])) if isinstance(data.get("optimizer_parameters"), list) else None
    for name, item in parameters.items():
        if not isinstance(item, dict):
            continue
        trainable = bool(item.get("trainable", False))
        grad = item.get("grad_norm")
        update = item.get("update_norm")
        if trainable and name not in {"optimizer_parameters", "metadata"}:
            if grad is None:
                report.add("gradients.missing_measurement", "Trainable parameter has no gradient measurement", "info", f"No `grad_norm` recorded for {name}.", parameter=name)
            elif not isinstance(grad, (int, float)) or grad != grad or abs(grad) == float("inf"):
                report.add("gradients.nonfinite", "Nonfinite gradient norm", "error", f"Gradient norm for {name} is nonfinite or invalid.", parameter=name, grad_norm=grad)
            elif grad == 0:
                report.add("gradients.zero", "Trainable parameter received a zero gradient", "warning", f"Gradient norm for {name} is zero.", parameter=name)
            if update == 0:
                report.add("gradients.no_update", "Parameter did not change", "warning", f"Update norm for {name} is zero.", parameter=name)
            parameter_norm = item.get("parameter_norm")
            if isinstance(update, (int, float)) and isinstance(parameter_norm, (int, float)) and parameter_norm > 0 and update > 0 and update / parameter_norm <= 1e-8:
                report.add("gradients.small_update", "Parameter update is very small relative to its weight norm", "warning", f"Update/parameter norm ratio for {name} is {update / parameter_norm:.2e}.", "Compare the ratio with your training scale and optimizer settings; this threshold is a screening heuristic.", parameter=name, update_ratio=update / parameter_norm)
            if optimizer is not None and name not in optimizer:
                report.add("gradients.not_in_optimizer", "Trainable parameter is absent from optimizer groups", "error", f"{name} is trainable but not listed in optimizer_parameters.", parameter=name)
    report.metadata["parameters_inspected"] = len(parameters)
    if not report.issues:
        report.add("gradients.no_findings", "No recorded gradient issue found", "ok", "Only fields present in the supplied snapshot were evaluated.")
    return report


def snapshot_trainable_parameters(model: Any) -> dict[str, Any]:
    """Capture detached copies of trainable parameters before an optimizer step."""
    before: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if getattr(parameter, "requires_grad", False):
            before[name] = parameter.detach().clone()
    return before


def collect_gradient_snapshot(model: Any, optimizer: Any = None, before: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Create a JSON-serializable gradient/update summary from a live model."""
    optimizer_ids: Optional[set[int]] = None
    if optimizer is not None:
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group.get("params", [])
        }
    parameters: dict[str, dict[str, Any]] = {}
    optimizer_names: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = bool(getattr(parameter, "requires_grad", False))
        if not trainable:
            continue
        gradient = getattr(parameter, "grad", None)
        grad_norm = None
        if gradient is not None:
            try:
                grad_norm = float(gradient.detach().float().norm().item())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                grad_norm = None
        update_norm = None
        parameter_norm = None
        try:
            parameter_norm = float(parameter.detach().float().norm().item())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        if before is not None and name in before:
            try:
                update_norm = float((parameter.detach() - before[name].to(parameter.device)).float().norm().item())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                update_norm = None
        parameters[name] = {"trainable": True, "grad_norm": grad_norm, "update_norm": update_norm, "parameter_norm": parameter_norm}
        if optimizer_ids is not None and id(parameter) in optimizer_ids:
            optimizer_names.append(name)
    result: dict[str, Any] = {"parameters": parameters}
    if optimizer_ids is not None:
        result["optimizer_parameters"] = optimizer_names
    return result


def minimize_failure(spec_path: Path, *, allow_execution: bool, budget_runs: int, timeout_seconds: int = 60) -> DiagnosisReport:
    """Delta-debug a JSONL dataset using an explicitly authorized argv command."""
    report = DiagnosisReport(metadata={"spec": str(spec_path), "execution": "enabled" if allow_execution else "disabled"})
    if not allow_execution:
        report.add("minimize.execution_required", "Execution permission is required", "error", "The minimizer runs the command in the JSON specification against temporary dataset subsets.", "Review the command, then pass --allow-execution and a run budget.")
        return report
    try:
        spec_path = spec_path.expanduser().resolve()
        spec = _read_object(spec_path)
        dataset_path = Path(spec["dataset"]).expanduser()
        argv = spec["argv"]
        if not dataset_path.is_absolute():
            dataset_path = (spec_path.parent / dataset_path).resolve()
        if not isinstance(argv, list) or not argv or not all(isinstance(part, str) for part in argv) or not any("{dataset}" in part for part in argv):
            raise ValueError("argv must be a non-empty string list containing {dataset}.")
        if dataset_path.suffix.lower() != ".jsonl":
            raise ValueError("The minimizer currently accepts a JSONL dataset.")
        if not dataset_path.is_file():
            raise ValueError(f"Dataset file does not exist: {dataset_path}")
        if any(part in {"-c", "--command"} for part in argv):
            raise ValueError("Inline interpreter commands are not accepted; use a reviewed script path in argv.")
        expected_exit_code = int(spec.get("failure_exit_code", 1))
        if not -255 <= expected_exit_code <= 255:
            raise ValueError("failure_exit_code must be between -255 and 255.")
        rows = _rows(dataset_path, limit=100000)
        if not rows:
            raise ValueError("Dataset contains no readable object rows.")
    except Exception as exc:
        report.add("minimize.spec_invalid", "Failure specification is invalid", "error", str(exc), "Provide JSON with `dataset` and an argv array containing a `{dataset}` placeholder.")
        return report
    budget_runs = max(1, min(int(budget_runs), 100))
    runs = 0

    def reproduces(candidate: list[dict[str, Any]]) -> bool:
        nonlocal runs
        if runs >= budget_runs:
            return False
        runs += 1
        with tempfile.TemporaryDirectory(prefix="peft-doctor-minimize-") as directory:
            candidate_path = Path(directory) / "candidate.jsonl"
            candidate_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidate), encoding="utf-8")
            command = [part.replace("{dataset}", str(candidate_path)) for part in argv]
            try:
                result = subprocess.run(command, cwd=spec_path.parent, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=max(1, min(timeout_seconds, 600)), check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                report.metadata["last_execution_error"] = str(exc)
                return False
        return result.returncode == expected_exit_code

    if not reproduces(rows):
        report.add("minimize.not_reproduced", "Failure did not reproduce on the full dataset", "warning", "The command did not return the configured failure_exit_code on the supplied dataset.", runs_used=runs)
        return report
    current = rows
    granularity = 2
    while len(current) >= 2 and runs < budget_runs:
        chunk_size = max(1, (len(current) + granularity - 1) // granularity)
        reduced = False
        for start in range(0, len(current), chunk_size):
            candidate = current[:start] + current[start + chunk_size:]
            if candidate and reproduces(candidate):
                current = candidate
                granularity = max(2, granularity - 1)
                reduced = True
                break
            if runs >= budget_runs:
                break
        if not reduced:
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)
    output = spec_path.with_name(f"{spec_path.stem}.minimized.jsonl")
    try:
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in current), encoding="utf-8")
    except OSError as exc:
        report.add("minimize.output_error", "Could not write minimized dataset", "error", str(exc), output=str(output), runs_used=runs)
        return report
    report.metadata.update({"runs_used": runs, "run_budget": budget_runs, "original_rows": len(rows), "remaining_rows": len(current), "output": str(output)})
    report.add("minimize.complete", "Failure subset written", "ok", f"The failure reproduced on {len(current)} of {len(rows)} rows after {runs} command runs.", f"Review {output} before sharing it; it may contain private training data.", output=str(output), budget_exhausted=runs >= budget_runs)
    return report


def find_dataset_conflicts(dataset: Path, *, threshold: float = 0.82, limit: int = 5000, max_pairs: int = 100) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"dataset": str(dataset), "similarity_threshold": threshold})
    try:
        rows = _rows(dataset, limit)
    except Exception as exc:
        report.add("conflicts.dataset_unreadable", "Dataset could not be read", "error", str(exc))
        return report
    pairs = []
    for left in range(len(rows)):
        prompt_a = str(rows[left].get("prompt", rows[left].get("instruction", rows[left].get("question", rows[left].get("input", ""))))).strip().lower()
        if not prompt_a:
            continue
        for right in range(left + 1, len(rows)):
            prompt_b = str(rows[right].get("prompt", rows[right].get("instruction", rows[right].get("question", rows[right].get("input", ""))))).strip().lower()
            if not prompt_b:
                continue
            similarity = SequenceMatcher(None, prompt_a, prompt_b, autojunk=False).ratio()
            answer_a, answer_b = _answer(rows[left]), _answer(rows[right])
            if similarity >= threshold and answer_a and answer_b and " ".join(answer_a.split()).casefold() != " ".join(answer_b.split()).casefold():
                pairs.append((similarity, left + 1, right + 1, prompt_a[:240], answer_a[:240], answer_b[:240]))
                if len(pairs) >= max_pairs:
                    break
        if len(pairs) >= max_pairs:
            break
    for similarity, a, b, prompt, answer_a, answer_b in pairs:
        report.add("conflicts.similar_prompt_different_answer", "Similar prompts have different answers", "warning", f"Rows {a} and {b} have similar prompts ({similarity:.2f}) and non-identical answers. Review whether the difference is valid.", row_a=a, row_b=b, prompt=prompt, answer_a=answer_a, answer_b=answer_b, similarity=round(similarity, 4))
    report.metadata.update({"rows_inspected": len(rows), "pairs_reported": len(pairs), "interpretation": "Candidates for human review; differing answers are not automatically treated as errors."})
    if not pairs:
        report.add("conflicts.none_found", "No candidate conflicts found", "ok", "No sufficiently similar prompt pair with different answers was found in the inspected rows.")
    return report


def plan_training_coverage(dataset: Path, tasks_path: Path, *, limit: int = 10000) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"dataset": str(dataset), "tasks": str(tasks_path)})
    try:
        rows = _rows(dataset, limit)
        config = _read_object(tasks_path)
    except Exception as exc:
        report.add("coverage.input_invalid", "Coverage inputs could not be read", "error", str(exc))
        return report
    tasks = config.get("tasks", [])
    if not isinstance(tasks, list):
        report.add("coverage.tasks_invalid", "Task list is invalid", "error", "Expected `tasks` to be a list of task mappings with `name` and `keywords`.")
        return report
    languages = config.get("languages", [])
    patterns = config.get("conversation_patterns", [])
    if not isinstance(languages, list) or not isinstance(patterns, list):
        report.add("coverage.optional_sections_invalid", "Optional coverage sections must be lists", "error", "Use list values for `languages` and `conversation_patterns`.")
        return report
    corpus = "\n".join(_text(row).casefold() for row in rows)
    role_counts: Counter[str] = Counter()
    role_sequences: Counter[str] = Counter()
    tool_call_rows = 0
    scripts: Counter[str] = Counter()
    for row in rows:
        messages = row.get("messages", row.get("conversations"))
        if isinstance(messages, list):
            has_tool_call = False
            roles = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "unknown")).lower()
                role_counts[role] += 1
                roles.append(role)
                has_tool_call = has_tool_call or bool(message.get("tool_calls")) or str(message.get("role", "")).lower() == "tool"
            tool_call_rows += int(has_tool_call)
            if roles:
                role_sequences[" -> ".join(roles)] += 1
        for char in _text(row):
            if char.isalpha():
                name = unicodedata.name(char, "")
                for script in ("LATIN", "CYRILLIC", "ARABIC", "HEBREW", "HIRAGANA", "KATAKANA", "HANGUL", "CJK"):
                    if script in name:
                        scripts[script.lower()] += 1
                        break
    for task in tasks:
        if isinstance(task, str):
            name, keywords = task, [task]
        elif isinstance(task, dict):
            name = str(task.get("name", "unnamed"))
            keywords = task.get("keywords", [name])
        else:
            continue
        if not isinstance(keywords, list):
            keywords = [str(keywords)]
        hits = [str(word) for word in keywords if str(word).casefold() in corpus]
        severity = "ok" if hits else "warning"
        report.add("coverage.task", f"Task coverage: {name}", severity, f"Matched {len(hits)} of {len(keywords)} configured keywords in {len(rows)} rows.", "Collect representative examples for this task and rerun the coverage check." if not hits else None, matched=hits, unmatched=[str(word) for word in keywords if str(word).casefold() not in corpus])
    for language in languages:
        if not isinstance(language, dict):
            continue
        name = str(language.get("name", "unnamed"))
        markers = language.get("keywords", [])
        if not isinstance(markers, list):
            markers = [str(markers)]
        matched = [str(marker) for marker in markers if str(marker).casefold() in corpus]
        report.add("coverage.language_markers", f"Language markers: {name}", "ok" if matched else "warning", f"Matched {len(matched)} of {len(markers)} supplied markers. This is only a text-marker check, not language detection.", "Add examples in this language and rerun the check." if not matched else None, matched=matched, unmatched=[str(marker) for marker in markers if str(marker).casefold() not in corpus])
    for pattern in patterns:
        if not isinstance(pattern, dict):
            continue
        name = str(pattern.get("name", "unnamed"))
        expected = pattern.get("roles", [])
        if not isinstance(expected, list):
            expected = []
        key = " -> ".join(str(role).lower() for role in expected)
        count = role_sequences.get(key, 0)
        report.add("coverage.conversation_pattern", f"Conversation pattern: {name}", "ok" if count else "warning", f"Observed this exact role sequence in {count} rows.", "Collect examples with this role sequence if it is a required task." if not count else None, role_sequence=key, rows=count)
    report.metadata["rows_inspected"] = len(rows)
    report.metadata["chat_role_counts"] = dict(role_counts)
    report.metadata["conversation_role_sequences"] = dict(role_sequences)
    report.metadata["rows_with_tool_calls_or_tool_role"] = tool_call_rows
    report.metadata["character_script_counts"] = dict(scripts)
    report.add("coverage.structure", "Observed dataset structure", "info", "Role counts, role sequences, and tool-call rows come from explicit message fields. Character scripts are script hints, not language identification.", role_counts=dict(role_counts), role_sequences=dict(role_sequences), rows_with_tool_calls_or_tool_role=tool_call_rows, script_character_counts=dict(scripts))
    report.add("coverage.limits", "Coverage is a keyword scan", "info", "This check does not measure task quality, language proficiency, or model performance. Supply representative keywords for each intended task.")
    return report


_PRIVACY_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)"),
    "secret": re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|pypi-[A-Za-z0-9_-]{40,}|sk-[A-Za-z0-9_-]{20,})\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
_SENSITIVE_FIELD = re.compile(r"(?:^|[_-])(email|phone|mobile|ssn|tax_id|passport|address|birth|dob|password|secret|api_key|access_token|credential)(?:$|[_-])", re.I)


def inspect_privacy(dataset: Path, *, redacted_output: Optional[Path] = None, limit: int = 100000) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"dataset": str(dataset), "redaction": "preview only" if not redacted_output else str(redacted_output)})
    try:
        rows = _rows(dataset, limit)
    except Exception as exc:
        report.add("privacy.dataset_unreadable", "Dataset could not be read", "error", str(exc))
        return report
    if redacted_output and redacted_output.suffix.lower() not in {".json", ".jsonl"}:
        report.add("privacy.output_format", "Redacted output must be JSON or JSONL", "error", "The input reader normalizes records before redaction; CSV and TXT output are not supported.")
        return report
    counts: Counter[str] = Counter()
    findings = []

    def clean(value: Any, row_no: int, key: str = "") -> Any:
        if key and _SENSITIVE_FIELD.search(key) and value not in (None, "", [], {}):
            counts["sensitive_field"] += 1
            findings.append((row_no, "sensitive_field", key))
            return "[REDACTED_SENSITIVE_FIELD]"
        if isinstance(value, dict):
            return {k: clean(v, row_no, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v, row_no, key) for v in value]
        if not isinstance(value, str):
            return value
        result = value
        for kind, pattern in _PRIVACY_PATTERNS.items():
            matches = list(pattern.finditer(result))
            if matches:
                counts[kind] += len(matches)
                findings.append((row_no, kind, key))
                result = pattern.sub(f"[REDACTED_{kind.upper()}]", result)
        return result

    cleaned = [clean(row, index) for index, row in enumerate(rows, 1)]
    for row, kind, field in findings[:200]:
        report.add("privacy.possible_sensitive_value", f"Possible {kind.replace('_', ' ')}", "warning", f"A pattern matched in row {row}, field `{field}`. This is a review signal, not proof that the value identifies a person or is a valid credential.", row=row, field=field, category=kind)
    if redacted_output:
        try:
            source_path = dataset.expanduser().resolve()
            output_path = redacted_output.expanduser().resolve()
            if source_path == output_path:
                report.add("privacy.output_overwrite", "Refusing to overwrite the input dataset", "error", "Choose a separate path for the redacted copy.")
                return report
            redacted_output = output_path
            redacted_output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            report.add("privacy.output_error", "Could not prepare redacted output", "error", str(exc))
            return report
        if redacted_output.suffix.lower() == ".jsonl":
            content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cleaned)
        else:
            content = json.dumps(cleaned, ensure_ascii=False, indent=2)
        try:
            redacted_output.write_text(content, encoding="utf-8")
        except OSError as exc:
            report.add("privacy.output_error", "Could not write redacted output", "error", str(exc), output=str(redacted_output))
            return report
        report.add("privacy.redacted_copy", "Redacted dataset copy written", "ok", f"Wrote {redacted_output}. Review it before use; pattern redaction may alter examples or miss sensitive data.", output=str(redacted_output), counts=dict(counts))
    if not findings:
        report.add("privacy.no_patterns", "No configured privacy pattern matched", "ok", "This scan does not prove the dataset contains no sensitive information.")
    report.metadata.update({"rows_inspected": len(rows), "matches_by_category": dict(counts), "complete_scan": len(rows) < limit})
    return report


def advise_strategy(requirements_path: Path) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"requirements": str(requirements_path), "method": "transparent rule-based guidance"})
    try:
        data = _read_object(requirements_path)
    except Exception as exc:
        report.add("strategy.input_invalid", "Requirements could not be read", "error", str(exc))
        return report
    changing = bool(data.get("facts_change_often", False))
    behavior = bool(data.get("consistent_behavior", False))
    has_examples = int(data.get("representative_examples", 0) or 0)
    needs_private_data = bool(data.get("private_data", False))
    if changing:
        report.add("strategy.retrieval", "Retrieval is a strong candidate for changing facts", "ok", "Requirements say the source facts change often. Retrieval can update the source without retraining; assess retrieval quality separately.", "Build and evaluate retrieval over the authoritative source.")
    if behavior and has_examples >= 100:
        report.add("strategy.finetune", "Fine-tuning may help consistent behavior", "info", f"The requirements describe stable behavior and {has_examples} representative examples. Compare a small fine-tuning run against a prompt-only baseline.", "Hold out examples and evaluate against task-specific criteria.")
    elif behavior:
        report.add("strategy.more_examples", "Behavior goal needs representative examples", "warning", f"Only {has_examples} representative examples were reported.", "Try a prompt baseline and collect diverse examples before committing to fine-tuning.")
    if not changing and not behavior:
        report.add("strategy.prompt_baseline", "Start with a prompt baseline", "info", "The requirements do not identify frequently changing facts or a stable behavior gap that calls for training.", "Write an evaluation set before choosing a more complex approach.")
    if needs_private_data:
        report.add("strategy.privacy", "Keep data handling in the decision", "warning", "Requirements mark the data as private. Review provider retention and access terms before using hosted retrieval or training.")
    report.add("strategy.evaluation", "Compare approaches on the same held-out cases", "info", "This advisor does not predict accuracy. Measure prompt, retrieval, and fine-tuning candidates against the same evaluation set.")
    return report


def plan_hardware(model: str, vram_gb: float, hours: float, *, seq_len: int = 2048, batch_size: int = 1, dataset_tokens: Optional[int] = None, params_billion: Optional[float] = None, gpu_count: int = 1, disk_gb: Optional[float] = None, required_disk_gb: Optional[float] = None, measured_tokens_per_second: Optional[float] = None) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"model": model, "available_vram_gb_per_gpu": vram_gb, "gpu_count": gpu_count, "time_budget_hours": hours})
    if vram_gb <= 0 or hours <= 0 or gpu_count <= 0:
        report.add("plan.invalid_budget", "Hardware budget inputs must be positive", "error", "VRAM, time budget, and GPU count must be greater than zero.")
        return report
    estimate = estimate_vram_gb(model=model, seq_len=seq_len, batch_size=batch_size, qlora=True, params_billion=params_billion)
    report.metadata["estimate_method"] = estimate.metadata.get("method")
    if estimate.has_errors:
        report.extend(estimate.issues)
        return report
    need = float(estimate.metadata.get("estimated_total_gb", 0))
    if need <= vram_gb:
        report.add("plan.memory_fit", "Starting configuration fits the supplied VRAM estimate", "ok", f"Estimated {need:.1f} GB against {vram_gb:.1f} GB available. Reserve headroom for runtime, evaluation, and fragmentation.", estimated_gb=need, available_gb=vram_gb)
    else:
        suggestions = []
        for seq in (seq_len, max(256, seq_len // 2), max(256, seq_len // 4)):
            for batch in (batch_size, 1):
                candidate = estimate_vram_gb(model=model, seq_len=seq, batch_size=batch, qlora=True, params_billion=params_billion)
                candidate_gb = float(candidate.metadata.get("estimated_total_gb", 0))
                if candidate_gb <= vram_gb:
                    suggestions.append((seq, batch, candidate_gb))
                    break
            if suggestions:
                break
        if suggestions:
            seq, batch, candidate_gb = suggestions[0]
            report.add("plan.memory_adjustment", "Lower sequence length or batch to fit the estimate", "warning", f"Current estimate {need:.1f} GB exceeds {vram_gb:.1f} GB. A formula estimate at sequence length {seq} and batch {batch} is {candidate_gb:.1f} GB.", "Validate with a short real run; estimates do not account for every kernel or allocator behavior.", suggested_seq_len=seq, suggested_batch_size=batch)
        else:
            report.add("plan.memory_no_fit", "No tested simple adjustment fits the supplied VRAM", "error", f"The starting estimate is {need:.1f} GB for {vram_gb:.1f} GB available.", "Consider a smaller model, more memory, CPU offload, or a supported distributed setup.")
    if dataset_tokens and hours > 0:
        required_tps = dataset_tokens / (hours * 3600)
        report.add("plan.throughput_needed", "Throughput target from supplied time budget", "info", f"To process {dataset_tokens} tokens in {hours} hours requires at least {required_tps:.2f} aggregate tokens per second, excluding evaluation and checkpoint time.", required_tokens_per_second=required_tps, required_per_gpu_tokens_per_second=required_tps / max(1, gpu_count))
        if measured_tokens_per_second and measured_tokens_per_second > 0:
            duration = dataset_tokens / measured_tokens_per_second / 3600
            report.add("plan.measured_duration", "Duration from supplied measured throughput", "ok" if duration <= hours else "warning", f"At the supplied aggregate rate of {measured_tokens_per_second:.2f} tokens/second, token processing alone is estimated at {duration:.2f} hours.", "Use throughput measured on the same model, sequence length, batch, and GPU setup.", measured_tokens_per_second=measured_tokens_per_second, estimated_hours=duration)
    if disk_gb is not None and required_disk_gb is not None:
        report.add("plan.disk_budget", "Disk budget comparison", "ok" if disk_gb >= required_disk_gb else "error", f"Supplied free disk: {disk_gb:.1f} GB; supplied run requirement: {required_disk_gb:.1f} GB.", "Include checkpoints, model/cache files, logs, and a reserve when estimating required disk.", available_disk_gb=disk_gb, required_disk_gb=required_disk_gb)
    report.add("plan.measurement_note", "Planning inputs are explicit", "info", "VRAM uses a formula estimate. GPU count, free disk, required disk, and throughput are used only when you supply them; no benchmark or provider price is inferred.")
    return report


def plan_adapter_composition(adapters: list[Path]) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"adapters": [str(path) for path in adapters], "operation": "inspection only; no weights are combined"})
    configs = []
    for path in adapters:
        config_path = path / "adapter_config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report.add("compose.config_unavailable", "Adapter config unavailable", "error", f"Could not read {config_path}: {exc}")
            continue
        configs.append((path, config))
    bases = {str(config.get("base_model_name_or_path", "")) for _, config in configs}
    if len(configs) != len(adapters):
        report.add("compose.incomplete_metadata", "Composition checks are incomplete", "warning", "At least one adapter config could not be read; compatibility cannot be assessed for every input.")
    if len(bases) > 1:
        report.add("compose.base_mismatch", "Adapters name different base models", "error", "The adapter configs do not identify one shared base model.", base_models=sorted(bases))
    target_sets = [set(config.get("target_modules", [])) for _, config in configs if isinstance(config.get("target_modules"), list)]
    if target_sets and any(targets != target_sets[0] for targets in target_sets[1:]):
        report.add("compose.target_mismatch", "Adapters target different module sets", "warning", "Composition may require a method that supports these module differences and must be evaluated.", target_modules=[sorted(targets) for targets in target_sets])
    revisions = {str(config.get("revision")) for _, config in configs if config.get("revision")}
    if len(revisions) > 1:
        report.add("compose.revision_mismatch", "Recorded base revisions differ", "warning", "Adapter metadata records more than one base revision.", revisions=sorted(revisions))
    for path, config in configs:
        report.add("compose.adapter", f"Adapter inspected: {path.name}", "info", "Composition method should be chosen and evaluated for the target task.", base=config.get("base_model_name_or_path"), peft_type=config.get("peft_type"), rank=config.get("r"), modules_to_save=config.get("modules_to_save"), tokenizer_files=any((path / filename).exists() for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")))
    tokenizer_hashes: dict[str, dict[str, str]] = {}
    for path, _ in configs:
        hashes: dict[str, str] = {}
        for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            artifact = path / filename
            if artifact.is_file():
                digest = hashlib.sha256()
                with artifact.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[filename] = digest.hexdigest()
        tokenizer_hashes[path.name] = hashes
    tokenizer_names = set.intersection(*(set(files) for files in tokenizer_hashes.values())) if tokenizer_hashes else set()
    differing_tokenizers = [name for name in sorted(tokenizer_names) if len({files[name] for files in tokenizer_hashes.values()}) > 1]
    if differing_tokenizers:
        report.add("compose.tokenizer_mismatch", "Tokenizer artifacts differ between adapters", "warning", "Matching tokenizer files have different SHA-256 hashes. Confirm vocabulary and special-token alignment before composition.", files=differing_tokenizers)
    if len(configs) == len(adapters) and len(adapters) >= 2 and len(bases) == 1 and "" not in bases:
        report.add("compose.plan", "Shared base model recorded", "ok", "Configs share the same base-model identifier. This does not prove identical model weights or tokenizer behavior; compare pinned revisions and evaluate a composed adapter before deployment.")
        same_targets = bool(target_sets) and all(targets == target_sets[0] for targets in target_sets[1:])
        all_lora = all(str(config.get("peft_type", "")).upper() == "LORA" for _, config in configs)
        if all_lora and same_targets and not differing_tokenizers:
            report.add("compose.candidate", "Weighted LoRA composition is a candidate to evaluate", "info", "The inspected adapters name one base and share target modules and tokenizer artifacts. Use a PEFT-supported weighted adapter method only after verifying the exact base revision; compare the composed result on held-out data.", candidate="weighted LoRA composition")
    return report


def inspect_serving(adapter: Path, engine: str) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"adapter": str(adapter), "serving_engine": engine})
    config_path = adapter / "adapter_config.json"
    if not config_path.exists():
        report.add("serving.adapter_config_missing", "Adapter config is missing", "error", f"Expected {config_path}.")
        return report
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add("serving.adapter_config_invalid", "Adapter config could not be parsed", "error", str(exc))
        return report
    if config.get("base_model_name_or_path"):
        report.add("serving.base_model", "Base model identifier recorded", "ok", str(config["base_model_name_or_path"]), base_model=config["base_model_name_or_path"], revision=config.get("revision"))
    else:
        report.add("serving.base_model_missing", "Base model identifier missing", "warning", "Adapter config does not record base_model_name_or_path.")
    if config.get("modules_to_save"):
        report.add("serving.modules_to_save", "Additional modules are saved with this adapter", "warning", "Confirm that the selected serving engine loads modules_to_save for this PEFT adapter.", modules_to_save=config["modules_to_save"])
    if config.get("target_parameters"):
        report.add("serving.target_parameters", "Adapter uses target_parameters", "warning", f"Verify support for target_parameters in {engine}; engine compatibility depends on model architecture and version.", target_parameters=config["target_parameters"])
    if config.get("trainable_token_indices"):
        report.add("serving.trainable_tokens", "Adapter uses trainable token indices", "warning", "Verify the serving stack loads the saved token embeddings and tokenizer additions.")
    tokenizer_files = [name for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json") if (adapter / name).exists()]
    report.add("serving.tokenizer_files", "Tokenizer files checked", "ok" if tokenizer_files else "warning", f"Found: {', '.join(tokenizer_files) if tokenizer_files else 'none in adapter directory'}. Load the tokenizer from the same source used for training.", files=tokenizer_files)
    report.add("serving.engine_review", "Engine compatibility requires runtime confirmation", "info", f"Static inspection cannot prove compatibility with {engine}; check its supported PEFT types, model architecture, quantization, and adapter loading path.")
    return report


def recommend_next_experiment(experiments: Path) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"experiments": str(experiments)})
    history_file = experiments / ".peft-doctor" / "history.jsonl"
    if not history_file.exists():
        report.add("next_experiment.history_missing", "Experiment history not found", "warning", f"Expected {history_file}.", f"Record runs with `peft-doctor history {experiments} --add-status oom --note 'reduced sequence length'` and include the tested change.")
        return report
    events = []
    for line_number, line in enumerate(history_file.read_text(encoding="utf-8").splitlines(), 1):
        try:
            events.append((line_number, json.loads(line)))
        except json.JSONDecodeError:
            report.add("next_experiment.bad_line", "Malformed history line", "warning", f"Could not parse line {line_number}.", line=line_number)
    statuses: list[str] = []
    combined_notes: list[str] = []
    for line_number, event in events:
        note = str(event.get("note", ""))
        statuses.append(str(event.get("status", "")).casefold())
        combined_notes.append(note.casefold())
        report.add("next_experiment.evidence", f"Run record at line {line_number}", "info", note or str(event.get("metric") or event.get("status", "run")), status=event.get("status"), metric=event.get("metric"), time=event.get("time"))
    evidence = " ".join(statuses + combined_notes)
    if any(word in evidence for word in ("oom", "out of memory", "cuda")):
        suggestion = "Record peak allocated VRAM and compare one memory setting at a time, starting with sequence length or per-device batch size. Preserve effective batch size where possible."
    elif any(word in evidence for word in ("nan", "overflow", "inf")):
        suggestion = "Check supervised-token counts and data validity first; then compare one precision or learning-rate change while holding the evaluation set fixed."
    elif any(word in evidence for word in ("completed", "success", "best")):
        suggestion = "Compare the completed run with a held-out metric and a fixed evaluation set. Change one documented factor in the next run."
    else:
        suggestion = "Record the failure symptom, one configuration change, and a comparable held-out metric. Change one factor in the next run."
    report.add("next_experiment.recommendation", "Suggested next comparison", "info", suggestion + " The history lacks structured settings and evaluation results, so it cannot name a winning configuration or predict quality.")
    report.metadata["records_read"] = len(events)
    return report


def investigate_distributed_logs(directory: Path) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"log_directory": str(directory), "files": []})
    files = sorted(path for path in directory.glob("**/*") if path.is_file() and (path.suffix.lower() in {".log", ".txt", ".out", ".err"} or "rank" in path.name.lower()))
    if not files:
        report.add("distributed_logs.none", "No rank logs found", "warning", "Pass a directory containing per-rank log text files.")
        return report
    signatures = {"cuda_oom": re.compile(r"out of memory|cuda oom", re.I), "nccl": re.compile(r"nccl|collective|all_reduce|all_gather", re.I), "timeout": re.compile(r"timed out|timeout", re.I), "uneven_input": re.compile(r"uneven|different number of batches|exhausted", re.I), "device": re.compile(r"device-side assert|illegal memory access|device mismatch", re.I)}
    occurrences = []
    for path in files:
        report.metadata["files"].append(str(path))
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for kind, pattern in signatures.items():
                if pattern.search(line):
                    occurrences.append((number, path, kind, line.strip()[:500]))
    for number, path, kind, line in sorted(occurrences, key=lambda item: (item[0], str(item[1])))[:100]:
        report.add(f"distributed_logs.{kind}", f"Possible {kind.replace('_', ' ')} event", "warning" if kind in {"timeout", "nccl"} else "error", f"{path.name}:{number}: {line}", file=str(path), line=number)
    if occurrences:
        earliest = min(occurrences, key=lambda item: (item[0], str(item[1])))
        report.metadata["earliest_line_number_candidate"] = {"file": str(earliest[1]), "line": earliest[0], "event": earliest[2]}
        report.add("distributed_logs.order_limit", "Cross-file time ordering is uncertain", "info", "Line numbers are not comparable across different files. Provide timestamps synchronized across ranks to establish the earliest event.")
    else:
        report.add("distributed_logs.no_signatures", "No configured distributed failure signature found", "ok", "No known text pattern matched the inspected logs.")
    return report


def answer_with_evidence(project: Path, question: str) -> DiagnosisReport:
    """Return local evidence references relevant to a question, without a remote model."""
    report = DiagnosisReport(metadata={"project": str(project), "question": question, "answer_mode": "local evidence retrieval"})
    query_terms = {term for term in re.findall(r"[a-z0-9_]+", question.casefold()) if len(term) > 2}
    paths = [project / "train.py", project / "trainer.log", project / "training.log"]
    paths.extend(sorted(project.glob("*.json"))[:10])
    paths.extend(sorted(project.glob("*.jsonl"))[:10])
    matches = 0
    for path in paths:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            lowered = line.casefold()
            terms = [term for term in query_terms if term in lowered]
            if terms:
                report.add("assistant.evidence", f"Evidence: {path.name}:{number}", "info", line[:1000], source=str(path), line=number, matched_terms=terms)
                matches += 1
                if matches >= 30:
                    break
        if matches >= 30:
            break
    if not matches:
        report.add("assistant.no_evidence", "No matching local evidence found", "info", "No matching lines were found in train.py, common training logs, or the first ten local JSON/JSONL files.")
    report.add("assistant.limits", "Answer is grounded in matching local lines", "info", "This command retrieves evidence; it does not infer facts absent from the files and does not contact a remote service.")
    return report
