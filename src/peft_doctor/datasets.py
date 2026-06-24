"""Dataset and prompt-format checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from .report import DiagnosisReport
from .utils import get_value


TEXT_COLUMNS = ["text", "prompt", "completion", "response", "output", "instruction", "input"]
RESPONSE_COLUMNS = ["response", "completion", "output", "answer"]
INSTRUCTION_COLUMNS = ["instruction", "prompt", "question", "input"]


def load_dataset_records(path: Union[str, Path], limit: int = 50) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if len(records) >= limit:
                    break
        return records

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [row for row in data[:limit] if isinstance(row, dict)]
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    return [row for row in value[:limit] if isinstance(row, dict)]
            return [data]
        return []

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [row for _, row in zip(range(limit), reader)]

    if suffix == ".txt":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    records.append({"text": text})
                if len(records) >= limit:
                    break
        return records

    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def sample_dataset(dataset: Any, limit: int = 50) -> list[Any]:
    if dataset is None:
        return []
    if isinstance(dataset, (str, Path)):
        return load_dataset_records(dataset, limit=limit)

    records = []
    try:
        length = min(len(dataset), limit)
        for index in range(length):
            records.append(dataset[index])
        return records
    except Exception:
        pass

    if isinstance(dataset, Iterable):
        for index, row in enumerate(dataset):
            if index >= limit:
                break
            records.append(row)
    return records


def _row_text(row: Any) -> str:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return ""
    pieces = []
    for column in TEXT_COLUMNS:
        value = row.get(column)
        if isinstance(value, str):
            pieces.append(value)
    return "\n".join(pieces)


def has_instruction_markers(text: str) -> bool:
    lowered = text.lower()
    marker_pairs = [
        ("### instruction", "### response"),
        ("instruction:", "response:"),
        ("user:", "assistant:"),
        ("human:", "assistant:"),
        ("<|user|>", "<|assistant|>"),
    ]
    return any(left in lowered and right in lowered for left, right in marker_pairs)


def has_chat_messages(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    messages = row.get("messages") or row.get("conversations")
    if not isinstance(messages, list) or not messages:
        return False
    first = messages[0]
    return isinstance(first, dict) and "role" in first and "content" in first


def has_instruction_response_columns(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    has_instruction = any(column in row for column in INSTRUCTION_COLUMNS)
    has_response = any(column in row for column in RESPONSE_COLUMNS)
    return has_instruction and has_response


def has_pretokenized_columns(row: Any) -> bool:
    return isinstance(row, dict) and "input_ids" in row and "labels" in row


def check_dataset(
    report: DiagnosisReport,
    train_dataset: Any = None,
    dataset_path: Optional[Union[str, Path]] = None,
    tokenizer: Any = None,
    sample_size: int = 50,
) -> None:
    dataset = dataset_path or train_dataset
    if dataset is None:
        report.add(
            "dataset.not_provided",
            "Dataset not provided",
            "info",
            "No dataset was provided, so prompt format and empty-row checks were skipped.",
        )
        return

    try:
        rows = sample_dataset(dataset, limit=sample_size)
    except Exception as exc:
        report.add(
            "dataset.read_failed",
            "Dataset could not be read",
            "error",
            f"PEFT Doctor could not read the dataset sample: {exc}",
            "Check the path and make sure the file is valid JSON, JSONL, CSV, or TXT.",
        )
        return

    if not rows:
        report.add(
            "dataset.empty",
            "Dataset sample is empty",
            "error",
            "The dataset sample has no rows.",
            "Check the dataset path, split name, and filtering code before training.",
        )
        return

    empty_rows = [index for index, row in enumerate(rows) if not _row_text(row).strip() and not has_chat_messages(row)]
    if empty_rows:
        report.add(
            "dataset.empty_rows",
            "Empty rows found",
            "warning",
            "Some sampled rows have no usable text fields.",
            "Remove empty samples and rows with missing instruction or response text.",
            count=len(empty_rows),
            first_index=empty_rows[0],
        )

    first = rows[0]
    if isinstance(first, dict):
        report.add(
            "dataset.columns",
            "Dataset columns detected",
            "info",
            "The first sampled row exposes these columns.",
            columns=", ".join(first.keys()),
        )

    chat_count = sum(1 for row in rows if has_chat_messages(row))
    instruction_columns = sum(1 for row in rows if has_instruction_response_columns(row))
    pretokenized = sum(1 for row in rows if has_pretokenized_columns(row))
    marker_count = sum(1 for row in rows if has_instruction_markers(_row_text(row)))

    if chat_count:
        chat_template = get_value(tokenizer, "chat_template")
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if tokenizer is not None and not chat_template and not callable(apply_chat_template):
            report.add(
                "dataset.chat_template_missing",
                "Chat dataset without tokenizer chat template",
                "warning",
                "The dataset uses chat-style messages, but the tokenizer does not expose a chat template.",
                "Set tokenizer.chat_template or format messages into text before tokenization.",
                chat_rows=chat_count,
            )
        else:
            report.add(
                "dataset.chat_format",
                "Chat message format detected",
                "ok",
                "Sampled rows use a `messages` style chat format.",
                chat_rows=chat_count,
            )
    elif instruction_columns:
        report.add(
            "dataset.instruction_columns",
            "Instruction and response columns detected",
            "ok",
            "The sampled rows look like instruction-tuning data.",
            matching_rows=instruction_columns,
        )
    elif pretokenized:
        report.add(
            "dataset.pretokenized",
            "Pretokenized dataset detected",
            "ok",
            "The sampled rows include `input_ids` and `labels`.",
            matching_rows=pretokenized,
        )
    elif marker_count:
        report.add(
            "dataset.prompt_markers",
            "Prompt markers detected",
            "ok",
            "Text rows include instruction/response style markers.",
            matching_rows=marker_count,
        )
    else:
        report.add(
            "dataset.prompt_format_unclear",
            "Prompt format is unclear",
            "warning",
            "The sampled rows do not look like chat messages, instruction/response columns, or marked text.",
            "Use a clear prompt template such as `### Instruction:` and `### Response:` or apply the tokenizer chat template.",
        )


def check_tokenizer(report: DiagnosisReport, tokenizer: Any = None) -> None:
    if tokenizer is None:
        report.add(
            "tokenizer.not_provided",
            "Tokenizer not provided",
            "info",
            "Tokenizer checks were skipped.",
        )
        return

    pad_token = get_value(tokenizer, "pad_token")
    eos_token = get_value(tokenizer, "eos_token")
    if pad_token is None:
        report.add(
            "tokenizer.pad_token_missing",
            "Tokenizer has no pad token",
            "warning",
            "Batch collation may fail because the tokenizer has no pad token.",
            "For causal LM fine-tuning, a common fix is `tokenizer.pad_token = tokenizer.eos_token`.",
            eos_token=eos_token,
        )
    else:
        report.add(
            "tokenizer.pad_token_ok",
            "Tokenizer pad token is set",
            "ok",
            "The tokenizer exposes a pad token.",
        )

    model_max_length = get_value(tokenizer, "model_max_length")
    if isinstance(model_max_length, int) and model_max_length > 1_000_000:
        report.add(
            "tokenizer.model_max_length_placeholder",
            "Tokenizer max length looks like a placeholder",
            "info",
            "The tokenizer reports a very large model_max_length value.",
            "Pass an explicit max sequence length in your data collator or training script.",
            model_max_length=model_max_length,
        )
