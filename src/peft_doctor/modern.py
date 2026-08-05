"""Checks for current PEFT, TRL, Accelerate, and bitsandbytes workflows."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Union

from .datasets import load_dataset_records
from .operations import inspect_trainer_checkpoint, read_safetensors_header
from .report import DiagnosisReport


def _load_structured(path: Union[str, Path]) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    if source.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {source}")
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not isinstance(value, dict):
        return result
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        result[name] = item
        if isinstance(item, dict):
            result.update(_flatten(item, name))
    return result


def _find(flat: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in flat:
            return flat[key]
        matches = [value for name, value in flat.items() if name.endswith(f".{key}")]
        if matches:
            return matches[-1]
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _load_report_config(path: Union[str, Path], report: DiagnosisReport) -> Optional[dict[str, Any]]:
    try:
        return _load_structured(path)
    except Exception as exc:
        report.add(
            "config.read_failed",
            "Configuration could not be read",
            "error",
            str(exc),
            "Pass a readable JSON or YAML object.",
        )
        return None


def _dataset_rows(path: Path, limit: int, report: DiagnosisReport) -> list[dict[str, Any]]:
    try:
        return load_dataset_records(path, limit=limit)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.add(
            "dataset.read_failed",
            "Dataset could not be read",
            "error",
            str(exc),
            "Repair malformed rows before running this check.",
        )
        return []


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("messages", row.get("conversations"))
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def check_sft_config(
    config_path: Union[str, Path],
    *,
    dataset_path: Optional[Union[str, Path]] = None,
    chat_template_path: Optional[Union[str, Path]] = None,
    limit: int = 500,
) -> DiagnosisReport:
    """Check TRL SFTConfig options against the dataset and chat template."""

    report = DiagnosisReport(
        metadata={
            "config": str(config_path),
            "dataset": str(dataset_path) if dataset_path else None,
            "chat_template": str(chat_template_path) if chat_template_path else None,
        }
    )
    config = _load_report_config(config_path, report)
    if config is None:
        return report
    flat = _flatten(config)
    assistant_only = _as_bool(_find(flat, "assistant_only_loss", default=False))
    completion_only = _as_bool(_find(flat, "completion_only_loss", default=False))
    packing = _as_bool(_find(flat, "packing", default=False))
    padding_free = _as_bool(_find(flat, "padding_free", default=False))
    strategy = str(_find(flat, "packing_strategy", default="bfd"))
    max_length = _find(flat, "max_length", "max_seq_length")
    attention = str(_find(flat, "attn_implementation", default="")).lower()

    report.metadata.update(
        {
            "assistant_only_loss": assistant_only,
            "completion_only_loss": completion_only,
            "packing": packing,
            "packing_strategy": strategy,
            "padding_free": padding_free,
        }
    )
    if assistant_only and completion_only:
        report.add(
            "sft.loss_modes_conflict",
            "Two selective loss modes are enabled",
            "warning",
            "assistant_only_loss and completion_only_loss are both enabled.",
            "Choose the loss mode that matches the dataset contract and verify one tokenized batch.",
        )
    if max_length is not None and (not isinstance(max_length, int) or max_length <= 0):
        report.add(
            "sft.max_length_invalid",
            "SFT max length is invalid",
            "error",
            f"max_length is {max_length!r}.",
            "Set max_length to a positive integer.",
        )
    if strategy not in {"bfd", "bfd_split", "wrapped"}:
        report.add(
            "sft.packing_strategy_unknown",
            "Packing strategy is not recognized",
            "warning",
            f"packing_strategy is {strategy!r}.",
            "Check the installed TRL version and use a strategy it documents.",
        )
    elif strategy == "bfd_split":
        report.add(
            "sft.packing_strategy_version",
            "BFD split needs a recent TRL release",
            "info",
            "packing_strategy='bfd_split' is newer than the baseline BFD and wrapped strategies.",
            "Check the installed TRL documentation before launching the run.",
        )
    if padding_free and attention not in {"flash_attention_2", "flash_attention_3"}:
        report.add(
            "sft.padding_free_attention",
            "Padding-free training needs compatible attention",
            "warning",
            "padding_free is enabled without FlashAttention 2 or 3 in this config.",
            "Use a supported FlashAttention implementation or disable padding_free.",
        )
    if packing and strategy == "bfd":
        report.add(
            "sft.bfd_padding_free",
            "BFD packing uses padding-free batches",
            "info",
            "TRL enables padding-free mode when BFD packing is selected.",
            "Confirm the attention backend supports padding-free execution.",
        )

    rows: list[dict[str, Any]] = []
    if dataset_path:
        rows = _dataset_rows(Path(dataset_path), limit, report)
        conversational = sum(bool(_messages(row)) for row in rows)
        prompt_completion = sum("prompt" in row and "completion" in row for row in rows)
        report.metadata.update(
            {
                "rows_sampled": len(rows),
                "conversational_rows": conversational,
                "prompt_completion_rows": prompt_completion,
            }
        )
        if assistant_only and rows and conversational != len(rows):
            report.add(
                "sft.assistant_loss_dataset",
                "Assistant-only loss does not match every sampled row",
                "error",
                f"{len(rows) - conversational} of {len(rows)} sampled rows are not conversational.",
                "Use messages-formatted data or disable assistant_only_loss.",
            )
        if completion_only and rows and prompt_completion != len(rows):
            report.add(
                "sft.completion_loss_dataset",
                "Completion-only loss does not match every sampled row",
                "error",
                f"{len(rows) - prompt_completion} of {len(rows)} sampled rows lack prompt/completion fields.",
                "Convert the dataset to prompt/completion rows or disable completion_only_loss.",
            )

    if assistant_only:
        if chat_template_path:
            try:
                template = Path(chat_template_path).read_text(encoding="utf-8-sig")
            except OSError as exc:
                report.add(
                    "sft.template_read_failed",
                    "Chat template could not be read",
                    "error",
                    str(exc),
                    "Pass a readable local Jinja template.",
                )
                return report
            if "{% generation %}" not in template or "{% endgeneration %}" not in template:
                report.add(
                    "sft.generation_markers_missing",
                    "Chat template has no generation markers",
                    "error",
                    "assistant_only_loss needs a template that marks assistant output with generation blocks.",
                    "Add balanced {% generation %} and {% endgeneration %} blocks around assistant text.",
                )
            else:
                report.add(
                    "sft.generation_markers_found",
                    "Chat-template generation markers found",
                    "ok",
                    "The supplied template contains both generation block markers.",
                )
        else:
            report.add(
                "sft.template_not_checked",
                "Chat template was not checked",
                "warning",
                "assistant_only_loss is enabled, but no template file was supplied.",
                "Pass --chat-template to verify generation markers.",
            )
    if not report.has_errors and not report.has_warnings:
        report.add("sft.config_ok", "SFT checks passed", "ok", "No checked SFTConfig conflict was found.")
    return report


def check_fsdp_config(
    config_path: Union[str, Path],
    *,
    training_args_path: Optional[Union[str, Path]] = None,
) -> DiagnosisReport:
    """Check Accelerate FSDP state, loading, wrapping, and compile settings."""

    report = DiagnosisReport(metadata={"config": str(config_path)})
    config = _load_report_config(config_path, report)
    if config is None:
        return report
    flat = _flatten(config)
    distributed_type = str(_find(flat, "distributed_type", default="")).upper()
    if distributed_type and distributed_type != "FSDP":
        report.add(
            "fsdp.distributed_type",
            "Accelerate is not configured for FSDP",
            "error",
            f"distributed_type is {distributed_type!r}.",
            "Set distributed_type: FSDP or run the general distributed-check command instead.",
        )
    state_type = str(_find(flat, "fsdp_state_dict_type", "state_dict_type", default="")).upper()
    if state_type and state_type != "SHARDED_STATE_DICT":
        report.add(
            "fsdp.state_dict_type",
            "FSDP state dict is not sharded",
            "warning",
            f"The configured state dict type is {state_type}.",
            "Use SHARDED_STATE_DICT for scalable checkpointing unless a full state dict is deliberate.",
        )
    cpu_efficient = _as_bool(_find(flat, "fsdp_cpu_ram_efficient_loading", default=False))
    sync_states = _as_bool(_find(flat, "fsdp_sync_module_states", default=False))
    if cpu_efficient and not sync_states:
        report.add(
            "fsdp.cpu_loading_sync",
            "CPU-efficient loading needs synchronized module states",
            "error",
            "fsdp_cpu_ram_efficient_loading is true while fsdp_sync_module_states is false.",
            "Set fsdp_sync_module_states: true.",
        )
    backend = str(_find(flat, "dynamo_backend", default="NO")).upper()
    compile_enabled = backend not in {"", "NO", "NONE", "FALSE"} or _as_bool(
        _find(flat, "torch_compile", default=False)
    )
    use_orig_params = _as_bool(_find(flat, "fsdp_use_orig_params", default=False))
    if compile_enabled and not use_orig_params:
        report.add(
            "fsdp.compile_orig_params",
            "Compiled FSDP should use original parameters",
            "warning",
            "A Dynamo/compile backend is enabled while fsdp_use_orig_params is false or absent.",
            "Set fsdp_use_orig_params: true and verify the installed Torch/Accelerate combination.",
        )
    processes = _find(flat, "num_processes")
    if isinstance(processes, int) and processes < 2:
        report.add(
            "fsdp.process_count",
            "FSDP has fewer than two processes",
            "warning",
            f"num_processes is {processes}.",
            "Use at least two processes for an FSDP run, or remove FSDP for a single-process job.",
        )
    wrap_policy = _find(flat, "fsdp_auto_wrap_policy")
    if not wrap_policy:
        report.add(
            "fsdp.wrap_policy_missing",
            "FSDP auto-wrap policy is missing",
            "warning",
            "No fsdp_auto_wrap_policy was found.",
            "Choose the transformer-based or size-based policy that matches the model.",
        )

    activation_checkpointing = _as_bool(_find(flat, "fsdp_activation_checkpointing", default=False))
    if training_args_path:
        args = _load_report_config(training_args_path, report)
        if args is not None:
            args_flat = _flatten(args)
            if activation_checkpointing and _as_bool(
                _find(args_flat, "gradient_checkpointing", default=False)
            ):
                report.add(
                    "fsdp.double_checkpointing",
                    "Two checkpointing mechanisms are enabled",
                    "warning",
                    "FSDP activation checkpointing and Trainer gradient checkpointing are both enabled.",
                    "Use one checkpointing mechanism unless the combination has been tested for this model.",
                )
    if not report.has_errors and not report.has_warnings:
        report.add("fsdp.config_ok", "FSDP checks passed", "ok", "No checked FSDP conflict was found.")
    return report


def _adapter_config(path: Union[str, Path], report: DiagnosisReport, label: str) -> Optional[dict[str, Any]]:
    source = Path(path)
    config = source if source.is_file() else source / "adapter_config.json"
    try:
        value = json.loads(config.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("adapter config is not an object")
        return value
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.add(
            f"hotswap.{label}_config",
            f"{label.capitalize()} adapter config could not be read",
            "error",
            str(exc),
            "Pass an adapter directory or adapter_config.json file.",
        )
        return None


def _string_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def check_hotswap_compatibility(
    loaded_adapter: Union[str, Path],
    incoming_adapter: Union[str, Path],
    *,
    compiled: bool = False,
    prepared_max_rank: Optional[int] = None,
) -> DiagnosisReport:
    """Compare two local adapters for PEFT LoRA hotswap constraints."""

    report = DiagnosisReport(
        metadata={
            "loaded_adapter": str(loaded_adapter),
            "incoming_adapter": str(incoming_adapter),
            "compiled": compiled,
            "prepared_max_rank": prepared_max_rank,
        }
    )
    loaded = _adapter_config(loaded_adapter, report, "loaded")
    incoming = _adapter_config(incoming_adapter, report, "incoming")
    if loaded is None or incoming is None:
        return report
    for label, config in (("loaded", loaded), ("incoming", incoming)):
        if str(config.get("peft_type", "LORA")).upper() != "LORA":
            report.add(
                f"hotswap.{label}_not_lora",
                f"{label.capitalize()} adapter is not LoRA",
                "error",
                f"peft_type is {config.get('peft_type')!r}.",
                "PEFT hotswap currently requires LoRA adapters.",
            )
    loaded_base = loaded.get("base_model_name_or_path")
    incoming_base = incoming.get("base_model_name_or_path")
    if loaded_base and incoming_base and loaded_base != incoming_base:
        report.add(
            "hotswap.base_model_mismatch",
            "Adapters reference different base models",
            "error",
            f"Loaded base: {loaded_base}; incoming base: {incoming_base}.",
            "Hotswap adapters trained for the same base model revision.",
        )
    loaded_targets = _string_set(loaded.get("target_modules"))
    incoming_targets = _string_set(incoming.get("target_modules"))
    if loaded_targets and incoming_targets and not incoming_targets.issubset(loaded_targets):
        extra = sorted(incoming_targets - loaded_targets)
        report.add(
            "hotswap.target_modules",
            "Incoming adapter targets new layers",
            "error",
            f"Incoming-only target modules: {', '.join(extra)}.",
            "Load the adapter with the superset of target layers first.",
        )
    loaded_parameters = _string_set(loaded.get("target_parameters"))
    incoming_parameters = _string_set(incoming.get("target_parameters"))
    if compiled and (loaded_parameters or incoming_parameters):
        report.add(
            "hotswap.compiled_target_parameters",
            "Compiled hotswap targets raw parameters",
            "warning",
            "target_parameters can cause recompilation or graph breaks during compiled hotswap.",
            "Avoid this combination or test graph counts before serving traffic.",
        )
    incoming_rank = incoming.get("r") if isinstance(incoming.get("r"), int) else 0
    loaded_rank = loaded.get("r") if isinstance(loaded.get("r"), int) else 0
    max_rank = prepared_max_rank if prepared_max_rank is not None else loaded_rank
    if compiled and incoming_rank > max_rank:
        report.add(
            "hotswap.rank_exceeds_prepared",
            "Incoming rank exceeds the prepared rank",
            "error",
            f"Incoming rank is {incoming_rank}; prepared rank is {max_rank}.",
            "Prepare compiled hotswap with a max_rank at least as large as every adapter rank.",
        )
    loaded_modules = _string_set(loaded.get("modules_to_save"))
    incoming_modules = _string_set(incoming.get("modules_to_save"))
    if loaded_modules != incoming_modules and (loaded_modules or incoming_modules):
        report.add(
            "hotswap.modules_to_save",
            "modules_to_save differs between adapters",
            "warning",
            "Hotswap is intended for LoRA weights; extra saved modules differ.",
            "Reload the model normally when modules_to_save changes.",
        )
    if not report.has_errors and not report.has_warnings:
        report.add(
            "hotswap.compatible",
            "No checked hotswap conflict found",
            "ok",
            "Base model, adapter type, target layers, and prepared rank satisfy the checked constraints.",
        )
    return report


def check_lora_config(
    config_path: Union[str, Path],
    *,
    compiled: bool = False,
    hotswap: bool = False,
) -> DiagnosisReport:
    """Validate modern LoraConfig fields and combinations."""

    report = DiagnosisReport(metadata={"config": str(config_path), "compiled": compiled, "hotswap": hotswap})
    config = _load_report_config(config_path, report)
    if config is None:
        return report
    flat = _flatten(config)
    rank = _find(flat, "r")
    alpha = _find(flat, "lora_alpha")
    dropout = _find(flat, "lora_dropout")
    if not isinstance(rank, int) or rank <= 0:
        report.add("lora.rank", "LoRA rank is invalid", "error", f"r is {rank!r}.", "Set r to a positive integer.")
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        report.add("lora.alpha", "LoRA alpha is invalid", "error", f"lora_alpha is {alpha!r}.", "Set lora_alpha to a positive number.")
    if not isinstance(dropout, (int, float)) or not 0 <= float(dropout) < 1:
        report.add("lora.dropout", "LoRA dropout is invalid", "error", f"lora_dropout is {dropout!r}.", "Use a value from 0 up to, but not including, 1.")
    targets = _string_set(_find(flat, "target_modules"))
    parameters = _string_set(_find(flat, "target_parameters"))
    risky = sorted(name for name in targets if any(part in name.lower() for part in ("lm_head", "embed", "wte")))
    if risky:
        report.add(
            "lora.risky_targets",
            "Output or embedding layers are targeted",
            "warning",
            f"Review these target modules: {', '.join(risky)}.",
            "Use trainable_token_indices for selected tokens or modules_to_save when full-layer training is deliberate.",
        )
    if targets == {"all-linear"}:
        report.add(
            "lora.all_linear",
            "All linear layers are targeted",
            "info",
            "target_modules='all-linear' follows the QLoRA-style coverage option.",
            "Confirm the larger adapter and trainable-parameter count fit the run budget.",
        )
    token_indices = _find(flat, "trainable_token_indices")
    if token_indices:
        report.add(
            "lora.trainable_tokens",
            "Selected token embeddings are trainable",
            "info",
            "trainable_token_indices is configured.",
            "Verify new tokens were added and model embeddings were resized before training.",
        )
        if not _as_bool(_find(flat, "ensure_weight_tying", default=False)):
            report.add(
                "lora.weight_tying_review",
                "Weight tying needs review",
                "warning",
                "Selected token training is enabled without ensure_weight_tying.",
                "Enable ensure_weight_tying when the model ties input and output embeddings.",
            )
    if _find(flat, "alora_invocation_tokens"):
        report.add(
            "lora.alora_merge",
            "Activated LoRA cannot be merged into the base model",
            "warning",
            "alora_invocation_tokens is configured.",
            "Keep the adapter separate and load it at inference time.",
        )
    if _as_bool(_find(flat, "use_qalora", default=False)):
        group_size = _find(flat, "qalora_group_size", default=16)
        if not isinstance(group_size, int) or group_size <= 0:
            report.add(
                "lora.qalora_group_size",
                "QALoRA group size is invalid",
                "error",
                f"qalora_group_size is {group_size!r}.",
                "Set qalora_group_size to a positive integer.",
            )
    init_mode = str(_find(flat, "init_lora_weights", default="true")).lower()
    known_init = {"true", "false", "gaussian", "eva", "olora", "corda", "loftq", "orthogonal"}
    if init_mode not in known_init and not init_mode.startswith("pissa"):
        report.add(
            "lora.init_mode",
            "LoRA initialization mode is not recognized",
            "warning",
            f"init_lora_weights is {init_mode!r}.",
            "Check the installed PEFT version and the exact initializer spelling.",
        )
    if compiled and hotswap and parameters:
        report.add(
            "lora.target_parameters_compile",
            "Compiled hotswap with target_parameters is fragile",
            "warning",
            "The config targets raw parameters and is marked for compiled hotswap.",
            "Expect recompilation or graph breaks; prefer target_modules for this serving path.",
        )
    if not targets and not parameters:
        report.add(
            "lora.targets_missing",
            "No LoRA target is configured",
            "warning",
            "Neither target_modules nor target_parameters was found.",
            "Set target_modules explicitly, use all-linear, or rely on a verified PEFT architecture mapping.",
        )
    if not report.has_errors and not report.has_warnings:
        report.add("lora.config_ok", "LoRA config checks passed", "ok", "No checked LoraConfig conflict was found.")
    return report


def inventory_checkpoints(path: Union[str, Path]) -> DiagnosisReport:
    """List checkpoint health and choose the newest checkpoint without errors."""

    root = Path(path)
    report = DiagnosisReport(metadata={"path": str(root), "checkpoints": []})
    if not root.is_dir():
        report.add("checkpoint_list.missing", "Checkpoint directory is missing", "error", f"{root} is not a directory.")
        return report
    candidates = [item for item in root.iterdir() if item.is_dir() and re.fullmatch(r"checkpoint[-_]\d+", item.name)]
    if re.fullmatch(r"checkpoint[-_]\d+", root.name):
        candidates = [root]
    candidates.sort(key=lambda item: int(re.search(r"\d+$", item.name).group(0)))
    valid: list[Path] = []
    for checkpoint in candidates:
        checked = inspect_trainer_checkpoint(checkpoint)
        status = "valid" if not checked.has_errors else "invalid"
        if status == "valid":
            valid.append(checkpoint)
        entry = {
            "path": str(checkpoint),
            "step": checked.metadata.get("checkpoint_step"),
            "status": status,
            "errors": checked.summary.get("error", 0),
            "warnings": checked.summary.get("warning", 0),
        }
        report.metadata["checkpoints"].append(entry)
        report.add(
            f"checkpoint_list.{checkpoint.name}",
            f"{checkpoint.name}: {status}",
            "ok" if status == "valid" else "error",
            f"{entry['errors']} error(s), {entry['warnings']} warning(s).",
            "Inspect this checkpoint directly with checkpoint-check." if status == "invalid" else None,
        )
    if not candidates:
        report.add("checkpoint_list.empty", "No checkpoints found", "warning", "No checkpoint-N directories were found.")
    elif valid:
        newest = valid[-1]
        report.metadata["latest_valid"] = str(newest)
        report.add(
            "checkpoint_list.latest_valid",
            f"Latest valid checkpoint: {newest.name}",
            "ok",
            "This is the newest checkpoint without a structural error.",
            f"Run peft-doctor resume-check {newest} --config config.json before resuming.",
        )
    else:
        report.add(
            "checkpoint_list.no_valid",
            "No structurally valid checkpoint found",
            "error",
            "Every checkpoint has at least one structural error.",
            "Inspect the errors and restore an earlier copy before resuming.",
        )
    return report


def _row_text(row: dict[str, Any]) -> str:
    messages = _messages(row)
    if messages:
        return "\n".join(str(item.get("content", "")) for item in messages)
    return "\n".join(
        str(row.get(key, ""))
        for key in ("text", "prompt", "completion", "instruction", "input", "response", "output", "answer")
        if row.get(key) is not None
    )


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[\w']+", text.lower(), flags=re.UNICODE))


def check_dataset_leakage(
    train_path: Union[str, Path],
    eval_path: Union[str, Path],
    *,
    threshold: float = 0.9,
    limit: int = 5000,
    max_examples: int = 20,
) -> DiagnosisReport:
    """Find exact and high-overlap train/eval examples with bounded local work."""

    report = DiagnosisReport(
        metadata={"train": str(train_path), "eval": str(eval_path), "threshold": threshold, "limit": limit}
    )
    if not 0 < threshold <= 1:
        report.add("leakage.threshold", "Leakage threshold is invalid", "error", f"threshold is {threshold}.", "Use a value above 0 and at most 1.")
        return report
    train_rows = _dataset_rows(Path(train_path), limit, report)
    eval_rows = _dataset_rows(Path(eval_path), limit, report)
    train_texts = [_normalize_text(_row_text(row)) for row in train_rows]
    eval_texts = [_normalize_text(_row_text(row)) for row in eval_rows]
    exact_index: defaultdict[str, list[int]] = defaultdict(list)
    token_index: defaultdict[str, set[int]] = defaultdict(set)
    train_tokens: list[set[str]] = []
    for index, text in enumerate(train_texts):
        exact_index[text].append(index)
        tokens = set(text.split())
        train_tokens.append(tokens)
        for token in tokens:
            if len(token_index[token]) < 250:
                token_index[token].add(index)
    exact_pairs: list[dict[str, Any]] = []
    near_pairs: list[dict[str, Any]] = []
    for eval_index, text in enumerate(eval_texts):
        if not text:
            continue
        for train_index in exact_index.get(text, []):
            if len(exact_pairs) < max_examples:
                exact_pairs.append({"train_row": train_index + 1, "eval_row": eval_index + 1})
        eval_tokens = set(text.split())
        candidates: Counter[int] = Counter()
        for token in eval_tokens:
            candidates.update(token_index.get(token, set()))
        for train_index, _overlap in candidates.most_common(100):
            if train_texts[train_index] == text:
                continue
            union = eval_tokens | train_tokens[train_index]
            similarity = len(eval_tokens & train_tokens[train_index]) / max(1, len(union))
            if similarity >= threshold:
                if len(near_pairs) < max_examples:
                    near_pairs.append(
                        {"train_row": train_index + 1, "eval_row": eval_index + 1, "jaccard": round(similarity, 4)}
                    )
                break
    report.metadata.update(
        {
            "train_rows_sampled": len(train_rows),
            "eval_rows_sampled": len(eval_rows),
            "exact_examples": exact_pairs,
            "near_examples": near_pairs,
        }
    )
    if exact_pairs:
        report.add(
            "leakage.exact",
            "Exact train/eval overlap found",
            "error",
            f"Found at least {len(exact_pairs)} exact pair(s) in the retained examples.",
            "Remove overlapping examples and rebuild the evaluation split.",
        )
    if near_pairs:
        report.add(
            "leakage.near",
            "High token-overlap examples found",
            "warning",
            f"Found at least {len(near_pairs)} pair(s) at or above Jaccard {threshold}.",
            "Review the reported row pairs; paraphrases and templated records may still leak evaluation content.",
        )
    if not exact_pairs and not near_pairs and not report.has_errors:
        report.add(
            "leakage.none",
            "No sampled overlap found",
            "ok",
            f"No exact or threshold-level overlap was found in {len(train_rows)} train and {len(eval_rows)} eval rows.",
            "Increase --limit for broader coverage when the datasets are larger.",
        )
    return report


def inspect_adapter_tensors(adapter_path: Union[str, Path]) -> DiagnosisReport:
    """Summarize exact safetensors shapes, dtypes, parameters, and file sizes."""

    adapter = Path(adapter_path)
    report = DiagnosisReport(metadata={"adapter": str(adapter)})
    files = sorted(adapter.glob("adapter_model*.safetensors")) if adapter.is_dir() else [adapter]
    files = [path for path in files if path.is_file() and path.suffix == ".safetensors"]
    if not files:
        report.add("adapter_tensors.missing", "No adapter safetensors found", "error", "No adapter_model*.safetensors file was found.")
        return report
    dtype_counts: Counter[str] = Counter()
    rank_counts: Counter[int] = Counter()
    tensor_count = 0
    parameter_count = 0
    for file in files:
        try:
            header = read_safetensors_header(file)
        except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            report.add("adapter_tensors.invalid", "Safetensors metadata is invalid", "error", f"{file.name}: {exc}")
            continue
        for name, info in header.items():
            if name == "__metadata__":
                continue
            shape = info["shape"]
            tensor_count += 1
            parameter_count += math.prod(shape)
            dtype_counts[str(info.get("dtype", "unknown"))] += 1
            rank_counts[len(shape)] += 1
    size_bytes = sum(path.stat().st_size for path in files)
    report.metadata.update(
        {
            "files": [path.name for path in files],
            "file_size_bytes": size_bytes,
            "tensor_count": tensor_count,
            "parameter_count": parameter_count,
            "dtype_tensor_counts": dict(dtype_counts),
            "tensor_rank_counts": {str(key): value for key, value in rank_counts.items()},
        }
    )
    if not report.has_errors:
        report.add(
            "adapter_tensors.summary",
            f"{tensor_count} tensors, {parameter_count:,} parameters",
            "ok",
            f"Read {len(files)} safetensors file(s), {size_bytes / (1024 * 1024):.2f} MiB on disk.",
            dtypes=dict(dtype_counts),
            tensor_ranks=dict(rank_counts),
        )
    return report


def check_chat_template(
    template_path: Union[str, Path],
    *,
    dataset_path: Optional[Union[str, Path]] = None,
    assistant_only_loss: bool = False,
    limit: int = 500,
) -> DiagnosisReport:
    """Check chat-template structure and whether sampled dataset roles are handled."""

    source = Path(template_path)
    report = DiagnosisReport(metadata={"template": str(source), "dataset": str(dataset_path) if dataset_path else None})
    try:
        if source.suffix.lower() == ".json":
            data = json.loads(source.read_text(encoding="utf-8-sig"))
            template = data.get("chat_template", "") if isinstance(data, dict) else ""
            if isinstance(template, dict):
                template = template.get("default") or next(iter(template.values()), "")
        else:
            template = source.read_text(encoding="utf-8-sig")
    except (OSError, json.JSONDecodeError) as exc:
        report.add("chat_template.read_failed", "Chat template could not be read", "error", str(exc))
        return report
    if not isinstance(template, str) or not template.strip():
        report.add("chat_template.missing", "Chat template is missing", "error", "No non-empty chat_template value was found.")
        return report
    generation_open = template.count("{% generation %}")
    generation_close = template.count("{% endgeneration %}")
    report.metadata.update({"generation_blocks": generation_open, "template_chars": len(template)})
    if generation_open != generation_close:
        report.add(
            "chat_template.generation_unbalanced",
            "Generation markers are unbalanced",
            "error",
            f"Found {generation_open} opening and {generation_close} closing generation markers.",
            "Balance every generation block before tokenization.",
        )
    elif assistant_only_loss and generation_open == 0:
        report.add(
            "chat_template.generation_missing",
            "Assistant masking markers are missing",
            "error",
            "assistant_only_loss was requested but the template has no generation block.",
            "Wrap assistant output in {% generation %} and {% endgeneration %}.",
        )
    roles: Counter[str] = Counter()
    if dataset_path:
        for row in _dataset_rows(Path(dataset_path), limit, report):
            roles.update(str(item.get("role", item.get("from", ""))).lower() for item in _messages(row))
        report.metadata["dataset_roles"] = dict(roles)
        for role in sorted(roles):
            if role and role not in template.lower():
                report.add(
                    f"chat_template.role_{re.sub(r'[^a-z0-9]+', '_', role)}",
                    f"Dataset role {role!r} is not visible in the template",
                    "warning",
                    f"The sampled dataset uses role {role!r}, but that word does not appear in the template.",
                    "Review how the template maps this role; aliases may be valid but need a tokenization test.",
                )
    if "messages" not in template:
        report.add(
            "chat_template.messages_missing",
            "Template does not reference messages",
            "warning",
            "The template text does not contain a messages variable.",
            "Confirm the template uses the conversation variable expected by the tokenizer.",
        )
    if not report.has_errors and not report.has_warnings:
        report.add("chat_template.ok", "Chat-template checks passed", "ok", "No checked template conflict was found.")
    return report


def audit_label_masking(
    dataset_path: Union[str, Path],
    *,
    pad_token_id: Optional[int] = None,
    limit: int = 1000,
) -> DiagnosisReport:
    """Measure supervision and pad masking in pretokenized dataset rows."""

    report = DiagnosisReport(metadata={"dataset": str(dataset_path), "pad_token_id": pad_token_id})
    rows = _dataset_rows(Path(dataset_path), limit, report)
    checked = 0
    missing = 0
    mismatched = 0
    all_ignored = 0
    no_ignored = 0
    pad_unmasked = 0
    supervised = 0
    total = 0
    examples: list[int] = []
    for index, row in enumerate(rows, start=1):
        input_ids = row.get("input_ids")
        labels = row.get("labels")
        if not isinstance(input_ids, list) or not isinstance(labels, list):
            missing += 1
            continue
        checked += 1
        if len(input_ids) != len(labels):
            mismatched += 1
            examples.append(index)
            continue
        row_supervised = sum(label != -100 for label in labels)
        supervised += row_supervised
        total += len(labels)
        if row_supervised == 0:
            all_ignored += 1
        if row_supervised == len(labels) and labels:
            no_ignored += 1
        if pad_token_id is not None:
            bad = any(token == pad_token_id and label != -100 for token, label in zip(input_ids, labels))
            if bad:
                pad_unmasked += 1
                examples.append(index)
    ratio = supervised / max(1, total)
    report.metadata.update(
        {
            "rows_sampled": len(rows),
            "pretokenized_rows": checked,
            "rows_without_token_labels": missing,
            "length_mismatches": mismatched,
            "all_labels_ignored": all_ignored,
            "no_labels_ignored": no_ignored,
            "pad_rows_unmasked": pad_unmasked,
            "supervised_token_ratio": round(ratio, 6),
            "example_rows": sorted(set(examples))[:20],
        }
    )
    if not checked:
        report.add(
            "labels.not_pretokenized",
            "No input_ids/labels rows found",
            "warning",
            "The sampled dataset is not pretokenized, so label masks cannot be inspected from disk.",
            "Save one tokenized batch or inspect the data collator output.",
        )
        return report
    if missing:
        report.add(
            "labels.rows_missing",
            "Some rows do not contain token labels",
            "warning",
            f"{missing} of {len(rows)} sampled row(s) lack list-valued input_ids and labels.",
            "Keep raw and pretokenized formats in separate splits or generate labels consistently.",
        )
    if mismatched:
        report.add("labels.length_mismatch", "Input and label lengths differ", "error", f"Found {mismatched} row(s).", "Make every labels array match input_ids length.")
    if all_ignored:
        report.add("labels.all_ignored", "Rows have no supervised tokens", "error", f"Found {all_ignored} row(s) where every label is -100.", "Fix response or assistant masking before training.")
    if pad_unmasked:
        report.add("labels.pad_unmasked", "Pad tokens contribute to loss", "warning", f"Found {pad_unmasked} row(s) with unmasked pad positions.", "Set labels to -100 wherever input_ids equals pad_token_id.")
    if no_ignored:
        report.add("labels.full_sequence", "Rows supervise every token", "info", f"Found {no_ignored} row(s) with no -100 labels.", "This is valid for full-sequence LM loss; verify it is intentional for instruction tuning.")
    if not report.has_errors and not report.has_warnings:
        report.add("labels.masking_ok", "Label-mask checks passed", "ok", f"Supervised token ratio is {ratio:.1%} across {checked} row(s).")
    return report


def check_quantization_config(
    config_path: Union[str, Path],
    *,
    gpu: Optional[str] = None,
    training: bool = True,
) -> DiagnosisReport:
    """Check bitsandbytes 4-bit/8-bit settings used by PEFT training."""

    report = DiagnosisReport(metadata={"config": str(config_path), "gpu": gpu, "training": training})
    config = _load_report_config(config_path, report)
    if config is None:
        return report
    flat = _flatten(config)
    four_bit = _as_bool(_find(flat, "load_in_4bit", default=False))
    eight_bit = _as_bool(_find(flat, "load_in_8bit", default=False))
    quant_type = str(_find(flat, "bnb_4bit_quant_type", default="")).lower()
    compute_dtype = str(_find(flat, "bnb_4bit_compute_dtype", default="")).lower()
    storage_dtype = str(_find(flat, "bnb_4bit_quant_storage", default="")).lower()
    model_dtype = str(_find(flat, "torch_dtype", default="")).lower()
    fsdp = _as_bool(_find(flat, "fsdp", default=False)) or "fsdp" in str(_find(flat, "distributed_type", default="")).lower()
    report.metadata.update(
        {
            "load_in_4bit": four_bit,
            "load_in_8bit": eight_bit,
            "quant_type": quant_type,
            "compute_dtype": compute_dtype,
            "storage_dtype": storage_dtype,
        }
    )
    if four_bit and eight_bit:
        report.add("quantization.mode_conflict", "4-bit and 8-bit loading are both enabled", "error", "Only one bitsandbytes loading mode can be active.", "Disable one quantization mode.")
    if not four_bit and not eight_bit:
        report.add("quantization.disabled", "bitsandbytes quantization is disabled", "info", "Neither load_in_4bit nor load_in_8bit is enabled.")
    if training and four_bit and quant_type != "nf4":
        report.add(
            "quantization.nf4",
            "4-bit training is not configured for NF4",
            "warning",
            f"bnb_4bit_quant_type is {quant_type or 'unset'}.",
            "Use nf4 for training normally distributed model weights unless a different format is deliberate.",
        )
    if four_bit and not compute_dtype:
        report.add(
            "quantization.compute_dtype_missing",
            "4-bit compute dtype is not explicit",
            "warning",
            "bnb_4bit_compute_dtype was not found.",
            "Set an explicit compute dtype supported by the GPU, commonly bfloat16 or float16.",
        )
    if gpu and "t4" in gpu.lower() and "bfloat16" in compute_dtype:
        report.add(
            "quantization.t4_bf16",
            "T4 does not provide native bf16 Tensor Core training",
            "warning",
            "The compute dtype is bfloat16 for a T4 plan.",
            "Use float16 on T4 and monitor overflow, or move to hardware with native bf16 support.",
        )
    if fsdp and four_bit and storage_dtype and model_dtype and storage_dtype != model_dtype:
        report.add(
            "quantization.fsdp_storage_dtype",
            "FSDP storage dtype differs from the model dtype",
            "warning",
            f"bnb_4bit_quant_storage is {storage_dtype}; torch_dtype is {model_dtype}.",
            "Use matching floating dtypes so FSDP can shard quantized layers consistently.",
        )
    if not report.has_errors and not report.has_warnings:
        report.add("quantization.config_ok", "Quantization checks passed", "ok", "No checked bitsandbytes conflict was found.")
    return report
