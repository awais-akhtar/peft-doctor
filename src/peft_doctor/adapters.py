"""Adapter artifact checks and LoRA merge helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .report import DiagnosisReport


ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHT_NAMES = [
    "adapter_model.safetensors",
    "adapter_model.bin",
]
TOKENIZER_NAMES = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


@dataclass
class MergeResult:
    """Result returned after a successful adapter merge."""

    output_dir: Optional[str]
    hub_model_id: Optional[str]
    tokenizer_saved: bool
    pushed_to_hub: bool
    safe_serialization: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "hub_model_id": self.hub_model_id,
            "tokenizer_saved": self.tokenizer_saved,
            "pushed_to_hub": self.pushed_to_hub,
            "safe_serialization": self.safe_serialization,
        }


def _is_local_path(value: str) -> bool:
    try:
        return Path(value).expanduser().exists()
    except OSError:
        return False


def _read_adapter_config(adapter: str) -> Optional[dict[str, Any]]:
    if not _is_local_path(adapter):
        return None
    path = Path(adapter).expanduser() / ADAPTER_CONFIG
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _has_local_tokenizer_files(path: str) -> bool:
    if not _is_local_path(path):
        return False
    root = Path(path).expanduser()
    return any((root / name).exists() for name in TOKENIZER_NAMES)


def _local_weight_files(path: str) -> list[str]:
    if not _is_local_path(path):
        return []
    root = Path(path).expanduser()
    found = [name for name in ADAPTER_WEIGHT_NAMES if (root / name).exists()]
    found.extend(file.name for file in sorted(root.glob("adapter_model-*.safetensors")))
    found.extend(file.name for file in sorted(root.glob("adapter_model-*.bin")))
    return found


def diagnose_adapter_merge(
    base_model: Optional[str] = None,
    adapter: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    overwrite: bool = False,
    merge_plan: bool = True,
) -> DiagnosisReport:
    """Check whether an adapter merge plan looks healthy."""

    report = DiagnosisReport(
        metadata={
            "base_model": base_model,
            "adapter": adapter,
            "output_dir": str(output_dir) if output_dir else None,
            "push_to_hub": push_to_hub,
            "hub_model_id": hub_model_id,
            "merge_plan": merge_plan,
        }
    )

    adapter_config = _read_adapter_config(adapter) if adapter else None
    adapter_base = adapter_config.get("base_model_name_or_path") if adapter_config else None
    effective_base = base_model or adapter_base

    if not adapter:
        report.add(
            "adapter_merge.adapter_missing",
            "Adapter is required",
            "error",
            "No adapter path or Hub id was provided.",
            "Pass `--adapter your-user/your-adapter` or a local adapter directory.",
        )
    elif _is_local_path(adapter):
        weights = _local_weight_files(adapter)
        if adapter_config:
            report.add(
                "adapter_merge.config_found",
                "Adapter config found",
                "ok",
                "A local adapter_config.json file was found.",
                peft_type=adapter_config.get("peft_type"),
                base_model_name_or_path=adapter_base,
            )
        else:
            report.add(
                "adapter_merge.config_missing",
                "Adapter config is missing",
                "error",
                "The local adapter directory does not contain adapter_config.json.",
                "Make sure this is a PEFT adapter directory saved with `model.save_pretrained(adapter_dir)`.",
            )

        if weights:
            report.add(
                "adapter_merge.weights_found",
                "Adapter weights found",
                "ok",
                "Adapter weight files were found.",
                files=", ".join(weights),
            )
        else:
            report.add(
                "adapter_merge.weights_missing",
                "Adapter weights are missing",
                "error",
                "No adapter_model.safetensors or adapter_model.bin file was found.",
                "Check that the adapter upload completed and that you are pointing at the adapter directory.",
            )
    else:
        report.add(
            "adapter_merge.hub_adapter",
            "Hub adapter will be loaded",
            "info",
            "The adapter looks like a Hugging Face Hub id or a remote path.",
            "Make sure you are logged in if the adapter is private or gated.",
        )

    if not effective_base:
        report.add(
            "adapter_merge.base_missing",
            "Base model is required",
            "error",
            "No base model was provided and the adapter config did not include base_model_name_or_path.",
            "Pass `--base-model` with the original base model used for LoRA training.",
        )
    elif base_model and adapter_base and base_model != adapter_base:
        report.add(
            "adapter_merge.base_mismatch",
            "Base model differs from adapter metadata",
            "warning",
            "The requested base model is different from adapter_config.json metadata.",
            "Use the same base model that was used during LoRA training unless you know the adapter is compatible.",
            requested_base_model=base_model,
            adapter_base_model=adapter_base,
        )
    else:
        report.add(
            "adapter_merge.base_ok",
            "Base model is available",
            "ok",
            "A base model id is available for the merge.",
            base_model=effective_base,
        )

    if merge_plan:
        if load_in_4bit or load_in_8bit:
            report.add(
                "adapter_merge.quantized_merge_risky",
                "Quantized merge is risky",
                "warning",
                "Merging adapters after 4-bit or 8-bit loading can fail or produce a model that is not saved as expected.",
                "For final export, load the base model in fp16, bf16, or fp32, merge, then save with safetensors.",
            )
        else:
            report.add(
                "adapter_merge.full_precision_merge",
                "Merge will use non-quantized loading",
                "ok",
                "This is the safest path for exporting a merged model.",
            )

        if output_dir:
            out = Path(output_dir)
            if out.exists() and any(out.iterdir()) and not overwrite:
                report.add(
                    "adapter_merge.output_exists",
                    "Output directory already has files",
                    "warning",
                    "The output directory exists and is not empty.",
                    "Pass `--overwrite` if you want save_pretrained to write into this directory.",
                    output_dir=str(out),
                )
            else:
                report.add(
                    "adapter_merge.output_ok",
                    "Output directory is usable",
                    "ok",
                    "The output directory can be used for save_pretrained.",
                    output_dir=str(out),
                )
        elif not push_to_hub:
            report.add(
                "adapter_merge.no_output",
                "No output target provided",
                "error",
                "The merge command needs a local output directory or a Hub repository.",
                "Pass `--output-dir merged-model` or use `--push-to-hub --hub-model-id user/repo`.",
            )

        if push_to_hub and not hub_model_id:
            report.add(
                "adapter_merge.hub_id_missing",
                "Hub model id is missing",
                "error",
                "Push-to-Hub was requested but no target repo id was provided.",
                "Pass `--hub-model-id username/model-name`.",
            )
        elif push_to_hub:
            report.add(
                "adapter_merge.hub_push_ready",
                "Hub push target is set",
                "ok",
                "The merged model and tokenizer will be pushed to the Hub.",
                hub_model_id=hub_model_id,
            )

    report.add(
        "adapter_merge.secret_note",
        "Use environment login for Hub tokens",
        "info",
        "Do not hard-code Hugging Face tokens in notebooks or scripts.",
        "Use `huggingface-cli login`, Colab secrets, or the `HF_TOKEN` environment variable.",
    )

    return report


def _resolve_torch_dtype(dtype: str) -> Any:
    if dtype == "auto":
        return "auto"

    import torch

    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[dtype.lower()]
    except KeyError as exc:
        raise ValueError("dtype must be one of: auto, bf16, bfloat16, fp16, float16, fp32, float32") from exc


def _maybe_resize_embeddings(model: Any, tokenizer: Any) -> bool:
    if tokenizer is None:
        return False
    try:
        embedding = model.get_input_embeddings()
        current_size = int(embedding.weight.shape[0])
        tokenizer_size = int(len(tokenizer))
    except Exception:
        return False
    if tokenizer_size and tokenizer_size != current_size:
        model.resize_token_embeddings(tokenizer_size)
        return True
    return False


def merge_lora_adapter(
    base_model: str,
    adapter: str,
    output_dir: Optional[Union[str, Path]] = None,
    *,
    tokenizer_source: Optional[str] = None,
    torch_dtype: str = "auto",
    device_map: Optional[str] = "auto",
    offload_folder: Optional[str] = "offload",
    low_cpu_mem_usage: bool = True,
    trust_remote_code: bool = False,
    safe_serialization: bool = True,
    max_shard_size: str = "5GB",
    save_tokenizer: bool = True,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    allow_quantized_merge: bool = False,
    resize_token_embeddings: bool = True,
) -> MergeResult:
    """Merge a PEFT LoRA adapter into its base model and optionally save/push it.

    This helper intentionally loads the base model through `AutoModelForCausalLM`
    and merges with `PeftModel.merge_and_unload()`, which is the standard export
    path for causal language model adapters.
    """

    if (load_in_4bit or load_in_8bit) and not allow_quantized_merge:
        raise ValueError(
            "Quantized merge was requested. For a final merged export, load the base model "
            "in fp16, bf16, or fp32. Pass allow_quantized_merge=True only if you know this "
            "PEFT/Transformers combination supports it."
        )

    if not output_dir and not push_to_hub:
        raise ValueError("Provide output_dir or enable push_to_hub with hub_model_id.")
    if push_to_hub and not hub_model_id:
        raise ValueError("hub_model_id is required when push_to_hub=True.")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _resolve_torch_dtype(torch_dtype)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    if offload_folder:
        model_kwargs["offload_folder"] = offload_folder
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    tokenizer_id = tokenizer_source or (adapter if _has_local_tokenizer_files(adapter) else base_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if resize_token_embeddings:
        _maybe_resize_embeddings(model, tokenizer)

    peft_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "is_trainable": False,
    }
    if device_map:
        peft_kwargs["device_map"] = device_map
    if offload_folder:
        peft_kwargs["offload_folder"] = offload_folder

    model = PeftModel.from_pretrained(model, adapter, **peft_kwargs)
    model = model.merge_and_unload()
    model.eval()

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(
            out,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )
        if save_tokenizer:
            tokenizer.save_pretrained(out)

    if push_to_hub:
        model.push_to_hub(
            hub_model_id,
            private=private,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
            commit_message=commit_message,
        )
        if save_tokenizer:
            tokenizer.push_to_hub(hub_model_id, private=private, commit_message=commit_message)

    return MergeResult(
        output_dir=str(output_dir) if output_dir else None,
        hub_model_id=hub_model_id,
        tokenizer_saved=save_tokenizer,
        pushed_to_hub=push_to_hub,
        safe_serialization=safe_serialization,
    )
