"""Memory and precision checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .report import DiagnosisReport
from .utils import bool_value, coerce_int, first_value, get_value


@dataclass
class GPUInfo:
    index: int
    name: str
    total_gb: float
    free_gb: Optional[float] = None


def get_gpu_info() -> list[GPUInfo]:
    try:
        import torch
    except Exception:
        return []

    try:
        if not torch.cuda.is_available():
            return []
        gpus = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free_gb = None
            try:
                free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
                free_gb = free_bytes / (1024**3)
            except Exception:
                pass
            gpus.append(
                GPUInfo(
                    index=index,
                    name=props.name,
                    total_gb=props.total_memory / (1024**3),
                    free_gb=free_gb,
                )
            )
        return gpus
    except Exception:
        return []


def estimate_model_size_gb(model: Any) -> Optional[float]:
    if model is None:
        return None

    num_parameters = getattr(model, "num_parameters", None)
    if callable(num_parameters):
        try:
            params = int(num_parameters())
            return params * 2 / (1024**3)
        except Exception:
            pass

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        total = 0
        try:
            for param in parameters():
                element_size = param.element_size() if callable(getattr(param, "element_size", None)) else 2
                total += int(param.numel()) * int(element_size)
            if total:
                return total / (1024**3)
        except Exception:
            return None

    return None


def is_quantized(model: Any = None, peft_config: Any = None, training_args: Any = None) -> bool:
    if bool_value(training_args, ["load_in_4bit"], False) or bool_value(
        training_args,
        ["load_in_8bit"],
        False,
    ):
        return True
    if bool_value(peft_config, ["load_in_4bit"], False) or bool_value(
        peft_config,
        ["load_in_8bit"],
        False,
    ):
        return True
    if bool_value(model, ["is_loaded_in_4bit"], False) or bool_value(
        model,
        ["is_loaded_in_8bit"],
        False,
    ):
        return True

    quant_config = get_value(model, "quantization_config") or get_value(get_value(model, "config"), "quantization_config")
    if quant_config is not None:
        return bool_value(quant_config, ["load_in_4bit"], False) or bool_value(
            quant_config,
            ["load_in_8bit"],
            False,
        )
    return False


def check_memory(
    report: DiagnosisReport,
    model: Any = None,
    training_args: Any = None,
    peft_config: Any = None,
    sequence_length: Optional[int] = None,
) -> None:
    gpus = get_gpu_info()
    quantized = is_quantized(model=model, peft_config=peft_config, training_args=training_args)
    batch_size = coerce_int(
        first_value(training_args, ["per_device_train_batch_size", "train_batch_size"], 1),
        1,
    )
    eval_batch_size = coerce_int(
        first_value(training_args, ["per_device_eval_batch_size", "eval_batch_size"], None),
        None,
    )
    grad_checkpointing = bool_value(training_args, ["gradient_checkpointing"], False) or bool_value(
        model, ["is_gradient_checkpointing"], False
    )
    bf16 = bool_value(training_args, ["bf16"], False)
    fp16 = bool_value(training_args, ["fp16"], False)
    seq_len = sequence_length or coerce_int(
        first_value(training_args, ["max_seq_length", "model_max_length", "block_size"], None),
        None,
    )
    model_size_gb = estimate_model_size_gb(model)

    if not gpus:
        report.add(
            "memory.no_cuda",
            "No CUDA GPU detected",
            "warning",
            "Torch did not report a CUDA GPU. Fine-tuning can run on CPU, but PEFT training will be very slow.",
            "Use a CUDA machine for practical LoRA or QLoRA training, or keep the run tiny for debugging.",
        )
    else:
        smallest = min(gpus, key=lambda gpu: gpu.total_gb)
        report.add(
            "memory.cuda_detected",
            "CUDA GPU detected",
            "ok",
            f"Detected {len(gpus)} CUDA GPU(s). Smallest GPU has {smallest.total_gb:.1f} GB VRAM.",
            gpu_count=len(gpus),
            smallest_gpu_gb=round(smallest.total_gb, 2),
        )
        if smallest.total_gb < 16 and not quantized:
            report.add(
                "memory.qlora_recommended",
                "4-bit QLoRA is recommended",
                "warning",
                "The available VRAM is tight for common 7B class fine-tuning without quantization.",
                "Load the model with load_in_4bit=True and use an NF4 BitsAndBytesConfig.",
                vram_gb=round(smallest.total_gb, 2),
            )
        elif smallest.total_gb < 24 and batch_size and batch_size > 1:
            report.add(
                "memory.batch_size_risky",
                "Batch size may cause OOM",
                "warning",
                "The per-device train batch size is above 1 on a GPU with limited VRAM.",
                "Try per_device_train_batch_size=1 and use gradient_accumulation_steps for throughput.",
                batch_size=batch_size,
                vram_gb=round(smallest.total_gb, 2),
            )

    if model_size_gb:
        report.add(
            "memory.model_size_estimate",
            "Model size estimated",
            "info",
            f"The loaded model parameters are roughly {model_size_gb:.2f} GB before optimizer and activations.",
            model_size_gb=round(model_size_gb, 3),
        )

    if not quantized:
        report.add(
            "memory.not_quantized",
            "Model does not look quantized",
            "info",
            "The model does not expose a 4-bit or 8-bit loading flag.",
            "For low VRAM machines, use load_in_4bit=True with NF4 and double quantization.",
        )
    else:
        report.add(
            "memory.quantized",
            "Quantized loading detected",
            "ok",
            "The setup looks like it uses 4-bit or 8-bit loading.",
        )

    if not grad_checkpointing:
        report.add(
            "memory.gradient_checkpointing_off",
            "Gradient checkpointing is off",
            "warning",
            "Activation memory can dominate LoRA training.",
            "Set gradient_checkpointing=True and call model.gradient_checkpointing_enable() when needed.",
        )

    if seq_len and seq_len > 4096:
        report.add(
            "memory.sequence_length_high",
            "Sequence length is high",
            "warning",
            "Long sequences increase activation memory quickly.",
            "Start with 1024 or 2048 tokens, then raise the length after the run is stable.",
            sequence_length=seq_len,
        )

    if eval_batch_size and batch_size and eval_batch_size > batch_size:
        report.add(
            "memory.eval_batch_larger",
            "Eval batch is larger than train batch",
            "warning",
            "Evaluation can run out of memory even when training fits.",
            "Use per_device_eval_batch_size <= per_device_train_batch_size or disable eval during debugging.",
            eval_batch_size=eval_batch_size,
            train_batch_size=batch_size,
        )

    if fp16 and not bf16:
        report.add(
            "precision.fp16_nan_risk",
            "fp16 can be fragile",
            "warning",
            "fp16 training is more likely to overflow on some PEFT runs.",
            "Use bf16=True on supported GPUs, lower the learning rate, and keep max_grad_norm enabled.",
        )
