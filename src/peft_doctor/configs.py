"""Safe starter configs for PEFT and QLoRA."""

from __future__ import annotations

from typing import Any, Optional

from .targets import recommend_target_modules


def create_safe_lora_config(
    model: Any = None,
    model_name: Optional[str] = None,
    model_family: Optional[str] = None,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    include_mlp: bool = True,
    as_dict: bool = False,
) -> Any:
    """Return a safe LoRA config.

    If PEFT is installed and `as_dict` is false, this returns `peft.LoraConfig`.
    Otherwise it returns a plain dictionary.
    """

    config = {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": bias,
        "task_type": task_type,
        "target_modules": recommend_target_modules(
            model=model,
            model_name=model_name,
            model_family=model_family,
            include_mlp=include_mlp,
        ),
    }

    if as_dict:
        return config

    try:
        from peft import LoraConfig

        return LoraConfig(**config)
    except Exception:
        return config


def create_safe_bnb_config(
    load_in_4bit: bool = True,
    quant_type: str = "nf4",
    compute_dtype: str = "bf16",
    use_double_quant: bool = True,
    as_dict: bool = False,
) -> Any:
    """Return a safe BitsAndBytesConfig or a serializable dictionary."""

    dtype_name = compute_dtype.lower().replace("bfloat16", "bf16")
    config = {
        "load_in_4bit": load_in_4bit,
        "bnb_4bit_quant_type": quant_type,
        "bnb_4bit_compute_dtype": "bfloat16" if dtype_name == "bf16" else compute_dtype,
        "bnb_4bit_use_double_quant": use_double_quant,
    }

    if as_dict:
        return config

    try:
        import torch
        from transformers import BitsAndBytesConfig

        torch_dtype = torch.bfloat16 if dtype_name == "bf16" else getattr(torch, compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
        )
    except Exception:
        return config


def create_safe_training_args(
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    num_train_epochs: int = 3,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    warmup_ratio: float = 0.03,
    lr_scheduler_type: str = "cosine",
    max_grad_norm: float = 1.0,
    logging_steps: int = 10,
    save_steps: int = 500,
    save_total_limit: int = 2,
    seed: int = 42,
    eval_strategy: str = "no",
) -> dict[str, Any]:
    """Return conservative TrainingArguments-style values."""

    return {
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "bf16": bf16,
        "gradient_checkpointing": gradient_checkpointing,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "max_grad_norm": max_grad_norm,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "seed": seed,
        "eval_strategy": eval_strategy,
    }
