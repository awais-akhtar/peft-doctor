"""Main PEFT diagnosis entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from ._version import __version__
from .datasets import check_dataset, check_tokenizer
from .memory import check_memory
from .report import DiagnosisReport
from .targets import infer_model_family, missing_target_modules, recommend_target_modules
from .utils import as_list, bool_value, coerce_float, coerce_int, first_value, get_value


def _check_target_modules(
    report: DiagnosisReport,
    model: Any = None,
    model_name: Optional[str] = None,
    peft_config: Any = None,
) -> None:
    family = infer_model_family(model=model, model_name=model_name)
    recommended = recommend_target_modules(model=model, model_name=model_name, model_family=family)
    configured = as_list(get_value(peft_config, "target_modules"))

    if family:
        report.add(
            "targets.family_detected",
            "Model family detected",
            "info",
            f"Model family looks like `{family}`.",
            family=family,
        )
    else:
        report.add(
            "targets.family_unknown",
            "Model family is unknown",
            "info",
            "PEFT Doctor could not confidently infer the model family.",
            "Pass `model_family` to `recommend_target_modules` or inspect `model.named_modules()`.",
        )

    if configured:
        missing = missing_target_modules(model, configured)
        if missing:
            report.add(
                "targets.missing",
                "Configured target modules were not found",
                "warning",
                "Some LoRA target modules do not appear in the model module names.",
                f"Try target_modules={recommended!r}.",
                missing=", ".join(str(item) for item in missing),
                configured=", ".join(str(item) for item in configured),
            )
        else:
            report.add(
                "targets.configured",
                "LoRA target modules are configured",
                "ok",
                "A target_modules value is present on the PEFT config.",
                target_modules=", ".join(str(item) for item in configured),
            )
    else:
        report.add(
            "targets.not_configured",
            "LoRA target modules are missing",
            "warning",
            "No target_modules value was found on the PEFT config.",
            f"Try target_modules={recommended!r}.",
            recommended=", ".join(recommended),
        )

    report.add(
        "targets.recommendation",
        "Suggested target modules",
        "info",
        "These target modules are a practical starting point for this model family.",
        target_modules=", ".join(recommended),
    )


def _check_training_args(report: DiagnosisReport, training_args: Any = None, peft_config: Any = None) -> None:
    if training_args is None:
        report.add(
            "training_args.not_provided",
            "Training arguments not provided",
            "info",
            "Training argument checks were skipped.",
        )
        return

    learning_rate = coerce_float(first_value(training_args, ["learning_rate", "lr"], None), None)
    batch_size = coerce_int(first_value(training_args, ["per_device_train_batch_size", "train_batch_size"], None), None)
    grad_accum = coerce_int(first_value(training_args, ["gradient_accumulation_steps"], None), None)
    epochs = coerce_float(first_value(training_args, ["num_train_epochs"], None), None)
    max_grad_norm = coerce_float(first_value(training_args, ["max_grad_norm"], None), None)
    eval_strategy = first_value(training_args, ["eval_strategy", "evaluation_strategy"], None)
    optim = first_value(training_args, ["optim", "optimizer"], None)
    warmup_steps = coerce_int(first_value(training_args, ["warmup_steps"], None), None)
    warmup_ratio = coerce_float(first_value(training_args, ["warmup_ratio"], None), None)
    lr_scheduler_type = first_value(training_args, ["lr_scheduler_type"], None)
    save_steps = coerce_int(first_value(training_args, ["save_steps"], None), None)
    save_total_limit = coerce_int(first_value(training_args, ["save_total_limit"], None), None)
    seed = first_value(training_args, ["seed", "data_seed"], None)
    dataloader_workers = coerce_int(first_value(training_args, ["dataloader_num_workers"], None), None)
    max_steps = coerce_int(first_value(training_args, ["max_steps"], None), None)
    load_in_4bit = bool_value(training_args, ["load_in_4bit"], False)
    load_in_8bit = bool_value(training_args, ["load_in_8bit"], False)

    is_lora = peft_config is not None or get_value(training_args, "peft_config") is not None
    if learning_rate is not None:
        if learning_rate > 1e-3:
            report.add(
                "training.lr_extreme",
                "Learning rate is extremely high",
                "error",
                f"learning_rate={learning_rate:g} is likely to destabilize fine-tuning.",
                "For LoRA, start around 2e-4, 1e-4, or 5e-5. For full fine-tuning, use a much smaller LR.",
                learning_rate=learning_rate,
            )
        elif learning_rate > 3e-4:
            report.add(
                "training.lr_high",
                "Learning rate looks high",
                "warning",
                f"learning_rate={learning_rate:g} can cause NaN loss or poor outputs.",
                "Try 1e-4 or 5e-5 if loss spikes, repeats, or becomes NaN.",
                learning_rate=learning_rate,
            )
        elif not is_lora and learning_rate > 5e-5:
            report.add(
                "training.full_finetune_lr",
                "Learning rate may be high for full fine-tuning",
                "warning",
                "This looks higher than a common full fine-tuning starting point.",
                "If you are not using LoRA, try 5e-5 or lower.",
                learning_rate=learning_rate,
            )
        else:
            report.add(
                "training.lr_ok",
                "Learning rate is in a common range",
                "ok",
                "The learning rate is within a common PEFT starting range.",
                learning_rate=learning_rate,
            )

    if batch_size and batch_size > 2:
        report.add(
            "training.batch_size_high",
            "Train batch size is above the usual QLoRA starting point",
            "warning",
            "Many PEFT runs start with per_device_train_batch_size=1 or 2.",
            "If you hit OOM, use batch size 1 and raise gradient_accumulation_steps.",
            batch_size=batch_size,
        )

    if grad_accum is None or grad_accum < 4:
        report.add(
            "training.grad_accum_low",
            "Gradient accumulation is low",
            "info",
            "Low gradient accumulation can make batch size 1 training noisy.",
            "Try gradient_accumulation_steps=4 or 8 when using tiny per-device batches.",
            gradient_accumulation_steps=grad_accum,
        )

    if epochs and epochs > 5:
        report.add(
            "training.epochs_high",
            "Epoch count may overfit",
            "warning",
            "Small instruction datasets often overfit when trained for many epochs.",
            "Watch eval samples and try fewer epochs, more data, dropout, or a smaller LoRA rank.",
            num_train_epochs=epochs,
        )

    if max_grad_norm is None:
        report.add(
            "training.grad_clip_missing",
            "Gradient clipping not set",
            "info",
            "max_grad_norm was not found in the training arguments.",
            "Use max_grad_norm=1.0 if you see spikes or NaN loss.",
        )

    if str(eval_strategy).lower() not in {"none", "no", ""} and eval_strategy is not None:
        report.add(
            "training.eval_enabled",
            "Evaluation is enabled",
            "info",
            "Evaluation can add memory pressure during fine-tuning.",
            "If eval causes OOM, set eval_strategy='no' while debugging or use a small eval batch.",
            eval_strategy=eval_strategy,
        )

    if bool_value(training_args, ["fp16"], False) and not bool_value(training_args, ["bf16"], False):
        report.add(
            "training.fp16_without_bf16",
            "fp16 is enabled without bf16",
            "warning",
            "fp16 can overflow more easily than bf16 on supported hardware.",
            "Try bf16=True on Ampere or newer NVIDIA GPUs.",
        )

    if (load_in_4bit or load_in_8bit) and optim:
        optim_text = str(optim).lower()
        if "paged" not in optim_text and "8bit" not in optim_text:
            report.add(
                "training.quantized_optimizer",
                "Optimizer may use more memory than needed",
                "info",
                "The setup looks quantized, but the optimizer is not a paged or 8-bit optimizer.",
                "For QLoRA, try `optim='paged_adamw_8bit'` or `optim='paged_adamw_32bit'`.",
                optim=optim,
            )
    elif (load_in_4bit or load_in_8bit) and not optim:
        report.add(
            "training.optimizer_missing",
            "Optimizer is not specified",
            "info",
            "No optimizer choice was found for a quantized training plan.",
            "For QLoRA, a common choice is `optim='paged_adamw_8bit'`.",
        )

    if warmup_steps in {None, 0} and warmup_ratio in {None, 0.0}:
        report.add(
            "training.warmup_missing",
            "Warmup is not configured",
            "info",
            "No warmup_steps or warmup_ratio value was found.",
            "Try warmup_ratio=0.03 for many SFT runs, especially when loss spikes early.",
        )

    if lr_scheduler_type is None:
        report.add(
            "training.scheduler_missing",
            "LR scheduler is not explicit",
            "info",
            "No lr_scheduler_type value was found.",
            "Common stable choices are `cosine` or `linear` with a small warmup.",
        )

    if save_steps and save_total_limit is None:
        report.add(
            "training.save_total_limit_missing",
            "Checkpoint limit is missing",
            "warning",
            "save_steps is configured, but save_total_limit was not found.",
            "Set save_total_limit=2 or 3 to avoid filling disk during long runs.",
            save_steps=save_steps,
        )

    if seed is None:
        report.add(
            "training.seed_missing",
            "Training seed is missing",
            "info",
            "No seed or data_seed value was found.",
            "Set seed=42 or another fixed value when comparing experiments.",
        )

    if dataloader_workers == 0:
        report.add(
            "training.dataloader_workers_zero",
            "Dataloader workers are zero",
            "info",
            "dataloader_num_workers=0 can slow training when tokenization or collation is heavy.",
            "Try dataloader_num_workers=2 or 4 if your platform supports it.",
        )

    if max_steps and max_steps > 0 and epochs:
        report.add(
            "training.max_steps_overrides_epochs",
            "max_steps overrides epochs",
            "info",
            "Transformers Trainer stops by max_steps when it is positive.",
            "Confirm this is intentional; num_train_epochs may not control the run length.",
            max_steps=max_steps,
            num_train_epochs=epochs,
        )


def _parameter_counts(model: Any) -> tuple[int, int]:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return 0, 0
    total = 0
    trainable = 0
    try:
        for param in parameters():
            numel = int(param.numel())
            total += numel
            if bool(getattr(param, "requires_grad", False)):
                trainable += numel
    except Exception:
        return 0, 0
    return total, trainable


def _embedding_size(model: Any) -> Optional[int]:
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(get_embeddings):
        return None
    try:
        embeddings = get_embeddings()
        return int(embeddings.weight.shape[0])
    except Exception:
        return None


def _tokenizer_size(tokenizer: Any) -> Optional[int]:
    if tokenizer is None:
        return None
    try:
        return int(len(tokenizer))
    except Exception:
        return None


def _check_model_state(
    report: DiagnosisReport,
    model: Any = None,
    tokenizer: Any = None,
    training_args: Any = None,
    peft_config: Any = None,
    sequence_length: Optional[int] = None,
) -> None:
    if model is None:
        return

    config = get_value(model, "config") or model
    if bool_value(training_args, ["gradient_checkpointing"], False) and bool_value(config, ["use_cache"], False):
        report.add(
            "model.use_cache_with_checkpointing",
            "use_cache conflicts with gradient checkpointing",
            "warning",
            "The model config has use_cache=True while gradient checkpointing is enabled.",
            "Set `model.config.use_cache = False` before training to avoid warnings or extra memory use.",
        )

    vocab_size = _tokenizer_size(tokenizer)
    embedding_size = _embedding_size(model)
    if vocab_size and embedding_size and vocab_size > embedding_size:
        report.add(
            "model.embedding_resize_needed",
            "Tokenizer is larger than model embeddings",
            "warning",
            "The tokenizer has more tokens than the model input embeddings.",
            "Call `model.resize_token_embeddings(len(tokenizer))` after adding special tokens.",
            tokenizer_size=vocab_size,
            embedding_size=embedding_size,
        )

    total_params, trainable_params = _parameter_counts(model)
    if total_params:
        ratio = trainable_params / total_params
        report.add(
            "model.trainable_params",
            "Trainable parameter ratio detected",
            "info",
            f"{trainable_params:,} of {total_params:,} parameters are trainable.",
            trainable_params=trainable_params,
            total_params=total_params,
            trainable_percent=round(ratio * 100, 4),
        )
        if peft_config is not None and trainable_params == 0:
            report.add(
                "model.no_trainable_params",
                "No trainable parameters",
                "error",
                "The model reports zero trainable parameters.",
                "After applying LoRA, call `model.print_trainable_parameters()` and verify adapters are active.",
            )
        elif peft_config is not None and ratio > 0.2:
            report.add(
                "model.too_many_trainable_params",
                "Too many parameters are trainable for PEFT",
                "warning",
                "More than 20% of parameters are trainable.",
                "Check that the base model is frozen and only adapter parameters are trainable.",
                trainable_percent=round(ratio * 100, 4),
            )

    attn_impl = first_value(config, ["attn_implementation", "_attn_implementation"], None)
    if sequence_length and sequence_length >= 2048 and attn_impl not in {"flash_attention_2", "flash_attention_3"}:
        report.add(
            "model.flash_attention_recommended",
            "Flash Attention may speed up long-context training",
            "info",
            "Sequence length is 2048 or higher and the config does not show Flash Attention.",
            "If your GPU and model support it, load with `attn_implementation='flash_attention_2'`.",
            attn_implementation=attn_impl,
        )


def _check_peft_config(report: DiagnosisReport, peft_config: Any = None) -> None:
    if peft_config is None:
        return

    r = coerce_int(get_value(peft_config, "r"), None)
    alpha = coerce_float(get_value(peft_config, "lora_alpha"), None)
    dropout = coerce_float(get_value(peft_config, "lora_dropout"), None)

    if r is not None and r < 4:
        report.add(
            "peft.rank_low",
            "LoRA rank may be too low",
            "info",
            "Very small LoRA ranks can underfit instruction data.",
            "Try r=8 or r=16 if the model is not learning.",
            r=r,
        )
    elif r is not None and r > 128:
        report.add(
            "peft.rank_high",
            "LoRA rank is high",
            "warning",
            "High LoRA rank increases memory use and overfitting risk.",
            "Try r=16, r=32, or r=64 unless you have a strong reason for a larger rank.",
            r=r,
        )

    if r and alpha:
        alpha_ratio = alpha / r
        if alpha_ratio > 4:
            report.add(
                "peft.alpha_high",
                "LoRA alpha is high relative to rank",
                "info",
                "A very high lora_alpha/r ratio can make adapter updates aggressive.",
                "A common starting point is lora_alpha around 2x the rank.",
                r=r,
                lora_alpha=alpha,
            )
        elif alpha_ratio < 1:
            report.add(
                "peft.alpha_low",
                "LoRA alpha is low relative to rank",
                "info",
                "A low lora_alpha/r ratio can make adapter updates weak.",
                "A common starting point is lora_alpha around 2x the rank.",
                r=r,
                lora_alpha=alpha,
            )

    if dropout is not None and dropout > 0.2:
        report.add(
            "peft.dropout_high",
            "LoRA dropout is high",
            "warning",
            "High LoRA dropout can slow learning or cause underfitting.",
            "Try lora_dropout between 0.0 and 0.1 for many SFT runs.",
            lora_dropout=dropout,
        )


def _check_adapter_flow(report: DiagnosisReport, peft_config: Any = None) -> None:
    if peft_config is None:
        return
    bias = get_value(peft_config, "bias")
    task_type = get_value(peft_config, "task_type")
    if bias and str(bias) != "none":
        report.add(
            "adapter.bias_trainable",
            "LoRA bias is trainable",
            "info",
            "Training LoRA bias can be valid, but it makes adapter behavior less minimal.",
            "Use bias='none' unless you have a reason to train bias terms.",
            bias=bias,
        )
    if task_type is None:
        report.add(
            "adapter.task_type_missing",
            "PEFT task type is missing",
            "warning",
            "The PEFT config does not expose task_type.",
            "For causal language model fine-tuning, set task_type='CAUSAL_LM'.",
        )

    report.add(
        "adapter.save_load_note",
        "Adapter save/load reminder",
        "info",
        "Save adapters with `model.save_pretrained(adapter_dir)` and load with `PeftModel.from_pretrained(base_model, adapter_dir)`.",
    )


def diagnose_peft(
    model: Any = None,
    tokenizer: Any = None,
    peft_config: Any = None,
    training_args: Any = None,
    train_dataset: Any = None,
    eval_dataset: Any = None,
    sequence_length: Optional[int] = None,
    model_name: Optional[str] = None,
    dataset_path: Optional[Union[str, Path]] = None,
) -> DiagnosisReport:
    """Run PEFT Doctor checks and return a report."""

    report = DiagnosisReport(
        metadata={
            "peft_doctor_version": __version__,
            "model_name": model_name,
            "sequence_length": sequence_length,
            "has_model": model is not None,
            "has_tokenizer": tokenizer is not None,
            "has_peft_config": peft_config is not None,
            "has_training_args": training_args is not None,
            "has_train_dataset": train_dataset is not None or dataset_path is not None,
            "has_eval_dataset": eval_dataset is not None,
        }
    )

    check_memory(report, model=model, training_args=training_args, peft_config=peft_config, sequence_length=sequence_length)
    _check_target_modules(report, model=model, model_name=model_name, peft_config=peft_config)
    check_tokenizer(report, tokenizer=tokenizer)
    _check_model_state(
        report,
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        peft_config=peft_config,
        sequence_length=sequence_length,
    )
    _check_peft_config(report, peft_config=peft_config)
    _check_training_args(report, training_args=training_args, peft_config=peft_config)
    check_dataset(
        report,
        train_dataset=train_dataset,
        dataset_path=dataset_path,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
    )
    _check_adapter_flow(report, peft_config=peft_config)

    return report
