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
    _check_training_args(report, training_args=training_args, peft_config=peft_config)
    check_dataset(report, train_dataset=train_dataset, dataset_path=dataset_path, tokenizer=tokenizer)
    _check_adapter_flow(report, peft_config=peft_config)

    return report
