"""Command line interface for PEFT Doctor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ._version import __version__
from .advanced import (
    advise_hyperparameters,
    ai_diagnosis_report,
    audit_policy_report,
    auto_tune_report,
    chat_answer_report,
    cloud_plan_report,
    compare_adapters_report,
    dataset_intelligence_report,
    diagnosis_text,
    estimate_cost_report,
    gpu_fingerprint_report,
    history_report,
    knowledge_base_report,
    lora_efficiency_report,
    memory_timeline,
    monitor_report,
    optimize_project_report,
    score_report,
    simulate_training,
    upgrade_suggestions_report,
    write_dataset_report_html,
)
from .adapters import diagnose_adapter_merge, merge_lora_adapter
from .configs import create_safe_bnb_config, create_safe_lora_config, create_safe_training_args
from .datasets import check_dataset
from .diagnostics import diagnose_peft
from .environment import diagnose_environment
from .estimator import estimate_vram_gb
from .explain import explanation_text, write_html_report, write_pdf_report
from .fixer import repair_config_file, repair_dataset_file, repair_python_file
from .logs import scan_training_log
from .notebooks import scan_notebook
from .profiles import list_model_profiles, profile_for
from .recipes import (
    PROJECT_RECIPE_NAMES,
    RECIPE_NAMES,
    benchmark_recipe_report,
    copy_recipe_project,
    create_training_recipe,
    normalize_recipe_name,
    validate_recipe_project,
)
from .report import DiagnosisReport, DiagnosticIssue
from .targets import infer_model_family, recommend_target_modules

app = typer.Typer(
    help="Pre-flight checks for PEFT, LoRA, and QLoRA fine-tuning.",
    no_args_is_help=True,
)
console = Console()


def _load_model_metadata(
    model_name: Optional[str],
    local_files_only: bool,
) -> tuple[object, object, list[DiagnosticIssue]]:
    if not model_name:
        return None, None, []

    issues: list[DiagnosticIssue] = []
    config = None
    tokenizer = None

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception as exc:
        issues.append(
            DiagnosticIssue(
                code="cli.model_config_unavailable",
                title="Model config could not be loaded",
                severity="info",
                message=f"Could not load the model config for `{model_name}`: {exc}",
                fix="Install transformers, check the model id, or run again without --local-files-only.",
            )
        )

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception as exc:
        issues.append(
            DiagnosticIssue(
                code="cli.tokenizer_unavailable",
                title="Tokenizer could not be loaded",
                severity="info",
                message=f"Could not load the tokenizer for `{model_name}`: {exc}",
                fix="Install transformers, check the model id, or pass a tokenizer in Python.",
            )
        )

    return config, tokenizer, issues


def _print_report(report: DiagnosisReport, output: str) -> None:
    if output == "json":
        console.print_json(report.to_json())
        return
    if output == "markdown":
        console.print(report.to_markdown())
        return

    table = Table(title="PEFT Doctor Report", show_lines=False)
    table.add_column("Severity", style="bold")
    table.add_column("Code")
    table.add_column("Finding")
    table.add_column("Fix")
    for issue in report.sorted_issues():
        style = {
            "error": "red",
            "warning": "yellow",
            "ok": "green",
            "info": "cyan",
        }.get(issue.severity, "white")
        table.add_row(
            issue.severity.upper(),
            issue.code,
            f"[{style}]{issue.title}[/{style}]\n{issue.message}",
            issue.fix or "",
        )
    console.print(table)
    summary = report.summary
    console.print(
        f"errors={summary.get('error', 0)} "
        f"warnings={summary.get('warning', 0)} "
        f"ok={summary.get('ok', 0)} "
        f"info={summary.get('info', 0)}"
    )


def _write_optional_reports(
    report: DiagnosisReport,
    html_report: Optional[Path],
    markdown_report: Optional[Path],
    pdf_report: Optional[Path] = None,
) -> None:
    if html_report is not None:
        write_html_report(report, html_report)
        console.print(f"[green]HTML report written to {html_report}[/green]")
    if markdown_report is not None:
        markdown_report.write_text(report.to_markdown(), encoding="utf-8")
        console.print(f"[green]Markdown report written to {markdown_report}[/green]")
    if pdf_report is not None:
        write_pdf_report(report, pdf_report)
        console.print(f"[green]PDF report written to {pdf_report}[/green]")


def _python_literal(value: object) -> str:
    if value in {"bnb_config", "torch.bfloat16", "torch.float16", "torch.float32", None}:
        return str(value)
    return repr(value)


@app.command()
def check(
    script: Optional[Path] = typer.Argument(None, help="Optional Python training script for explain/fix pre-checks."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Hugging Face model id or local model path."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Local JSON, JSONL, CSV, or TXT dataset."),
    eval_dataset: Optional[Path] = typer.Option(None, "--eval-dataset", help="Optional eval dataset for train/eval overlap checks."),
    batch_size: int = typer.Option(1, "--batch-size", help="per_device_train_batch_size to check."),
    eval_batch_size: Optional[int] = typer.Option(None, "--eval-batch-size", help="per_device_eval_batch_size to check."),
    grad_accum: int = typer.Option(8, "--grad-accum", help="gradient_accumulation_steps to check."),
    sequence_length: int = typer.Option(2048, "--sequence-length", help="Training sequence length."),
    learning_rate: float = typer.Option(2e-4, "--learning-rate", help="Learning rate to check."),
    optim: Optional[str] = typer.Option(None, "--optim", help="Trainer optimizer name, for example paged_adamw_8bit."),
    warmup_ratio: Optional[float] = typer.Option(None, "--warmup-ratio", help="Trainer warmup_ratio value."),
    warmup_steps: Optional[int] = typer.Option(None, "--warmup-steps", help="Trainer warmup_steps value."),
    lr_scheduler_type: Optional[str] = typer.Option(None, "--lr-scheduler-type", help="Trainer scheduler, for example cosine or linear."),
    save_steps: Optional[int] = typer.Option(500, "--save-steps", help="Trainer save_steps value."),
    save_total_limit: Optional[int] = typer.Option(None, "--save-total-limit", help="Checkpoint retention limit."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Training seed."),
    max_grad_norm: Optional[float] = typer.Option(None, "--max-grad-norm", help="Gradient clipping value."),
    dataloader_num_workers: Optional[int] = typer.Option(None, "--dataloader-num-workers", help="Trainer dataloader workers."),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="Model device_map, for example auto."),
    world_size: Optional[int] = typer.Option(None, "--world-size", help="Distributed world size."),
    local_rank: Optional[int] = typer.Option(None, "--local-rank", help="Distributed local rank."),
    fsdp: Optional[str] = typer.Option(None, "--fsdp", help="Trainer FSDP setting."),
    deepspeed: Optional[str] = typer.Option(None, "--deepspeed", help="DeepSpeed config path or setting."),
    torch_compile: bool = typer.Option(False, "--torch-compile", help="Tell the checker torch_compile is enabled."),
    packing: bool = typer.Option(False, "--packing", help="Tell the checker dataset packing is enabled."),
    group_by_length: bool = typer.Option(False, "--group-by-length", help="Tell the checker length grouping is enabled."),
    response_template: Optional[str] = typer.Option(None, "--response-template", help="Completion-only response template string."),
    completion_only_loss: bool = typer.Option(False, "--completion-only-loss", help="Tell the checker completion-only loss is enabled."),
    assistant_only_loss: bool = typer.Option(False, "--assistant-only-loss", help="Tell the checker assistant-only loss is enabled."),
    remove_unused_columns: Optional[bool] = typer.Option(None, "--remove-unused-columns/--keep-unused-columns", help="Trainer remove_unused_columns setting."),
    gradient_checkpointing_use_reentrant: Optional[bool] = typer.Option(None, "--gradient-checkpointing-use-reentrant/--gradient-checkpointing-non-reentrant", help="Gradient checkpointing use_reentrant setting."),
    load_in_4bit: bool = typer.Option(False, "--load-in-4bit", help="Tell the checker the model will use 4-bit loading."),
    load_in_8bit: bool = typer.Option(False, "--load-in-8bit", help="Tell the checker the model will use 8-bit loading."),
    bf16: bool = typer.Option(True, "--bf16/--no-bf16", help="Tell the checker whether bf16 is enabled."),
    fp16: bool = typer.Option(False, "--fp16/--no-fp16", help="Tell the checker whether fp16 is enabled."),
    attn_implementation: Optional[str] = typer.Option(None, "--attn-implementation", help="Attention implementation, such as flash_attention_2, sdpa, or eager."),
    ddp_find_unused_parameters: Optional[bool] = typer.Option(None, "--ddp-find-unused-parameters/--ddp-no-find-unused-parameters", help="Distributed DDP find_unused_parameters setting."),
    logging_steps: Optional[int] = typer.Option(None, "--logging-steps", help="Trainer logging_steps value."),
    gradient_checkpointing: bool = typer.Option(
        True,
        "--gradient-checkpointing/--no-gradient-checkpointing",
        help="Tell the checker whether gradient checkpointing is enabled.",
    ),
    local_files_only: bool = typer.Option(
        False,
        "--local-files-only",
        help="Only read model metadata already present in the local Hugging Face cache.",
    ),
    explain: bool = typer.Option(False, "--explain", help="Print risk score, reasons, and copy-paste fixes."),
    html_report: Optional[Path] = typer.Option(None, "--html-report", help="Write an HTML report to this path."),
    pdf_report: Optional[Path] = typer.Option(None, "--pdf-report", help="Write a simple PDF report to this path."),
    markdown_report: Optional[Path] = typer.Option(None, "--report", help="Write a markdown report to this path."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Run a pre-flight PEFT/LoRA/QLoRA diagnosis."""

    config, tokenizer, metadata_issues = _load_model_metadata(model, local_files_only)
    training_args = {
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "bf16": bf16,
        "gradient_checkpointing": gradient_checkpointing,
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
        "fp16": fp16,
    }
    if eval_batch_size is not None:
        training_args["per_device_eval_batch_size"] = eval_batch_size
    if optim is not None:
        training_args["optim"] = optim
    if warmup_ratio is not None:
        training_args["warmup_ratio"] = warmup_ratio
    if warmup_steps is not None:
        training_args["warmup_steps"] = warmup_steps
    if lr_scheduler_type is not None:
        training_args["lr_scheduler_type"] = lr_scheduler_type
    if save_steps is not None:
        training_args["save_steps"] = save_steps
    if save_total_limit is not None:
        training_args["save_total_limit"] = save_total_limit
    if seed is not None:
        training_args["seed"] = seed
    if max_grad_norm is not None:
        training_args["max_grad_norm"] = max_grad_norm
    if dataloader_num_workers is not None:
        training_args["dataloader_num_workers"] = dataloader_num_workers
    if device_map is not None:
        training_args["device_map"] = device_map
    if world_size is not None:
        training_args["world_size"] = world_size
    if local_rank is not None:
        training_args["local_rank"] = local_rank
    if fsdp is not None:
        training_args["fsdp"] = fsdp
    if deepspeed is not None:
        training_args["deepspeed"] = deepspeed
    if torch_compile:
        training_args["torch_compile"] = torch_compile
    if packing:
        training_args["packing"] = packing
    if group_by_length:
        training_args["group_by_length"] = group_by_length
    if response_template is not None:
        training_args["response_template"] = response_template
    if completion_only_loss:
        training_args["completion_only_loss"] = completion_only_loss
    if assistant_only_loss:
        training_args["assistant_only_loss"] = assistant_only_loss
    if remove_unused_columns is not None:
        training_args["remove_unused_columns"] = remove_unused_columns
    if gradient_checkpointing_use_reentrant is not None:
        training_args["gradient_checkpointing_kwargs"] = {
            "use_reentrant": gradient_checkpointing_use_reentrant
        }
    if attn_implementation is not None:
        training_args["attn_implementation"] = attn_implementation
    if ddp_find_unused_parameters is not None:
        training_args["ddp_find_unused_parameters"] = ddp_find_unused_parameters
    if logging_steps is not None:
        training_args["logging_steps"] = logging_steps

    family = infer_model_family(config, model_name=model)
    peft_config = create_safe_lora_config(model=config, model_name=model, model_family=family, as_dict=True)

    report = diagnose_peft(
        model=config,
        tokenizer=tokenizer,
        peft_config=peft_config,
        training_args=training_args,
        dataset_path=dataset,
        eval_dataset=eval_dataset,
        sequence_length=sequence_length,
        model_name=model,
    )
    report.extend(metadata_issues)
    if script is not None:
        script_report = repair_python_file(script, model_family=family, dry_run=True)
        report.extend(script_report.issues)
        report.metadata["script"] = str(script)
        report.metadata["script_safe_fixes"] = script_report.metadata.get("safe_fixes", 0)
    _write_optional_reports(report, html_report, markdown_report, pdf_report)
    _print_report(report, output)
    if explain:
        console.print()
        console.print(explanation_text(report), markup=False)


def _print_fix_report(report: DiagnosisReport, output: str, *, will_write: bool) -> None:
    if output in {"json", "markdown"}:
        _print_report(report, output)
        return
    found = int(report.metadata.get("issues_found", len(report.issues)))
    safe = int(report.metadata.get("safe_fixes", 0))
    console.print(f"Found {found} issue{'s' if found != 1 else ''}.")
    console.print(f"Safe auto-fixes available for {safe}.")
    if report.metadata.get("written_to"):
        console.print(f"Patched file written to {report.metadata['written_to']}.")
    elif report.metadata.get("changed") and not will_write:
        console.print("Run with --write or --output to apply.")
    _print_report(report, "table")


@app.command("fix")
def fix_command(
    target: Optional[Path] = typer.Argument(None, help="Optional Python or JSON config file to fix."),
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="Python training script to patch."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="JSONL or JSON dataset to patch."),
    config: Optional[Path] = typer.Option(None, "--config", help="JSON config file to patch."),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Write patched output to this file."),
    write: bool = typer.Option(False, "--write", help="Write changes in place when no --output is provided."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the auto-fix report without writing files."),
    family: Optional[str] = typer.Option(None, "--family", "-f", help="Model family for target module repairs."),
    pad_token_id: int = typer.Option(0, "--pad-token-id", help="Pad token id to mask in dataset labels."),
    report_format: str = typer.Option("table", "--format", help="Report format: table, json, or markdown."),
) -> None:
    """Safely patch common PEFT training, config, and dataset problems."""

    selected = input_path or config or dataset or target
    if selected is None:
        console.print("[red]Pass a Python file, --input, --config, or --dataset.[/red]")
        raise typer.Exit(2)

    is_dataset = dataset is not None
    is_config = config is not None or (selected.suffix.lower() == ".json" and input_path is None)
    is_python = input_path is not None or selected.suffix.lower() == ".py"
    will_write = (write or output_path is not None) and not dry_run

    if is_dataset:
        report = repair_dataset_file(
            selected,
            pad_token_id=pad_token_id,
            output=output_path,
            write=write,
            dry_run=dry_run,
        )
    elif is_config and not is_python:
        report = repair_config_file(
            selected,
            model_family=family,
            output=output_path,
            write=write,
            dry_run=dry_run,
        )
    else:
        report = repair_python_file(
            selected,
            model_family=family,
            output=output_path,
            write=write,
            dry_run=dry_run,
        )
    _print_fix_report(report, report_format, will_write=will_write)


@app.command("estimate")
def estimate_command(
    model: str = typer.Option(..., "--model", "-m", help="Model name or family, for example llama-3-8b."),
    seq_len: int = typer.Option(2048, "--seq-len", help="Training sequence length."),
    batch_size: int = typer.Option(1, "--batch-size", help="Per-device train batch size."),
    qlora: bool = typer.Option(False, "--qlora", help="Estimate a 4-bit QLoRA run."),
    no_lora: bool = typer.Option(False, "--no-lora", help="Estimate full fine-tuning instead of LoRA."),
    no_gradient_checkpointing: bool = typer.Option(False, "--no-gradient-checkpointing", help="Estimate without gradient checkpointing."),
    target_vram: Optional[float] = typer.Option(None, "--target-vram", help="Target GPU VRAM in GB."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Estimate VRAM before starting a PEFT run."""

    _print_report(
        estimate_vram_gb(
            model,
            seq_len=seq_len,
            batch_size=batch_size,
            qlora=qlora,
            lora=not no_lora,
            gradient_checkpointing=not no_gradient_checkpointing,
            target_vram_gb=target_vram,
        ),
        output,
    )


@app.command("init")
def init_command(
    output_dir: Path = typer.Option(Path("peft-doctor-run"), "--output-dir", "-o", help="Project directory to create."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model or family. Prompted when omitted."),
    gpu: Optional[str] = typer.Option(None, "--gpu", help="GPU name, for example T4, L4, A100, or local."),
    dataset_type: Optional[str] = typer.Option(None, "--dataset-type", help="chat, completion, instruction, or text."),
    target_vram: Optional[float] = typer.Option(None, "--target-vram", help="Target VRAM in GB."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow writing into a non-empty project directory."),
) -> None:
    """Interactive wizard that generates a full training project."""

    model_value = model or typer.prompt("Model or family", default="llama3")
    gpu_value = gpu or typer.prompt("GPU", default="T4")
    dataset_value = dataset_type or typer.prompt("Dataset type", default="chat")
    vram_value = target_vram if target_vram is not None else float(typer.prompt("Target VRAM GB", default="16"))

    lowered = f"{model_value} {gpu_value} {dataset_value}".lower()
    if "completion" in lowered:
        recipe = "completion-only-sft"
    elif "qwen" in lowered:
        recipe = "qwen2-qlora-colab"
    elif "mistral" in lowered and vram_value >= 24:
        recipe = "mistral-lora-local"
    elif "gemma" in lowered or vram_value <= 12:
        recipe = "gemma-low-vram"
    else:
        recipe = "llama3-qlora-colab"

    report = copy_recipe_project(recipe, output_dir, overwrite=overwrite)
    report.metadata.update(
        {
            "wizard_model": model_value,
            "wizard_gpu": gpu_value,
            "wizard_dataset_type": dataset_value,
            "wizard_target_vram": vram_value,
            "selected_recipe": normalize_recipe_name(recipe),
        }
    )
    _print_report(report, "table")
    if report.has_errors:
        raise typer.Exit(2)
    console.print(f"[green]Created project from {recipe}. Next: cd {output_dir} && python train.py --dry-run[/green]")


@app.command("diagnose")
def diagnose_command(
    script: Optional[Path] = typer.Argument(None, help="Optional training script, usually train.py."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Dataset path to include in diagnosis."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model id or family, for example llama-3-8b."),
    gpu: Optional[str] = typer.Option(None, "--gpu", help="GPU name, for example RTX 4090, T4, L4, or A100."),
    sequence_length: int = typer.Option(2048, "--sequence-length", help="Training sequence length."),
    batch_size: int = typer.Option(1, "--batch-size", help="Per-device train batch size."),
    qlora: bool = typer.Option(True, "--qlora/--no-qlora", help="Whether the run uses QLoRA."),
    output: str = typer.Option("diagnosis", "--output", "-o", help="diagnosis, table, json, or markdown."),
) -> None:
    """Explain why a PEFT run is likely to fail and what to fix first."""

    report = ai_diagnosis_report(
        script,
        dataset=dataset,
        model=model,
        gpu=gpu,
        sequence_length=sequence_length,
        batch_size=batch_size,
        qlora=qlora,
    )
    if output == "diagnosis":
        console.print(diagnosis_text(report), markup=False)
    else:
        _print_report(report, output)


@app.command("simulate")
def simulate_command(
    model: str = typer.Option("llama-3-8b", "--model", "-m", help="Model id or family."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Dataset path for row-count planning."),
    gpu: Optional[str] = typer.Option(None, "--gpu", help="GPU name, for example T4, L4, A100, RTX 4090."),
    seq_len: int = typer.Option(2048, "--seq-len", help="Training sequence length."),
    batch_size: int = typer.Option(1, "--batch-size", help="Per-device train batch size."),
    qlora: bool = typer.Option(True, "--qlora/--no-qlora", help="Whether the run uses QLoRA."),
    eval_batch_size: int = typer.Option(1, "--eval-batch-size", help="Evaluation batch size."),
    save_steps: int = typer.Option(500, "--save-steps", help="Checkpoint save interval."),
    total_steps: int = typer.Option(1000, "--total-steps", help="Planned training steps."),
    disk_free_gb: Optional[float] = typer.Option(None, "--disk-free-gb", help="Free disk estimate for checkpoint planning."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Dry-run a training plan without loading or training a model."""

    report = simulate_training(
        model=model,
        dataset=dataset,
        gpu=gpu,
        seq_len=seq_len,
        batch_size=batch_size,
        qlora=qlora,
        eval_batch_size=eval_batch_size,
        save_steps=save_steps,
        total_steps=total_steps,
        disk_free_gb=disk_free_gb,
    )
    if output == "table":
        for step in report.metadata.get("steps", []):
            console.print(f"{step}... simulated")
    _print_report(report, output)


@app.command("memory-timeline")
def memory_timeline_command(
    model: str = typer.Option("llama-3-8b", "--model", "-m", help="Model id or family."),
    seq_len: int = typer.Option(2048, "--seq-len", help="Training sequence length."),
    batch_size: int = typer.Option(1, "--batch-size", help="Per-device train batch size."),
    qlora: bool = typer.Option(True, "--qlora/--no-qlora", help="Whether the run uses QLoRA."),
    gradient_checkpointing: bool = typer.Option(True, "--gradient-checkpointing/--no-gradient-checkpointing", help="Whether checkpointing is enabled."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Show where memory spikes happen during a training step."""

    _print_report(
        memory_timeline(
            model,
            seq_len=seq_len,
            batch_size=batch_size,
            qlora=qlora,
            gradient_checkpointing=gradient_checkpointing,
        ),
        output,
    )


@app.command("estimate-cost")
def estimate_cost_command(
    model: str = typer.Option("llama-3-8b", "--model", "-m", help="Model id or family."),
    dataset_size: int = typer.Option(8000, "--dataset-size", help="Number of training examples."),
    seq_len: int = typer.Option(2048, "--seq-len", help="Training sequence length."),
    batch_size: int = typer.Option(1, "--batch-size", help="Per-device train batch size."),
    qlora: bool = typer.Option(True, "--qlora/--no-qlora", help="Whether the run uses QLoRA."),
    gpu: Optional[list[str]] = typer.Option(None, "--gpu", help="GPU option to compare. Repeat for multiple GPUs."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Estimate cloud GPU time and cost for a fine-tuning plan."""

    _print_report(
        estimate_cost_report(
            model=model,
            dataset_size=dataset_size,
            seq_len=seq_len,
            batch_size=batch_size,
            qlora=qlora,
            gpus=gpu,
        ),
        output,
    )


@app.command("advise-hparams")
def advise_hparams_command(
    model: str = typer.Option("llama-3-8b", "--model", "-m", help="Model id or family."),
    dataset_size: int = typer.Option(..., "--dataset-size", help="Number of training examples."),
    gpu_vram: Optional[float] = typer.Option(None, "--gpu-vram", help="GPU VRAM in GB."),
    task: str = typer.Option("chat", "--task", help="Task type: chat, completion, instruction, or tool."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Recommend LoRA rank, alpha, and dropout for the run."""

    _print_report(
        advise_hyperparameters(model=model, dataset_size=dataset_size, gpu_vram_gb=gpu_vram, task=task),
        output,
    )


@app.command("monitor")
def monitor_command(
    log_file: Optional[Path] = typer.Argument(None, help="Optional Trainer log file."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Analyze live or saved training logs for health and NaN risk."""

    _print_report(monitor_report(log_file), output)


@app.command("auto-tune")
def auto_tune_command(
    model: str = typer.Option("llama-3-8b", "--model", "-m", help="Model id or family."),
    batch_size: int = typer.Option(4, "--batch-size", help="Current per-device train batch size."),
    grad_accum: int = typer.Option(1, "--grad-accum", help="Current gradient accumulation steps."),
    seq_len: int = typer.Option(2048, "--seq-len", help="Current sequence length."),
    target_vram: float = typer.Option(16.0, "--target-vram", help="Target GPU VRAM in GB."),
    qlora: bool = typer.Option(True, "--qlora/--no-qlora", help="Whether the run uses QLoRA."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Suggest a lower-memory setup while preserving effective batch size."""

    _print_report(
        auto_tune_report(
            model=model,
            batch_size=batch_size,
            grad_accum=grad_accum,
            seq_len=seq_len,
            target_vram_gb=target_vram,
            qlora=qlora,
        ),
        output,
    )


@app.command("score")
def score_command(
    script: Optional[Path] = typer.Argument(None, help="Optional training script."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model id or family."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Dataset path."),
    gpu: Optional[str] = typer.Option(None, "--gpu", help="GPU name."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Score dataset, configuration, hardware, and trainer readiness."""

    _print_report(score_report(model=model, dataset=dataset, gpu=gpu, script=script), output)


@app.command("dataset-intel")
def dataset_intel_command(
    dataset: Path = typer.Argument(..., help="Dataset path."),
    limit: int = typer.Option(1000, "--limit", help="Maximum rows to scan."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Find duplicates, empty answers, prompt injections, outliers, and quality score."""

    _print_report(dataset_intelligence_report(dataset, limit=limit), output)


@app.command("dataset-report")
def dataset_report_command(
    dataset: Path = typer.Argument(..., help="Dataset path."),
    output_file: Path = typer.Option(Path("dataset-report.html"), "--output", "-o", help="HTML report path."),
    limit: int = typer.Option(1000, "--limit", help="Maximum rows to scan."),
) -> None:
    """Generate a static HTML dataset visualizer report."""

    write_dataset_report_html(dataset, output_file, limit=limit)
    console.print(f"[green]Dataset report written to {output_file}[/green]")


@app.command("lora-efficiency")
def lora_efficiency_command(
    model: str = typer.Option("llama-3-8b", "--model", "-m", help="Model id or family."),
    rank: int = typer.Option(16, "--rank", "-r", help="LoRA rank."),
    dataset_size: int = typer.Option(8000, "--dataset-size", help="Number of training examples."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Predict adapter size, expected gain, slowdown, and merge compatibility."""

    _print_report(lora_efficiency_report(model=model, rank=rank, dataset_size=dataset_size), output)


@app.command("compare-adapters")
def compare_adapters_command(
    adapter_a: Path = typer.Argument(..., help="First adapter directory."),
    adapter_b: Path = typer.Argument(..., help="Second adapter directory."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Compare two local LoRA adapter directories."""

    _print_report(compare_adapters_report(adapter_a, adapter_b), output)


@app.command("upgrade-suggestions")
def upgrade_suggestions_command(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Check package versions and suggest safe upgrades."""

    _print_report(upgrade_suggestions_report(), output)


@app.command("gpu-fingerprint")
def gpu_fingerprint_command(
    gpu: Optional[str] = typer.Argument(None, help="GPU name. If omitted, PEFT Doctor checks local CUDA."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Identify GPU-specific fine-tuning risks and precision advice."""

    _print_report(gpu_fingerprint_report(gpu), output)


@app.command("history")
def history_command(
    root: Path = typer.Argument(Path("."), help="Project root."),
    add_status: Optional[str] = typer.Option(None, "--add-status", help="Record a run status such as OOM, NaN, completed, or best."),
    metric: Optional[str] = typer.Option(None, "--metric", help="Optional metric text, for example 'BLEU +3.1'."),
    note: Optional[str] = typer.Option(None, "--note", help="Optional note for the run history."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Show or append lightweight PEFT experiment history."""

    _print_report(history_report(root, add_status=add_status, metric=metric, note=note), output)


@app.command("knowledge-base")
def knowledge_base_command(
    query: str = typer.Argument(..., help="Issue to search, for example 'CUDA illegal memory access'."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Search the built-in offline PEFT failure knowledge base."""

    _print_report(knowledge_base_report(query), output)


@app.command("chat")
def chat_command(
    question: Optional[str] = typer.Argument(None, help="Question, for example 'Why is my loss exploding?'"),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Dataset path to inspect."),
    log_file: Optional[Path] = typer.Option(None, "--log", help="Trainer log file to inspect."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Ask a local PEFT Doctor question without sending data to a remote service."""

    question_text = question or typer.prompt("Question")
    _print_report(chat_answer_report(question_text, dataset=dataset, log_file=log_file), output)


@app.command("optimize")
def optimize_command(
    project: Path = typer.Argument(Path("."), help="Project directory."),
    write: bool = typer.Option(False, "--write", help="Apply safe file fixes. Default is dry-run."),
    html_report: Optional[Path] = typer.Option(None, "--html-report", help="Write an HTML optimization report."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Run one local project optimizer pass over config, dataset, trainer, and score."""

    _print_report(optimize_project_report(project, write=write, html_report=html_report), output)


@app.command("audit")
def audit_command(
    project: Path = typer.Argument(Path("."), help="Project directory."),
    policy: Path = typer.Option(..., "--policy", "-p", help="Policy file in simple YAML or JSON."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Audit a project against team PEFT policies."""

    _print_report(audit_policy_report(project, policy), output)


@app.command("cloud")
def cloud_command(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Show the privacy-first PEFT Doctor Cloud roadmap."""

    _print_report(cloud_plan_report(), output)


def _print_recipe(recipe_data: dict[str, object], output: str) -> None:
    if output == "json":
        console.print_json(json.dumps(recipe_data, indent=2))
        return
    if output == "markdown":
        console.print(f"# {recipe_data['recipe']}")
        console.print()
        console.print(str(recipe_data.get("description", "")))
        console.print()
        for key in ["install", "commands", "checks", "notes"]:
            value = recipe_data.get(key)
            if not value:
                continue
            console.print(f"## {key.replace('_', ' ').title()}")
            if isinstance(value, list):
                for item in value:
                    console.print(f"- `{item}`" if key == "commands" else f"- {item}", markup=False)
            else:
                console.print(f"`{value}`", markup=False)
            console.print()
        return

    console.print("# Generated by peft-doctor recipe")
    console.print(f"# Recipe: {recipe_data['recipe']}")
    if "install" in recipe_data:
        console.print(f"# Install: {recipe_data['install']}", markup=False)
    if "commands" in recipe_data:
        for command in recipe_data["commands"]:  # type: ignore[index]
            console.print(f"# {command}")
        return

    console.print("from peft import LoraConfig")
    console.print("from transformers import BitsAndBytesConfig")
    console.print("import torch")
    console.print()
    console.print("model_kwargs = {")
    for key, value in dict(recipe_data["model_kwargs"]).items():  # type: ignore[arg-type]
        console.print(f"    {key!r}: {_python_literal(value)},")
    console.print("}")
    console.print()
    console.print("peft_config = LoraConfig(")
    for key, value in dict(recipe_data["lora_config"]).items():  # type: ignore[arg-type]
        console.print(f"    {key}={value!r},")
    console.print(")")
    console.print()
    console.print("bnb_config = BitsAndBytesConfig(")
    bnb_config = dict(recipe_data["bnb_config"])  # type: ignore[arg-type]
    for key, value in bnb_config.items():
        if key == "bnb_4bit_compute_dtype":
            console.print("    bnb_4bit_compute_dtype=torch.bfloat16,")
        else:
            console.print(f"    {key}={value!r},")
    console.print(")")
    console.print()
    console.print("training_args = {")
    for key, value in dict(recipe_data["training_args"]).items():  # type: ignore[arg-type]
        console.print(f"    {key!r}: {value!r},")
    console.print("}")


@app.command("recipe")
def recipe_command(
    recipe_name: Optional[str] = typer.Argument(None, help="Recipe name, for example llama3-qlora-colab."),
    kind: Optional[str] = typer.Option(None, "--kind", "-k", help=f"Starter recipe name: {', '.join(RECIPE_NAMES)}."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model id used to infer the family."),
    family: Optional[str] = typer.Option(None, "--family", "-f", help="Known family, for example llama, qwen, or mistral."),
    copy_to: Optional[Path] = typer.Option(None, "--copy", help="Copy a runnable recipe project to this directory."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow copying into a non-empty directory."),
    output: str = typer.Option("python", "--output", "-o", help="Output format: python, json, or markdown."),
) -> None:
    """Generate a practical PEFT/QLoRA fine-tuning recipe."""

    selected = recipe_name or kind or "qlora-sft"
    if copy_to is not None:
        report = copy_recipe_project(selected, copy_to, overwrite=overwrite)
        _print_report(report, "table" if output == "python" else output)
        if report.has_errors:
            raise typer.Exit(2)
        return

    try:
        recipe_data = create_training_recipe(kind=selected, model_name=model, model_family=family)
    except ValueError as exc:
        if selected in PROJECT_RECIPE_NAMES:
            console.print("[yellow]This is a project recipe. Use --copy ./my-run to create files.[/yellow]")
        raise typer.BadParameter(str(exc)) from exc
    _print_recipe(recipe_data, output)


@app.command("validate-recipe")
def validate_recipe_command(
    path: Path = typer.Argument(..., help="Copied recipe project directory."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Validate a copied recipe project."""

    _print_report(validate_recipe_project(path), output)


@app.command("benchmark")
def benchmark_command(
    recipe: str = typer.Option(..., "--recipe", "-r", help="Recipe name, for example llama3-qlora-colab."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Show packaged pre-flight benchmark evidence for a recipe."""

    _print_report(benchmark_recipe_report(recipe), output)


@app.command("validate")
def validate_command(
    model: str = typer.Option(..., "--model", "-m", help="Model family or model id, for example qwen."),
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Dataset path to validate."),
    report_path: Optional[Path] = typer.Option(None, "--report", help="Write a markdown report to this path."),
    sequence_length: int = typer.Option(2048, "--sequence-length", help="Sequence length to validate."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Run a lightweight validation report without loading the full model."""

    family = infer_model_family(model_name=model) or model
    peft_config = create_safe_lora_config(model_family=family, model_name=model, as_dict=True)
    training_args = create_safe_training_args()
    training_args.update({"load_in_4bit": True, "gradient_checkpointing_kwargs": {"use_reentrant": False}})
    report = diagnose_peft(
        peft_config=peft_config,
        training_args=training_args,
        dataset_path=dataset,
        sequence_length=sequence_length,
        model_name=model,
    )
    if report_path is not None:
        report_path.write_text(report.to_markdown(), encoding="utf-8")
        report.metadata["written_to"] = str(report_path)
    _print_report(report, output)


@app.command("dataset-doctor")
def dataset_doctor_command(
    dataset: Path = typer.Argument(..., help="Local JSON, JSONL, CSV, or TXT dataset."),
    sequence_length: Optional[int] = typer.Option(None, "--sequence-length", help="Sequence length for long-row checks."),
    response_template: Optional[str] = typer.Option(None, "--response-template", help="Completion-only response template."),
    packing: bool = typer.Option(False, "--packing", help="Check packed dataset EOS boundaries."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Dataset doctor for bad rows, duplicates, roles, and too-long examples."""

    report = DiagnosisReport(metadata={"dataset": str(dataset), "doctor": "dataset"})
    check_dataset(
        report,
        dataset_path=dataset,
        sequence_length=sequence_length,
        training_args={"response_template": response_template, "packing": packing},
    )
    _print_report(report, output)


@app.command("inspect-adapter")
def inspect_adapter_command(
    adapter: str = typer.Argument(..., help="PEFT adapter path or Hugging Face Hub id."),
    base_model: Optional[str] = typer.Option(None, "--base-model", "-m", help="Expected original base model id."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Adapter doctor for saved LoRA adapters before upload or merge."""

    _print_report(diagnose_adapter_merge(base_model=base_model, adapter=adapter, merge_plan=False), output)


@app.command("analyze-log")
def analyze_log_command(
    log_file: Path = typer.Argument(..., help="Trainer log file, JSONL log, or text log."),
    explain: bool = typer.Option(True, "--explain/--no-explain", help="Print risk and fix explanations."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Analyze a training log and explain the failure."""

    report = DiagnosisReport(metadata={"log_file": str(log_file), "doctor": "log"})
    report.extend(scan_training_log(log_file))
    _print_report(report, output)
    if explain:
        console.print()
        console.print(explanation_text(report), markup=False)


@app.command("profiles")
def profiles_command(
    family: Optional[str] = typer.Argument(None, help="Optional family: llama, qwen, mistral, gemma, phi, falcon."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Show built-in model-family profiles."""

    report = DiagnosisReport(metadata={"profiles": "model families"})
    profiles = [profile_for(family)] if family else list_model_profiles()
    for profile in profiles:
        if profile is None:
            report.add(
                "profiles.unknown",
                "Model profile not found",
                "warning",
                f"No profile matched `{family}`.",
                "Use one of: llama, qwen, mistral, gemma, phi, falcon.",
            )
            continue
        report.add(
            "profiles.family",
            f"{profile.name} profile",
            "ok",
            profile.tokenizer_notes,
            "Use the listed target modules as a practical LoRA starting point.",
            target_modules=", ".join(profile.target_modules),
            default_seq_len=profile.default_seq_len,
            common_risks=", ".join(profile.common_risks),
        )
    _print_report(report, output)


@app.command()
def targets(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model id used to infer the family."),
    family: Optional[str] = typer.Option(None, "--family", "-f", help="Known family, for example llama, qwen, gpt2."),
    no_mlp: bool = typer.Option(False, "--no-mlp", help="Return attention targets only."),
    output: str = typer.Option("text", "--output", "-o", help="Output format: text or json."),
) -> None:
    """Recommend LoRA target modules."""

    modules = recommend_target_modules(model_name=model, model_family=family, include_mlp=not no_mlp)
    if output == "json":
        console.print_json(json.dumps({"target_modules": modules}, indent=2))
        return
    console.print("target_modules = [")
    for module in modules:
        console.print(f'    "{module}",')
    console.print("]")


@app.command("safe-config")
def safe_config(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model id used to infer target modules."),
    family: Optional[str] = typer.Option(None, "--family", "-f", help="Known family, for example llama, qwen, gpt2."),
    qlora: bool = typer.Option(True, "--qlora/--no-qlora", help="Include a QLoRA BitsAndBytesConfig block."),
    output: str = typer.Option("python", "--output", "-o", help="Output format: python or json."),
) -> None:
    """Print a safe LoRA or QLoRA starter config."""

    lora = create_safe_lora_config(model_name=model, model_family=family, as_dict=True)
    bnb = create_safe_bnb_config(as_dict=True) if qlora else None
    if output == "json":
        console.print_json(json.dumps({"lora_config": lora, "bnb_config": bnb}, indent=2))
        return

    console.print("from peft import LoraConfig")
    if qlora:
        console.print("from transformers import BitsAndBytesConfig")
        console.print("import torch")
        console.print()
    console.print("peft_config = LoraConfig(")
    for key, value in lora.items():
        console.print(f"    {key}={value!r},")
    console.print(")")

    if qlora and bnb:
        console.print()
        console.print("bnb_config = BitsAndBytesConfig(")
        console.print("    load_in_4bit=True,")
        console.print(f"    bnb_4bit_quant_type={bnb['bnb_4bit_quant_type']!r},")
        console.print("    bnb_4bit_compute_dtype=torch.bfloat16,")
        console.print(f"    bnb_4bit_use_double_quant={bnb['bnb_4bit_use_double_quant']!r},")
        console.print(")")


@app.command("inspect-dataset")
def inspect_dataset(
    dataset: Path = typer.Argument(..., help="Local JSON, JSONL, CSV, or TXT dataset."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Inspect dataset rows for prompt-format problems."""

    report = DiagnosisReport(metadata={"dataset": str(dataset)})
    check_dataset(report, dataset_path=dataset)
    _print_report(report, output)


@app.command("scan-log")
def scan_log(
    log_file: Path = typer.Argument(..., help="Trainer log file, JSONL log, or text log."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Scan a training log for loss, runtime, device, disk, and shape failures."""

    report = DiagnosisReport(metadata={"log_file": str(log_file)})
    report.extend(scan_training_log(log_file))
    _print_report(report, output)


@app.command("scan-notebook")
def scan_notebook_command(
    notebook: Path = typer.Argument(..., help="Notebook to scan."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Scan a notebook for PEFT, Colab, and token-handling mistakes."""

    _print_report(scan_notebook(notebook), output)


@app.command("notebook-check")
def notebook_check_command(
    notebook: Path = typer.Argument(..., help="Notebook to scan."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Scan a Colab or Jupyter notebook for PEFT mistakes."""

    _print_report(scan_notebook(notebook), output)


@app.command("adapter-check")
def adapter_check(
    adapter: str = typer.Option(..., "--adapter", "-a", help="PEFT adapter path or Hugging Face Hub id."),
    base_model: Optional[str] = typer.Option(None, "--base-model", "-m", help="Original base model id."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Local directory planned for the merged model."),
    push_to_hub: bool = typer.Option(False, "--push-to-hub", help="Check a Hub push plan."),
    hub_model_id: Optional[str] = typer.Option(None, "--hub-model-id", help="Hub repo id for the merged model."),
    load_in_4bit: bool = typer.Option(False, "--load-in-4bit", help="Check a 4-bit merge plan."),
    load_in_8bit: bool = typer.Option(False, "--load-in-8bit", help="Check an 8-bit merge plan."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow writing into a non-empty output directory."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Check a LoRA adapter merge/export plan without loading the model."""

    report = diagnose_adapter_merge(
        base_model=base_model,
        adapter=adapter,
        output_dir=output_dir,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        overwrite=overwrite,
    )
    _print_report(report, output)


@app.command("merge-adapter")
def merge_adapter(
    adapter: str = typer.Option(..., "--adapter", "-a", help="PEFT adapter path or Hugging Face Hub id."),
    base_model: Optional[str] = typer.Option(None, "--base-model", "-m", help="Original base model id."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Local directory for the merged model."),
    tokenizer_source: Optional[str] = typer.Option(None, "--tokenizer-source", help="Tokenizer id/path. Defaults to adapter tokenizer if present, otherwise base model."),
    dtype: str = typer.Option("auto", "--dtype", help="auto, bf16, fp16, or fp32."),
    device_map: Optional[str] = typer.Option("auto", "--device-map", help="Transformers device_map value."),
    offload_folder: Optional[str] = typer.Option("offload", "--offload-folder", help="Folder used by accelerate offloading."),
    max_shard_size: str = typer.Option("5GB", "--max-shard-size", help="Shard size passed to save_pretrained."),
    safe_serialization: bool = typer.Option(True, "--safe-serialization/--no-safe-serialization", help="Save safetensors when supported."),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code", help="Allow custom model code from the Hub."),
    push_to_hub: bool = typer.Option(False, "--push-to-hub", help="Push the merged model and tokenizer to the Hub."),
    hub_model_id: Optional[str] = typer.Option(None, "--hub-model-id", help="Target Hub repo id, for example username/model-name."),
    private: bool = typer.Option(False, "--private", help="Create or push to a private Hub repo."),
    commit_message: Optional[str] = typer.Option(None, "--commit-message", help="Hub commit message."),
    load_in_4bit: bool = typer.Option(False, "--load-in-4bit", help="Load base model in 4-bit before merging."),
    load_in_8bit: bool = typer.Option(False, "--load-in-8bit", help="Load base model in 8-bit before merging."),
    allow_quantized_merge: bool = typer.Option(False, "--allow-quantized-merge", help="Allow a quantized merge attempt."),
    no_tokenizer: bool = typer.Option(False, "--no-tokenizer", help="Do not save or push tokenizer files."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow writing into a non-empty output directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only check the merge plan."),
    report_output: str = typer.Option("table", "--report", help="Dry-run report format: table, json, or markdown."),
) -> None:
    """Merge a LoRA adapter into its base model, then save or push the result."""

    plan = diagnose_adapter_merge(
        base_model=base_model,
        adapter=adapter,
        output_dir=output_dir,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        overwrite=overwrite,
    )
    if dry_run or plan.has_errors:
        _print_report(plan, report_output)
        if plan.has_errors:
            raise typer.Exit(2)
        return

    if output_dir and output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        console.print("[red]Output directory is not empty. Pass --overwrite to continue.[/red]")
        raise typer.Exit(2)

    adapter_config_base = None
    if not base_model:
        # Reuse the checker's metadata path without importing heavy ML packages.
        from .adapters import _read_adapter_config

        adapter_config = _read_adapter_config(adapter)
        adapter_config_base = adapter_config.get("base_model_name_or_path") if adapter_config else None

    resolved_base_model = base_model or adapter_config_base
    if not resolved_base_model:
        console.print("[red]Base model could not be inferred. Pass --base-model.[/red]")
        raise typer.Exit(2)

    console.print("[cyan]Loading base model and adapter. This can take a while for 7B+ models.[/cyan]")
    try:
        result = merge_lora_adapter(
            base_model=resolved_base_model,
            adapter=adapter,
            output_dir=output_dir,
            tokenizer_source=tokenizer_source,
            torch_dtype=dtype,
            device_map=device_map,
            offload_folder=offload_folder,
            trust_remote_code=trust_remote_code,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
            save_tokenizer=not no_tokenizer,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            private=private,
            commit_message=commit_message,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            allow_quantized_merge=allow_quantized_merge,
        )
    except Exception as exc:
        console.print(f"[red]Merge failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print("[green]Adapter merge finished.[/green]")
    console.print_json(json.dumps(result.to_dict(), indent=2))


@app.command("env")
def env_command(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json, or markdown."),
) -> None:
    """Check Python, CUDA, and common fine-tuning packages."""

    _print_report(diagnose_environment(), output)


@app.command()
def colab() -> None:
    """Print a Google Colab setup cell."""

    snippet = """# Run this in a fresh Colab notebook cell.
%pip install -U "peft-doctor[ml]"

!peft-doctor env

# Optional for private or gated models:
# 1. Add your own Hugging Face token to Colab Secrets as HF_TOKEN.
# 2. Uncomment these lines after the secret is saved.
#
# from google.colab import userdata
# from huggingface_hub import login
# login(token=userdata.get("HF_TOKEN"))

from peft_doctor import (
    diagnose_peft,
    create_safe_lora_config,
    create_safe_bnb_config,
    create_training_recipe,
)

model_name = "meta-llama/Llama-3-8B"
peft_config = create_safe_lora_config(model_name=model_name)
bnb_config = create_safe_bnb_config()
recipe = create_training_recipe(kind="low-vram-colab", model_name=model_name)

# After you create model, tokenizer, train_dataset, and training_args:
# report = diagnose_peft(
#     model=model,
#     tokenizer=tokenizer,
#     peft_config=peft_config,
#     training_args=training_args,
#     train_dataset=train_dataset,
#     sequence_length=2048,
#     model_name=model_name,
# )
# print(report.to_markdown())
"""
    console.print(snippet, markup=False, soft_wrap=True)


@app.command()
def version() -> None:
    """Print the installed version."""

    console.print(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
