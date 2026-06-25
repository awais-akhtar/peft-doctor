"""Command line interface for PEFT Doctor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ._version import __version__
from .adapters import diagnose_adapter_merge, merge_lora_adapter
from .configs import create_safe_bnb_config, create_safe_lora_config, create_safe_training_args
from .datasets import check_dataset
from .diagnostics import diagnose_peft
from .environment import diagnose_environment
from .fixer import repair_config_file, repair_dataset_file, repair_python_file
from .logs import scan_training_log
from .notebooks import scan_notebook
from .recipes import (
    PROJECT_RECIPE_NAMES,
    RECIPE_NAMES,
    benchmark_recipe_report,
    copy_recipe_project,
    create_training_recipe,
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


def _python_literal(value: object) -> str:
    if value in {"bnb_config", "torch.bfloat16", "torch.float16", "torch.float32", None}:
        return str(value)
    return repr(value)


@app.command()
def check(
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
    _print_report(report, output)


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
