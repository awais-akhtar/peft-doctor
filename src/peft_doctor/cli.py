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
from .logs import scan_training_log
from .notebooks import scan_notebook
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


@app.command()
def check(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Hugging Face model id or local model path."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Local JSON, JSONL, CSV, or TXT dataset."),
    batch_size: int = typer.Option(1, "--batch-size", help="per_device_train_batch_size to check."),
    grad_accum: int = typer.Option(8, "--grad-accum", help="gradient_accumulation_steps to check."),
    sequence_length: int = typer.Option(2048, "--sequence-length", help="Training sequence length."),
    learning_rate: float = typer.Option(2e-4, "--learning-rate", help="Learning rate to check."),
    load_in_4bit: bool = typer.Option(False, "--load-in-4bit", help="Tell the checker the model will use 4-bit loading."),
    bf16: bool = typer.Option(True, "--bf16/--no-bf16", help="Tell the checker whether bf16 is enabled."),
    fp16: bool = typer.Option(False, "--fp16/--no-fp16", help="Tell the checker whether fp16 is enabled."),
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
    training_args = create_safe_training_args(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
    )
    training_args["load_in_4bit"] = load_in_4bit
    training_args["fp16"] = fp16

    family = infer_model_family(config, model_name=model)
    peft_config = create_safe_lora_config(model=config, model_name=model, model_family=family, as_dict=True)

    report = diagnose_peft(
        model=config,
        tokenizer=tokenizer,
        peft_config=peft_config,
        training_args=training_args,
        dataset_path=dataset,
        sequence_length=sequence_length,
        model_name=model,
    )
    report.extend(metadata_issues)
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
    """Scan a training log for NaN, infinity, OOM, overflow, and unstable loss jumps."""

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

    console.print(
        """# Run this in a fresh Colab notebook cell.
%pip install -U "peft-doctor[ml]"

!peft-doctor env

# Optional for private or gated models:
# 1. Add your own Hugging Face token to Colab Secrets as HF_TOKEN.
# 2. Uncomment these lines after the secret is saved.
#
# from google.colab import userdata
# from huggingface_hub import login
# login(token=userdata.get("HF_TOKEN"))

from peft_doctor import diagnose_peft, create_safe_lora_config, create_safe_bnb_config

model_name = "meta-llama/Llama-3-8B"
peft_config = create_safe_lora_config(model_name=model_name)
bnb_config = create_safe_bnb_config()

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
    )


@app.command()
def version() -> None:
    """Print the installed version."""

    console.print(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
