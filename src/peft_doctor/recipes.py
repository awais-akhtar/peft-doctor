"""Ready-to-use fine-tuning recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .configs import create_safe_bnb_config, create_safe_lora_config, create_safe_training_args
from .report import DiagnosisReport


RECIPE_NAMES = [
    "qlora-sft",
    "low-vram-colab",
    "completion-only",
    "long-context",
    "distributed-qlora",
    "moe-lora",
    "adapter-merge",
]

PROJECT_RECIPE_NAMES = [
    "llama3-qlora-colab",
    "qwen2-qlora-colab",
    "qwen-low-vram",
    "mistral-lora-local",
    "gemma-low-vram",
    "completion-only-sft",
]

RECIPE_ALIASES = {
    "llama3_qlora_colab": "llama3-qlora-colab",
    "qwen2_qlora_colab": "qwen2-qlora-colab",
    "qwen-low-vram": "qwen2-qlora-colab",
    "mistral_lora_local": "mistral-lora-local",
    "gemma_low_vram": "gemma-low-vram",
    "completion_only_sft": "completion-only-sft",
}


def create_training_recipe(
    kind: str = "qlora-sft",
    model_family: Optional[str] = None,
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    """Return a practical PEFT fine-tuning recipe as plain Python data."""

    recipe = kind.strip().lower().replace("_", "-")
    if recipe not in RECIPE_NAMES:
        known = ", ".join(RECIPE_NAMES)
        raise ValueError(f"Unknown recipe `{kind}`. Choose one of: {known}.")

    lora = create_safe_lora_config(
        model_name=model_name,
        model_family=model_family,
        as_dict=True,
    )
    bnb = create_safe_bnb_config(as_dict=True)
    training = create_safe_training_args()
    model_kwargs: dict[str, Any] = {
        "quantization_config": "bnb_config",
        "device_map": "auto",
        "torch_dtype": "torch.bfloat16",
    }
    notes = [
        "Run `peft-doctor check` before starting the full training job.",
        "Keep the first run small, confirm loss moves, then increase sequence length or dataset size.",
    ]

    if recipe == "low-vram-colab":
        training.update(
            {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "learning_rate": 1e-4,
                "eval_strategy": "no",
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
            }
        )
        notes.extend(
            [
                "Use a GPU runtime before loading the model.",
                "Start with max_seq_length=1024 or 2048 on T4/L4 GPUs.",
                "Avoid evaluation during the first memory-debugging run.",
            ]
        )

    elif recipe == "completion-only":
        training.update(
            {
                "completion_only_loss": True,
                "response_template": "### Response:",
                "remove_unused_columns": False,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
            }
        )
        notes.extend(
            [
                "Make sure every formatted sample contains the exact response template.",
                "If every label becomes -100, the response template did not match the formatted text.",
            ]
        )

    elif recipe == "long-context":
        training.update(
            {
                "gradient_accumulation_steps": 16,
                "learning_rate": 1e-4,
                "max_seq_length": 4096,
                "group_by_length": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
            }
        )
        model_kwargs["attn_implementation"] = "flash_attention_2"
        notes.extend(
            [
                "Verify the model actually supports the chosen context length.",
                "Use Flash Attention only when your GPU, dtype, and model stack support it.",
            ]
        )

    elif recipe == "distributed-qlora":
        model_kwargs["device_map"] = None
        training.update(
            {
                "ddp_find_unused_parameters": False,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "optim": "paged_adamw_8bit",
            }
        )
        notes.extend(
            [
                "Do not combine device_map='auto' with torchrun or multi-process Accelerate.",
                "Let each process own its local GPU.",
            ]
        )

    elif recipe == "moe-lora":
        lora["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora["target_parameters"] = [
            "feed_forward.experts.gate_up_proj",
            "feed_forward.experts.down_proj",
        ]
        training.update({"gradient_checkpointing_kwargs": {"use_reentrant": False}})
        notes.extend(
            [
                "MoE models often need target_parameters for expert weights instead of only target_modules.",
                "Inspect the model parameter names and adjust target_parameters to match the architecture.",
            ]
        )

    elif recipe == "adapter-merge":
        return {
            "recipe": recipe,
            "description": "Safe LoRA adapter merge and export checklist.",
            "commands": [
                "peft-doctor adapter-check --base-model BASE --adapter ADAPTER --output-dir merged-model",
                "peft-doctor merge-adapter --base-model BASE --adapter ADAPTER --output-dir merged-model --dtype fp16",
            ],
            "checks": [
                "Confirm adapter_config.json exists.",
                "Use the same base model that the adapter was trained from.",
                "Merge from fp16, bf16, or fp32 when possible instead of k-bit weights.",
                "Save tokenizer files with the merged model.",
                "Use HF_TOKEN, huggingface-cli login, or Colab Secrets for Hub pushes.",
            ],
        }

    else:
        training.update(
            {
                "optim": "paged_adamw_8bit",
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
            }
        )
        notes.extend(
            [
                "Use target_modules='all-linear' only when you understand the memory tradeoff.",
                "For most Llama/Qwen/Mistral runs, attention plus MLP projection targets are a stable start.",
            ]
        )

    return {
        "recipe": recipe,
        "description": "Practical PEFT/QLoRA fine-tuning starter setup.",
        "install": 'python -m pip install -U "peft-doctor[ml]"',
        "model_kwargs": model_kwargs,
        "lora_config": lora,
        "bnb_config": bnb,
        "training_args": training,
        "notes": notes,
    }


def normalize_recipe_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    return RECIPE_ALIASES.get(normalized, normalized)


def _recipe_profile(name: str) -> dict[str, str]:
    recipe = normalize_recipe_name(name)
    profiles = {
        "llama3-qlora-colab": {
            "title": "Llama 3 QLoRA Colab",
            "model": "meta-llama/Meta-Llama-3-8B",
            "family": "llama",
            "sequence_length": "2048",
            "batch": "1",
            "grad_accum": "16",
            "lr": "1e-4",
            "qlora": "true",
            "notes": "Designed for Colab T4/L4 style low-memory debugging runs.",
        },
        "qwen2-qlora-colab": {
            "title": "Qwen2.5 QLoRA Colab",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "family": "qwen",
            "sequence_length": "2048",
            "batch": "1",
            "grad_accum": "16",
            "lr": "1e-4",
            "qlora": "true",
            "notes": "Includes Qwen EOS guidance for instruct chat data.",
        },
        "mistral-lora-local": {
            "title": "Mistral LoRA Local",
            "model": "mistralai/Mistral-7B-v0.1",
            "family": "mistral",
            "sequence_length": "2048",
            "batch": "1",
            "grad_accum": "8",
            "lr": "2e-4",
            "qlora": "false",
            "notes": "Local CUDA recipe with LoRA settings and no Hub token in code.",
        },
        "gemma-low-vram": {
            "title": "Gemma Low VRAM QLoRA",
            "model": "google/gemma-2-2b-it",
            "family": "gemma",
            "sequence_length": "1024",
            "batch": "1",
            "grad_accum": "16",
            "lr": "1e-4",
            "qlora": "true",
            "notes": "Small-model low-VRAM recipe for fast sanity checks.",
        },
        "completion-only-sft": {
            "title": "Completion-Only SFT",
            "model": "mistralai/Mistral-7B-v0.1",
            "family": "mistral",
            "sequence_length": "1024",
            "batch": "1",
            "grad_accum": "8",
            "lr": "1e-4",
            "qlora": "true",
            "notes": "Prompt/completion recipe with response template masking.",
        },
    }
    if recipe not in profiles:
        known = ", ".join(PROJECT_RECIPE_NAMES)
        raise ValueError(f"Unknown project recipe `{name}`. Choose one of: {known}.")
    return profiles[recipe]


def _sample_data(recipe: str) -> str:
    if recipe == "completion-only-sft":
        return (
            '{"prompt": "Explain LoRA in one sentence.", "completion": "LoRA trains small adapter matrices while keeping the base model mostly frozen."}\n'
            '{"prompt": "Give one QLoRA memory tip.", "completion": "Use 4-bit NF4 loading, batch size 1, and gradient checkpointing first."}\n'
        )
    return (
        '{"messages": [{"role": "user", "content": "Explain LoRA simply."}, {"role": "assistant", "content": "LoRA trains small adapter weights while leaving the base model frozen."}]}\n'
        '{"messages": [{"role": "user", "content": "Why use QLoRA?"}, {"role": "assistant", "content": "QLoRA lowers memory by loading the base model in 4-bit while training adapters."}]}\n'
    )


def _train_py(profile: dict[str, str], recipe: str) -> str:
    completion_only = recipe == "completion-only-sft"
    completion_setting = "        completion_only_loss=True,\n" if completion_only else ""
    use_qlora = profile["qlora"] == "true"
    bnb_import = ", prepare_model_for_kbit_training" if use_qlora else ""
    quantization_import = ", BitsAndBytesConfig" if use_qlora else ""
    quantization_block = """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
""" if use_qlora else ""
    quantization_argument = "        quantization_config=bnb_config,\n" if use_qlora else ""
    prepare_call = "    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)\n" if use_qlora else ""
    optimizer = "paged_adamw_8bit" if use_qlora else "adamw_torch"
    return f'''"""Runnable PEFT Doctor recipe: {profile["title"]}."""

from __future__ import annotations

import argparse

import torch
from datasets import Dataset
from peft import LoraConfig{bnb_import}
from transformers import AutoModelForCausalLM, AutoTokenizer{quantization_import}
from trl import SFTConfig, SFTTrainer

from peft_doctor import diagnose_peft, recommend_target_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="{profile["model"]}")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data", default="sample_data.jsonl")
    parser.add_argument("--output-dir", default="outputs/{recipe}")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = Dataset.from_json(args.data)

    dry_run_args = {{
        "per_device_train_batch_size": {profile["batch"]},
        "gradient_accumulation_steps": {profile["grad_accum"]},
        "learning_rate": {profile["lr"]},
        "max_length": {profile["sequence_length"]},
        "bf16": False,
        "fp16": False,
        "gradient_checkpointing": True,
        "load_in_4bit": {str(use_qlora)},
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "save_total_limit": 2,
    }}
    dry_run_peft = {{
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": recommend_target_modules(model_name=args.model),
    }}
    if args.dry_run:
        report = diagnose_peft(
            peft_config=dry_run_peft,
            training_args=dry_run_args,
            train_dataset=dataset,
            sequence_length={profile["sequence_length"]},
            model_name=args.model,
        )
        print(report.to_markdown())
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    if "qwen" in args.model.lower() and "instruct" in args.model.lower():
        tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
{quantization_block.rstrip()}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
{quantization_argument.rstrip()}
        torch_dtype=compute_dtype,
    )
{prepare_call.rstrip()}
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=recommend_target_modules(model_name=args.model),
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size={profile["batch"]},
        gradient_accumulation_steps={profile["grad_accum"]},
        learning_rate={profile["lr"]},
        max_length={profile["sequence_length"]},
        max_steps=args.max_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={{"use_reentrant": False}},
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        optim="{optimizer}",
{completion_setting}    )

    report = diagnose_peft(
        model=model,
        processing_class=tokenizer,
        peft_config=peft_config,
        training_args=training_args,
        train_dataset=dataset,
        sequence_length={profile["sequence_length"]},
        model_name=args.model,
    )
    print(report.to_markdown())
    if report.has_errors:
        return

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
'''


def run_project_recipe(name: str) -> None:
    """Run one packaged training recipe from its local project wrapper."""

    import argparse

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from .diagnostics import diagnose_peft
    from .targets import recommend_target_modules

    recipe = normalize_recipe_name(name)
    profile = _recipe_profile(recipe)
    use_qlora = profile["qlora"] == "true"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=profile["model"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data", default="sample_data.jsonl")
    parser.add_argument("--output-dir", default=f"outputs/{recipe}")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset = Dataset.from_json(args.data)
    target_modules = recommend_target_modules(model_name=args.model)
    dry_run_args = {
        "per_device_train_batch_size": int(profile["batch"]),
        "gradient_accumulation_steps": int(profile["grad_accum"]),
        "learning_rate": float(profile["lr"]),
        "max_length": int(profile["sequence_length"]),
        "gradient_checkpointing": True,
        "load_in_4bit": use_qlora,
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "save_total_limit": 2,
    }
    dry_run_peft = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": target_modules,
    }
    if args.dry_run:
        report = diagnose_peft(
            peft_config=dry_run_peft,
            training_args=dry_run_args,
            train_dataset=dataset,
            sequence_length=int(profile["sequence_length"]),
            model_name=args.model,
        )
        print(report.to_markdown())
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    if "qwen" in args.model.lower() and "instruct" in args.model.lower():
        tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model_kwargs: dict[str, Any] = {"torch_dtype": compute_dtype}
    if use_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        **model_kwargs,
    )
    if use_qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    sft_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": int(profile["batch"]),
        "gradient_accumulation_steps": int(profile["grad_accum"]),
        "learning_rate": float(profile["lr"]),
        "max_length": int(profile["sequence_length"]),
        "max_steps": args.max_steps,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 2,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit" if use_qlora else "adamw_torch",
    }
    if recipe == "completion-only-sft":
        sft_kwargs["completion_only_loss"] = True
    training_args = SFTConfig(**sft_kwargs)

    report = diagnose_peft(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        training_args=training_args,
        train_dataset=dataset,
        sequence_length=int(profile["sequence_length"]),
        model_name=args.model,
    )
    print(report.to_markdown())
    if report.has_errors:
        return

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def project_recipe_files(name: str) -> dict[str, str]:
    recipe = normalize_recipe_name(name)
    profile = _recipe_profile(recipe)
    return {
        "README.md": f"""# {profile["title"]}

{profile["notes"]}

## Quick Start

```bash
python -m pip install -r requirements.txt
python train.py --dry-run
python train.py --max-steps 10
```

Run PEFT Doctor before a full job:

```bash
peft-doctor fix --dry-run train.py
peft-doctor validate --model {profile["family"]} --dataset sample_data.jsonl --report report.md
```

For gated models, log in with `huggingface-cli login` or use `HF_TOKEN` from your environment. Do not paste tokens into this file.
""",
        "train.py": _train_py(profile, recipe),
        "requirements.txt": """peft-doctor[ml]>=0.9
torch
transformers
peft
trl
datasets
accelerate
safetensors
sentencepiece
""",
        "sample_data.jsonl": _sample_data(recipe),
        "expected_output.md": """# Expected Output

The dry run should print a PEFT Doctor report with no errors after dependencies and model access are ready.

Expected early signs:

- tokenizer pad token is set
- use_cache is disabled for checkpointing
- LoRA target modules are configured
- QLoRA quantized loading is detected
- training logs appear every 10 steps
""",
        "tested_environment.md": f"""# Environment Notes

Recipe: `{recipe}`

This file records the intended starting environment. It is not a claim that every model/GPU combination completed training.

Starting point:

- Python 3.10 or 3.11
- CUDA GPU runtime
- Recent `torch`, `transformers`, `peft`, `trl`, `accelerate`, and `bitsandbytes`
- Sequence length: {profile["sequence_length"]}
- Train batch size: {profile["batch"]}
- Gradient accumulation: {profile["grad_accum"]}

Use `peft-doctor env` to verify the actual runtime before training.
""",
    }


def copy_recipe_project(name: str, destination: Path, *, overwrite: bool = False) -> DiagnosisReport:
    files = project_recipe_files(name)
    report = DiagnosisReport(metadata={"recipe": normalize_recipe_name(name), "destination": str(destination)})
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        report.add(
            "recipe.copy_destination_exists",
            "Destination is not empty",
            "error",
            "The recipe destination already contains files.",
            "Pass --overwrite or choose a new directory.",
        )
        return report

    destination.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        path = destination / relative
        path.write_text(content, encoding="utf-8")
        report.add(
            "recipe.file_written",
            "Recipe file written",
            "ok",
            f"Wrote {relative}.",
            path=str(path),
        )
    return report


def validate_recipe_project(path: Path) -> DiagnosisReport:
    report = DiagnosisReport(metadata={"recipe_path": str(path)})
    required = [
        "README.md",
        "train.py",
        "requirements.txt",
        "sample_data.jsonl",
        "expected_output.md",
        "tested_environment.md",
    ]
    for filename in required:
        full_path = path / filename
        if full_path.exists():
            report.add("recipe.required_file", "Required file exists", "ok", f"{filename} exists.")
        else:
            report.add(
                "recipe.required_file_missing",
                "Required file missing",
                "error",
                f"{filename} is missing.",
                "Copy a fresh recipe or restore the missing file.",
            )

    train_py = path / "train.py"
    if train_py.exists():
        text = train_py.read_text(encoding="utf-8")
        for expected in ["diagnose_peft", "tokenizer.pad_token", "use_cache = False", "warmup_ratio", "logging_steps"]:
            if expected in text:
                report.add("recipe.train_check", "Training script check passed", "ok", f"`{expected}` found in train.py.")
            else:
                report.add(
                    "recipe.train_check_missing",
                    "Training script safety check missing",
                    "warning",
                    f"`{expected}` was not found in train.py.",
                    "Run `peft-doctor fix --dry-run train.py`.",
                )

    sample = path / "sample_data.jsonl"
    if sample.exists() and not sample.read_text(encoding="utf-8").strip():
        report.add(
            "recipe.sample_empty",
            "Sample data is empty",
            "error",
            "sample_data.jsonl is empty.",
            "Add at least one small instruction or chat sample.",
        )
    return report


def benchmark_recipe_report(name: str) -> DiagnosisReport:
    recipe = normalize_recipe_name(name)
    report = DiagnosisReport(metadata={"recipe": recipe, "benchmark": "packaged fixture validation"})
    if recipe not in PROJECT_RECIPE_NAMES:
        report.add(
            "benchmark.recipe_unknown",
            "No benchmark row for recipe",
            "warning",
            "This recipe does not have a packaged validation row yet.",
            "Run validate-recipe after copying it.",
        )
        return report
    files = project_recipe_files(recipe)
    required = {"README.md", "train.py", "requirements.txt", "sample_data.jsonl", "expected_output.md", "tested_environment.md"}
    missing = sorted(required - files.keys())
    script = files.get("train.py", "")
    profile = _recipe_profile(recipe)
    expected_markers = ["gradient_checkpointing=True", "model.config.use_cache = False", "processing_class=tokenizer"]
    if profile["qlora"] == "true":
        expected_markers.extend(["load_in_4bit=True", 'bnb_4bit_quant_type="nf4"', "prepare_model_for_kbit_training"])
    if recipe == "completion-only-sft":
        expected_markers.append("completion_only_loss=True")
    missing_markers = [marker for marker in expected_markers if marker not in script]
    report.metadata.update(
        {
            "packaged_files": sorted(files),
            "missing_files": missing,
            "missing_training_markers": missing_markers,
        }
    )
    report.add(
        "benchmark.recipe_validation",
        "Packaged recipe fixture",
        "error" if missing or missing_markers else "ok",
        f"Checked {len(files)} generated file(s) and {len(expected_markers)} required training marker(s).",
        "Restore missing files or training markers before copying this recipe." if missing or missing_markers else None,
    )
    return report
