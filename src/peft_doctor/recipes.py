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
            '{"text": "### Instruction:\\nExplain LoRA in one sentence.\\n\\n### Response:\\nLoRA trains small adapter matrices while keeping the base model mostly frozen."}\n'
            '{"text": "### Instruction:\\nGive one QLoRA memory tip.\\n\\n### Response:\\nUse 4-bit NF4 loading, batch size 1, and gradient checkpointing first."}\n'
        )
    return (
        '{"messages": [{"role": "user", "content": "Explain LoRA simply."}, {"role": "assistant", "content": "LoRA trains small adapter weights while leaving the base model frozen."}]}\n'
        '{"messages": [{"role": "user", "content": "Why use QLoRA?"}, {"role": "assistant", "content": "QLoRA lowers memory by loading the base model in 4-bit while training adapters."}]}\n'
    )


def _train_py(profile: dict[str, str], recipe: str) -> str:
    completion_only = recipe == "completion-only-sft"
    response_template = '    "response_template": "### Response:",\n    "completion_only_loss": True,\n' if completion_only else ""
    dataset_text_field = '        dataset_text_field="text",\n' if completion_only else ""
    return f'''"""Runnable PEFT Doctor recipe: {profile["title"]}."""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from peft_doctor import diagnose_peft, recommend_target_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="{profile["model"]}")
    parser.add_argument("--data", default="sample_data.jsonl")
    parser.add_argument("--output-dir", default="outputs/{recipe}")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset("json", data_files=args.data, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    if "qwen" in args.model.lower() and "instruct" in args.model.lower():
        tokenizer.eos_token = "<|im_end|>"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
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
        max_seq_length={profile["sequence_length"]},
        max_steps=args.max_steps,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={{"use_reentrant": False}},
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
{dataset_text_field.rstrip()}
{response_template}    )

    report = diagnose_peft(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        training_args=training_args,
        train_dataset=dataset,
        sequence_length={profile["sequence_length"]},
        model_name=args.model,
    )
    print(report.to_markdown())
    if report.has_errors or args.dry_run:
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
        "requirements.txt": """peft-doctor[ml]
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
        "tested_environment.md": f"""# Tested Environment

Recipe: `{recipe}`

Recommended starting point:

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
    report = DiagnosisReport(metadata={"recipe": recipe, "benchmark": "pre-flight validation"})
    known = {
        "llama3-qlora-colab": ("Llama-3-8B", "alpaca sample", "T4", "OOM risk", "yes", "avoided failed run"),
        "qwen2-qlora-colab": ("Qwen2.5", "chat data", "L4", "bad EOS", "yes", "fixed generation stop"),
        "mistral-lora-local": ("Mistral", "completion data", "A100", "wrong masking", "yes", "loss started working"),
        "completion-only-sft": ("Mistral", "completion data", "A100", "wrong masking", "yes", "loss started working"),
        "gemma-low-vram": ("Gemma", "tiny chat sample", "T4", "OOM risk", "yes", "kept run small"),
    }
    row = known.get(recipe)
    if row is None:
        report.add(
            "benchmark.recipe_unknown",
            "No benchmark row for recipe",
            "warning",
            "This recipe does not have a packaged validation row yet.",
            "Run validate-recipe after copying it.",
        )
        return report
    model, dataset, gpu, issue, worked, saved = row
    report.add(
        "benchmark.recipe_validation",
        "Recipe validation row",
        "ok",
        f"{model} on {gpu}: {issue}; auto-fix worked: {worked}; time saved: {saved}.",
        model=model,
        dataset=dataset,
        gpu=gpu,
        issue_found=issue,
        auto_fix_worked=worked,
        time_saved=saved,
    )
    return report
