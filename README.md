# PEFT Doctor: LoRA and QLoRA Fine-Tuning Debugger

PEFT Doctor is a pre-flight checker and troubleshooting toolkit for PEFT, LoRA, and QLoRA fine-tuning. It catches the problems that usually waste a training run: CUDA out of memory, NaN loss, risky learning rates, missing tokenizer padding, wrong LoRA target modules, broken prompt formats, bitsandbytes setup issues, and adapter save/load or merge failures.

It is built for the way people actually fine-tune models today: Hugging Face Transformers, PEFT, TRL, bitsandbytes, Google Colab, local CUDA machines, and common Llama, Mistral, Qwen, Gemma, Phi, GPT-2, Falcon, Bloom, and T5-style model families.

The package works in two ways:

- Use `peft-doctor` from the terminal before training.
- Use `diagnose_peft(...)` inside your training script with real `model`, `tokenizer`, `peft_config`, `training_args`, and dataset objects.

## Problems PEFT Doctor Helps Fix

Developers often find this package while trying to fix one of these PEFT fine-tuning problems:

- `CUDA out of memory` during LoRA or QLoRA training
- QLoRA 4-bit loading problems with `bitsandbytes`
- `loss=nan`, infinite loss, fp16 overflow, or unstable training loss
- wrong `target_modules` for Llama, Mistral, Qwen, Gemma, Phi, GPT-2, Falcon, Bloom, or T5
- tokenizer padding errors such as `tokenizer has no pad_token`
- model not learning after PEFT fine-tuning
- bad output, repeated text, or prompt template mistakes
- PEFT adapter not saving, loading, or merging correctly
- `PeftModel.from_pretrained` adapter loading issues
- `merge_and_unload()` problems when exporting a merged LoRA model
- Colab PEFT setup problems, missing GPU runtime, or broken install cells
- dataset format problems for instruction tuning, chat templates, SFT, and prompt/completion data
- labels fully masked with `-100`, label/input length mismatches, or bad data collators
- train/eval leakage, duplicate samples, and long rows getting truncated
- `use_cache=True` conflicts with gradient checkpointing
- tokenizer size larger than model embeddings after adding special tokens
- too many or zero trainable parameters after applying LoRA
- missing warmup, scheduler, seed, checkpoint retention, or QLoRA optimizer choices
- slow long-context training that could use Flash Attention
- `device_map="auto"` conflicts with DDP, Accelerate, or torchrun
- DeepSpeed, FSDP, and QLoRA setup risks
- `torch_compile` instability with k-bit loading or gradient checkpointing
- sequence length larger than model context window or RoPE setup
- completion-only response template mismatch
- packed dataset examples without EOS separators
- pad tokens left inside labels instead of being masked to `-100`
- LoRA targeting `lm_head` or embedding layers by accident
- `inference_mode=True` or disabled LoRA initialization in a training config
- assistant-only or completion-only loss masking that hides the wrong tokens
- chat templates without assistant generation blocks
- Qwen instruct EOS token mistakes that make generations fail to stop cleanly
- mixed chat/instruction schemas inside one training file
- tool-calling and vision-language rows that need special formatting or collators
- 4-bit and 8-bit loading accidentally enabled together
- `bf16` and `fp16` both enabled in the same training run
- DDP `find_unused_parameters` settings that slow or break LoRA training
- MoE models where expert parameters may need `target_parameters`
- newer PEFT choices such as `all-linear`, rsLoRA, LoftQ, and DoRA tradeoffs
- disk-full, device mismatch, shape mismatch, overlong sequence, and gradient-norm failures in logs

## Install

Minimal install:

```bash
python -m pip install peft-doctor
```

Install with the normal fine-tuning stack:

```bash
python -m pip install "peft-doctor[ml]"
```

In Google Colab:

```python
%pip install -U "peft-doctor[ml]"
!peft-doctor env
```

Use a GPU runtime in Colab before loading a model: `Runtime` -> `Change runtime type` -> `T4`, `L4`, `A100`, or another GPU.

Development install from this repository:

```bash
git clone https://github.com/awais-akhtar/peft-doctor.git
cd peft-doctor
python -m pip install -e ".[dev,ml]"
```

## Quick Start

Run a pre-flight check from the terminal:

```bash
peft-doctor check \
  --model meta-llama/Llama-3-8B \
  --dataset data.jsonl \
  --batch-size 4 \
  --sequence-length 4096 \
  --learning-rate 2e-4
```

Generate a practical starter recipe:

```bash
peft-doctor recipe --kind qlora-sft --family llama
peft-doctor recipe --kind low-vram-colab --family qwen --output markdown
peft-doctor recipe --kind completion-only --family mistral --output json
```

Use it in Python:

```python
from peft_doctor import diagnose_peft

report = diagnose_peft(
    model=model,
    tokenizer=tokenizer,
    peft_config=peft_config,
    training_args=training_args,
    train_dataset=train_dataset,
    sequence_length=2048,
)

print(report.to_markdown())
```

Generate safe starter configs:

```python
from peft_doctor import (
    create_safe_lora_config,
    create_safe_bnb_config,
    create_safe_training_args,
    create_training_recipe,
)

peft_config = create_safe_lora_config(model)
bnb_config = create_safe_bnb_config()
training_args = create_safe_training_args()
recipe = create_training_recipe(kind="completion-only", model_family="llama")
```

## What It Checks

| Area | Common problem | Typical fix |
| --- | --- | --- |
| GPU memory | CUDA out of memory | Use QLoRA, batch size 1, gradient checkpointing, shorter sequence length |
| Target modules | LoRA attached to the wrong layers | Use model-aware targets like `q_proj`, `v_proj`, `c_attn`, or `query_key_value` |
| Prompt format | Dataset does not teach the response shape | Use instruction/response text or a proper chat template |
| Learning rate | Loss spikes or NaN | Try `1e-4`, `5e-5`, bf16, cleaner samples, and label checks |
| Tokenizer | Padding crash during batching | Set `tokenizer.pad_token = tokenizer.eos_token` when appropriate |
| Evaluation | Eval OOM after training works | Disable eval or use a tiny eval batch |
| Adapter flow | Adapter not found after training | Use `model.save_pretrained()` and `PeftModel.from_pretrained()` |
| Data quality | Duplicate rows, split leakage, masked labels | Deduplicate, fix labels, separate train/eval |
| Model state | `use_cache`, embeddings, trainable params | Disable cache, resize embeddings, verify LoRA trainables |
| Trainer config | Missing warmup, seed, scheduler, checkpoint limit | Add stable defaults before long runs |
| Distributed runs | DDP/FSDP/DeepSpeed/device map conflicts | Check launcher, quantization, and sharding settings |
| Completion masking | Response template missing, pad labels, packing leaks | Fix collator templates, EOS, and label masks |
| Advanced PEFT | rsLoRA, LoftQ, DoRA, all-linear, MoE targeting | Use `check` and `recipe` before long experiments |
| Runtime logs | Device mismatch, disk full, shape mismatch, grad norm spikes | Run `scan-log` on trainer output |

## Troubleshooting Recipes

For a longer problem-by-problem guide, see [docs/troubleshooting.md](docs/troubleshooting.md).

### Fix CUDA Out of Memory in PEFT or QLoRA

```bash
peft-doctor check \
  --model meta-llama/Llama-3-8B \
  --dataset train.jsonl \
  --eval-dataset eval.jsonl \
  --batch-size 4 \
  --sequence-length 4096 \
  --learning-rate 2e-4 \
  --packing \
  --response-template "### Response:" \
  --device-map auto
```

If the report warns about memory, start with:

```python
training_args = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing": True,
    "bf16": True,
}
```

For QLoRA:

```python
from peft_doctor import create_safe_bnb_config

bnb_config = create_safe_bnb_config()
```

### Fix Wrong LoRA Target Modules

```bash
peft-doctor targets --model meta-llama/Llama-3-8B
peft-doctor targets --model Qwen/Qwen2.5-7B
peft-doctor targets --family gpt2
```

### Fix NaN Loss in LoRA Fine-Tuning

```bash
peft-doctor scan-log trainer_log.jsonl
```

Common fixes are lower learning rate, bf16 instead of fp16, cleaner samples, valid labels, gradient clipping, and shorter sequences while debugging.

### Fix Tokenizer Padding Errors

```python
tokenizer.pad_token = tokenizer.eos_token
```

PEFT Doctor warns when a causal language model tokenizer has no pad token.

### Merge a LoRA Adapter Into the Base Model

```bash
peft-doctor adapter-check \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter your-user/your-lora-adapter \
  --output-dir merged-model

peft-doctor merge-adapter \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter your-user/your-lora-adapter \
  --output-dir merged-model \
  --dtype fp16
```

## Commands

Full command reference with examples: [docs/commands.md](docs/commands.md).

### `peft-doctor check`

Runs the main pre-flight check.

```bash
peft-doctor check --model meta-llama/Llama-3-8B --dataset data.jsonl
```

Useful options:

```bash
peft-doctor check \
  --model Qwen/Qwen2.5-7B \
  --dataset train.jsonl \
  --batch-size 2 \
  --grad-accum 8 \
  --sequence-length 2048 \
  --learning-rate 2e-4 \
  --load-in-4bit \
  --bf16 \
  --gradient-checkpointing
```

Machine-readable output:

```bash
peft-doctor check --model mistralai/Mistral-7B-v0.1 --dataset train.jsonl --output json
```

Markdown output for issues or pull requests:

```bash
peft-doctor check --model gpt2 --dataset train.jsonl --output markdown
```

### `peft-doctor targets`

Recommends LoRA `target_modules`.

```bash
peft-doctor targets --model meta-llama/Llama-3-8B
peft-doctor targets --family gpt2
```

Print as JSON:

```bash
peft-doctor targets --family qwen --output json
```

### `peft-doctor safe-config`

Prints a safe LoRA or QLoRA starter config.

```bash
peft-doctor safe-config --model meta-llama/Llama-3-8B
```

Only LoRA:

```bash
peft-doctor safe-config --family gpt2 --no-qlora
```

JSON:

```bash
peft-doctor safe-config --family llama --output json
```

### `peft-doctor recipe`

Generates ready-to-use starter recipes for common PEFT jobs.

```bash
peft-doctor recipe --kind qlora-sft --family llama
peft-doctor recipe --kind low-vram-colab --family qwen
peft-doctor recipe --kind completion-only --family mistral --output json
peft-doctor recipe --kind long-context --family llama --output markdown
peft-doctor recipe --kind distributed-qlora --family qwen
peft-doctor recipe --kind moe-lora --family deepseek
peft-doctor recipe --kind adapter-merge
```

Available recipes: `qlora-sft`, `low-vram-colab`, `completion-only`, `long-context`, `distributed-qlora`, `moe-lora`, and `adapter-merge`.

### `peft-doctor inspect-dataset`

Checks a local `.json`, `.jsonl`, `.csv`, or `.txt` dataset sample.

```bash
peft-doctor inspect-dataset data.jsonl
```

The command looks for common training shapes:

- `messages` chat rows with `role` and `content`
- `instruction` and `response` style columns
- single `text` rows containing instruction/response markers
- pre-tokenized `input_ids` and `labels`

### `peft-doctor scan-log`

Scans a training log for NaN, infinity, CUDA OOM, overflow, device mismatch, disk-full errors, shape mismatch, overlong sequence warnings, gradient-norm spikes, and unstable loss jumps.

```bash
peft-doctor scan-log trainer_log.jsonl
peft-doctor scan-log run.log --output markdown
```

### `peft-doctor adapter-check`

Checks a LoRA adapter merge plan without loading the full model.

```bash
peft-doctor adapter-check \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter awaisakhtar/llama-2-7b-summarization-finetuned-on-xsum-lora \
  --output-dir merged-llama
```

### `peft-doctor merge-adapter`

Merges a PEFT LoRA adapter into the base model and saves a normal Transformers model.

```bash
peft-doctor merge-adapter \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter awaisakhtar/llama-2-7b-summarization-finetuned-on-xsum-lora \
  --output-dir Llama-2-7b-summarization-finetuned-on-xsum \
  --dtype fp16
```

Push the merged model and tokenizer to the Hugging Face Hub:

```bash
huggingface-cli login

peft-doctor merge-adapter \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter your-user/your-lora-adapter \
  --output-dir merged-model \
  --push-to-hub \
  --hub-model-id your-user/merged-model \
  --dtype fp16
```

For Colab or private/gated models, store your own Hugging Face token as a secret named
`HF_TOKEN` and read it from the notebook environment. Do not paste access tokens into
notebooks, scripts, shell history, or GitHub issues.

For final exports, do not merge from a 4-bit or 8-bit loaded model unless you know your PEFT/Transformers versions support it. The safest export path is fp16, bf16, or fp32, then `save_pretrained(..., safe_serialization=True)`.

### `peft-doctor scan-notebook`

Scans a notebook for common PEFT and Colab mistakes, including exposed Hugging Face tokens.

```bash
peft-doctor scan-notebook model_merge.ipynb
```

### `peft-doctor env`

Checks the local Python, CUDA, and fine-tuning package stack.

```bash
peft-doctor env
peft-doctor env --output json
```

This is especially useful in Colab because many setup problems come from the notebook runtime, not the training script.

### `peft-doctor colab`

Prints a notebook-friendly setup cell.

```bash
peft-doctor colab
```

### `peft-doctor version`

Prints the installed version.

```bash
peft-doctor version
```

## Python API

### Diagnose a training setup

```python
from peft_doctor import diagnose_peft

report = diagnose_peft(
    model=model,
    tokenizer=tokenizer,
    peft_config=peft_config,
    training_args={
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "bf16": True,
        "gradient_checkpointing": True,
    },
    train_dataset=train_dataset,
)

if report.has_errors:
    raise RuntimeError(report.to_markdown())
```

### Generate target modules

```python
from peft_doctor import recommend_target_modules

targets = recommend_target_modules(model_name="meta-llama/Llama-3-8B")
```

### Generate safe LoRA and QLoRA configs

```python
from peft_doctor import create_safe_bnb_config, create_safe_lora_config

peft_config = create_safe_lora_config(model, r=16, lora_alpha=32)
bnb_config = create_safe_bnb_config()
```

When `peft`, `transformers`, and `torch` are installed, these helpers return real `LoraConfig` and `BitsAndBytesConfig` objects. Without those packages, they return plain dictionaries so you can still inspect the recommendation.

### Generate a full recipe

```python
from peft_doctor import create_training_recipe

recipe = create_training_recipe(kind="low-vram-colab", model_family="llama")
print(recipe["training_args"])
```

### Guard training logs

```python
from peft_doctor import NanLossGuard

guard = NanLossGuard()

for log in trainer_state_log_history:
    issues = guard.update(log)
    for issue in issues:
        print(issue.title, issue.fix)
```

### Merge a LoRA adapter

```python
from peft_doctor import merge_lora_adapter

result = merge_lora_adapter(
    base_model="meta-llama/Llama-2-7b-hf",
    adapter="your-user/your-lora-adapter",
    output_dir="merged-model",
    torch_dtype="fp16",
)

print(result.to_dict())
```

## Colab Notebook Pattern

```python
%pip install -U "peft-doctor[ml]"

from peft_doctor import diagnose_peft, create_safe_lora_config, create_safe_bnb_config

peft_config = create_safe_lora_config(model_name="meta-llama/Llama-3-8B")
bnb_config = create_safe_bnb_config()

report = diagnose_peft(
    model_name="meta-llama/Llama-3-8B",
    peft_config=peft_config,
    training_args={
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "bf16": True,
        "gradient_checkpointing": True,
        "load_in_4bit": True,
    },
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

print(report.to_markdown())
```

## Dependency Note

PEFT Doctor uses open-source Python packages from the normal PyData and Hugging Face fine-tuning stack: `torch`, `transformers`, `peft`, `datasets`, `accelerate`, `rich`, `typer`, and related optional packages. Model weights and datasets can have their own licenses, so always check the license of the model and data you fine-tune.

## Common Safe Config

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
```

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

## Publishing

The repository includes GitHub Actions for CI and PyPI publishing.

1. Push this repository to `awais-akhtar/peft-doctor`.
2. On PyPI, create a trusted publisher for package `peft-doctor`:
   - owner: `awais-akhtar`
   - repository: `peft-doctor`
   - workflow: `publish.yml`
   - environment: `pypi`
3. On TestPyPI, create the same trusted publisher with environment `testpypi`.
4. Push a version tag to publish to PyPI:

```bash
git tag v0.2.0
git push origin v0.2.0
```

Manual TestPyPI publishing is available from the `Publish Python Package` workflow in GitHub Actions.

## Project Status

This is alpha software. The checks are deliberately conservative: the package should warn early, explain the reason, and give a fix that a developer can actually try.

## License

MIT
