# PEFT Doctor: LoRA and QLoRA Fine-Tuning Debugger

PEFT Doctor is a local diagnosis layer, pre-flight checker, auto-fixer, VRAM and cost estimator, and troubleshooting toolkit for PEFT, LoRA, and QLoRA fine-tuning. It catches the problems that usually waste a training run: CUDA out of memory, NaN loss, risky learning rates, missing tokenizer padding, wrong LoRA target modules, broken prompt formats, bitsandbytes setup issues, and adapter save/load or merge failures.

It is built for the way people actually fine-tune models today: Hugging Face Transformers, PEFT, TRL, bitsandbytes, Google Colab, local CUDA machines, and common Llama, Mistral, Qwen, Gemma, Phi, GPT-2, Falcon, Bloom, and T5-style model families.

The package works in two ways:

- Use `peft-doctor` from the terminal before training.
- Use `diagnose_peft(...)` inside your training script with real `model`, `tokenizer`, `peft_config`, `training_args`, and dataset objects.
- Use `peft-doctor fix --dry-run train.py` to preview safe auto-repairs before writing a patched file.
- Use `peft-doctor diagnose train.py` for a local expert-style explanation of why a run may fail and what to fix first.

Privacy note: PEFT Doctor's diagnosis, chat, knowledge-base, optimizer, and cloud roadmap commands are local. They do not upload scripts, datasets, logs, adapters, or tokens.

Positioning: **PEFT Doctor is the quality assurance and optimization layer for LoRA/QLoRA fine-tuning.**

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
- training scripts and JSON configs that can be safely patched before a failed run

Common search phrases this project is built to answer: fix CUDA out of memory in QLoRA, LoRA NaN loss, PEFT wrong target_modules, tokenizer has no pad_token, Qwen EOS token issue, merge LoRA adapter into base model, Colab PEFT fine-tuning setup, TRL SFTTrainer label masking, adapter not saving, adapter loading error, bitsandbytes 4-bit training, VRAM estimator for LLM fine-tuning, dataset doctor for chat templates, and training log analyzer for failed fine-tuning runs.

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
peft-doctor recipe llama3-qlora-colab --copy ./my-run
peft-doctor validate-recipe ./my-run
```

Preview safe auto-fixes:

```bash
peft-doctor fix --dry-run train.py
peft-doctor fix --input train.py --output train.fixed.py
peft-doctor fix --dataset data.jsonl --write --pad-token-id 0
peft-doctor fix --config config.json --dry-run
peft-doctor estimate --model llama-3-8b --seq-len 2048 --batch-size 2 --qlora
peft-doctor init --model llama3 --gpu T4 --dataset-type chat --target-vram 16
peft-doctor dataset-doctor data.jsonl --sequence-length 2048
peft-doctor inspect-adapter ./adapter
peft-doctor analyze-log trainer.log
peft-doctor profiles qwen
peft-doctor check train.py --explain --html-report report.html --pdf-report report.pdf
```

Advanced local diagnosis and planning:

```bash
peft-doctor diagnose train.py --dataset data.jsonl --model llama-3-8b --gpu "RTX 4090"
peft-doctor simulate --model llama-3-8b --dataset data.jsonl --gpu L4 --seq-len 2048 --batch-size 2
peft-doctor memory-timeline --model llama-3-8b --seq-len 4096 --batch-size 1 --qlora
peft-doctor estimate-cost --model llama-3-8b --dataset-size 8000 --gpu L4 --gpu A100
peft-doctor advise-hparams --model llama-3-8b --dataset-size 8000 --gpu-vram 24
peft-doctor auto-tune --model llama-3-8b --batch-size 4 --grad-accum 1 --target-vram 16
peft-doctor score train.py --dataset data.jsonl --gpu T4
peft-doctor dataset-intel data.jsonl
peft-doctor dataset-report data.jsonl --output dataset-report.html
peft-doctor lora-efficiency --model llama-3-8b --rank 32 --dataset-size 8000
peft-doctor compare-adapters ./adapter-r16 ./adapter-r64
peft-doctor upgrade-suggestions
peft-doctor gpu-fingerprint "RTX 3060"
peft-doctor monitor trainer.log
peft-doctor history . --add-status completed --metric "BLEU +3.1"
peft-doctor knowledge-base "CUDA illegal memory access"
peft-doctor chat "Why is my loss exploding?" --dataset data.jsonl --log trainer.log
peft-doctor optimize . --html-report optimize-report.html
peft-doctor audit . --policy peft-policy.yml
peft-doctor cloud
```

## Beginner-Friendly Usage

If you are new to LoRA or QLoRA, start with the beginner guide:

[Beginner Command Guide](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/beginner-command-guide.md)

It explains every command in this format:

- when to use the command
- a copy-paste example
- how to understand the output
- what to do next

The shortest safe workflow for a new project is:

```bash
peft-doctor env
peft-doctor dataset-doctor data.jsonl
peft-doctor check train.py --dataset data.jsonl --model llama-3-8b --explain
peft-doctor fix --dry-run train.py
peft-doctor optimize . --html-report peft-doctor-report.html
```

If a command prints `ERROR`, fix that before training. If it prints `WARNING`, run a tiny smoke training before spending hours on a full run.

## Real Output, Not Fake Claims

PEFT Doctor separates real checks from planning estimates:

- Commands like `check`, `dataset-doctor`, `inspect-adapter`, `scan-log`, `notebook-check`, `fix`, `validate-recipe`, and `audit` inspect real local files.
- Commands like `estimate`, `simulate`, `memory-timeline`, `estimate-cost`, `advise-hparams`, and `lora-efficiency` are planning estimates. They help choose safer settings, but they do not replace a real training/evaluation run.
- `benchmark` prints a validation-matrix style entry for documented recipe checks. It is not a live ML benchmark.
- `cloud` describes the long-term roadmap. The local command does not upload anything.

## Feature Checklist

This README is the PyPI long description, so the checklist and examples below are visible from the package page after a release is published.

### Product Features

| Status | Feature | Command or example |
| --- | --- | --- |
| [x] | Init wizard that asks model, GPU, dataset type, target VRAM, and writes a training project | `peft-doctor init --model llama3 --gpu T4 --dataset-type chat --target-vram 16 --output-dir my-run` |
| [x] | VRAM estimator before training | `peft-doctor estimate --model llama-3-8b --seq-len 2048 --batch-size 2 --qlora` |
| [x] | Dataset doctor for bad rows, empty answers, duplicates, long examples, and chat-role mistakes | `peft-doctor dataset-doctor data.jsonl --sequence-length 2048` |
| [x] | Adapter doctor before upload or merge | `peft-doctor inspect-adapter ./adapter` |
| [x] | Training log analyzer for OOM, NaN, overflow, disk, shape, and device errors | `peft-doctor analyze-log trainer.log` |
| [x] | Model-family profiles for Llama, Qwen, Mistral, Gemma, Phi, Falcon, GPT-2, Bloom, and T5-style models | `peft-doctor profiles qwen` |
| [x] | Notebook checker for Colab and Jupyter mistakes, including pasted token patterns | `peft-doctor notebook-check notebook.ipynb` |
| [x] | Explain mode with reasons and copy-paste fixes | `peft-doctor check train.py --explain` |
| [x] | Risk score for training readiness | `peft-doctor check train.py --explain` |
| [x] | HTML and PDF reports for teams and GitHub issues | `peft-doctor check train.py --html-report report.html --pdf-report report.pdf` |

### Auto-Repair

| Status | Auto-repair item | Command or behavior |
| --- | --- | --- |
| [x] | Dry-run repair report | `peft-doctor fix --dry-run train.py` |
| [x] | Patch a Python training script to a new file | `peft-doctor fix --input train.py --output train.fixed.py` |
| [x] | Patch a dataset in place when requested | `peft-doctor fix --dataset data.jsonl --write --pad-token-id 0` |
| [x] | Dry-run JSON config repair | `peft-doctor fix --config config.json --dry-run` |
| [x] | Add `tokenizer.pad_token = tokenizer.eos_token` when missing | `peft-doctor fix --dry-run train.py` |
| [x] | Set `model.config.use_cache = False` when gradient checkpointing is active | `peft-doctor fix --dry-run train.py` |
| [x] | Block `fp16=True` and `bf16=True` being enabled together | `peft-doctor fix --dry-run train.py` |
| [x] | Suggest or replace safer LoRA `target_modules` | `peft-doctor fix --input train.py --output train.fixed.py --model-family llama` |
| [x] | Reduce high-risk batch size and sequence length | `peft-doctor fix --dry-run train.py` |
| [x] | Add stable `warmup_ratio`, `logging_steps`, and `save_strategy` values | `peft-doctor fix --dry-run train.py` |
| [x] | Mask pad labels to `-100` in JSON/JSONL data | `peft-doctor fix --dataset data.jsonl --write --pad-token-id 0` |
| [x] | Warn before targeting `lm_head` or embedding layers | `peft-doctor check train.py --explain` |

### Reproducible Recipes

| Status | Recipe item | Command or folder |
| --- | --- | --- |
| [x] | Llama 3 QLoRA Colab project | `recipes/llama3_qlora_colab/` |
| [x] | Qwen2 QLoRA Colab project | `recipes/qwen2_qlora_colab/` |
| [x] | Mistral LoRA local project | `recipes/mistral_lora_local/` |
| [x] | Gemma low-VRAM project | `recipes/gemma_low_vram/` |
| [x] | Completion-only SFT project | `recipes/completion_only_sft/` |
| [x] | Each recipe includes README, train script, requirements, sample data, expected output, and tested environment notes | `peft-doctor validate-recipe ./my-run` |
| [x] | Copy a complete runnable recipe | `peft-doctor recipe llama3-qlora-colab --copy ./my-run` |
| [x] | Copy a low-VRAM Qwen recipe | `peft-doctor recipe qwen-low-vram --copy ./my-run` |
| [x] | Validate a copied recipe | `peft-doctor validate-recipe ./my-run` |

### Validation And Trust

| Status | Trust-building item | Location or command |
| --- | --- | --- |
| [x] | Before/after broken config and fixed config examples | [docs/before-after.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/before-after.md) |
| [x] | Failure gallery with common PEFT failures and fixes | [docs/failure-gallery.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/failure-gallery.md) |
| [x] | Compatibility matrix for Transformers, PEFT, bitsandbytes, CUDA, and GPUs | [docs/compatibility-matrix.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/compatibility-matrix.md) |
| [x] | Real CLI/report screenshot assets | [docs/reports-and-screenshots.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/reports-and-screenshots.md) |
| [x] | GitHub Actions CI on every push and pull request | `.github/workflows/ci.yml` |
| [x] | GitHub Actions PyPI publishing with `PYPI_API_TOKEN1` secret | `.github/workflows/publish.yml` |
| [x] | Validation matrix with model, dataset, GPU, issue, fix, and time saved | [benchmarks/validation_matrix.md](https://github.com/awais-akhtar/peft-doctor/blob/main/benchmarks/validation_matrix.md) |
| [x] | Benchmark command | `peft-doctor benchmark --recipe llama3-qlora-colab` |
| [x] | Validation report command | `peft-doctor validate --model qwen --dataset sample.jsonl --report report.md` |
| [x] | Case study: CUDA OOM fixed | [docs/case-studies/cuda-oom-fixed.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/case-studies/cuda-oom-fixed.md) |
| [x] | Case study: NaN loss fixed | [docs/case-studies/nan-loss-fixed.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/case-studies/nan-loss-fixed.md) |
| [x] | Case study: wrong target modules fixed | [docs/case-studies/wrong-target-modules-fixed.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/case-studies/wrong-target-modules-fixed.md) |

### Advanced Local Features

| Status | Feature | Command |
| --- | --- | --- |
| [x] | Local expert diagnosis with confidence and recommended fixes | `peft-doctor diagnose train.py --dataset data.jsonl --model llama-3-8b` |
| [x] | Dry-run training simulation | `peft-doctor simulate --model llama-3-8b --dataset data.jsonl --gpu L4` |
| [x] | Memory timeline showing load, forward, backward, optimizer, and peak VRAM | `peft-doctor memory-timeline --model llama-3-8b --seq-len 2048 --batch-size 2 --qlora` |
| [x] | Cloud GPU cost estimator | `peft-doctor estimate-cost --model llama-3-8b --dataset-size 8000 --gpu L4 --gpu A100` |
| [x] | Hyperparameter advisor for LoRA rank, alpha, and dropout | `peft-doctor advise-hparams --model llama-3-8b --dataset-size 8000 --gpu-vram 24` |
| [x] | Training health monitor with loss trend, NaN chance, and GPU snapshot when available | `peft-doctor monitor trainer.log` |
| [x] | Smart auto-tuning for batch size, gradient accumulation, and sequence length | `peft-doctor auto-tune --model llama-3-8b --batch-size 4 --grad-accum 1 --target-vram 16` |
| [x] | Fine-tuning project score | `peft-doctor score train.py --dataset data.jsonl --gpu T4` |
| [x] | Dataset intelligence with quality score and issue counts | `peft-doctor dataset-intel data.jsonl` |
| [x] | LoRA efficiency predictor | `peft-doctor lora-efficiency --model llama-3-8b --rank 32 --dataset-size 8000` |
| [x] | Adapter comparison with rank, params, memory, size, and quality estimate | `peft-doctor compare-adapters ./adapter-r16 ./adapter-r64` |
| [x] | Automatic upgrade suggestions for fine-tuning packages | `peft-doctor upgrade-suggestions` |
| [x] | GPU fingerprinting for common local and cloud GPUs | `peft-doctor gpu-fingerprint "RTX 3060"` |
| [x] | Dataset visualizer with histograms, role distribution, duplicate clusters, and outliers | `peft-doctor dataset-report data.jsonl --output dataset-report.html` |
| [x] | Lightweight experiment history | `peft-doctor history . --add-status completed --metric "BLEU +3.1"` |
| [x] | Offline community knowledge base | `peft-doctor knowledge-base "CUDA illegal memory access"` |
| [x] | Local chat mode using dataset/log checks and the offline knowledge base | `peft-doctor chat "Why is my loss exploding?" --dataset data.jsonl --log trainer.log` |
| [x] | One-click project optimizer | `peft-doctor optimize . --html-report optimize-report.html` |
| [x] | Organization policy audit | `peft-doctor audit . --policy peft-policy.yml` |
| [x] | PEFT Doctor Cloud roadmap command, local and non-uploading | `peft-doctor cloud` |

## Screenshots

![PEFT Doctor diagnose CLI](https://raw.githubusercontent.com/awais-akhtar/peft-doctor/main/docs/assets/cli-diagnose.svg)

![PEFT Doctor dry-run auto-fix](https://raw.githubusercontent.com/awais-akhtar/peft-doctor/main/docs/assets/fix-dry-run.svg)

![PEFT Doctor dataset report](https://raw.githubusercontent.com/awais-akhtar/peft-doctor/main/docs/assets/dataset-report.svg)

![PEFT Doctor Colab recipe validation](https://raw.githubusercontent.com/awais-akhtar/peft-doctor/main/docs/assets/colab-success.svg)

## Every Command At A Glance

| Command | What it does | Example |
| --- | --- | --- |
| `check` | Main pre-flight check with optional risk explanation and reports | `peft-doctor check train.py --explain` |
| `fix` | Safe auto-repair for scripts, configs, and datasets | `peft-doctor fix --dry-run train.py` |
| `estimate` | VRAM estimate before loading a model | `peft-doctor estimate --model llama-3-8b --seq-len 2048 --batch-size 2 --qlora` |
| `init` | Generate a full training project | `peft-doctor init --model llama3 --gpu T4 --dataset-type chat --target-vram 16` |
| `diagnose` | Local expert-style diagnosis | `peft-doctor diagnose train.py --dataset data.jsonl` |
| `simulate` | Predict start success, VRAM, ETA, and likely failures | `peft-doctor simulate --model llama-3-8b --gpu L4` |
| `memory-timeline` | Show where memory spikes | `peft-doctor memory-timeline --model llama-3-8b --qlora` |
| `estimate-cost` | Compare cloud GPU cost and time | `peft-doctor estimate-cost --model llama-3-8b --dataset-size 8000 --gpu L4 --gpu A100` |
| `advise-hparams` | Recommend LoRA rank, alpha, and dropout | `peft-doctor advise-hparams --model llama-3-8b --dataset-size 8000` |
| `monitor` | Analyze training health from logs and local GPU state | `peft-doctor monitor trainer.log` |
| `auto-tune` | Keep effective batch while lowering memory | `peft-doctor auto-tune --model llama-3-8b --batch-size 4 --grad-accum 1 --target-vram 16` |
| `score` | Score dataset, config, hardware, trainer, and project readiness | `peft-doctor score train.py --dataset data.jsonl` |
| `dataset-intel` | Dataset quality intelligence | `peft-doctor dataset-intel data.jsonl` |
| `dataset-report` | HTML dataset visualizer | `peft-doctor dataset-report data.jsonl --output dataset-report.html` |
| `dataset-doctor` | Dataset pre-flight checker | `peft-doctor dataset-doctor data.jsonl --sequence-length 2048` |
| `inspect-dataset` | Inspect local dataset samples | `peft-doctor inspect-dataset data.jsonl` |
| `lora-efficiency` | Predict adapter size, gain, slowdown, and merge compatibility | `peft-doctor lora-efficiency --model llama-3-8b --rank 32` |
| `compare-adapters` | Compare two adapters | `peft-doctor compare-adapters ./adapter-a ./adapter-b` |
| `inspect-adapter` | Check a saved adapter before upload or merge | `peft-doctor inspect-adapter ./adapter` |
| `adapter-check` | Check an adapter merge plan | `peft-doctor adapter-check --base-model base --adapter adapter --output-dir merged` |
| `merge-adapter` | Merge a LoRA adapter into a base model | `peft-doctor merge-adapter --base-model base --adapter adapter --output-dir merged` |
| `upgrade-suggestions` | Check installed package versions | `peft-doctor upgrade-suggestions` |
| `gpu-fingerprint` | GPU-specific advice | `peft-doctor gpu-fingerprint "RTX 3060"` |
| `history` | Local experiment history | `peft-doctor history . --add-status completed --metric "BLEU +3.1"` |
| `knowledge-base` | Search bundled PEFT failure guidance | `peft-doctor knowledge-base "CUDA illegal memory access"` |
| `chat` | Ask a local troubleshooting question | `peft-doctor chat "Why is my loss exploding?" --dataset data.jsonl` |
| `optimize` | Combine fixer, dataset checks, score, and report | `peft-doctor optimize . --html-report optimize-report.html` |
| `audit` | Enforce team fine-tuning policy | `peft-doctor audit . --policy peft-policy.yml` |
| `cloud` | Show the long-term cloud reporting roadmap | `peft-doctor cloud` |
| `recipe` | Generate config recipes or copy runnable projects | `peft-doctor recipe llama3-qlora-colab --copy ./my-run` |
| `validate-recipe` | Validate copied recipe files | `peft-doctor validate-recipe ./my-run` |
| `benchmark` | Print recipe validation benchmark entry | `peft-doctor benchmark --recipe llama3-qlora-colab` |
| `validate` | Write a markdown validation report | `peft-doctor validate --model qwen --dataset sample.jsonl --report report.md` |
| `profiles` | Show built-in model-family profile | `peft-doctor profiles llama` |
| `targets` | Recommend LoRA target modules | `peft-doctor targets --model meta-llama/Llama-3-8B` |
| `safe-config` | Print safe LoRA/QLoRA starter config | `peft-doctor safe-config --family llama` |
| `scan-log` | Scan logs for runtime failures | `peft-doctor scan-log trainer.log` |
| `analyze-log` | Alias-friendly log analyzer | `peft-doctor analyze-log trainer.log` |
| `scan-notebook` | Scan notebooks for PEFT and Colab mistakes | `peft-doctor scan-notebook notebook.ipynb` |
| `notebook-check` | Notebook checker alias | `peft-doctor notebook-check notebook.ipynb` |
| `env` | Show Python, CUDA, and package environment | `peft-doctor env` |
| `colab` | Print a Colab setup cell | `peft-doctor colab` |
| `version` | Print installed version | `peft-doctor version` |

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
| Auto-repair | Common config mistakes repeated across projects | Run `fix --dry-run`, then write a patched copy |
| Recipes | Beginners need a complete first run | Use `recipe NAME --copy ./my-run` and `validate-recipe` |
| Local diagnosis | Need an expert explanation before training | Run `diagnose`, `simulate`, `score`, and `optimize` |
| Memory timeline | Need to know where VRAM spikes | Run `memory-timeline` |
| Cloud planning | Need cost estimates before renting GPUs | Run `estimate-cost` |
| Hyperparameters | Unsure about LoRA rank/alpha/dropout | Run `advise-hparams` |
| Dataset intelligence | Need quality score, outliers, and HTML visualizer | Run `dataset-intel` and `dataset-report` |
| Adapter comparison | Need to choose between adapters | Run `compare-adapters` |
| Team policies | Need standards for every fine-tuning project | Run `audit --policy peft-policy.yml` |
| VRAM estimate | Guessing memory before training | Run `estimate` before loading the model |
| Explain mode | Warnings without context | Use `--explain` for risk score, reasons, and copy-paste fixes |

## Troubleshooting Recipes

For a longer problem-by-problem guide, see [docs/troubleshooting.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/troubleshooting.md).

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

Full command reference with examples: [docs/commands.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/commands.md).

Beginner command guide: [docs/beginner-command-guide.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/beginner-command-guide.md).

Advanced feature guide: [docs/advanced-features.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/advanced-features.md).

Privacy and security notes: [docs/privacy-and-security.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/privacy-and-security.md).

### `peft-doctor fix`

Safely patches common PEFT training mistakes.

```bash
peft-doctor fix --dry-run train.py
peft-doctor fix --input train.py --output train.fixed.py
peft-doctor fix --config config.json --dry-run
peft-doctor fix --dataset data.jsonl --write --pad-token-id 0
```

It can add `tokenizer.pad_token = tokenizer.eos_token`, set `model.config.use_cache = False`, resolve `bf16`/`fp16` conflicts, replace risky LoRA target modules, lower high-risk batch/sequence values, add warmup/logging/save settings, and mask pad labels to `-100`.

### Product Commands

```bash
peft-doctor init --model llama3 --gpu T4 --dataset-type chat --target-vram 16 --output-dir my-run
peft-doctor estimate --model llama-3-8b --seq-len 2048 --batch-size 2 --qlora --target-vram 16
peft-doctor dataset-doctor data.jsonl --sequence-length 2048
peft-doctor inspect-adapter ./adapter
peft-doctor analyze-log trainer.log
peft-doctor notebook-check notebook.ipynb
peft-doctor profiles llama
peft-doctor check train.py --explain --html-report report.html --pdf-report report.pdf
```

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
peft-doctor recipe llama3-qlora-colab --copy ./my-run
peft-doctor recipe qwen-low-vram --copy ./my-run
```

Available recipes: `qlora-sft`, `low-vram-colab`, `completion-only`, `long-context`, `distributed-qlora`, `moe-lora`, and `adapter-merge`.

Copyable project recipes: `llama3-qlora-colab`, `qwen2-qlora-colab`, `qwen-low-vram`, `mistral-lora-local`, `gemma-low-vram`, and `completion-only-sft`.

Validate a copied project:

```bash
peft-doctor validate-recipe ./my-run
```

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

### Validation And Case Studies

- [benchmarks/validation_matrix.md](https://github.com/awais-akhtar/peft-doctor/blob/main/benchmarks/validation_matrix.md)
- [docs/before-after.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/before-after.md)
- [docs/failure-gallery.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/failure-gallery.md)
- [docs/compatibility-matrix.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/compatibility-matrix.md)
- [docs/reports-and-screenshots.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/reports-and-screenshots.md)
- [docs/case-studies/cuda-oom-fixed.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/case-studies/cuda-oom-fixed.md)
- [docs/case-studies/nan-loss-fixed.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/case-studies/nan-loss-fixed.md)
- [docs/case-studies/wrong-target-modules-fixed.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/case-studies/wrong-target-modules-fixed.md)

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
git tag v0.7.2
git push origin v0.7.2
```

Manual TestPyPI publishing is available from the `Publish Python Package` workflow in GitHub Actions.

## Project Status

This is alpha software. The checks are deliberately conservative: the package should warn early, explain the reason, and give a fix that a developer can actually try.

## License

MIT
