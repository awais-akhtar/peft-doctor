# PEFT Doctor

PEFT Doctor provides pre-flight checks and repair tools for LoRA and QLoRA fine-tuning projects using PEFT, Transformers, TRL, bitsandbytes, and Accelerate. It reviews training settings, datasets, memory estimates, adapters, checkpoints, generation configuration, and distributed setup before or after a run.

Checks report the rules that matched and suggested follow-up actions. They help identify common configuration and data problems; they do not predict model quality or guarantee a successful training run.

```bash
python -m pip install -U peft-doctor
peft-doctor check train.py --dataset data.jsonl --model llama-3-8b --explain
peft-doctor fix --input train.py --dry-run
```

The base install provides the CLI and file-based checks. Install the optional training stack for commands that load model or tokenizer metadata or inspect a live model:

```bash
python -m pip install -U "peft-doctor[ml]"
```

## Start Here

For an existing project, run these commands in order:

```bash
# 1. Check the environment, script, and dataset.
peft-doctor env
peft-doctor check train.py --dataset data.jsonl --model llama-3-8b --explain

# 2. Preview supported edits. This does not change the file.
peft-doctor fix --input train.py --dry-run

# 3. Write a separate patched script after reviewing the preview.
peft-doctor fix --input train.py --output train.fixed.py

# 4. Check the data and estimate memory.
peft-doctor dataset-doctor data.jsonl --sequence-length 2048
peft-doctor estimate --model llama-3-8b --seq-len 2048 --batch-size 1 --qlora --target-vram 16
```

For a new project, use the wizard or copy a recipe:

```bash
peft-doctor init --model llama3 --gpu T4 --dataset-type chat --target-vram 16 --output-dir my-run
peft-doctor recipe qwen2-qlora-colab --copy qwen-run
peft-doctor validate-recipe qwen-run
```

## What The Results Mean

Every check returns one of four severities:

| Severity | Meaning |
| --- | --- |
| `ERROR` | A checked contradiction or invalid artifact was found. |
| `WARNING` | A setting or artifact needs review. |
| `OK` | The named check passed for the inputs examined. |
| `INFO` | Context, a limitation, or a manual follow-up. |

Reports can be printed as a table, JSON, or Markdown. The main `check` command can also write HTML and PDF:

```bash
peft-doctor check train.py --dataset data.jsonl --output json
peft-doctor check train.py --dataset data.jsonl --html-report report.html --pdf-report report.pdf
```

## Auto-Repair

`fix` supports Python scripts, JSON configs, and JSON/JSONL datasets.

```bash
# Preview a training-script patch.
peft-doctor fix --input train.py --dry-run

# Write to a new path.
peft-doctor fix --input train.py --output train.fixed.py --family llama

# Preview config changes.
peft-doctor fix --config config.json --dry-run

# Mask pad-token labels in place after making a backup.
peft-doctor fix --dataset data.jsonl --write --pad-token-id 0
```

Supported script and config repairs include:

- add `tokenizer.pad_token = tokenizer.eos_token` when no pad-token setup is present;
- set `model.config.use_cache = False` when gradient checkpointing is enabled;
- turn off `fp16` when `fp16` and `bf16` are both enabled;
- replace high-risk batch size or sequence length literals;
- add `warmup_ratio`, `logging_steps`, and `save_strategy` where supported;
- replace missing or risky LoRA target module lists with a selected model-family profile;
- flag `lm_head` and embedding targets for manual review;
- set label values to `-100` at the supplied pad-token positions.

The fixer is intentionally narrow. It will report changes it cannot make safely instead of rewriting arbitrary Python.

## Dataset Checks

Use `dataset-doctor` for structure and training-format checks:

```bash
peft-doctor dataset-doctor data.jsonl --sequence-length 2048
```

Use `dataset-intel` for row counts, duplicates, empty assistant messages, assistant-only conversations, malformed data, phrase-based review markers, and character-set distribution:

```bash
peft-doctor dataset-intel data.jsonl --limit 5000
peft-doctor dataset-report data.jsonl --output dataset-report.html
```

The phrase checks are review filters. They do not determine whether an answer is true, hallucinated, malicious, or safe.

Check tokenized labels directly when a dataset contains `input_ids` and `labels`:

```bash
peft-doctor label-audit tokenized.jsonl --pad-token-id 0
```

This reports length mismatches, rows where every label is `-100`, rows with no ignored labels, unmasked pad positions, and the measured supervised-token ratio.

Check train/eval leakage with exact matching and token-set Jaccard similarity:

```bash
peft-doctor leakage-check train.jsonl eval.jsonl --threshold 0.90 --limit 10000
```

## TRL And Chat Templates

Check an `SFTConfig` saved as JSON or YAML against the dataset:

```bash
peft-doctor sft-check sft-config.yaml \
  --dataset data.jsonl \
  --chat-template chat-template.jinja
```

The command checks `assistant_only_loss`, `completion_only_loss`, packing strategy, padding-free attention requirements, maximum length, dataset shape, and `{% generation %}` markers.

Check a template by itself or compare its role handling with a dataset:

```bash
peft-doctor chat-template-check tokenizer_config.json \
  --dataset chat.jsonl \
  --assistant-only-loss
```

## LoRA And QLoRA Configuration

Generate target modules or a starter configuration:

```bash
peft-doctor targets --model meta-llama/Llama-3-8B
peft-doctor targets --family gpt2 --no-mlp
peft-doctor safe-config --family qwen --output json
```

Check a current `LoraConfig` or `adapter_config.json`:

```bash
peft-doctor lora-config-check adapter_config.json
peft-doctor lora-config-check lora.yaml --compiled --hotswap
```

This covers rank, alpha, dropout, `all-linear`, risky embedding targets, `trainable_token_indices`, weight tying, aLoRA merge limitations, QALoRA group size, initialization mode, and `target_parameters` with compiled hotswap.

Check a bitsandbytes configuration:

```bash
peft-doctor quantization-check bnb-config.json --gpu T4 --training
```

This catches simultaneous 4-bit and 8-bit loading, reviews NF4 for 4-bit training, checks explicit compute dtype, flags T4/bf16 plans, and compares FSDP storage and model dtypes when both are present.

## Memory, Runtime, And Cost

Estimate VRAM before model loading:

```bash
peft-doctor estimate \
  --model llama-3-8b \
  --seq-len 2048 \
  --batch-size 1 \
  --qlora \
  --target-vram 16
```

The VRAM estimate is formula-based. Framework allocations, kernels, padding, optimizer choice, and model architecture can change the real peak.

If the model name does not contain its size, pass it explicitly. Architecture overrides are useful for an unrecognized family:

```bash
peft-doctor estimate --model ./local-model --params-billion 8 --hidden-size 4096 --num-layers 32 --qlora
```

Show the same estimate by training phase:

```bash
peft-doctor memory-timeline --model llama-3-8b --seq-len 4096 --batch-size 1 --qlora
```

Run a planning simulation without loading a model:

```bash
peft-doctor simulate \
  --model llama-3-8b \
  --dataset data.jsonl \
  --gpu T4 \
  --tokens-per-second 750 \
  --epochs 3 \
  --checkpoint-gb 0.8 \
  --save-total-limit 2 \
  --disk-free-gb 20
```

Runtime is shown only when measured tokens per second is supplied. Disk use is shown only when checkpoint size is supplied.

Calculate cloud cost from provider prices and throughput you measured on the same workload:

```bash
peft-doctor estimate-cost \
  --model llama-3-8b \
  --dataset-size 8000 \
  --seq-len 2048 \
  --epochs 3 \
  --offer "L4:0.80:750" \
  --offer "A100:1.90:1800"
```

Each offer is `NAME:HOURLY_RATE:TOKENS_PER_SECOND`. PEFT Doctor does not ship a price table because provider rates and measured throughput change.

## Adapters, Hotswap, And Merge

Inspect a saved adapter before upload or merge:

```bash
peft-doctor inspect-adapter ./adapter --base-model meta-llama/Llama-3-8B
peft-doctor adapter-integrity ./adapter --write-checksums adapter_checksums.json
peft-doctor adapter-tensors ./adapter --output json
```

`adapter-tensors` reads safetensors headers without loading tensor data. It reports exact tensor count, parameter count, dtype counts, tensor ranks, and file bytes.

Compare two adapters using their configuration and actual local files:

```bash
peft-doctor compare-adapters ./adapter-r16 ./adapter-r64
```

The comparison does not predict which adapter is better. Use held-out evaluation metrics for that decision.

Check PEFT hotswap compatibility:

```bash
peft-doctor hotswap-check ./adapter-loaded ./adapter-next
peft-doctor hotswap-check ./adapter-loaded ./adapter-next --compiled --prepared-max-rank 64
```

Before a final merge, inspect the plan:

```bash
peft-doctor adapter-check \
  --base-model meta-llama/Llama-3-8B \
  --adapter ./adapter \
  --output-dir ./merged
```

Run the merge only after the plan is clean:

```bash
peft-doctor merge-adapter \
  --base-model meta-llama/Llama-3-8B \
  --adapter ./adapter \
  --output-dir ./merged \
  --revision MODEL_COMMIT_SHA \
  --dtype bf16
```

The merge command does not need a token for local public artifacts. For a private or gated model, log in with your own Hugging Face account outside the script.
`--revision` accepts a branch, tag, or commit SHA; a commit SHA makes the base-model and tokenizer download reproducible.

## Checkpoints And Reproducibility

Find the latest structurally valid checkpoint:

```bash
peft-doctor checkpoint-list ./outputs
```

Inspect one checkpoint and compare it with current settings:

```bash
peft-doctor checkpoint-check ./outputs/checkpoint-500
peft-doctor resume-check ./outputs/checkpoint-500 --config training-config.json
```

Record and verify project files, package versions, platform details, and Git commit:

```bash
peft-doctor manifest . --output run-manifest.json
peft-doctor verify-manifest run-manifest.json
```

## Distributed Training

Use the general Accelerate/DeepSpeed checker:

```bash
peft-doctor distributed-check accelerate.yaml --training-args training.json
```

Use the FSDP-specific checker for state dictionaries, CPU-efficient loading, synchronized module state, auto-wrapping, activation checkpointing, process count, and compile settings:

```bash
peft-doctor fsdp-check accelerate-fsdp.yaml --training-args training.json
```

## Logs And Notebooks

Scan a Trainer log for NaN/Inf loss, gradient spikes, CUDA OOM, illegal memory access, disk errors, device mismatch, shape mismatch, and overlong input warnings:

```bash
peft-doctor analyze-log trainer.log
peft-doctor monitor trainer.log
```

`monitor` reports parsed loss direction and a local CUDA memory snapshot when Torch can read one. It does not assign a probability to future NaN loss.

Check a notebook before running it in Colab or Jupyter:

```bash
peft-doctor notebook-check model_merge.ipynb
```

The notebook checker looks for pasted Hugging Face token patterns, tokens passed in shell commands, brittle install cells, and quantized adapter-merge plans.

## Recipes

Five complete projects are included:

- `llama3-qlora-colab`
- `qwen2-qlora-colab` (alias: `qwen-low-vram`)
- `mistral-lora-local`
- `gemma-low-vram`
- `completion-only-sft`

Copy and validate one:

```bash
peft-doctor recipe llama3-qlora-colab --copy ./my-run
peft-doctor validate-recipe ./my-run
cd my-run
python train.py --dry-run
python train.py --revision MODEL_COMMIT_SHA --max-steps 10
```

Each project contains `README.md`, `train.py`, `requirements.txt`, `sample_data.jsonl`, `expected_output.md`, and `tested_environment.md`.

## Investigation And Planning Commands

These commands add local evidence review for causes that are hard to spot from a single config warning:

```bash
peft-doctor investigate ./run
peft-doctor inspect-objective data.jsonl --row 42
peft-doctor gradients gradient-summary.json
peft-doctor minimize failure.json --allow-execution --budget-runs 10
peft-doctor conflicts data.jsonl
peft-doctor coverage data.jsonl --tasks tasks.yaml
peft-doctor privacy data.jsonl --redacted-output data.redacted.jsonl
peft-doctor strategy requirements.yaml
peft-doctor plan --model llama-3-8b --vram 16 --hours 4
peft-doctor compose-plan ./adapter-a ./adapter-b
peft-doctor serving-check ./adapter --engine vllm
peft-doctor next-experiment ./my-run
peft-doctor investigate-distributed ./rank-logs
peft-doctor assistant ./my-run --question "Why did training stop?"
```

The [investigation guide](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/intelligent-investigations.md) describes each command's inputs, examples, and limits. The failure minimizer runs a user-supplied process only when `--allow-execution` is set; it uses an argument array and enforces a run budget. The evidence assistant searches local files without calling a remote model.

| Command | Purpose | Example |
| --- | --- | --- |
| `investigate` | Correlate dataset, script, and log findings | `peft-doctor investigate ./run --output markdown` |
| `inspect-objective` | Review a row's saved text, chat turns, IDs, and labels | `peft-doctor inspect-objective data.jsonl --row 42` |
| `gradients` | Inspect a JSON summary of trainable gradients and updates | `peft-doctor gradients gradient-summary.json` |
| `minimize` | Reduce a reproducible JSONL failure under a run budget | `peft-doctor minimize failure.json --allow-execution --budget-runs 10` |
| `conflicts` | Find similar questions with differing answers for review | `peft-doctor conflicts data.jsonl --threshold 0.82` |
| `coverage` | Compare data against task markers supplied in YAML/JSON | `peft-doctor coverage data.jsonl --tasks tasks.yaml` |
| `privacy` | Scan common sensitive patterns and optionally redact a copy | `peft-doctor privacy data.jsonl --redacted-output clean.jsonl` |
| `strategy` | Compare prompt, retrieval, and fine-tuning considerations | `peft-doctor strategy requirements.yaml` |
| `plan` | Compare a formula-based memory estimate with supplied budgets | `peft-doctor plan --model llama-3-8b --vram 16 --hours 4` |
| `compose-plan` | Compare local adapter metadata before evaluation | `peft-doctor compose-plan ./adapter-a ./adapter-b` |
| `serving-check` | Review adapter metadata relevant to serving | `peft-doctor serving-check ./adapter --engine vllm` |
| `next-experiment` | Suggest a comparison from notes saved by `history` | `peft-doctor history ./my-run --add-status oom --note "Reduced sequence length"` then `peft-doctor next-experiment ./my-run` |
| `investigate-distributed` | Find common error signatures in per-rank logs | `peft-doctor investigate-distributed ./rank-logs` |
| `assistant` | Retrieve matching lines from local run files for a question | `peft-doctor assistant ./run --question "Why did training stop?"` |

These commands inspect local inputs and provide planning guidance. They do not evaluate model quality. `inspect-objective` does not reproduce a trainer's full collator and packing path; `coverage` checks user-supplied markers; and `privacy` uses pattern matching that may miss sensitive values. See the detailed guide for input formats and limitations.

## All Commands

Examples operate on the named inputs. Some commands write reports or repaired copies; merge and Hub operations can download or upload model artifacts when configured.

| Command | What to use it for | Example |
| --- | --- | --- |
| `check` | Main script, model, dataset, trainer, tokenizer, memory, and risk check | `peft-doctor check train.py --dataset data.jsonl --model llama-3-8b --revision MODEL_COMMIT_SHA --explain` |
| `fix` | Preview or write supported script, config, and dataset repairs | `peft-doctor fix --input train.py --output train.fixed.py` |
| `estimate` | Formula-based VRAM planning | `peft-doctor estimate --model llama-3-8b --seq-len 2048 --batch-size 1 --qlora` |
| `init` | Generate a complete project from model, GPU, dataset type, and VRAM | `peft-doctor init --model qwen2 --gpu L4 --dataset-type chat --target-vram 24 --output-dir run` |
| `diagnose` | Prioritize local script, dataset, model, and memory findings | `peft-doctor diagnose train.py --dataset data.jsonl --model llama-3-8b --gpu T4` |
| `simulate` | Check VRAM, evaluation, runtime inputs, and checkpoint schedule without training | `peft-doctor simulate --model llama-3-8b --dataset data.jsonl --gpu T4` |
| `memory-timeline` | Break the VRAM formula into load, forward, backward, and optimizer phases | `peft-doctor memory-timeline --model llama-3-8b --qlora` |
| `estimate-cost` | Calculate cost from supplied rate and measured throughput | `peft-doctor estimate-cost --model llama-3-8b --dataset-size 8000 --offer "L4:0.80:750"` |
| `advise-hparams` | Get a rule-based rank, alpha, and dropout starting point | `peft-doctor advise-hparams --model llama-3-8b --dataset-size 8000 --gpu-vram 24` |
| `monitor` | Summarize log failures, loss direction, and local GPU memory | `peft-doctor monitor trainer.log` |
| `auto-tune` | Lower estimated memory while preserving effective batch | `peft-doctor auto-tune --model llama-3-8b --batch-size 4 --grad-accum 1 --target-vram 16` |
| `score` | Summarize checked errors and warnings with the documented rule score | `peft-doctor score train.py --dataset data.jsonl --model llama-3-8b --gpu T4` |
| `dataset-intel` | Count dataset rule matches and rows without flagged issues | `peft-doctor dataset-intel data.jsonl` |
| `dataset-report` | Write a static HTML dataset report | `peft-doctor dataset-report data.jsonl --output dataset-report.html` |
| `dataset-doctor` | Check rows, formats, roles, duplicates, lengths, and labels | `peft-doctor dataset-doctor data.jsonl --sequence-length 2048` |
| `inspect-dataset` | Print a compact dataset pre-flight report | `peft-doctor inspect-dataset data.jsonl` |
| `leakage-check` | Find exact and high-overlap train/eval pairs | `peft-doctor leakage-check train.jsonl eval.jsonl --threshold 0.9` |
| `label-audit` | Inspect pretokenized supervision and pad masking | `peft-doctor label-audit tokenized.jsonl --pad-token-id 0` |
| `sft-check` | Check TRL selective loss, packing, padding-free mode, and dataset format | `peft-doctor sft-check sft.yaml --dataset data.jsonl --chat-template template.jinja` |
| `chat-template-check` | Check template variables, roles, and generation markers | `peft-doctor chat-template-check tokenizer_config.json --dataset chat.jsonl --assistant-only-loss` |
| `lora-config-check` | Check current PEFT LoRA fields and variant combinations | `peft-doctor lora-config-check adapter_config.json` |
| `quantization-check` | Check bitsandbytes 4/8-bit, NF4, dtype, and FSDP settings | `peft-doctor quantization-check bnb.json --gpu T4` |
| `investigate` | Join project, data, and log findings into possible causal chains | `peft-doctor investigate ./run --output markdown` |
| `inspect-objective` | Inspect stored text, conversation turns, and label masks for one row | `peft-doctor inspect-objective data.jsonl --row 42` |
| `gradients` | Find zero/nonfinite gradients, zero updates, and optimizer omissions from a JSON snapshot | `peft-doctor gradients gradient-summary.json` |
| `minimize` | Reduce a reproducible dataset failure with explicit permission and a run budget | `peft-doctor minimize failure.json --allow-execution --budget-runs 10` |
| `conflicts` | Find similar prompts with different answers for human review | `peft-doctor conflicts data.jsonl --threshold 0.82` |
| `coverage` | Compare user-supplied task/language markers and conversation patterns with the dataset | `peft-doctor coverage data.jsonl --tasks tasks.yaml` |
| `privacy` | Flag common sensitive-data patterns and optionally write a redacted copy | `peft-doctor privacy data.jsonl --redacted-output clean.jsonl` |
| `strategy` | Give rule-based prompting, retrieval, or fine-tuning guidance from requirements | `peft-doctor strategy requirements.yaml` |
| `plan` | Compare supplied VRAM, disk, GPU count, time, and measured throughput inputs | `peft-doctor plan --model llama-3-8b --vram 16 --hours 4 --gpu-count 2` |
| `compose-plan` | Compare local adapter metadata before a separately evaluated composition | `peft-doctor compose-plan ./adapter-a ./adapter-b` |
| `serving-check` | Review adapter metadata that can affect a named serving engine | `peft-doctor serving-check ./adapter --engine vllm` |
| `next-experiment` | Review local run history and identify missing comparison evidence | `peft-doctor history ./my-run --add-status oom --note "Reduced sequence length"` then `peft-doctor next-experiment ./my-run` |
| `investigate-distributed` | Scan per-rank log files for common distributed failure signatures | `peft-doctor investigate-distributed ./rank-logs` |
| `assistant` | Retrieve matching lines from local project files for a question | `peft-doctor assistant ./my-run --question "Why did training stop?"` |
| `lora-efficiency` | Review rank and target coverage without quality guessing | `peft-doctor lora-efficiency --model llama-3-8b --rank 32` |
| `compare-adapters` | Compare two local adapter configs and actual file metadata | `peft-doctor compare-adapters ./adapter-a ./adapter-b` |
| `inspect-adapter` | Check adapter config, weights, and base-model metadata | `peft-doctor inspect-adapter ./adapter --base-model org/model` |
| `adapter-integrity` | Validate safetensors structure and write/verify SHA-256 hashes | `peft-doctor adapter-integrity ./adapter --write-checksums adapter_checksums.json` |
| `adapter-tensors` | Count exact tensors, parameters, dtypes, and file bytes | `peft-doctor adapter-tensors ./adapter --output json` |
| `hotswap-check` | Compare two adapters before PEFT hotswap | `peft-doctor hotswap-check ./loaded ./incoming --compiled --prepared-max-rank 64` |
| `adapter-check` | Validate an adapter merge plan without loading the model | `peft-doctor adapter-check --base-model org/model --adapter ./adapter --output-dir ./merged` |
| `merge-adapter` | Merge a LoRA adapter into its base model | `peft-doctor merge-adapter --base-model org/model --revision MODEL_COMMIT_SHA --adapter ./adapter --output-dir ./merged` |
| `checkpoint-list` | Inventory every checkpoint and select the latest structurally valid one | `peft-doctor checkpoint-list ./outputs` |
| `checkpoint-check` | Inspect one Trainer/Accelerate checkpoint | `peft-doctor checkpoint-check ./outputs/checkpoint-500` |
| `resume-check` | Compare checkpoint metadata with current settings | `peft-doctor resume-check ./outputs/checkpoint-500 --config config.json` |
| `training-plan` | Calculate effective batch, optimizer steps, warmup, and saves | `peft-doctor training-plan --dataset-size 8000 --batch-size 1 --grad-accum 8 --epochs 3` |
| `disk-check` | Compare real free disk with an explicit checkpoint budget | `peft-doctor disk-check ./outputs --checkpoint-gb 1.2 --save-total-limit 2 --reserve-gb 5` |
| `token-lengths` | Measure tokenizer-specific or approximate length percentiles | `peft-doctor token-lengths data.jsonl --model Qwen/Qwen2.5-7B --revision MODEL_COMMIT_SHA --max-length 2048` |
| `distributed-check` | Check Accelerate and DeepSpeed precision, process, and offload conflicts | `peft-doctor distributed-check accelerate.yaml --training-args training.json` |
| `fsdp-check` | Check FSDP-specific loading, wrapping, state, and compile settings | `peft-doctor fsdp-check accelerate-fsdp.yaml --training-args training.json` |
| `generation-check` | Validate EOS, padding, lengths, sampling, and beam settings | `peft-doctor generation-check ./merged-model` |
| `analyze-log` | Scan and explain common training-log failures | `peft-doctor analyze-log trainer.log` |
| `scan-log` | Print the lower-level log scan | `peft-doctor scan-log trainer.log --output markdown` |
| `notebook-check` | Check a Colab/Jupyter notebook | `peft-doctor notebook-check train.ipynb` |
| `scan-notebook` | Run the same notebook scanner under its original command name | `peft-doctor scan-notebook train.ipynb` |
| `profiles` | Show a built-in model-family profile | `peft-doctor profiles qwen` |
| `targets` | Recommend LoRA target module names | `peft-doctor targets --model Qwen/Qwen2.5-7B` |
| `safe-config` | Print starter LoRA, QLoRA, and Trainer values | `peft-doctor safe-config --family llama --output json` |
| `recipe` | Print a config recipe or copy a runnable project | `peft-doctor recipe gemma-low-vram --copy ./gemma-run` |
| `validate-recipe` | Confirm all required recipe files and values are present | `peft-doctor validate-recipe ./gemma-run` |
| `benchmark` | Run the packaged pre-flight validation scenario for a recipe | `peft-doctor benchmark --recipe llama3-qlora-colab` |
| `validate` | Write a local Markdown validation report | `peft-doctor validate --model qwen --dataset sample.jsonl --report report.md` |
| `manifest` | Create a reproducibility manifest | `peft-doctor manifest . --output run-manifest.json` |
| `verify-manifest` | Detect changed files and package-version drift | `peft-doctor verify-manifest run-manifest.json` |
| `upgrade-suggestions` | Compare installed versions with declared package minimums | `peft-doctor upgrade-suggestions` |
| `gpu-fingerprint` | Show local GPU identity or a built-in reference VRAM note | `peft-doctor gpu-fingerprint "RTX 4090"` |
| `history` | Append or read local experiment notes | `peft-doctor history . --add-status completed --metric "eval_loss=1.82"` |
| `knowledge-base` | Search bundled troubleshooting rules | `peft-doctor knowledge-base "CUDA illegal memory access"` |
| `chat` | Match a question with local rules plus optional dataset/log checks | `peft-doctor chat "Why is my loss exploding?" --dataset data.jsonl --log trainer.log` |
| `optimize` | Combine fixer, dataset, memory, and report checks | `peft-doctor optimize . --html-report optimize-report.html` |
| `audit` | Enforce a local team policy file | `peft-doctor audit . --policy peft-policy.yml` |
| `cloud` | Show the hosted-service roadmap; it uploads nothing | `peft-doctor cloud` |
| `env` | Show Python, CUDA, Colab, and package status | `peft-doctor env --output json` |
| `colab` | Print a Colab installation and pre-flight cell | `peft-doctor colab` |
| `version` | Print the installed package version | `peft-doctor version` |

The detailed option guide is in [docs/commands.md](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/commands.md). All documentation links are absolute so they work from both GitHub and PyPI.

## Python API

```python
from peft_doctor import diagnose_peft

report = diagnose_peft(
    model=model,
    tokenizer=tokenizer,
    peft_config=peft_config,
    training_args=training_args,
    train_dataset=train_dataset,
)

for issue in report.sorted_issues():
    print(issue.severity, issue.code, issue.message)
```

Create starter configs:

```python
from peft_doctor import (
    create_safe_bnb_config,
    create_safe_lora_config,
    create_safe_training_args,
)

peft_config = create_safe_lora_config(model_name="meta-llama/Llama-3-8B")
bnb_config = create_safe_bnb_config()
training_args = create_safe_training_args()
```

Guard Trainer logs:

```python
from peft_doctor import NanLossGuard

guard = NanLossGuard()
for issue in guard.update({"loss": float("nan"), "grad_norm": 145.0}):
    print(issue.code, issue.fix)
```

## Colab

In a fresh notebook cell:

```python
%pip install -q -U "peft-doctor[ml]"
```

Then run:

```python
!peft-doctor env
!peft-doctor dataset-doctor /content/data.jsonl
!peft-doctor estimate --model llama-3-8b --seq-len 2048 --batch-size 1 --qlora --target-vram 16
```

Do not paste an access token into a notebook, command, config, or repository. Use your own Colab Secret, environment variable, or interactive Hugging Face login for private and gated models.

## Scope And Evidence

PEFT Doctor performs static checks, bounded dataset sampling, local file validation, and formula-based planning. These checks do not establish model quality, training success, factual correctness, or production readiness. Review each finding against the model, trainer, and dataset in use.

- [Compatibility matrix](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/compatibility-matrix.md)
- [Command guide](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/commands.md)
- [Troubleshooting guide](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/troubleshooting.md)
- [Operational safety](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/operational-safety.md)
- [Privacy and security](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/privacy-and-security.md)
- [Contributing](https://github.com/awais-akhtar/peft-doctor/blob/main/CONTRIBUTING.md)

## License

PEFT Doctor is released under the [MIT License](https://github.com/awais-akhtar/peft-doctor/blob/main/LICENSE).
