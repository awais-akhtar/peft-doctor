# PEFT Doctor Command Reference

This page explains every `peft-doctor` command, when to use it, and practical examples.

If you are new to PEFT fine-tuning, start with the beginner guide first:

[Beginner Command Guide](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/beginner-command-guide.md)

That guide explains each command with a copy-paste example, how to read the result, and what to do next.

Install first:

```bash
python -m pip install "peft-doctor[ml]"
```

In Colab:

```python
%pip install -U "peft-doctor[ml]"
```

## Command Index

```bash
peft-doctor check
peft-doctor fix
peft-doctor init
peft-doctor estimate
peft-doctor diagnose
peft-doctor simulate
peft-doctor memory-timeline
peft-doctor estimate-cost
peft-doctor advise-hparams
peft-doctor monitor
peft-doctor auto-tune
peft-doctor score
peft-doctor dataset-intel
peft-doctor dataset-report
peft-doctor lora-efficiency
peft-doctor compare-adapters
peft-doctor upgrade-suggestions
peft-doctor gpu-fingerprint
peft-doctor history
peft-doctor knowledge-base
peft-doctor chat
peft-doctor optimize
peft-doctor audit
peft-doctor cloud
peft-doctor dataset-doctor
peft-doctor inspect-adapter
peft-doctor analyze-log
peft-doctor profiles
peft-doctor notebook-check
peft-doctor targets
peft-doctor safe-config
peft-doctor recipe
peft-doctor validate-recipe
peft-doctor benchmark
peft-doctor validate
peft-doctor inspect-dataset
peft-doctor scan-log
peft-doctor scan-notebook
peft-doctor adapter-check
peft-doctor merge-adapter
peft-doctor training-plan
peft-doctor disk-check
peft-doctor checkpoint-check
peft-doctor resume-check
peft-doctor manifest
peft-doctor verify-manifest
peft-doctor token-lengths
peft-doctor distributed-check
peft-doctor generation-check
peft-doctor adapter-integrity
peft-doctor sft-check
peft-doctor fsdp-check
peft-doctor hotswap-check
peft-doctor lora-config-check
peft-doctor checkpoint-list
peft-doctor leakage-check
peft-doctor adapter-tensors
peft-doctor chat-template-check
peft-doctor label-audit
peft-doctor quantization-check
peft-doctor env
peft-doctor colab
peft-doctor version
```

## `peft-doctor check`

Use this before training. It checks memory risk, model family, LoRA targets, tokenizer padding, prompt format, dataset quality, train/eval leakage, learning rate, optimizer, warmup, checkpoint settings, and common QLoRA problems.

Basic:

```bash
peft-doctor check --model meta-llama/Llama-3-8B --dataset train.jsonl
```

Pin remote metadata to a model commit for a reproducible check:

```bash
peft-doctor check --model meta-llama/Llama-3-8B --revision MODEL_COMMIT_SHA --dataset train.jsonl
```

QLoRA run:

```bash
peft-doctor check \
  --model Qwen/Qwen2.5-7B \
  --dataset train.jsonl \
  --eval-dataset eval.jsonl \
  --batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum 8 \
  --sequence-length 2048 \
  --learning-rate 2e-4 \
  --load-in-4bit \
  --optim paged_adamw_8bit \
  --warmup-ratio 0.03 \
  --lr-scheduler-type cosine \
  --save-total-limit 2 \
  --seed 42 \
  --bf16 \
  --gradient-checkpointing
```

Advanced distributed or packed SFT check:

```bash
peft-doctor check \
  --model meta-llama/Llama-3-8B \
  --dataset train.jsonl \
  --eval-dataset eval.jsonl \
  --load-in-4bit \
  --device-map auto \
  --world-size 2 \
  --fsdp full_shard \
  --deepspeed ds_config.json \
  --torch-compile \
  --packing \
  --completion-only-loss \
  --assistant-only-loss \
  --response-template "### Response:" \
  --attn-implementation flash_attention_2 \
  --ddp-find-unused-parameters \
  --gradient-checkpointing-use-reentrant \
  --remove-unused-columns
```

Debug a likely OOM:

```bash
peft-doctor check \
  --model mistralai/Mistral-7B-v0.1 \
  --dataset train.jsonl \
  --batch-size 4 \
  --sequence-length 4096 \
  --load-in-4bit
```

Use local Hugging Face cache only:

```bash
peft-doctor check --model meta-llama/Llama-3-8B --local-files-only
```

Output formats:

```bash
peft-doctor check --model gpt2 --dataset train.jsonl --output table
peft-doctor check --model gpt2 --dataset train.jsonl --output json
peft-doctor check --model gpt2 --dataset train.jsonl --output markdown
```

Explain mode and reports:

```bash
peft-doctor check train.py --explain
peft-doctor check train.py --html-report report.html
peft-doctor check train.py --pdf-report report.pdf
peft-doctor check train.py --report report.md
```

Useful options:

- `--model`: Hugging Face model id or local model path
- `--dataset`: train dataset path, `.json`, `.jsonl`, `.csv`, or `.txt`
- `--eval-dataset`: eval dataset path for train/eval overlap checks
- `--batch-size`: `per_device_train_batch_size`
- `--eval-batch-size`: `per_device_eval_batch_size`
- `--grad-accum`: `gradient_accumulation_steps`
- `--sequence-length`: max sequence length
- `--learning-rate`: training learning rate
- `--load-in-4bit`: check as a QLoRA 4-bit run
- `--load-in-8bit`: check as an 8-bit loaded run
- `--bf16` / `--no-bf16`: whether bf16 is enabled
- `--fp16` / `--no-fp16`: whether fp16 is enabled
- `--gradient-checkpointing` / `--no-gradient-checkpointing`: checkpointing state
- `--optim`: optimizer, such as `paged_adamw_8bit`
- `--warmup-ratio` or `--warmup-steps`: warmup setting
- `--lr-scheduler-type`: scheduler, such as `cosine` or `linear`
- `--save-steps`: checkpoint interval
- `--save-total-limit`: checkpoint retention limit
- `--seed`: training seed
- `--max-grad-norm`: gradient clipping value
- `--dataloader-num-workers`: dataloader worker count
- `--device-map`: model loading device map, such as `auto`
- `--world-size` and `--local-rank`: distributed launch metadata
- `--fsdp`: FSDP mode
- `--deepspeed`: DeepSpeed config path or setting
- `--torch-compile`: check torch compile risks
- `--attn-implementation`: attention backend, such as `flash_attention_2`, `sdpa`, or `eager`
- `--packing`: check packed dataset EOS risks
- `--group-by-length`: tell the checker length grouping is enabled
- `--completion-only-loss`: check prompt/completion masking behavior
- `--assistant-only-loss`: check chat assistant-span masking behavior
- `--response-template`: completion-only response marker
- `--remove-unused-columns` / `--keep-unused-columns`: Trainer column behavior
- `--gradient-checkpointing-use-reentrant` / `--gradient-checkpointing-non-reentrant`: checkpointing mode
- `--ddp-find-unused-parameters` / `--ddp-no-find-unused-parameters`: DDP unused parameter behavior
- `--logging-steps`: logging interval used to catch early loss failures
- `--explain`: print risk score, reasons, and copy-paste fixes
- `--html-report`: write an HTML report
- `--pdf-report`: write a small PDF report

## `peft-doctor init`

Use the wizard to generate a full project from model, GPU, dataset type, and target VRAM.

```bash
peft-doctor init
peft-doctor init --model llama3 --gpu T4 --dataset-type chat --target-vram 16 --output-dir my-run
```

## `peft-doctor estimate`

Estimate expected VRAM before training:

```bash
peft-doctor estimate --model llama-3-8b --seq-len 2048 --batch-size 2 --qlora
peft-doctor estimate --model qwen2.5-7b --seq-len 4096 --batch-size 1 --qlora --target-vram 24
peft-doctor estimate --model ./local-model --params-billion 8 --hidden-size 4096 --num-layers 32 --qlora
```

When the model name does not include a parameter count, the estimator reports its 7B fallback. Use the override options for an unknown architecture.

## `peft-doctor diagnose`

Prioritized diagnosis for a training script, dataset, model, and GPU. The script and dataset checks use local files. If a remote model identifier is supplied, metadata lookup may contact the configured model host.

```bash
peft-doctor diagnose train.py
peft-doctor diagnose train.py --dataset data.jsonl --model llama-3-8b --gpu "RTX 4090"
peft-doctor diagnose train.py --dataset data.jsonl --sequence-length 4096 --batch-size 2 --output markdown
```

It prints the checked risk level, matching findings, and recommended fixes. The risk level is not a probability of success.

## `peft-doctor simulate`

Dry-run a training plan without loading or training a model.

```bash
peft-doctor simulate --model llama-3-8b --dataset data.jsonl --gpu L4
peft-doctor simulate --model qwen2.5-7b --gpu T4 --seq-len 2048 --batch-size 1 --save-steps 100
peft-doctor simulate --model llama-3-8b --dataset data.jsonl --tokens-per-second 750 --epochs 3
peft-doctor simulate --model llama-3-8b --checkpoint-gb 0.8 --save-total-limit 2 --disk-free-gb 20 --total-steps 3000
```

It checks a formula-based VRAM plan, evaluation OOM risk, and checkpoint schedule. Runtime needs `--tokens-per-second`; disk use needs `--checkpoint-gb`. Use measurements from the same workload.

## `peft-doctor memory-timeline`

Show where memory spikes happen inside a training step.

```bash
peft-doctor memory-timeline --model llama-3-8b --seq-len 2048 --batch-size 2 --qlora
peft-doctor memory-timeline --model mistral-7b --seq-len 4096 --batch-size 1 --no-gradient-checkpointing
```

Phases include load, forward, backward, optimizer, and peak memory.

## `peft-doctor estimate-cost`

Calculate cloud GPU time and cost from values you supply.

```bash
peft-doctor estimate-cost --model llama-3-8b --dataset-size 8000 --epochs 3 --offer "L4:0.80:750"
peft-doctor estimate-cost --model qwen2.5-14b --dataset-size 20000 --seq-len 4096 --offer "L4:0.80:600" --offer "A100:1.90:1500"
```

Each offer is `NAME:HOURLY_RATE:TOKENS_PER_SECOND`. Rates and throughput are not bundled because both vary by provider and workload.

## `peft-doctor advise-hparams`

Choose a rule-based starting point for LoRA rank, alpha, and dropout from model size, dataset size, task, and VRAM.

```bash
peft-doctor advise-hparams --model llama-3-8b --dataset-size 8000 --gpu-vram 24
peft-doctor advise-hparams --model gemma-2b --dataset-size 700 --gpu-vram 12 --task completion
```

Compare multiple values on held-out data. This command does not predict final quality.

## `peft-doctor monitor`

Analyze a saved Trainer log and show a local GPU memory snapshot when available.

```bash
peft-doctor monitor trainer.log
peft-doctor monitor trainer_state.jsonl --output markdown
```

It reports parsed loss direction and checked log failures such as NaN/Inf, CUDA OOM, device mismatch, disk-full, shape mismatch, and gradient spikes. It does not assign a probability to future NaN loss.

## `peft-doctor auto-tune`

Suggest a lower-memory configuration while preserving effective batch size.

```bash
peft-doctor auto-tune --model llama-3-8b --batch-size 4 --grad-accum 1 --target-vram 16
peft-doctor auto-tune --model qwen2.5-7b --batch-size 2 --grad-accum 4 --seq-len 4096 --target-vram 24
```

Typical output recommends smaller `batch_size`, higher `gradient_accumulation_steps`, and safer sequence length.

## `peft-doctor score`

Summarize checked errors and warnings with a documented rule score.

```bash
peft-doctor score train.py --dataset data.jsonl --gpu T4
peft-doctor score --dataset data.jsonl --output json
```

The formula is `100 - 25 per error - 7 per warning`, with a minimum of zero. It covers only checks that had enough local input to run.

## `peft-doctor dataset-intel`

Dataset intelligence counts duplicates, empty assistant messages, assistant-only rows, prompt-injection phrase matches, malformed JSONL rows, boilerplate answers, unsupported-claim phrases, outliers, and character-set buckets.

```bash
peft-doctor dataset-intel data.jsonl
peft-doctor dataset-intel data.jsonl --limit 5000 --output markdown
```

The percentage is the share of observed rows without a matched rule, not a model-quality score. Phrase matches require human review.

## `peft-doctor dataset-report`

Generate a static HTML dataset visualizer.

```bash
peft-doctor dataset-report data.jsonl
peft-doctor dataset-report data.jsonl --output reports/dataset-report.html --limit 5000
```

The report includes conversation length histogram, token histogram, duplicate counts, character-set buckets, role distribution, average response length, longest conversations, and outliers. The HTML escapes dataset text and stays local.

## `peft-doctor lora-efficiency`

Review LoRA rank and target coverage without guessing quality, size, or runtime.

```bash
peft-doctor lora-efficiency --model llama-3-8b --rank 16 --dataset-size 8000
peft-doctor lora-efficiency --model qwen2.5-14b --rank 64 --dataset-size 50000 --output json
```

Use it to review the planned coverage. Run `adapter-tensors` after saving and compare held-out metrics between ranks.

## `peft-doctor compare-adapters`

Compare two local LoRA adapter directories.

```bash
peft-doctor compare-adapters ./adapter-r16 ./adapter-r64
peft-doctor compare-adapters ./run-a/adapter ./run-b/adapter --output markdown
```

It compares rank, alpha, target modules, actual adapter file size, and exact safetensors parameter counts. It does not recommend an adapter without evaluation metrics.

## `peft-doctor upgrade-suggestions`

Compare installed fine-tuning package versions with PEFT Doctor's declared minimums.

```bash
peft-doctor upgrade-suggestions
peft-doctor upgrade-suggestions --output json
```

This is useful when a project has old `transformers`, `peft`, `trl`, or `bitsandbytes` versions.

## `peft-doctor gpu-fingerprint`

Identify GPU-specific risks.

```bash
peft-doctor gpu-fingerprint
peft-doctor gpu-fingerprint "RTX 3060"
peft-doctor gpu-fingerprint "RTX 4090" --output markdown
```

It reports a built-in reference VRAM value and local CUDA data when available. Confirm the exact card, driver, Torch build, precision support, and attention backend in the current environment.

## `peft-doctor history`

Track lightweight PEFT run history without installing MLflow.

```bash
peft-doctor history .
peft-doctor history . --add-status OOM --note "batch_size=4 failed on T4"
peft-doctor history . --add-status completed --metric "BLEU +3.1"
peft-doctor history . --add-status best --metric "eval_loss 0.82"
```

History is written locally to `.peft-doctor/history.jsonl`.

## `peft-doctor knowledge-base`

Search the built-in offline failure knowledge base.

```bash
peft-doctor knowledge-base "CUDA illegal memory access"
peft-doctor knowledge-base "nan loss"
peft-doctor knowledge-base "adapter merge" --output markdown
```

This command uses bundled knowledge only. It does not query a remote service.

## `peft-doctor chat`

Ask a local PEFT Doctor question and optionally include a dataset or log file.

```bash
peft-doctor chat "Why is my loss exploding?"
peft-doctor chat "Why is my model answering blank?" --dataset data.jsonl
peft-doctor chat "What caused this failure?" --log trainer.log
```

The answer is built from local checks and the offline knowledge base. It does not send your question or files to a hosted model.

## `peft-doctor optimize`

Run a one-pass project optimizer across script, dataset, reports, and score.

```bash
peft-doctor optimize .
peft-doctor optimize . --html-report optimize-report.html
peft-doctor optimize . --write
```

Default mode is dry-run. Pass `--write` only after reviewing the proposed edits.

## `peft-doctor audit`

Audit a project against a team policy.

Example policy:

```yaml
policy:
  max_seq_len: 4096
  require_bf16: true
  forbid_fp16: true
  require_dataset_validation: true
```

Command:

```bash
peft-doctor audit . --policy peft-policy.yml
peft-doctor audit ./training-project --policy policy.json --output json
```

Use this in CI when a team wants consistent fine-tuning standards.

## `peft-doctor cloud`

Show the privacy-first PEFT Doctor Cloud roadmap.

```bash
peft-doctor cloud
peft-doctor cloud --output markdown
```

This command only prints roadmap information; it does not transmit files or contact a hosted service.

## `peft-doctor dataset-doctor`

Dataset doctor checks bad rows, empty answers, duplicates, too-long examples, missing assistant roles, prompt markers, and completion templates.

```bash
peft-doctor dataset-doctor data.jsonl --sequence-length 2048
peft-doctor dataset-doctor data.jsonl --packing --response-template "### Response:"
```

## `peft-doctor inspect-adapter`

Adapter doctor checks saved LoRA adapters before upload or merge.

```bash
peft-doctor inspect-adapter ./adapter
peft-doctor inspect-adapter ./adapter --base-model meta-llama/Meta-Llama-3-8B
```

## `peft-doctor analyze-log`

Analyze logs and explain failures.

```bash
peft-doctor analyze-log trainer.log
peft-doctor analyze-log trainer.log --output markdown
```

## `peft-doctor profiles`

Show built-in model-family profiles.

```bash
peft-doctor profiles
peft-doctor profiles qwen
```

## `peft-doctor notebook-check`

Scan a Colab or Jupyter notebook before sharing it or running a long job. This is the same checker as `scan-notebook`, with a clearer product name.

```bash
peft-doctor notebook-check model_merge.ipynb
peft-doctor notebook-check training_notebook.ipynb --output markdown
```

It checks hard-coded Hugging Face tokens, risky login cells, stale package pins, adapter merge mistakes, typo redirects, and quantized merge patterns.

## `peft-doctor fix`

Use this for the fixer's supported repair patterns. Start with dry-run mode:

```bash
peft-doctor fix --input train.py --dry-run
```

Typical output:

```text
Found 7 issues.
Automated edits available for 5.
Run with --write or --output to apply.
```

Patch a Python training script into a new file:

```bash
peft-doctor fix --input train.py --output train.fixed.py
```

Patch a JSON config:

```bash
peft-doctor fix --config config.json --dry-run
peft-doctor fix --config config.json --output config.fixed.json
```

Patch a JSONL dataset in place by masking pad labels:

```bash
peft-doctor fix --dataset data.jsonl --write --pad-token-id 0
```

Automatic repairs include:

- add `tokenizer.pad_token = tokenizer.eos_token`
- set `model.config.use_cache = False` when gradient checkpointing is on
- turn off `fp16` when both `fp16` and `bf16` are enabled
- replace risky LoRA targets such as `lm_head` or embeddings
- reduce risky batch size and long sequence length starter values
- add `warmup_ratio`, `logging_steps`, and `save_strategy`
- mask pad labels to `-100` in JSON/JSONL datasets

Useful options:

- `--family llama`, `--family qwen`, or `--family mistral` for target module repair
- `--format json` or `--format markdown` for automation
- `--write` for in-place edits
- `--output` for a patched copy

## `peft-doctor targets`

Use this when LoRA is not learning or you are unsure what `target_modules` should be.

Llama, Mistral, Qwen, Gemma, Phi style:

```bash
peft-doctor targets --model meta-llama/Llama-3-8B
peft-doctor targets --model Qwen/Qwen2.5-7B
peft-doctor targets --family mistral
```

GPT-2 style:

```bash
peft-doctor targets --family gpt2
```

Attention-only targets:

```bash
peft-doctor targets --family llama --no-mlp
```

JSON:

```bash
peft-doctor targets --family qwen --output json
```

## `peft-doctor safe-config`

Use this to generate a safe LoRA/QLoRA starter config.

Print Python code:

```bash
peft-doctor safe-config --model meta-llama/Llama-3-8B
```

Only LoRA, no QLoRA block:

```bash
peft-doctor safe-config --family gpt2 --no-qlora
```

JSON:

```bash
peft-doctor safe-config --family llama --output json
```

Copy the output into a training script, then adjust rank, dropout, and target modules only after the first stable run.

## `peft-doctor recipe`

Use this when you want a complete starting point instead of separate config pieces.

Basic QLoRA SFT:

```bash
peft-doctor recipe --kind qlora-sft --family llama
```

Copy a complete runnable recipe project:

```bash
peft-doctor recipe llama3-qlora-colab --copy ./my-run
peft-doctor recipe qwen-low-vram --copy ./my-run
peft-doctor recipe completion-only-sft --copy ./completion-run
```

Low-VRAM Colab:

```bash
peft-doctor recipe --kind low-vram-colab --family qwen --output markdown
```

Completion-only prompt/response training:

```bash
peft-doctor recipe --kind completion-only --family mistral --output json
```

Long-context experiment:

```bash
peft-doctor recipe --kind long-context --family llama
```

Distributed QLoRA:

```bash
peft-doctor recipe --kind distributed-qlora --family qwen
```

MoE LoRA:

```bash
peft-doctor recipe --kind moe-lora --family deepseek
```

Adapter merge/export checklist:

```bash
peft-doctor recipe --kind adapter-merge --output markdown
```

Available recipes:

- `qlora-sft`
- `low-vram-colab`
- `completion-only`
- `long-context`
- `distributed-qlora`
- `moe-lora`
- `adapter-merge`

Copyable project recipes:

- `llama3-qlora-colab`
- `qwen2-qlora-colab`
- `qwen-low-vram`
- `mistral-lora-local`
- `gemma-low-vram`
- `completion-only-sft`

## `peft-doctor validate-recipe`

Use this after copying a recipe project.

```bash
peft-doctor validate-recipe ./my-run
peft-doctor validate-recipe ./my-run --output markdown
```

It checks the required project files and confirms the training script still contains PEFT Doctor safety calls.

## `peft-doctor benchmark`

Show packaged pre-flight validation evidence for a recipe.

```bash
peft-doctor benchmark --recipe llama3-qlora-colab
peft-doctor benchmark --recipe qwen2-qlora-colab --output markdown
```

## `peft-doctor validate`

Create a lightweight validation report for a model family and dataset without loading the full model.

```bash
peft-doctor validate --model qwen --dataset sample_data.jsonl --report report.md
peft-doctor validate --model mistral --dataset sample_data.jsonl --output json
```

## `peft-doctor inspect-dataset`

Use this when the model is not learning, outputs repeated text, gives blank answers, or your prompt format may be wrong.

```bash
peft-doctor inspect-dataset train.jsonl
peft-doctor inspect-dataset train.csv --output markdown
peft-doctor inspect-dataset train.json --output json
```

It checks:

- supported local files: `.jsonl`, `.json`, `.csv`, `.txt`
- instruction/response columns
- prompt/completion columns
- chat `messages`
- pre-tokenized `input_ids` and `labels`
- empty rows
- duplicate rows
- missing assistant replies
- all-ignored labels
- label/input length mismatches
- rows likely to exceed sequence length when used through `check`

Good examples:

```json
{"instruction": "Explain LoRA simply.", "response": "LoRA trains small adapter matrices..."}
```

```json
{"messages": [{"role": "user", "content": "Explain LoRA."}, {"role": "assistant", "content": "LoRA is..."}]}
```

## `peft-doctor scan-log`

Use this after or during training when loss, runtime, or export logs look broken.

```bash
peft-doctor scan-log trainer_state.jsonl
peft-doctor scan-log run.log
peft-doctor scan-log run.log --output markdown
```

It detects:

- `loss=nan`
- `loss=inf`
- CUDA out of memory
- overflow messages
- sudden loss jumps
- invalid or spiking `grad_norm`
- disk-full and quota errors
- CUDA illegal memory access
- tensor device mismatch
- tensor shape mismatch
- tokenized sequence length above the model limit

Typical fixes:

- lower learning rate to `1e-4` or `5e-5`
- use `bf16=True` instead of fp16 when supported
- check labels and bad samples
- add `max_grad_norm=1.0`

## `peft-doctor scan-notebook`

Use this before sharing a Colab notebook or when an old notebook keeps failing.

```bash
peft-doctor scan-notebook model_merge.ipynb
peft-doctor scan-notebook training_notebook.ipynb --output markdown
```

It checks:

- exposed Hugging Face tokens
- token login patterns in notebook cells
- old `transformers` or `peft` pins
- duplicate `pip install pip install`
- typo redirects such as `>>dev>null`
- adapter merge code that can be replaced with `merge-adapter`
- risky quantized merge patterns

For private or gated models, use `huggingface-cli login`, `HF_TOKEN`, or Colab Secrets. Do not paste tokens into notebooks.

## `peft-doctor adapter-check`

Use this before merging or publishing a LoRA adapter.

```bash
peft-doctor adapter-check \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter your-user/your-lora-adapter \
  --output-dir merged-model
```

Local adapter directory:

```bash
peft-doctor adapter-check \
  --base-model ./base-model \
  --adapter ./adapter-checkpoint \
  --output-dir ./merged-model
```

Check a Hub push plan:

```bash
peft-doctor adapter-check \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter your-user/your-lora-adapter \
  --push-to-hub \
  --hub-model-id your-user/merged-model
```

It checks:

- `adapter_config.json`
- `adapter_model.safetensors` or `adapter_model.bin`
- base model mismatch
- risky 4-bit or 8-bit merge plans
- output directory safety
- missing Hub repo id

## `peft-doctor merge-adapter`

Use this when you want a normal merged Transformers model instead of a small LoRA adapter.

Dry run first:

```bash
peft-doctor merge-adapter \
  --base-model meta-llama/Llama-2-7b-hf \
  --revision MODEL_COMMIT_SHA \
  --adapter your-user/your-lora-adapter \
  --output-dir merged-model \
  --dry-run
```

Merge and save locally:

```bash
peft-doctor merge-adapter \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter your-user/your-lora-adapter \
  --output-dir merged-model \
  --dtype fp16
```

Merge and push to Hugging Face Hub:

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

Useful options:

- `--dtype auto`, `--dtype fp16`, `--dtype bf16`, or `--dtype fp32`
- `--device-map auto`
- `--offload-folder offload`
- `--max-shard-size 5GB`
- `--safe-serialization` to save safetensors
- `--trust-remote-code` for custom model code
- `--tokenizer-source` to load tokenizer from a different repo/path
- `--revision` to select a branch, tag, or commit SHA; use a commit SHA for reproducibility
- `--no-tokenizer` if you do not want tokenizer files saved or pushed
- `--overwrite` for non-empty output directories
- `--allow-quantized-merge` only if you know your stack supports it

For final model export, fp16, bf16, or fp32 loading is safer than 4-bit or 8-bit loading.

## `peft-doctor training-plan`

Use this before a run to confirm effective batch size, optimizer steps, warmup steps, and checkpoint count.

```bash
peft-doctor training-plan --dataset-size 8000 --batch-size 1 --grad-accum 8 --world-size 1 --epochs 3 --save-steps 500
```

Read `effective_batch_size` and `total_optimizer_steps` in JSON output when configuring a scheduler:

```bash
peft-doctor training-plan --dataset-size 8000 --batch-size 2 --grad-accum 4 --output json
```

Next: compare the result with the first Trainer log lines. This is a deterministic plan from your inputs; it does not measure model throughput.

## `peft-doctor disk-check`

Checks the actual free space on the output disk against your checkpoint, dataset/cache, and reserve budget.

```bash
peft-doctor disk-check ./outputs --checkpoint-gb 1.2 --save-total-limit 2 --dataset-gb 3 --reserve-gb 5
```

Next: if the margin is negative or small, reduce `save_total_limit`, use a different output disk, or clear old checkpoints. Checkpoint size is supplied by you; free space is read from the filesystem.

## `peft-doctor checkpoint-check`

Inspects the newest `checkpoint-*` directory, or a checkpoint path you provide. It checks weights, `trainer_state.json`, optimizer state, scheduler state, RNG state, and adapter config without loading pickle files.

```bash
peft-doctor checkpoint-check ./outputs
peft-doctor checkpoint-check ./outputs/checkpoint-500 --output json
```

Next: do not rely on an exact resume if optimizer, scheduler, or RNG state is missing. Adapter-only weights can still be useful for inference but are not the same as a full Trainer resume.

## `peft-doctor resume-check`

Runs the checkpoint check and compares saved settings with your current JSON or YAML configuration.

```bash
peft-doctor resume-check ./outputs/checkpoint-500 --config config.json
```

The comparison covers base model, LoRA targets/rank, learning rate, sequence length, batch/accumulation, precision, seed, and quantization when those values are present in both files. A base-model, target-module, or rank mismatch is an error.

Next: restore the original settings or begin a new run instead of silently resuming incompatible state.

## `peft-doctor manifest`

Creates a reproducibility JSON file containing SHA-256 hashes for project config/code/data files, installed fine-tuning package versions, Python/platform details, and the current Git commit when available.

```bash
peft-doctor manifest . --output run-manifest.json
peft-doctor manifest ./my-run --output reports/run-manifest.json --max-file-mb 64
```

Hashes are streamed in bounded chunks. Files above `--max-file-mb` are recorded but deliberately not hashed. Review paths before sharing the manifest; it records metadata, not file contents.

## `peft-doctor verify-manifest`

Detects missing or changed files and version drift from a previously recorded run.

```bash
peft-doctor verify-manifest run-manifest.json
peft-doctor verify-manifest run-manifest.json --max-file-mb 1024 --output markdown
```

Next: reproduce with the recorded files and versions, or create a new manifest to give the changed run a new identity.

## `peft-doctor token-lengths`

Profiles p50, p95, p99, maximum length, over-limit rows, and estimated static-padding waste.

Exact model-tokenizer counts:

```bash
peft-doctor token-lengths data.jsonl --model Qwen/Qwen2.5-7B --revision MODEL_COMMIT_SHA --max-length 2048 --limit 2000
```

Fast offline estimate:

```bash
peft-doctor token-lengths data.jsonl --max-length 2048 --limit 1000
```

Next: filter exceptional outliers, choose a deliberate truncation policy, or use packing/length grouping. Tokenizer-free output is labeled as an estimate.

## `peft-doctor distributed-check`

Validates Accelerate YAML or DeepSpeed JSON for precision conflicts, invalid process counts, missing referenced files, incompatible ZeRO/offload choices, accumulation mismatches, and `device_map=auto` with multi-process training.

```bash
peft-doctor distributed-check accelerate.yaml
peft-doctor distributed-check accelerate.yaml --training-args training.json --output json
```

Next: keep one source of truth for DeepSpeed settings and rerun the check before `accelerate launch`.

## `peft-doctor generation-check`

Checks `generation_config.json`, or a local model directory containing generation/tokenizer/model config files.

```bash
peft-doctor generation-check ./merged-model
peft-doctor generation-check generation_config.json --output markdown
```

It catches missing/mismatched EOS and pad ids, conflicting length controls, invalid numeric settings, ignored sampling controls, and impossible beam/return-sequence combinations.

Next: test the corrected settings on a fixed prompt set. This command validates configuration consistency; it does not judge response quality.

## `peft-doctor adapter-integrity`

Checks a local adapter before upload, merge, or handoff. It validates `adapter_config.json`, base-model identity, adapter weight presence, safetensors header/offset metadata, and any existing checksum manifest. It never deserializes `.bin` files.

```bash
peft-doctor adapter-integrity ./adapter
peft-doctor adapter-integrity ./adapter --write-checksums adapter_checksums.json --output json
```

Relative checksum paths are written inside the adapter directory. Run the first command again after copying or downloading an adapter; a checksum mismatch means that copy should not be trusted until its origin is verified.

## `peft-doctor sft-check`

Check a TRL `SFTConfig` in JSON or YAML against the dataset and chat template.

```bash
peft-doctor sft-check sft-config.yaml --dataset data.jsonl --chat-template chat-template.jinja
peft-doctor sft-check sft-config.json --dataset completions.jsonl --output json
```

Use this before `SFTTrainer` when enabling `assistant_only_loss`, `completion_only_loss`, packing, or padding-free batches. It checks the sampled dataset shape and requires generation markers when assistant-only masking is requested.

## `peft-doctor fsdp-check`

Check Accelerate FSDP settings that need to agree.

```bash
peft-doctor fsdp-check accelerate-fsdp.yaml
peft-doctor fsdp-check accelerate-fsdp.yaml --training-args training.json --output markdown
```

It checks state-dict type, CPU-efficient loading with synchronized states, process count, auto-wrap policy, compile with original parameters, and duplicate checkpointing mechanisms.

## `peft-doctor hotswap-check`

Compare the adapter loaded first with the adapter that will be swapped in.

```bash
peft-doctor hotswap-check ./adapter-loaded ./adapter-next
peft-doctor hotswap-check ./adapter-loaded ./adapter-next --compiled --prepared-max-rank 64
```

It checks LoRA type, base-model identity, target-layer subset rules, `modules_to_save`, raw target parameters, and the rank reserved for compiled hotswap.

## `peft-doctor lora-config-check`

Validate a current PEFT `LoraConfig`, adapter config, or equivalent YAML.

```bash
peft-doctor lora-config-check adapter_config.json
peft-doctor lora-config-check lora.yaml --compiled --hotswap
```

It covers rank, alpha, dropout, `all-linear`, risky embedding/output targets, trainable token indices, weight tying, aLoRA merge limitations, QALoRA group size, initialization mode, and raw target parameters.

## `peft-doctor checkpoint-list`

Inspect every `checkpoint-N` directory and select the newest one without a structural error.

```bash
peft-doctor checkpoint-list ./outputs
peft-doctor checkpoint-list ./outputs --output json
```

The selected checkpoint still needs `resume-check` before use because structural validity does not prove current settings are compatible.

## `peft-doctor leakage-check`

Compare local train and evaluation splits for exact and high-overlap text.

```bash
peft-doctor leakage-check train.jsonl eval.jsonl
peft-doctor leakage-check train.jsonl eval.jsonl --threshold 0.85 --limit 10000 --max-examples 50
```

Near overlap uses token-set Jaccard similarity. The command retains a bounded number of row pairs and does not decide whether an overlap is acceptable for the task.

## `peft-doctor adapter-tensors`

Read exact metadata from local adapter safetensors without loading tensor data.

```bash
peft-doctor adapter-tensors ./adapter
peft-doctor adapter-tensors ./adapter/adapter_model.safetensors --output json
```

It reports files, bytes, tensor count, parameter count, dtype counts, and tensor-rank counts. Use `adapter-integrity` as well when hashes or offset validation matter.

## `peft-doctor chat-template-check`

Check a Jinja template or `tokenizer_config.json` and optionally compare it with sampled dataset roles.

```bash
peft-doctor chat-template-check chat-template.jinja --dataset chat.jsonl
peft-doctor chat-template-check tokenizer_config.json --dataset chat.jsonl --assistant-only-loss
```

It checks the `messages` variable, balanced generation blocks, and whether sampled role names are visible in the template. Role aliases can be valid, so unmatched names are warnings for review.

## `peft-doctor label-audit`

Inspect pretokenized rows containing `input_ids` and `labels`.

```bash
peft-doctor label-audit tokenized.jsonl
peft-doctor label-audit tokenized.jsonl --pad-token-id 0 --limit 5000
```

It reports length mismatches, all-ignored rows, fully supervised rows, unmasked pad positions, and the measured supervised-token ratio. Raw text datasets need to be tokenized before this check can see masks.

## `peft-doctor quantization-check`

Check a bitsandbytes or training config in JSON/YAML.

```bash
peft-doctor quantization-check bnb-config.json --gpu T4
peft-doctor quantization-check inference.yaml --inference --output markdown
```

It checks simultaneous 4-bit/8-bit loading, NF4 for training, explicit compute dtype, T4/bf16 plans, and FSDP storage/model dtype mismatches.

## `peft-doctor env`

Use this at the start of a Colab or fresh machine.

```bash
peft-doctor env
peft-doctor env --output json
peft-doctor env --output markdown
```

It reports:

- Python version
- Colab or local runtime
- CUDA availability
- GPU name and VRAM
- `torch`, `transformers`, `peft`, `accelerate`, `datasets`
- `bitsandbytes`, `trl`, `safetensors`, `sentencepiece`, `protobuf`

In Colab, run it immediately after install:

```python
%pip install -U "peft-doctor[ml]"
!peft-doctor env
```

## `peft-doctor colab`

Prints a safe Colab starter cell.

```bash
peft-doctor colab
```

Use it inside Colab:

```python
!peft-doctor colab
```

The output includes:

- install command
- environment check
- optional Colab Secrets login pattern for `HF_TOKEN`
- safe config helpers
- `create_training_recipe(...)`
- `diagnose_peft(...)` example

## `peft-doctor version`

Prints the installed version.

```bash
peft-doctor version
```

Use it in bug reports:

```bash
python -m pip show peft-doctor
peft-doctor version
```

## `peft-doctor investigate`

Connect local dataset, training script, and log evidence into possible causes:

```bash
peft-doctor investigate ./run --output markdown
```

It does not load a model or execute training. Row numbers and filenames in the report point to the evidence used.

## `peft-doctor inspect-objective`

Inspect stored text, chat turns, and label masks for one one-based row:

```bash
peft-doctor inspect-objective data.jsonl --row 42
peft-doctor inspect-objective ./run --row 42 --tokenizer ./tokenizer
```

The optional tokenizer must be local. For chat rows it renders that tokenizer's chat template. It does not run the trainer's collator, packing, truncation, or final loss-mask pipeline.

## `peft-doctor gradients`

Inspect a JSON gradient summary for zero/nonfinite gradients, zero updates, and optimizer omissions:

```bash
peft-doctor gradients gradient-summary.json
```

Use a JSON summary with `parameters` entries containing `trainable`, `grad_norm`, and optionally `update_norm` and `parameter_norm`; optionally include `optimizer_parameters`. Relative updates at or below `1e-8` are flagged as a screening heuristic. Pickle-based `.pt` and `.bin` files are not loaded.

Create one around a real optimizer step with `snapshot_trainable_parameters(model)` and `collect_gradient_snapshot(model, optimizer, before)`; see the [investigation guide](https://github.com/awais-akhtar/peft-doctor/blob/main/docs/intelligent-investigations.md#inspect-gradient-flow).

## `peft-doctor minimize`

Reduce a reproducible JSONL failure using an argument-array command from a JSON specification:

```bash
peft-doctor minimize failure.json --allow-execution --budget-runs 10
```

The spec must include `dataset`, `argv` with a `{dataset}` placeholder, and optionally `failure_exit_code` (default `1`). Review the command before allowing it. It runs with `shell=False`, writes a minimized dataset beside the spec, and is capped at 100 runs. The command can still perform actions available to your account; inspect its output for private data.

## `peft-doctor conflicts`

Find similar prompts with different answers for human review:

```bash
peft-doctor conflicts data.jsonl --threshold 0.82 --limit 5000
```

String similarity produces candidates, not proof of contradictory or incorrect answers.

## `peft-doctor coverage`

Compare dataset text to user-supplied task keywords in YAML/JSON:

```bash
peft-doctor coverage data.jsonl --tasks tasks.yaml
```

Use a `tasks` list with task `name` and `keywords`. Optional `languages` entries can supply marker keywords; `conversation_patterns` entries can name exact role sequences. The report also shows observed roles, role sequences, tool-call rows, and character-script hints. It does not identify languages from text or score task quality.

## `peft-doctor privacy`

Scan local data for common email, phone, token-shaped secret, and private-key patterns:

```bash
peft-doctor privacy data.jsonl --redacted-output data.redacted.jsonl
```

Redacted output must use `.json` or `.jsonl`. Pattern checks can miss sensitive values or redact benign text; review the output.

## `peft-doctor strategy`

Get rule-based guidance for prompting, retrieval, and fine-tuning from a YAML/JSON requirements file:

```bash
peft-doctor strategy requirements.yaml
```

Fields include `facts_change_often`, `consistent_behavior`, `representative_examples`, and `private_data`. The command does not predict model quality.

## `peft-doctor plan`

Compare a QLoRA VRAM estimate and required throughput with supplied budgets:

```bash
peft-doctor plan --model llama-3-8b --vram 16 --hours 4 --seq-len 2048 --batch-size 1 --gpu-count 2 --disk-gb 80 --required-disk-gb 60 --dataset-tokens 5000000 --measured-tokens-per-second 500
```

Supply `--params-billion` when the model size is not recognized. Disk size and throughput are only compared when supplied; measured speed should come from a comparable run. VRAM remains a formula estimate, and provider prices are not inferred.

## `peft-doctor compose-plan`

Compare local adapter configs before selecting a composition method:

```bash
peft-doctor compose-plan ./adapter-a ./adapter-b
```

This inspects metadata only; it does not merge or prove weight compatibility.

## `peft-doctor serving-check`

Review adapter metadata that may affect deployment in a named serving engine:

```bash
peft-doctor serving-check ./adapter --engine vllm
```

Static checks cannot prove runtime support. Confirm the engine version, base revision, tokenizer, and adapter features in a small deployment test.

## `peft-doctor next-experiment`

Review recorded local experiment history:

```bash
peft-doctor history ./my-run --add-status oom --note "Reduced sequence length"
peft-doctor next-experiment ./my-run
```

History is stored at `.peft-doctor/history.jsonl`. Add records with `peft-doctor history ./my-run --add-status oom --note "Reduced sequence length"`. The command cites available records and identifies missing comparison evidence rather than guessing a best hyperparameter.

## `peft-doctor investigate-distributed`

Scan a directory of per-rank logs for common distributed failures:

```bash
peft-doctor investigate-distributed ./rank-logs --output markdown
```

It flags OOM, NCCL/collective, timeout, uneven-input, and device errors. Line numbers across files are not a shared clock.

## `peft-doctor assistant`

Search local project evidence for a question and cite matching files and lines:

```bash
peft-doctor assistant ./my-run --question "Why did training stop?"
```

This local command does not call an LLM or upload files. It retrieves matching evidence from a bounded set of scripts, logs, JSON, and JSONL files.

## Python API Examples

Diagnose a live training setup:

```python
from peft_doctor import diagnose_peft

report = diagnose_peft(
    model=model,
    tokenizer=tokenizer,
    peft_config=peft_config,
    training_args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    sequence_length=2048,
    model_name="meta-llama/Llama-3-8B",
)

print(report.to_markdown())
```

Generate safe configs:

```python
from peft_doctor import (
    create_safe_lora_config,
    create_safe_bnb_config,
    create_safe_training_args,
    create_training_recipe,
)

peft_config = create_safe_lora_config(model_name="Qwen/Qwen2.5-7B")
bnb_config = create_safe_bnb_config()
training_args = create_safe_training_args()
recipe = create_training_recipe(kind="qlora-sft", model_family="qwen")
```

`create_safe_training_args()` includes conservative defaults for batch size, gradient accumulation, bf16, gradient checkpointing, warmup, cosine scheduler, gradient clipping, checkpoint retention, and seed.

Monitor logs:

```python
from peft_doctor import NanLossGuard

guard = NanLossGuard()
for log in trainer.state.log_history:
    for issue in guard.update(log):
        print(issue.title, issue.fix)
```
