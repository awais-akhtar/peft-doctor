# PEFT Doctor Command Reference

This page explains every `peft-doctor` command, when to use it, and practical examples.

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
peft-doctor targets
peft-doctor safe-config
peft-doctor recipe
peft-doctor inspect-dataset
peft-doctor scan-log
peft-doctor scan-notebook
peft-doctor adapter-check
peft-doctor merge-adapter
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
- `--no-tokenizer` if you do not want tokenizer files saved or pushed
- `--overwrite` for non-empty output directories
- `--allow-quantized-merge` only if you know your stack supports it

For final model export, fp16, bf16, or fp32 loading is safer than 4-bit or 8-bit loading.

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
