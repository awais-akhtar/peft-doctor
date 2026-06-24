# Commands

This page is a practical command reference for PEFT Doctor.

## `peft-doctor check`

Runs the main pre-flight check. It combines model-family inference, target module recommendations, dataset format checks, memory checks, precision checks, and training-argument warnings.

```bash
peft-doctor check --model meta-llama/Llama-3-8B --dataset data.jsonl
```

With explicit training choices:

```bash
peft-doctor check \
  --model Qwen/Qwen2.5-7B \
  --dataset train.jsonl \
  --batch-size 1 \
  --grad-accum 8 \
  --sequence-length 2048 \
  --learning-rate 2e-4 \
  --load-in-4bit \
  --bf16 \
  --gradient-checkpointing
```

Output formats:

```bash
peft-doctor check --model gpt2 --dataset train.jsonl --output table
peft-doctor check --model gpt2 --dataset train.jsonl --output json
peft-doctor check --model gpt2 --dataset train.jsonl --output markdown
```

Use local Hugging Face cache only:

```bash
peft-doctor check --model meta-llama/Llama-3-8B --local-files-only
```

## `peft-doctor targets`

Prints LoRA target modules for a model id or family.

```bash
peft-doctor targets --model meta-llama/Llama-3-8B
peft-doctor targets --family llama
peft-doctor targets --family gpt2
peft-doctor targets --family falcon
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

Prints starter LoRA and QLoRA config code.

```bash
peft-doctor safe-config --model meta-llama/Llama-3-8B
```

Without QLoRA:

```bash
peft-doctor safe-config --family gpt2 --no-qlora
```

JSON:

```bash
peft-doctor safe-config --family llama --output json
```

## `peft-doctor inspect-dataset`

Checks whether sampled rows look usable for instruction tuning.

```bash
peft-doctor inspect-dataset data.jsonl
peft-doctor inspect-dataset data.csv --output markdown
```

Supported local files:

- `.jsonl`
- `.json`
- `.csv`
- `.txt`

Good dataset shapes include:

- `messages`: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`
- `instruction` plus `response`
- `prompt` plus `completion`
- `text` with clear instruction/response markers
- pre-tokenized `input_ids` plus `labels`

## `peft-doctor scan-log`

Scans logs for obvious training failures.

```bash
peft-doctor scan-log trainer_state.jsonl
peft-doctor scan-log run.log --output json
```

It looks for:

- `loss=nan`
- `loss=inf`
- CUDA out of memory
- loss overflow
- sharp loss jumps

## `peft-doctor adapter-check`

Checks a LoRA adapter merge plan without loading a large model.

```bash
peft-doctor adapter-check \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter your-user/your-lora-adapter \
  --output-dir merged-model
```

This command checks:

- local `adapter_config.json`
- local adapter weight files
- base model mismatch
- risky 4-bit or 8-bit merge plans
- existing output directory
- missing Hub repo id when pushing

## `peft-doctor merge-adapter`

Merges a PEFT LoRA adapter into its base model.

```bash
peft-doctor merge-adapter \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter your-user/your-lora-adapter \
  --output-dir merged-model \
  --dtype fp16
```

Dry run:

```bash
peft-doctor merge-adapter \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter your-user/your-lora-adapter \
  --output-dir merged-model \
  --dry-run
```

Push to the Hub:

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
- `--safe-serialization` for safetensors
- `--trust-remote-code` for models that require custom code
- `--overwrite` when saving into a non-empty directory

## `peft-doctor scan-notebook`

Scans a notebook for common PEFT and Colab mistakes.

```bash
peft-doctor scan-notebook model_merge.ipynb
peft-doctor scan-notebook model_merge.ipynb --output markdown
```

It checks for:

- hard-coded Hugging Face tokens
- command-line token login patterns in notebook cells
- old `transformers` or `peft` pins
- duplicate `pip install pip install`
- typo redirects such as `>>dev>null`
- adapter merge workflows that should use `peft-doctor merge-adapter`

## `peft-doctor env`

Checks the runtime before you start training.

```bash
peft-doctor env
peft-doctor env --output json
```

It reports:

- Python version
- Colab or local runtime
- CUDA availability
- GPU name and VRAM when Torch can see CUDA
- common package versions such as `torch`, `transformers`, `peft`, `accelerate`, `datasets`, and `bitsandbytes`

## `peft-doctor colab`

Prints a setup cell for Google Colab.

```bash
peft-doctor colab
```

## `peft-doctor version`

```bash
peft-doctor version
```
