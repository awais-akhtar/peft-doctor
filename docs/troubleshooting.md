# PEFT Troubleshooting Guide

This guide collects the problem phrases developers usually search for when a PEFT, LoRA, or QLoRA run breaks.

## CUDA Out Of Memory During QLoRA Training

Symptoms:

- `CUDA out of memory`
- training starts and crashes on the first batch
- evaluation crashes even though training works

Check the run:

```bash
peft-doctor check --model meta-llama/Llama-3-8B --dataset train.jsonl --batch-size 4
```

Common fixes:

```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
gradient_checkpointing = True
bf16 = True
```

For QLoRA, load in 4-bit:

```python
from peft_doctor import create_safe_bnb_config

bnb_config = create_safe_bnb_config()
```

## Loss Becomes NaN In LoRA Fine-Tuning

Symptoms:

- `loss=nan`
- `train_loss: nan`
- loss suddenly jumps and never recovers
- fp16 overflow warnings

Scan logs:

```bash
peft-doctor scan-log trainer_log.jsonl
```

Common fixes:

- lower the learning rate to `1e-4` or `5e-5`
- use `bf16=True` when the GPU supports it
- turn off fp16 if it is unstable
- remove empty or corrupted samples
- check labels and ignored label ids
- keep `max_grad_norm=1.0`

## Wrong LoRA Target Modules

Symptoms:

- model does not learn
- LoRA attaches to no useful layers
- target module names do not exist in the model

Get recommended targets:

```bash
peft-doctor targets --model meta-llama/Llama-3-8B
peft-doctor targets --model Qwen/Qwen2.5-7B
peft-doctor targets --family gpt2
```

Common target modules:

- Llama, Mistral, Qwen, Gemma, Phi: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- GPT-2: `c_attn`, `c_proj`, `c_fc`
- Falcon, Bloom, GPT-NeoX: `query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h`

## Tokenizer Has No Pad Token

Symptoms:

- tokenizer padding error
- batch collation fails
- `Using pad_token, but it is not set yet`

For causal LM fine-tuning, the common fix is:

```python
tokenizer.pad_token = tokenizer.eos_token
```

PEFT Doctor reports this during `diagnose_peft(...)` and `peft-doctor check`.

## PEFT Adapter Not Loading

Symptoms:

- `PeftModel.from_pretrained` fails
- adapter directory is missing files
- base model and adapter do not match

Check the adapter first:

```bash
peft-doctor adapter-check \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter your-user/your-lora-adapter \
  --output-dir merged-model
```

The adapter should include:

- `adapter_config.json`
- `adapter_model.safetensors` or `adapter_model.bin`

## Merge LoRA Adapter Into Base Model

Symptoms:

- merged model export fails
- `merge_and_unload()` fails
- pushed model only contains adapter weights

Use:

```bash
peft-doctor merge-adapter \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter your-user/your-lora-adapter \
  --output-dir merged-model \
  --dtype fp16
```

For final exports, load the base model in fp16, bf16, or fp32. Avoid 4-bit or 8-bit loading for the final merged model unless you know your PEFT and Transformers versions support that path.

## Google Colab PEFT Setup Problems

Symptoms:

- Torch does not see CUDA
- `bitsandbytes` import fails
- private model download fails
- notebook install cells keep breaking

Run:

```python
%pip install -U "peft-doctor[ml]"
!peft-doctor env
```

For private or gated Hugging Face models, use Colab Secrets with `HF_TOKEN`. Do not paste tokens into notebooks.
