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

## More Advanced Problems PEFT Doctor Checks

These are common problems that often show up after the first training run works.

1. Fully masked labels:

```text
Problem: labels are all -100, so the model learns nothing.
Fix: check the data collator and completion-only masking.
```

2. Label/input length mismatch:

```text
Problem: input_ids and labels have different lengths.
Fix: tokenize, truncate, pad, and mask labels in one consistent step.
```

3. Duplicate samples:

```text
Problem: repeated rows make the adapter memorize examples.
Fix: deduplicate train data before tokenization.
```

4. Train/eval leakage:

```text
Problem: eval rows also appear in train data.
Fix: split first, then deduplicate each split and remove overlap.
```

5. Long samples are truncated:

```text
Problem: prompt or answer tokens are cut off by max sequence length.
Fix: shorten examples, raise sequence_length, or preserve answer tokens during truncation.
```

6. `use_cache=True` with gradient checkpointing:

```text
Problem: extra memory use and Trainer warnings.
Fix: set model.config.use_cache = False before training.
```

7. Tokenizer larger than model embeddings:

```text
Problem: new special tokens were added but embeddings were not resized.
Fix: call model.resize_token_embeddings(len(tokenizer)).
```

8. No trainable LoRA parameters:

```text
Problem: adapters are not active or the model is frozen incorrectly.
Fix: call model.print_trainable_parameters() and verify LoRA modules are attached.
```

9. Missing warmup or scheduler:

```text
Problem: early loss spikes or unstable learning.
Fix: try warmup_ratio=0.03 and lr_scheduler_type="cosine" or "linear".
```

10. Checkpoints fill the disk:

```text
Problem: save_steps is set but save_total_limit is missing.
Fix: set save_total_limit=2 or save_total_limit=3.
```

Run the combined check:

```bash
peft-doctor check \
  --model meta-llama/Llama-3-8B \
  --dataset train.jsonl \
  --eval-dataset eval.jsonl \
  --load-in-4bit \
  --optim paged_adamw_8bit \
  --warmup-ratio 0.03 \
  --lr-scheduler-type cosine \
  --save-total-limit 2 \
  --seed 42
```

## Twelve Complicated Fine-Tuning Problems And Fixes

1. `device_map="auto"` with DDP:

```text
Problem: each distributed process may try to shard the model across all GPUs.
Fix: avoid device_map="auto" in torchrun/DDP; let each process own one GPU.
```

2. FSDP with QLoRA:

```text
Problem: FSDP wrapping and k-bit parameters can conflict.
Fix: start with single-process QLoRA or use a recipe known to support FSDP + PEFT.
```

3. DeepSpeed with QLoRA:

```text
Problem: ZeRO stage, bitsandbytes optimizer, and adapter parameters need compatible placement.
Fix: verify your DeepSpeed stage and PEFT recipe before long runs.
```

4. `torch_compile` with 4-bit or 8-bit loading:

```text
Problem: compile support varies across quantized modules.
Fix: disable torch_compile until the run is stable.
```

5. Reentrant gradient checkpointing:

```text
Problem: QLoRA can be more fragile with reentrant checkpointing.
Fix: use gradient_checkpointing_kwargs={"use_reentrant": False}.
```

6. Sequence length exceeds model context:

```text
Problem: sequence_length is larger than max_position_embeddings.
Fix: lower sequence length or use a model-supported RoPE/context extension.
```

7. RoPE scaling mismatch:

```text
Problem: long-context config does not match the model recipe.
Fix: verify rope_scaling and rope_theta before training.
```

8. Added special tokens not saved:

```text
Problem: tokenizer grows, but adapter export does not save embeddings.
Fix: resize embeddings and save embed_tokens and often lm_head when needed.
```

9. Completion template mismatch:

```text
Problem: response_template is not found, so labels can become fully masked.
Fix: make the template string exactly match formatted samples.
```

10. Packed examples without EOS:

```text
Problem: examples bleed into each other during packed SFT.
Fix: append tokenizer.eos_token to every formatted sample before packing.
```

11. Pad token trained as a label:

```text
Problem: labels contain pad_token_id.
Fix: mask padding labels to -100.
```

12. Training config accidentally in inference mode:

```text
Problem: inference_mode=True prevents adapter training.
Fix: create a training LoRA config with inference_mode=False.
```

Example command:

```bash
peft-doctor check \
  --model meta-llama/Llama-3-8B \
  --dataset train.jsonl \
  --eval-dataset eval.jsonl \
  --load-in-4bit \
  --device-map auto \
  --world-size 2 \
  --torch-compile \
  --packing \
  --completion-only-loss \
  --assistant-only-loss \
  --response-template "### Response:" \
  --attn-implementation flash_attention_2 \
  --ddp-find-unused-parameters \
  --gradient-checkpointing-use-reentrant \
  --optim adamw_torch
```

## Next-Level Problems PEFT Doctor Now Checks

1. Completion-only loss without a response marker:

```text
Problem: completion_only_loss is enabled, but no response_template is configured.
Fix: pass the exact marker used in formatted samples, such as "### Response:".
```

2. Assistant-only loss with the wrong chat template:

```text
Problem: assistant_only_loss needs assistant generation spans in the chat template.
Fix: use a template with generation blocks, or use completion_only_loss with prompt/response text.
```

3. Pad token equals EOS during masked-loss training:

```text
Problem: a collator can accidentally mask EOS labels when pad_token_id equals eos_token_id.
Fix: verify masking by position, or add a dedicated pad token when needed.
```

4. Qwen instruct generation does not stop:

```text
Problem: EOS does not match the instruct chat template.
Fix: review the tokenizer EOS and use <|im_end|> when the recipe requires it.
```

5. Tool-call rows without tool schemas:

```text
Problem: tool-call examples lose meaning when tools/tool_schema is missing.
Fix: keep tool schemas with rows or render tool calls into plain text.
```

6. Vision-language rows handled like text-only rows:

```text
Problem: image/video fields need a VLM processor and collator.
Fix: verify image tokens are inserted before PEFT training.
```

7. MoE expert weights not targeted:

```text
Problem: some MoE expert weights are parameters, not ordinary modules.
Fix: inspect parameter names and configure target_parameters.
```

8. High-rank LoRA instability:

```text
Problem: large ranks can be harder to stabilize.
Fix: compare against use_rslora=True for r=64 or higher.
```

9. QLoRA quality needs a better initialization:

```text
Problem: default LoRA initialization may underperform for some 4-bit runs.
Fix: compare a short run with init_lora_weights="loftq".
```

10. DoRA with QLoRA memory pressure:

```text
Problem: DoRA adds magnitude parameters.
Fix: benchmark a short run before using DoRA in long k-bit training.
```

11. Runtime crash after training starts:

```text
Problem: logs show device mismatch, shape mismatch, disk full, illegal CUDA memory access, or grad_norm spikes.
Fix: run peft-doctor scan-log run.log and apply the reported fix before restarting.
```

12. No ready baseline for a new project:

```text
Problem: too many config choices before the first stable run.
Fix: start with peft-doctor recipe --kind qlora-sft, low-vram-colab, completion-only, long-context, distributed-qlora, or moe-lora.
```
