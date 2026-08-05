# Checks

PEFT Doctor is built around practical checks that match the way LoRA and QLoRA runs usually fail.

## Memory

The memory checker looks at:

- CUDA availability
- GPU VRAM size
- train and eval batch size
- gradient checkpointing
- sequence length
- quantization flags
- rough loaded model size when a model object is available

Common fixes:

- `load_in_4bit=True`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_use_double_quant=True`
- `per_device_train_batch_size=1`
- `gradient_checkpointing=True`
- shorter sequence length while debugging

## Target Modules

Wrong target modules are one of the easiest ways to waste a run. PEFT Doctor can infer a family from the model object, config, or model id and suggest target modules.

Examples:

- Llama, Mistral, Qwen, Gemma, Phi: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- GPT-2 style: `c_attn`, `c_proj`, `c_fc`
- Falcon, Bloom, GPT-NeoX: `query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h`
- T5 style: `q`, `k`, `v`, `o`, `wi`, `wo`

## Dataset Format

The dataset checker samples local rows or an in-memory dataset and looks for:

- chat `messages`
- instruction/response columns
- prompt/completion columns
- marked text templates
- pre-tokenized `input_ids` and `labels`
- empty rows

If the format is unclear, the usual fix is to make the training text explicit:

```text
### Instruction:
Write a short answer.

### Response:
Here is the answer.
```

For chat models, prefer the tokenizer chat template:

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
```

## Learning Rate

LoRA often starts around `2e-4`, `1e-4`, or `5e-5`. If loss spikes, repeats, or becomes NaN, lower it.

Full fine-tuning usually needs a much smaller learning rate than LoRA.

## Tokenizer Padding

Many causal LM tokenizers do not ship with a pad token. If batching fails, a common causal-LM fix is:

```python
tokenizer.pad_token = tokenizer.eos_token
```

## NaN Loss

When loss becomes NaN, check these first:

- learning rate too high
- fp16 overflow
- empty samples
- corrupted labels
- labels set incorrectly
- very long samples
- missing gradient clipping

The log scanner and `NanLossGuard` point to the same short list of fixes.

## Advanced Data Checks

PEFT Doctor also checks for problems that make a run look successful while silently teaching the wrong thing:

- duplicate samples
- train/eval overlap
- empty or tiny response fields
- chat rows without assistant replies
- empty assistant messages
- unknown chat roles
- mixed chat and instruction schemas
- assistant-only loss without chat rows
- completion-only loss without a clear response boundary
- tool-call rows without tool schemas
- image or video rows that need VLM processors/collators
- pre-tokenized labels that are all `-100`
- `input_ids` and `labels` length mismatches
- padding tokens left in labels
- packed data without visible EOS separators
- completion template strings missing from sampled text
- text rows likely to exceed the configured sequence length

## Model State Checks

When a model object is available in Python, `diagnose_peft(...)` checks:

- `use_cache=True` with gradient checkpointing
- tokenizer size larger than model embeddings
- trainable parameter counts and ratios
- zero trainable parameters after applying PEFT
- unusually high trainable ratio for adapter training
- long-context runs that may benefit from Flash Attention
- sequence length beyond a sliding attention window
- MoE models that may need `target_parameters` for expert weights

## Trainer Config Checks

The training argument checker warns about:

- missing warmup
- missing scheduler
- missing seed
- missing checkpoint retention limit
- QLoRA runs without a paged or 8-bit optimizer
- dataloader workers set to zero
- `max_steps` overriding `num_train_epochs`
- both 4-bit and 8-bit loading enabled
- both `bf16` and `fp16` enabled
- DDP `find_unused_parameters` left enabled for LoRA
- Flash Attention without bf16/fp16
- missing or very sparse logging
- completion-only loss without a response template

## Advanced Distributed And Export Checks

PEFT Doctor now also warns about:

- `device_map="auto"` with DDP, Accelerate multi-process, or torchrun
- FSDP plus 4-bit or 8-bit quantized loading
- DeepSpeed ZeRO plus QLoRA recipes that need version-specific setup
- `torch_compile` with k-bit loading
- `torch_compile` with gradient checkpointing
- reentrant gradient checkpointing with QLoRA
- sequence length larger than the model context window
- RoPE scaling that should match the base model recipe
- tokenizer growth without `modules_to_save` for embeddings or `lm_head`
- LoRA target modules that accidentally include `lm_head` or embedding layers
- `inference_mode=True` in a training config
- `init_lora_weights=False` for a fresh adapter
- `all-linear` target shortcut handling
- rsLoRA, LoftQ, DoRA, and tied-weight export hints

## Runtime Log Checks

`peft-doctor scan-log` detects:

- `loss=nan` or `loss=inf`
- sudden loss jumps
- overflow messages
- CUDA out-of-memory
- CUDA illegal memory access
- disk-full or quota errors
- tensor device mismatch
- tensor shape mismatch
- sequence length above the model limit
- invalid or spiking gradient norms

## Auto-Repair Checks

`peft-doctor fix` can safely patch common repeat issues:

- tokenizer pad token fallback
- `use_cache=False` for gradient checkpointing
- `fp16` and `bf16` conflict
- risky LoRA target modules
- high-risk starter batch size or sequence length
- missing warmup, logging, and save strategy
- pad token ids inside dataset labels

Always run dry-run first:

```bash
peft-doctor fix --dry-run train.py
```
