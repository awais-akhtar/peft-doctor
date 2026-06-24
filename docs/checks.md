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
