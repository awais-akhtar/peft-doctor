# Adapter Merge

Many PEFT runs save only the LoRA adapter. That is good for training and sharing small artifacts, but deployment often needs a normal merged Transformers model.

PEFT Doctor wraps the standard merge pattern:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=...)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
```

## CLI

```bash
peft-doctor merge-adapter \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter your-user/your-lora-adapter \
  --output-dir merged-model \
  --dtype fp16
```

Check first:

```bash
peft-doctor adapter-check \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter your-user/your-lora-adapter \
  --output-dir merged-model
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

## Python

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

## Colab

```python
%pip install -U "peft-doctor[ml]"

!huggingface-cli login

!peft-doctor merge-adapter \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter your-user/your-lora-adapter \
  --output-dir merged-model \
  --dtype fp16
```

Use `bf16` on A100/L4/H100-class hardware when supported. Use `fp16` on T4. Use `fp32` if you are on CPU or debugging a strange merge.

## Common Problems

Base model mismatch:

```text
Use the exact base model from adapter_config.json when possible.
```

Adapter directory missing files:

```text
The adapter directory should contain adapter_config.json and adapter_model.safetensors or adapter_model.bin.
```

Quantized merge:

```text
4-bit and 8-bit loading are excellent for training/inference memory, but final merge exports are safest from fp16, bf16, or fp32.
```

Hard-coded tokens:

```text
Never paste Hugging Face tokens into notebooks. Use huggingface-cli login, Colab Secrets, or an HF_TOKEN environment variable.
```
