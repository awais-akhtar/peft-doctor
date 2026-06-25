# Before And After Auto-Fix Examples

## Broken Training Script

Before:

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
peft_config = LoraConfig(target_modules=["lm_head"])
training_args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=4,
    max_seq_length=8192,
    bf16=True,
    fp16=True,
    gradient_checkpointing=True,
)
```

Command:

```bash
peft-doctor fix --dry-run train.py --family llama
peft-doctor fix --input train.py --output train.fixed.py --family llama
```

After:

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.use_cache = False
peft_config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
training_args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=1,
    max_seq_length=2048,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="steps",
)
```

## Dataset Labels

Before:

```json
{"input_ids": [1, 2, 0], "labels": [1, 2, 0]}
```

Command:

```bash
peft-doctor fix --dataset data.jsonl --write --pad-token-id 0
```

After:

```json
{"input_ids": [1, 2, 0], "labels": [1, 2, -100]}
```

