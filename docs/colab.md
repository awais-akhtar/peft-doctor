# Google Colab

Most LoRA and QLoRA experiments start in Colab, so PEFT Doctor keeps the notebook path simple.

## Setup Cell

```python
%pip install -U "peft-doctor[ml]"
!peft-doctor env
```

Before loading a model, choose a GPU runtime:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

For private or gated models, add your own Hugging Face token to Colab Secrets as
`HF_TOKEN`, then log in from the notebook without showing the token value:

```python
from google.colab import userdata
from huggingface_hub import login

login(token=userdata.get("HF_TOKEN"))
```

## Pre-Flight Cell

```python
from peft_doctor import diagnose_peft, create_safe_lora_config, create_safe_bnb_config

model_name = "meta-llama/Llama-3-8B"

peft_config = create_safe_lora_config(model_name=model_name)
bnb_config = create_safe_bnb_config()

training_args = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "bf16": True,
    "gradient_checkpointing": True,
    "load_in_4bit": True,
}

report = diagnose_peft(
    model=model,
    tokenizer=tokenizer,
    peft_config=peft_config,
    training_args=training_args,
    train_dataset=train_dataset,
    sequence_length=2048,
    model_name=model_name,
)

print(report.to_markdown())
```

## Common Colab Fixes

No GPU:

```text
Runtime -> Change runtime type -> GPU
```

Out of memory:

```python
training_args["per_device_train_batch_size"] = 1
training_args["gradient_accumulation_steps"] = 8
training_args["gradient_checkpointing"] = True
```

QLoRA loading:

```python
from peft_doctor import create_safe_bnb_config

bnb_config = create_safe_bnb_config()
```

Tokenizer padding:

```python
tokenizer.pad_token = tokenizer.eos_token
```

NaN loss:

```python
training_args["learning_rate"] = 1e-4
training_args["bf16"] = True
training_args["fp16"] = False
```

## Notebook Command Helper

You can print the setup snippet from the terminal or a notebook shell cell:

```python
!peft-doctor colab
```
