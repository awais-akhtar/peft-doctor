# Compatibility Matrix

PEFT Doctor keeps the core CLI lightweight. The full fine-tuning stack is installed with:

```bash
python -m pip install "peft-doctor[ml]"
```

| Component | Minimum in package extras | Notes |
| --- | --- | --- |
| Python | 3.9 | Tested by CI across supported Python versions where available |
| torch | 2.1 | CUDA support depends on your install and GPU |
| transformers | 4.40 | Required for model/tokenizer loading checks |
| peft | 0.10 | Required for real `LoraConfig` objects and adapter loading |
| accelerate | 0.28 | Used by common PEFT/QLoRA training scripts |
| bitsandbytes | 0.43 | Linux/Colab recommended for QLoRA |
| datasets | 2.18 | Used by recipe train scripts |
| trl | 0.8 | Used by SFT recipe train scripts |
| safetensors | 0.4 | Recommended for adapter and merged model files |

GPU guidance:

| GPU | Good first recipe | Notes |
| --- | --- | --- |
| T4 16 GB | `llama3-qlora-colab`, `gemma-low-vram` | Keep batch size 1 and sequence length 1024-2048 |
| L4 24 GB | `qwen2-qlora-colab` | Good Colab Pro target |
| A100 40/80 GB | `mistral-lora-local`, long context recipes | Can test larger sequence lengths |
| Local CPU | none for real training | Use only tiny smoke tests |

