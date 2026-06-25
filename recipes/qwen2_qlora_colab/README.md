# Qwen2.5 QLoRA Colab

Colab starter recipe for Qwen2.5 instruct chat fine-tuning.

```bash
python -m pip install -r requirements.txt
python train.py --dry-run
```

For Qwen instruct models, review EOS handling. Many recipes use `<|im_end|>` so generation stops cleanly.

