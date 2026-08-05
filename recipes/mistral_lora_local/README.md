# Mistral LoRA Local

Local CUDA starter recipe for Mistral LoRA/QLoRA experiments.

```bash
python -m pip install -r requirements.txt
python train.py --dry-run --model mistralai/Mistral-7B-v0.1
```

For a reproducible model download, add `--revision MODEL_COMMIT_SHA`.
