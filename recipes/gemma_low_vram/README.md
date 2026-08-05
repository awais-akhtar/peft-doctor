# Gemma Low VRAM

Small-model recipe for fast low-VRAM sanity checks.

```bash
python -m pip install -r requirements.txt
python train.py --dry-run --model google/gemma-2-2b-it
```

For a reproducible model download, add `--revision MODEL_COMMIT_SHA`.
