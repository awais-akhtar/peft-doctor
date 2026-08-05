# Completion-Only SFT

Prompt/completion recipe for response-only label masking.

```bash
python -m pip install -r requirements.txt
python train.py --dry-run
```

For a reproducible model download, add `--revision MODEL_COMMIT_SHA`.

Every sample must include the exact `### Response:` marker.
