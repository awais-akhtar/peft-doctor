# Completion-Only SFT

Prompt/completion recipe for response-only label masking.

```bash
python -m pip install -r requirements.txt
python train.py --dry-run
```

Every sample must include the exact `### Response:` marker.

