# NaN Loss Fixed

Problem: a run used both `bf16=True` and `fp16=True`, had no warmup, and had sparse logging.

Commands:

```bash
peft-doctor scan-log run.log
peft-doctor fix --dry-run train.py
peft-doctor fix --input train.py --output train.fixed.py
```

What PEFT Doctor caught:

- `loss=nan`
- invalid precision configuration
- missing warmup
- missing frequent logging

Safe fix: keep bf16, turn fp16 off, add `warmup_ratio=0.03`, and set `logging_steps=10`.

Result: the user can restart with earlier warning signals and safer precision defaults.

