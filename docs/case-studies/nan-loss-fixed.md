# NaN Loss Repair Fixture

This example verifies log and configuration rules. It does not claim that changing precision alone guarantees stable training.

Fixture inputs:

- a log containing `loss=nan`;
- a script with both `bf16=True` and `fp16=True`;
- no warmup and sparse logging.

```bash
peft-doctor analyze-log run.log
peft-doctor fix --input train.py --dry-run
peft-doctor fix --input train.py --output train.fixed.py
```

Expected findings include invalid loss, conflicting precision, and missing warmup/logging settings. After the patch, inspect data labels and run a short smoke test because NaN loss can also come from bad batches, optimizer state, or kernels.
