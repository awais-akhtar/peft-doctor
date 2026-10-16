# CUDA OOM Repair Fixture

This is a reproducible static-check example. It is not a recorded GPU benchmark.

The broken script uses a large per-device batch, long sequence length, gradient checkpointing with cache still enabled, and no retention limit.

```bash
peft-doctor fix --input train.py --dry-run
peft-doctor estimate --model llama-3-8b --seq-len 8192 --batch-size 4 --qlora --target-vram 16
```

Expected findings:

- high batch and sequence literals are repair candidates;
- `model.config.use_cache = False` is missing;
- checkpoint retention needs review;
- the VRAM formula exceeds or approaches the supplied budget.

Write a separate patch, inspect it, then run a short real training smoke test:

```bash
peft-doctor fix --input train.py --output train.fixed.py
```
