# CUDA OOM Fixed Before Training

Problem: a Llama-3-8B QLoRA run was configured with a large batch and long sequence length on a T4-style GPU.

Command:

```bash
peft-doctor fix --dry-run train.py
```

What PEFT Doctor caught:

- train batch size above a safe QLoRA starting point
- long sequence length
- missing warmup and logging settings
- missing `model.config.use_cache = False`

Safe fix:

```bash
peft-doctor fix --input train.py --output train.fixed.py
```

Result: the patched script starts with batch size 1, conservative sequence length, gradient checkpointing-safe cache settings, and frequent logging.

