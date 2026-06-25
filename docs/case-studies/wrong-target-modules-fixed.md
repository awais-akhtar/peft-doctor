# Wrong Target Modules Fixed

Problem: a LoRA config targeted `lm_head`, causing unnecessary memory risk and poor adapter behavior for a normal causal LM SFT run.

Command:

```bash
peft-doctor fix --dry-run train.py --family llama
```

What PEFT Doctor caught:

- risky output-head or embedding targets
- missing practical projection targets

Safe fix:

```bash
peft-doctor fix --input train.py --output train.fixed.py --family llama
```

Result: the fixed script uses attention and MLP projection targets such as `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
