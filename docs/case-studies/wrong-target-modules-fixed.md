# Wrong Target Modules Repair Fixture

This fixture uses a causal-LM LoRA config that targets `lm_head` instead of the model-family projection layers.

```bash
peft-doctor fix --dry-run train.py --family llama
peft-doctor targets --family llama
```

Expected findings:

- output or embedding targets require manual review;
- the family profile supplies attention and MLP projection names;
- a patch can replace a literal target-module list.

```bash
peft-doctor fix --input train.py --output train.fixed.py --family llama
```

Load the actual model and inspect trainable parameter names before a full run. Family profiles are defaults, not proof that every derived architecture uses the same names.
