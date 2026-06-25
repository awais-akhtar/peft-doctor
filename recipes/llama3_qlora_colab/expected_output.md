# Expected Output

`python train.py --dry-run` should print a PEFT Doctor report before training.

Expected checks:

- tokenizer pad token is set
- `model.config.use_cache = False`
- QLoRA quantized loading is detected
- LoRA target modules are configured
- no hard-coded token is needed

