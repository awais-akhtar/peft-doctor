# Failure Gallery

Ten common PEFT failures and how PEFT Doctor fixes or explains them.

| Failure | Command | Fix |
| --- | --- | --- |
| CUDA out of memory | `peft-doctor estimate --model llama-3-8b --qlora --target-vram 16` | lower batch/sequence length, use QLoRA |
| `loss=nan` | `peft-doctor analyze-log trainer.log` | lower LR, prefer bf16, add gradient clipping |
| tokenizer has no pad token | `peft-doctor fix --dry-run train.py` | add `tokenizer.pad_token = tokenizer.eos_token` |
| `use_cache=True` with checkpointing | `peft-doctor fix --input train.py --output train.fixed.py` | set `model.config.use_cache = False` |
| fp16 and bf16 both enabled | `peft-doctor fix --config config.json --dry-run` | keep one precision mode |
| wrong target modules | `peft-doctor targets --family llama` | use attention/MLP projections |
| packed examples without EOS | `peft-doctor dataset-doctor data.jsonl --packing` | append EOS between examples |
| completion template mismatch | `peft-doctor dataset-doctor data.jsonl --response-template "### Response:"` | match formatted samples exactly |
| adapter missing config | `peft-doctor inspect-adapter ./adapter` | save with `model.save_pretrained()` |
| hard-coded notebook token | `peft-doctor notebook-check notebook.ipynb` | remove/revoke token and use `HF_TOKEN` |

