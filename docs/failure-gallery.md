# Failure Gallery

Examples of common PEFT failure symptoms, the command that can help investigate them, and a typical follow-up. Most recommendations are diagnostic; review changes before applying them.

| Failure | Command | Typical follow-up |
| --- | --- | --- |
| CUDA out of memory | `peft-doctor estimate --model llama-3-8b --qlora --target-vram 16` | Review the estimate; reduce per-device batch size or sequence length and consider quantization. |
| `loss=nan` | `peft-doctor analyze-log trainer.log` | Review reported log patterns, then check precision, learning rate, and data. |
| tokenizer has no pad token | `peft-doctor fix --input train.py --dry-run` | Review the proposed `tokenizer.pad_token = tokenizer.eos_token` patch. |
| `use_cache=True` with checkpointing | `peft-doctor fix --input train.py --output train.fixed.py` | Review the generated patch and confirm it matches the model-loading code. |
| fp16 and bf16 both enabled | `peft-doctor fix --config config.json --dry-run` | Review the patch and select a precision supported by the hardware and model. |
| wrong target modules | `peft-doctor targets --family llama` | Compare the suggested names with modules exposed by the selected model. |
| packed examples without EOS | `peft-doctor dataset-doctor data.jsonl --packing` | Check the report and confirm that the dataset packing format separates examples as intended. |
| completion template mismatch | `peft-doctor dataset-doctor data.jsonl --response-template "### Response:"` | Confirm the response marker matches the formatting in the training examples. |
| adapter missing config | `peft-doctor inspect-adapter ./adapter` | Check the report; adapters are normally saved with the PEFT model's `save_pretrained()` method. |
| hard-coded notebook token | `peft-doctor notebook-check notebook.ipynb` | Remove the credential from the notebook and revoke it if it was exposed; use an environment secret instead. |

