# Environment Notes

This is a short-context QLoRA starting point, not a guarantee that every runtime will fit.

- Python 3.10 or 3.11
- CUDA runtime
- sequence length 1024 and train batch size 1
- current package minimums from `peft-doctor env`

Run the VRAM estimate with the actual model and GPU budget before loading weights.
