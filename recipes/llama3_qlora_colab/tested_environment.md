# Environment Notes

This is the intended starting environment, not a recorded training benchmark.

- Python 3.10 or 3.11
- CUDA runtime with enough VRAM for the selected model
- T4 uses fp16; hardware with native bf16 uses bf16
- current package minimums from `peft-doctor env`

Run `peft-doctor env` and `python train.py --dry-run` before model loading.
