# Environment Notes

This recipe uses prompt/completion rows and TRL completion-only loss. It is not a recorded GPU benchmark.

- Python 3.10 or 3.11
- CUDA runtime with enough VRAM for the selected model
- TRL 0.23 or newer
- current package minimums from `peft-doctor env`

Run `peft-doctor sft-check` on a saved config when changing the loss or packing mode.
