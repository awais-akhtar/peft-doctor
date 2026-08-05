# Environment Notes

This recipe loads Mistral in a floating-point dtype and therefore needs more VRAM than the QLoRA recipes. The file is not a recorded performance benchmark.

- local NVIDIA CUDA runtime
- native bf16 preferred; fp16 is selected otherwise
- enough VRAM for the base model plus LoRA training state
- current package minimums from `peft-doctor env`
