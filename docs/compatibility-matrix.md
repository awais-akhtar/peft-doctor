# Package Requirements And Compatibility Notes

PEFT Doctor keeps the core CLI lightweight. The full fine-tuning stack is installed with:

```bash
python -m pip install "peft-doctor[ml]"
```

| Component | Minimum in package extras | Notes |
| --- | --- | --- |
| Python | 3.9 | Declared package minimum; CI runs on Python 3.9 through 3.12 |
| torch | 2.4 | CUDA support depends on your install and GPU |
| transformers | 4.56 | Required for model/tokenizer loading checks |
| peft | 0.17 | Covers trainable tokens and the LoRA fields checked in this release |
| accelerate | 1.10 | Covers the FSDP configuration fields checked in this release |
| bitsandbytes | 0.47 | Optional-extra minimum; platform and CUDA compatibility depend on its installation requirements. |
| datasets | 3.0 | Used by recipe train scripts |
| trl | 0.23 | Recipe API uses `SFTConfig(max_length=...)` and `processing_class` |
| safetensors | 0.4 | Used for adapter metadata and tensor inspection where applicable |
| PyYAML | 6.0 | Reads Accelerate and training YAML configs for operational checks |

Operational file support:

| Check | Supported local inputs | Network required |
| --- | --- | --- |
| Checkpoint/resume | Trainer or Accelerate checkpoint directories; JSON/YAML config comparison | No |
| Reproducibility manifest | Local project files and installed package metadata | No |
| Token lengths | JSON, JSONL, CSV, TXT; optional Transformers tokenizer | Only when downloading an uncached tokenizer |
| Distributed config | Accelerate YAML and DeepSpeed JSON | No |
| Generation config | `generation_config.json` or local model directory | No |
| Adapter integrity | PEFT adapter directory; safetensors or `.bin` weights | No |
| TRL SFT | JSON/YAML SFTConfig, local dataset, optional Jinja template | No |
| FSDP | Accelerate JSON/YAML, optional Trainer settings | No |
| LoRA hotswap/config | Local adapter config files | No |
| Leakage/labels | Local JSON, JSONL, CSV, TXT, or pretokenized rows | No |
| Adapter tensors | Local safetensors header metadata | No |

The checks follow the public contracts documented by [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), [Accelerate checkpointing](https://huggingface.co/docs/accelerate/main/usage_guides/checkpoint), [Accelerate DeepSpeed](https://huggingface.co/docs/accelerate/main/usage_guides/deepspeed), [PEFT checkpoints](https://huggingface.co/docs/peft/main/developer_guides/checkpoint), [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer), and [safetensors metadata parsing](https://huggingface.co/docs/safetensors/metadata_parsing). Exact behavior can still vary by installed version, model architecture, and trainer implementation.

Example starting points (not a benchmark or a guarantee that a run fits):

| GPU | Example recipe to evaluate | Notes |
| --- | --- | --- |
| T4 16 GB | `llama3-qlora-colab`, `gemma-low-vram` | Begin with a small batch and sequence length; use `estimate` before a run. |
| L4 24 GB | `qwen2-qlora-colab` | Confirm availability and memory headroom in the selected runtime. |
| A100 40/80 GB | `mistral-lora-local`, long context recipes | Estimate the actual model and sequence configuration before training. |
| Local CPU | none for real training | Use only tiny smoke tests |
