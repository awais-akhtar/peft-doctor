# Packaged Validation Matrix

`peft-doctor benchmark` is a fixture check, not a timed GPU benchmark. It verifies that each generated recipe contains the required files and the training markers expected for its mode.

| Recipe | Dataset shape | Required mode | Fixture check |
| --- | --- | --- | --- |
| `llama3-qlora-colab` | conversational `messages` | NF4 QLoRA | required files, 4-bit config, checkpointing, cache disabled |
| `qwen2-qlora-colab` | conversational `messages` | NF4 QLoRA | required files, 4-bit config, checkpointing, cache disabled |
| `mistral-lora-local` | conversational `messages` | NF4 PEFT starter | required files, 4-bit config, checkpointing, cache disabled |
| `gemma-low-vram` | conversational `messages` | short-context NF4 QLoRA | required files, 4-bit config, checkpointing, cache disabled |
| `completion-only-sft` | `prompt` and `completion` | completion-only loss | base markers plus `completion_only_loss=True` |

Run a fixture check:

```bash
peft-doctor benchmark --recipe llama3-qlora-colab
peft-doctor benchmark --recipe completion-only-sft --output json
```

Validate files after copying a recipe:

```bash
peft-doctor recipe llama3-qlora-colab --copy ./my-run
peft-doctor validate-recipe ./my-run
```

Neither command starts training or measures wall-clock time, peak GPU memory, convergence, or output quality.
