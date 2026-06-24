"""Ready-to-use fine-tuning recipes."""

from __future__ import annotations

from typing import Any, Optional

from .configs import create_safe_bnb_config, create_safe_lora_config, create_safe_training_args


RECIPE_NAMES = [
    "qlora-sft",
    "low-vram-colab",
    "completion-only",
    "long-context",
    "distributed-qlora",
    "moe-lora",
    "adapter-merge",
]


def create_training_recipe(
    kind: str = "qlora-sft",
    model_family: Optional[str] = None,
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    """Return a practical PEFT fine-tuning recipe as plain Python data."""

    recipe = kind.strip().lower().replace("_", "-")
    if recipe not in RECIPE_NAMES:
        known = ", ".join(RECIPE_NAMES)
        raise ValueError(f"Unknown recipe `{kind}`. Choose one of: {known}.")

    lora = create_safe_lora_config(
        model_name=model_name,
        model_family=model_family,
        as_dict=True,
    )
    bnb = create_safe_bnb_config(as_dict=True)
    training = create_safe_training_args()
    model_kwargs: dict[str, Any] = {
        "quantization_config": "bnb_config",
        "device_map": "auto",
        "torch_dtype": "torch.bfloat16",
    }
    notes = [
        "Run `peft-doctor check` before starting the full training job.",
        "Keep the first run small, confirm loss moves, then increase sequence length or dataset size.",
    ]

    if recipe == "low-vram-colab":
        training.update(
            {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "learning_rate": 1e-4,
                "eval_strategy": "no",
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
            }
        )
        notes.extend(
            [
                "Use a GPU runtime before loading the model.",
                "Start with max_seq_length=1024 or 2048 on T4/L4 GPUs.",
                "Avoid evaluation during the first memory-debugging run.",
            ]
        )

    elif recipe == "completion-only":
        training.update(
            {
                "completion_only_loss": True,
                "response_template": "### Response:",
                "remove_unused_columns": False,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
            }
        )
        notes.extend(
            [
                "Make sure every formatted sample contains the exact response template.",
                "If every label becomes -100, the response template did not match the formatted text.",
            ]
        )

    elif recipe == "long-context":
        training.update(
            {
                "gradient_accumulation_steps": 16,
                "learning_rate": 1e-4,
                "max_seq_length": 4096,
                "group_by_length": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
            }
        )
        model_kwargs["attn_implementation"] = "flash_attention_2"
        notes.extend(
            [
                "Verify the model actually supports the chosen context length.",
                "Use Flash Attention only when your GPU, dtype, and model stack support it.",
            ]
        )

    elif recipe == "distributed-qlora":
        model_kwargs["device_map"] = None
        training.update(
            {
                "ddp_find_unused_parameters": False,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "optim": "paged_adamw_8bit",
            }
        )
        notes.extend(
            [
                "Do not combine device_map='auto' with torchrun or multi-process Accelerate.",
                "Let each process own its local GPU.",
            ]
        )

    elif recipe == "moe-lora":
        lora["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora["target_parameters"] = [
            "feed_forward.experts.gate_up_proj",
            "feed_forward.experts.down_proj",
        ]
        training.update({"gradient_checkpointing_kwargs": {"use_reentrant": False}})
        notes.extend(
            [
                "MoE models often need target_parameters for expert weights instead of only target_modules.",
                "Inspect the model parameter names and adjust target_parameters to match the architecture.",
            ]
        )

    elif recipe == "adapter-merge":
        return {
            "recipe": recipe,
            "description": "Safe LoRA adapter merge and export checklist.",
            "commands": [
                "peft-doctor adapter-check --base-model BASE --adapter ADAPTER --output-dir merged-model",
                "peft-doctor merge-adapter --base-model BASE --adapter ADAPTER --output-dir merged-model --dtype fp16",
            ],
            "checks": [
                "Confirm adapter_config.json exists.",
                "Use the same base model that the adapter was trained from.",
                "Merge from fp16, bf16, or fp32 when possible instead of k-bit weights.",
                "Save tokenizer files with the merged model.",
                "Use HF_TOKEN, huggingface-cli login, or Colab Secrets for Hub pushes.",
            ],
        }

    else:
        training.update(
            {
                "optim": "paged_adamw_8bit",
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
            }
        )
        notes.extend(
            [
                "Use target_modules='all-linear' only when you understand the memory tradeoff.",
                "For most Llama/Qwen/Mistral runs, attention plus MLP projection targets are a stable start.",
            ]
        )

    return {
        "recipe": recipe,
        "description": "Practical PEFT/QLoRA fine-tuning starter setup.",
        "install": 'python -m pip install -U "peft-doctor[ml]"',
        "model_kwargs": model_kwargs,
        "lora_config": lora,
        "bnb_config": bnb,
        "training_args": training,
        "notes": notes,
    }
