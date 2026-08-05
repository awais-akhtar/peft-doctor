"""Run the packaged Mistral local LoRA recipe."""

from __future__ import annotations

from peft_doctor.recipes import run_project_recipe


if __name__ == "__main__":
    run_project_recipe("mistral-lora-local")
