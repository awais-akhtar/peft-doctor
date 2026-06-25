"""Mistral local LoRA recipe."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "mistralai/Mistral-7B-v0.1"])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "outputs/mistral_lora_local"])
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "llama3_qlora_colab" / "train.py"),
        run_name="__main__",
    )
