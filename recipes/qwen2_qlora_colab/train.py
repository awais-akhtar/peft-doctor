"""Qwen2.5 QLoRA Colab recipe."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "Qwen/Qwen2.5-7B-Instruct"])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "outputs/qwen2_qlora_colab"])
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "llama3_qlora_colab" / "train.py"),
        run_name="__main__",
    )
