"""Gemma low-VRAM recipe."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "google/gemma-2-2b-it"])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "outputs/gemma_low_vram"])
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "llama3_qlora_colab" / "train.py"),
        run_name="__main__",
    )
