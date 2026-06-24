"""Notebook checks for common PEFT/Colab mistakes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Union

from .report import DiagnosisReport


HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")


def _code_cells(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    code = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            code.append("".join(source))
        elif isinstance(source, str):
            code.append(source)
    return code


def scan_notebook(path: Union[str, Path]) -> DiagnosisReport:
    """Scan a notebook for PEFT setup and adapter-merge anti-patterns."""

    notebook = Path(path)
    report = DiagnosisReport(metadata={"notebook": str(notebook)})

    try:
        cells = _code_cells(notebook)
    except Exception as exc:
        report.add(
            "notebook.read_failed",
            "Notebook could not be read",
            "error",
            f"Could not read the notebook: {exc}",
            "Make sure the file is a valid .ipynb notebook.",
        )
        return report

    joined = "\n".join(cells)
    token_count = len(HF_TOKEN_RE.findall(joined))
    if token_count:
        report.add(
            "notebook.hf_token",
            "Hard-coded Hugging Face token found",
            "error",
            "The notebook contains one or more Hugging Face access tokens.",
            (
                "Revoke the exposed token, remove it from the notebook, and use "
                "Colab Secrets, `HF_TOKEN`, or interactive `huggingface-cli login`."
            ),
            count=token_count,
        )

    if "huggingface-cli login --token" in joined:
        report.add(
            "notebook.cli_token_login",
            "Token passed on the command line",
            "warning",
            (
                "Passing tokens in notebook shell commands can leak credentials through "
                "history, outputs, or shared notebooks."
            ),
            (
                "Use interactive `huggingface-cli login`, Colab Secrets, or the "
                "`HF_TOKEN` environment variable."
            ),
        )

    if "pip install pip install" in joined:
        report.add(
            "notebook.duplicate_pip",
            "Duplicate pip install command",
            "warning",
            "The notebook contains `pip install pip install`, which is usually a copy/paste mistake.",
            "Use one clean setup cell such as `%pip install -U \"peft-doctor[ml]\"`.",
        )

    if ">>dev>null" in joined or ">> dev>null" in joined:
        report.add(
            "notebook.redirect_typo",
            "Shell redirect typo found",
            "warning",
            "`>>dev>null` is not the same as redirecting output to `/dev/null`.",
            "In notebooks, prefer `%pip install -q ...` instead of shell redirects.",
        )

    old_pins = []
    for package in ("transformers", "peft"):
        match = re.search(rf"{package}==([0-9][^\s]+)", joined)
        if match:
            old_pins.append(f"{package}=={match.group(1)}")
    if old_pins:
        report.add(
            "notebook.old_pins",
            "Old fine-tuning package pins found",
            "warning",
            "The notebook pins older fine-tuning packages.",
            "Use current `transformers`, `peft`, and `accelerate` unless you are reproducing an old run.",
            packages=", ".join(old_pins),
        )

    if "merge_and_unload" in joined:
        report.add(
            "notebook.merge_flow",
            "Adapter merge flow detected",
            "ok",
            "The notebook uses the standard `PeftModel.from_pretrained(...).merge_and_unload()` flow.",
            "You can replace the manual cells with `peft-doctor merge-adapter` for a repeatable export.",
        )

    if "merge_and_unload" in joined and ("load_in_8bit=True" in joined or "load_in_4bit=True" in joined):
        report.add(
            "notebook.quantized_merge",
            "Quantized load appears near merge workflow",
            "warning",
            "Quantized loading is useful for inference, but final adapter merging is safest with fp16, bf16, or fp32.",
            "Load the base model without 4-bit or 8-bit quantization when exporting the merged model.",
        )

    if not report.issues:
        report.add(
            "notebook.clean",
            "No obvious notebook issues found",
            "ok",
            "No Hugging Face tokens, brittle install cells, or adapter merge anti-patterns were detected.",
        )

    return report
