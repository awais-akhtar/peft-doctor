"""Environment checks for Colab and local fine-tuning machines."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Optional

from packaging.version import InvalidVersion, Version

from .report import DiagnosisReport


CORE_PACKAGES = {
    "torch": "2.1",
    "transformers": "4.40",
    "peft": "0.10",
    "accelerate": "0.28",
    "datasets": "2.18",
    "rich": "13.7",
    "typer": "0.12",
}

OPTIONAL_FINETUNING_PACKAGES = {
    "bitsandbytes": "0.43",
    "trl": "0.8",
    "safetensors": "0.4",
    "sentencepiece": "0.1.99",
    "protobuf": "4.25",
}


@dataclass
class PackageStatus:
    name: str
    installed: bool
    version: Optional[str] = None
    minimum: Optional[str] = None

    @property
    def is_old(self) -> bool:
        if not self.installed or not self.version or not self.minimum:
            return False
        try:
            return Version(self.version) < Version(self.minimum)
        except InvalidVersion:
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "installed": self.installed,
            "version": self.version,
            "minimum": self.minimum,
            "is_old": self.is_old,
        }


def _package_status(name: str, minimum: Optional[str] = None) -> PackageStatus:
    try:
        version = metadata.version(name)
        return PackageStatus(name=name, installed=True, version=version, minimum=minimum)
    except metadata.PackageNotFoundError:
        return PackageStatus(name=name, installed=False, minimum=minimum)


def is_colab_runtime() -> bool:
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True
    try:
        __import__("google.colab")
        return True
    except Exception:
        return False


def _cuda_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"torch_import_error": str(exc), "available": False}

    info: dict[str, Any] = {
        "available": False,
        "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "device_count": 0,
        "devices": [],
    }
    try:
        available = bool(torch.cuda.is_available())
        info["available"] = available
        if not available:
            return info
        info["device_count"] = int(torch.cuda.device_count())
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_gb": round(props.total_memory / (1024**3), 2),
                }
            )
        info["devices"] = devices
    except Exception as exc:
        info["error"] = str(exc)
    return info


def collect_environment() -> dict[str, Any]:
    """Collect Python, package, CUDA, and Colab runtime metadata."""

    packages = {
        name: _package_status(name, minimum).to_dict()
        for name, minimum in {**CORE_PACKAGES, **OPTIONAL_FINETUNING_PACKAGES}.items()
    }
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "is_colab": is_colab_runtime(),
        "packages": packages,
        "cuda": _cuda_info(),
    }


def diagnose_environment() -> DiagnosisReport:
    """Return a report for the current fine-tuning environment."""

    env = collect_environment()
    report = DiagnosisReport(metadata=env)

    if env["is_colab"]:
        report.add(
            "env.colab",
            "Colab runtime detected",
            "ok",
            "This looks like a Google Colab runtime.",
        )
    else:
        report.add(
            "env.local",
            "Local runtime detected",
            "info",
            "This does not look like Google Colab.",
        )

    cuda = env["cuda"]
    if cuda.get("available"):
        devices = cuda.get("devices", [])
        device_text = ", ".join(
            f"{device['name']} ({device['total_gb']} GB)" for device in devices
        )
        report.add(
            "env.cuda_available",
            "CUDA is available",
            "ok",
            f"CUDA is available with {cuda.get('device_count', 0)} device(s): {device_text}.",
            torch_cuda=cuda.get("torch_cuda"),
        )
    else:
        report.add(
            "env.cuda_missing",
            "CUDA is not available",
            "warning",
            "Torch does not see a CUDA GPU.",
            "In Colab, choose Runtime -> Change runtime type -> GPU before loading the model.",
            torch_cuda=cuda.get("torch_cuda"),
        )

    packages = env["packages"]
    for name in CORE_PACKAGES:
        pkg = packages[name]
        if not pkg["installed"]:
            report.add(
                f"env.package_missing.{name}",
                f"{name} is missing",
                "warning",
                f"`{name}` is not installed.",
                "Install with `python -m pip install -U \"peft-doctor[ml]\"`.",
            )
        elif pkg["is_old"]:
            report.add(
                f"env.package_old.{name}",
                f"{name} is older than recommended",
                "warning",
                f"`{name}` {pkg['version']} is installed; PEFT Doctor recommends {pkg['minimum']} or newer.",
                f"Upgrade with `python -m pip install -U {name}`.",
                installed=pkg["version"],
                minimum=pkg["minimum"],
            )
        else:
            report.add(
                f"env.package_ok.{name}",
                f"{name} is installed",
                "ok",
                f"`{name}` {pkg['version']} is installed.",
            )

    for name in OPTIONAL_FINETUNING_PACKAGES:
        pkg = packages[name]
        if not pkg["installed"]:
            severity = "warning" if name == "bitsandbytes" else "info"
            report.add(
                f"env.optional_missing.{name}",
                f"{name} is not installed",
                severity,
                f"`{name}` is not installed.",
                "Install optional fine-tuning tools with `python -m pip install -U \"peft-doctor[ml]\"`.",
            )
        elif pkg["is_old"]:
            report.add(
                f"env.optional_old.{name}",
                f"{name} is older than recommended",
                "info",
                f"`{name}` {pkg['version']} is installed; {pkg['minimum']} or newer is recommended.",
                f"Upgrade with `python -m pip install -U {name}`.",
            )

    return report
