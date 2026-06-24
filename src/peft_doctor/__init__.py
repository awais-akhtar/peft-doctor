"""Public API for PEFT Doctor."""

from ._version import __version__
from .adapters import MergeResult, diagnose_adapter_merge, merge_lora_adapter
from .configs import (
    create_safe_bnb_config,
    create_safe_lora_config,
    create_safe_training_args,
)
from .diagnostics import diagnose_peft
from .environment import collect_environment, diagnose_environment
from .logs import NanLossGuard, scan_training_log
from .notebooks import scan_notebook
from .report import DiagnosisReport, DiagnosticIssue
from .targets import infer_model_family, recommend_target_modules

__all__ = [
    "DiagnosisReport",
    "DiagnosticIssue",
    "MergeResult",
    "NanLossGuard",
    "__version__",
    "create_safe_bnb_config",
    "create_safe_lora_config",
    "create_safe_training_args",
    "diagnose_adapter_merge",
    "collect_environment",
    "diagnose_environment",
    "diagnose_peft",
    "infer_model_family",
    "merge_lora_adapter",
    "recommend_target_modules",
    "scan_notebook",
    "scan_training_log",
]
