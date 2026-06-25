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
from .estimator import estimate_vram_gb, infer_params_billion
from .explain import (
    explanation_text,
    report_to_html,
    report_to_pdf_bytes,
    risk_summary,
    write_html_report,
    write_pdf_report,
)
from .fixer import repair_config_file, repair_dataset_file, repair_python_file, repair_python_source
from .logs import NanLossGuard, scan_training_log
from .notebooks import scan_notebook
from .profiles import MODEL_FAMILY_PROFILES, ModelFamilyProfile, list_model_profiles, profile_for
from .recipes import (
    PROJECT_RECIPE_NAMES,
    RECIPE_NAMES,
    benchmark_recipe_report,
    copy_recipe_project,
    create_training_recipe,
    validate_recipe_project,
)
from .report import DiagnosisReport, DiagnosticIssue
from .targets import infer_model_family, recommend_target_modules

__all__ = [
    "DiagnosisReport",
    "DiagnosticIssue",
    "MergeResult",
    "NanLossGuard",
    "PROJECT_RECIPE_NAMES",
    "RECIPE_NAMES",
    "MODEL_FAMILY_PROFILES",
    "ModelFamilyProfile",
    "__version__",
    "create_safe_bnb_config",
    "create_safe_lora_config",
    "create_safe_training_args",
    "create_training_recipe",
    "benchmark_recipe_report",
    "copy_recipe_project",
    "diagnose_adapter_merge",
    "collect_environment",
    "diagnose_environment",
    "diagnose_peft",
    "estimate_vram_gb",
    "explanation_text",
    "infer_model_family",
    "infer_params_billion",
    "list_model_profiles",
    "merge_lora_adapter",
    "recommend_target_modules",
    "profile_for",
    "report_to_html",
    "report_to_pdf_bytes",
    "repair_config_file",
    "repair_dataset_file",
    "repair_python_file",
    "repair_python_source",
    "scan_notebook",
    "scan_training_log",
    "risk_summary",
    "validate_recipe_project",
    "write_html_report",
    "write_pdf_report",
]
