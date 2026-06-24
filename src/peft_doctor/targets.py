"""LoRA target module recommendations."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Optional

from .utils import get_value


FAMILY_TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mixtral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "gpt_bigcode": ["c_attn", "c_proj", "c_fc"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "t5": ["q", "k", "v", "o", "wi", "wo"],
    "mt5": ["q", "k", "v", "o", "wi", "wo"],
    "bert": ["query", "key", "value", "dense"],
    "roberta": ["query", "key", "value", "dense"],
    "deberta": ["query_proj", "key_proj", "value_proj", "dense"],
}

ATTENTION_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "query",
    "key",
    "value",
    "query_key_value",
    "c_attn",
    "c_proj",
}

MLP_NAMES = {
    "gate_proj",
    "up_proj",
    "down_proj",
    "c_fc",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
    "wi",
    "wo",
}


def _model_config(model: Any) -> Any:
    return get_value(model, "config")


def _config_text(config: Any) -> str:
    if config is None:
        return ""
    values = [
        get_value(config, "model_type", ""),
        " ".join(get_value(config, "architectures", []) or []),
        config.__class__.__name__,
    ]
    return " ".join(str(value).lower() for value in values if value)


def infer_model_family(model: Any = None, model_name: Optional[str] = None) -> Optional[str]:
    """Infer a model family from a model object, config object, or model id."""

    config = _model_config(model) or model
    text = " ".join(
        part
        for part in [
            _config_text(config),
            str(model_name or "").lower(),
            model.__class__.__name__.lower() if model is not None else "",
        ]
        if part
    )

    aliases = [
        ("deepseek", ["deepseek"]),
        ("qwen3", ["qwen3"]),
        ("qwen2", ["qwen2", "qwen-2"]),
        ("qwen", ["qwen"]),
        ("mixtral", ["mixtral"]),
        ("mistral", ["mistral"]),
        ("llama", ["llama", "codellama", "vicuna", "alpaca"]),
        ("gemma", ["gemma"]),
        ("phi", ["phi"]),
        ("gpt_bigcode", ["starcoder", "gpt_bigcode", "bigcode"]),
        ("gpt_neox", ["gpt_neox", "gpt-neox", "pythia"]),
        ("falcon", ["falcon"]),
        ("bloom", ["bloom"]),
        ("gpt2", ["gpt2", "distilgpt2"]),
        ("mt5", ["mt5"]),
        ("t5", ["t5", "flan-t5"]),
        ("deberta", ["deberta"]),
        ("roberta", ["roberta"]),
        ("bert", ["bert"]),
    ]
    for family, needles in aliases:
        if any(needle in text for needle in needles):
            return family
    return None


def iter_leaf_module_names(model: Any) -> Iterable[str]:
    """Yield leaf module names from a torch-style model."""

    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return []

    names = []
    for full_name, module in named_modules():
        if not full_name:
            continue
        has_children = False
        children = getattr(module, "children", None)
        if callable(children):
            try:
                has_children = any(True for _ in children())
            except Exception:
                has_children = False
        if not has_children:
            names.append(full_name.rsplit(".", 1)[-1])
    return names


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def recommend_target_modules(
    model: Any = None,
    model_name: Optional[str] = None,
    model_family: Optional[str] = None,
    include_mlp: bool = True,
) -> list[str]:
    """Recommend LoRA target module names.

    The function prefers the actual module names present on the model. When a
    model object is not available, it falls back to a family map.
    """

    family = (model_family or infer_model_family(model, model_name) or "").lower()
    family_targets = FAMILY_TARGET_MODULES.get(family, [])
    allowed = set(ATTENTION_NAMES)
    if include_mlp:
        allowed.update(MLP_NAMES)

    present = Counter(iter_leaf_module_names(model) if model is not None else [])
    if present:
        if family_targets:
            found = [target for target in family_targets if target in present and target in allowed]
            if found:
                return found

        common = [name for name, _count in present.most_common() if name in allowed]
        if common:
            return _ordered_unique(common)

    if family_targets:
        if include_mlp:
            return list(family_targets)
        return [target for target in family_targets if target in ATTENTION_NAMES]

    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def missing_target_modules(model: Any, configured_targets: Iterable[str]) -> list[str]:
    present = set(iter_leaf_module_names(model))
    if not present:
        return []
    return [target for target in configured_targets if target not in present]
