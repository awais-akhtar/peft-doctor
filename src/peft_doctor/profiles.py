"""Model-family profiles used by estimates, recipes, and explanations."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

from .targets import recommend_target_modules


@dataclass(frozen=True)
class ModelFamilyProfile:
    name: str
    aliases: tuple[str, ...]
    target_modules: tuple[str, ...]
    default_seq_len: int
    hidden_size: int
    layers: int
    tokenizer_notes: str
    common_risks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_FAMILY_PROFILES: dict[str, ModelFamilyProfile] = {
    "llama": ModelFamilyProfile(
        name="llama",
        aliases=("llama", "llama2", "llama3", "meta-llama"),
        target_modules=tuple(recommend_target_modules(model_family="llama")),
        default_seq_len=2048,
        hidden_size=4096,
        layers=32,
        tokenizer_notes="Causal LM tokenizers often need pad_token=eos_token for batching.",
        common_risks=("CUDA OOM at long context", "wrong target modules", "use_cache with checkpointing"),
    ),
    "qwen": ModelFamilyProfile(
        name="qwen",
        aliases=("qwen", "qwen2", "qwen2.5", "qwen3"),
        target_modules=tuple(recommend_target_modules(model_family="qwen")),
        default_seq_len=2048,
        hidden_size=3584,
        layers=28,
        tokenizer_notes="Qwen instruct recipes often need EOS reviewed, commonly <|im_end|>.",
        common_risks=("bad EOS stop token", "chat template mismatch", "CUDA OOM"),
    ),
    "mistral": ModelFamilyProfile(
        name="mistral",
        aliases=("mistral", "mixtral"),
        target_modules=tuple(recommend_target_modules(model_family="mistral")),
        default_seq_len=2048,
        hidden_size=4096,
        layers=32,
        tokenizer_notes="Use the model chat template for instruct variants.",
        common_risks=("completion masking mismatch", "long-context memory", "wrong target modules"),
    ),
    "gemma": ModelFamilyProfile(
        name="gemma",
        aliases=("gemma", "google/gemma"),
        target_modules=tuple(recommend_target_modules(model_family="gemma")),
        default_seq_len=1024,
        hidden_size=2304,
        layers=26,
        tokenizer_notes="Gemma recipes are sensitive to chat template formatting.",
        common_risks=("low VRAM pressure", "chat formatting", "pad/eos masking"),
    ),
    "phi": ModelFamilyProfile(
        name="phi",
        aliases=("phi", "phi-2", "phi-3", "phi-4"),
        target_modules=tuple(recommend_target_modules(model_family="phi")),
        default_seq_len=2048,
        hidden_size=3072,
        layers=32,
        tokenizer_notes="Check tokenizer max length placeholders and chat template formatting.",
        common_risks=("target modules", "sequence length", "tokenizer max length"),
    ),
    "falcon": ModelFamilyProfile(
        name="falcon",
        aliases=("falcon", "tiiuae/falcon"),
        target_modules=tuple(recommend_target_modules(model_family="falcon")),
        default_seq_len=2048,
        hidden_size=4544,
        layers=32,
        tokenizer_notes="Falcon-style target modules differ from Llama/Qwen projection names.",
        common_risks=("wrong target modules", "memory pressure", "optimizer memory"),
    ),
}


def list_model_profiles() -> list[ModelFamilyProfile]:
    return list(MODEL_FAMILY_PROFILES.values())


def profile_for(name: Optional[str]) -> Optional[ModelFamilyProfile]:
    if not name:
        return None
    lowered = name.lower()
    for profile in MODEL_FAMILY_PROFILES.values():
        if lowered == profile.name or any(alias in lowered for alias in profile.aliases):
            return profile
    return None
