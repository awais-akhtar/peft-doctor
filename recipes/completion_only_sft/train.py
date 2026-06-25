"""Completion-only SFT recipe."""

from __future__ import annotations

from peft_doctor.recipes import project_recipe_files


if __name__ == "__main__":
    code = project_recipe_files("completion-only-sft")["train.py"]
    exec(  # noqa: S102
        compile(code, __file__, "exec"),
        {"__name__": "__main__", "__file__": __file__},
    )
