import re
from pathlib import Path

from typer.main import get_command

from peft_doctor.cli import app


ROOT = Path(__file__).resolve().parents[1]


def test_every_cli_command_has_a_documented_example():
    commands = set(get_command(app).commands)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    reference = (ROOT / "docs" / "commands.md").read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## `peft-doctor ([^`]+)`\s*$", reference, re.MULTILINE))
    documented = {match.group(1) for match in headings}

    assert documented == commands
    for index, match in enumerate(headings):
        command = match.group(1)
        end = headings[index + 1].start() if index + 1 < len(headings) else len(reference)
        section = reference[match.end() : end]
        assert f"peft-doctor {command}" in section
        assert "```" in section
        assert f"peft-doctor {command}" in readme


def test_readme_links_work_from_pypi():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    relative_links = re.findall(r"\]\((?!https?://|#|mailto:)([^)]+)\)", readme)

    assert relative_links == []


def test_repository_does_not_contain_literal_access_tokens():
    secret_patterns = [
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
        re.compile(r"pypi-[A-Za-z0-9_-]{40,}"),
    ]
    excluded = {".git", ".venv", "dist", "build"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in excluded or part.startswith(".pytest") for part in path.parts):
            continue
        if path.suffix.lower() not in {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".jsonl", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in secret_patterns:
            assert pattern.search(text) is None, f"Possible access token in {path.relative_to(ROOT)}"
