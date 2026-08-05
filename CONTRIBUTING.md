# Contributing

Thanks for helping improve PEFT Doctor.

## Local Setup

```bash
git clone https://github.com/awais-akhtar/peft-doctor.git
cd peft-doctor
python -m pip install -e ".[dev,ml]"
```

## Checks

```bash
python -m ruff check src tests
python -m pytest
python -m build
```

## Pull Requests

Good pull requests usually include:

- a clear bug or use case
- a small test when behavior changes
- docs when a command or public API changes

PEFT Doctor should stay practical. Warnings should explain what was detected, why it matters, and the next fix to try.
