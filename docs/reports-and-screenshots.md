# Reports And Screenshots

PEFT Doctor can generate terminal, markdown, JSON, HTML, and PDF reports.

## Dry-Run Auto-Fix

```text
Found 9 issues.
Safe auto-fixes available for 9.
Run with --write or --output to apply.
```

## Risk Explanation

```bash
peft-doctor check train.py --explain
```

Example:

```text
Run risk: HIGH (85/100)
Reasons:
- bf16 and fp16 are both enabled
- No CUDA GPU detected
- Checkpoint limit is missing
Copy-paste fixes:
- Use bf16 on supported NVIDIA GPUs, otherwise use fp16, but do not enable both.
- Set save_total_limit=2 or 3 to avoid filling disk during long runs.
```

## HTML Report

```bash
peft-doctor check train.py --html-report report.html
```

Attach `report.html` to GitHub issues, pull requests, or team debugging notes.

## PDF Report

```bash
peft-doctor check train.py --pdf-report report.pdf
```

Use PDF reports when a team needs a portable run summary without sharing local logs.
