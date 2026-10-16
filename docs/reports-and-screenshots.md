# Reports

PEFT Doctor can print terminal tables, JSON, and Markdown. Selected commands also write static HTML or PDF files.

## Capture Current Output

Use the installed version to capture output for an issue or pull request:

```bash
peft-doctor version
peft-doctor check train.py --dataset data.jsonl --output markdown
peft-doctor fix --input train.py --dry-run
```

Do not rely on an old screenshot when exact wording or issue codes matter. CLI output can change as checks are corrected.

## HTML Report

```bash
peft-doctor check train.py --dataset data.jsonl --html-report report.html
```

The report includes metadata, severity counts, findings, and copyable fixes. Review paths and dataset previews before attaching it to a public issue.

## PDF Report

```bash
peft-doctor check train.py --dataset data.jsonl --pdf-report report.pdf
```

The PDF contains the same local report in a portable format.

## Dataset Report

```bash
peft-doctor dataset-report data.jsonl --output dataset-report.html
```

The dataset report includes length and approximate-token histograms, role distribution, character-set buckets, response lengths, duplicate clusters, and rule-match counts. It escapes displayed dataset text and contains no JavaScript.

## Project Report

```bash
peft-doctor optimize . --html-report optimize-report.html
```

The default optimizer pass is dry-run. Add `--write` only after reviewing the repair candidates.
