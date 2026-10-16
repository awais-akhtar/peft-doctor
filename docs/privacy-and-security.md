# Privacy And Security

PEFT Doctor processes file-based checks locally and has no hosted analysis service. Commands that resolve model or tokenizer identifiers may contact the configured model host. Merge and Hub-upload operations can transfer model artifacts when explicitly configured.

## What Stays Local

These commands read files on your machine and generate local reports:

```bash
peft-doctor diagnose train.py --dataset data.jsonl
peft-doctor chat "Why is my loss exploding?" --dataset data.jsonl --log trainer.log
peft-doctor optimize . --html-report optimize-report.html
peft-doctor dataset-report data.jsonl
peft-doctor audit . --policy peft-policy.yml
```

The commands above do not send files to a PEFT Doctor service. If a command is given a remote model identifier, the relevant upstream library may contact the model host. Hub upload occurs only through an explicitly configured upload operation.

## Tokens

Do not paste access tokens into notebooks, scripts, shell commands, reports, or GitHub issues.

For private or gated Hugging Face models, use one of these patterns:

```bash
huggingface-cli login
```

or an environment variable set through your shell or secret manager:

```bash
export HF_TOKEN
```

In Colab, store your own token in Colab Secrets and read it from the notebook environment.

## Reports

HTML, PDF, markdown, and JSON reports are written only to the path you choose. Dataset report HTML escapes displayed values and does not include executable JavaScript.

## Cloud Roadmap

`peft-doctor cloud` only prints roadmap information and does not send data to a hosted PEFT Doctor service.
