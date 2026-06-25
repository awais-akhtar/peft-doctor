# Advanced PEFT Doctor Features

PEFT Doctor is the quality assurance and optimization layer for LoRA and QLoRA fine-tuning. These features are local-first: they do not upload scripts, datasets, adapters, logs, or tokens.

## 1. Local Expert Diagnosis

```bash
peft-doctor diagnose train.py --dataset data.jsonl --model llama-3-8b --gpu "RTX 4090"
```

Explains likely failure reasons, confidence, recommended fixes, and estimated success rate after fixes.

## 2. Dry-Run Simulation

```bash
peft-doctor simulate --model llama-3-8b --dataset data.jsonl --gpu L4
```

Predicts start success, peak VRAM, rough ETA, evaluation OOM risk, and checkpoint disk risk without starting training.

## 3. Memory Timeline

```bash
peft-doctor memory-timeline --model llama-3-8b --seq-len 2048 --batch-size 2 --qlora
```

Shows load, forward, backward, optimizer, and peak memory phases.

## 4. Cost Estimator

```bash
peft-doctor estimate-cost --model llama-3-8b --dataset-size 8000 --gpu L4 --gpu A100
```

Compares rough cloud GPU time and cost across GPU options.

## 5. Hyperparameter Advisor

```bash
peft-doctor advise-hparams --model llama-3-8b --dataset-size 8000 --gpu-vram 24
```

Recommends LoRA rank, alpha, and dropout.

## 6. Training Health Monitor

```bash
peft-doctor monitor trainer.log
```

Reads training logs and reports loss trend, NaN chance, and known runtime failures.

## 7. Smart Auto-Tuning

```bash
peft-doctor auto-tune --model llama-3-8b --batch-size 4 --grad-accum 1 --target-vram 16
```

Suggests smaller batch size, adjusted gradient accumulation, and safer sequence length.

## 8. Fine-Tuning Score

```bash
peft-doctor score train.py --dataset data.jsonl --gpu T4
```

Scores dataset, configuration, hardware, trainer, and overall readiness.

## 9. Dataset Intelligence

```bash
peft-doctor dataset-intel data.jsonl
```

Finds duplicated conversations, empty assistant replies, assistant-only rows, prompt injections, malformed rows, and quality score.

## 10. LoRA Efficiency Predictor

```bash
peft-doctor lora-efficiency --model llama-3-8b --rank 32 --dataset-size 8000
```

Estimates adapter size, expected quality lift, inference slowdown, and merge compatibility.

## 11. Adapter Comparison

```bash
peft-doctor compare-adapters ./adapter-r16 ./adapter-r64
```

Compares rank, alpha, target modules, adapter size, and expected tradeoffs.

## 12. Automatic Upgrade Suggestions

```bash
peft-doctor upgrade-suggestions
```

Checks installed package versions and suggests upgrades for common fine-tuning packages.

## 13. GPU Fingerprinting

```bash
peft-doctor gpu-fingerprint "RTX 3060"
```

Reports VRAM profile, bf16 support guidance, Flash Attention guidance, and GPU-specific cautions.

## 14. Dataset Visualizer

```bash
peft-doctor dataset-report data.jsonl --output dataset-report.html
```

Creates an HTML report with length histogram, token histogram, role distribution, language buckets, response length, and outliers.

## 15. Experiment Tracker

```bash
peft-doctor history . --add-status completed --metric "BLEU +3.1"
peft-doctor history .
```

Stores lightweight local history in `.peft-doctor/history.jsonl`.

## 16. Offline Knowledge Base

```bash
peft-doctor knowledge-base "CUDA illegal memory access"
```

Searches bundled PEFT failure guidance without contacting a remote service.

## 17. Local Chat Mode

```bash
peft-doctor chat "Why is my loss exploding?" --dataset data.jsonl --log trainer.log
```

Answers from local dataset/log checks and the offline knowledge base.

## 18. Project Optimizer

```bash
peft-doctor optimize . --html-report optimize-report.html
peft-doctor optimize . --write
```

Combines safe fixer, dataset intelligence, scoring, and an HTML report. Default mode is dry-run.

## 19. Organization Policies

```bash
peft-doctor audit . --policy peft-policy.yml
```

Policy example:

```yaml
policy:
  max_seq_len: 4096
  require_bf16: true
  forbid_fp16: true
  require_dataset_validation: true
```

Use this in CI to enforce fine-tuning standards.

## 20. PEFT Doctor Cloud Roadmap

```bash
peft-doctor cloud
```

Documents the long-term hosted-reporting idea. The local command does not upload anything.

