# Advanced PEFT Doctor Features

These checks run locally. They do not upload scripts, datasets, adapters, logs, or tokens.

## 1. Prioritized Diagnosis

```bash
peft-doctor diagnose train.py --dataset data.jsonl --model llama-3-8b --gpu "RTX 4090"
```

Combines matching script, dataset, model, and memory rules into a prioritized report. The risk level is not a probability.

## 2. Dry-Run Simulation

```bash
peft-doctor simulate --model llama-3-8b --dataset data.jsonl --gpu L4 --tokens-per-second 750
```

Checks formula-based VRAM, evaluation risk, measured-throughput runtime, and an explicit checkpoint budget without starting training.

## 3. Memory Timeline

```bash
peft-doctor memory-timeline --model llama-3-8b --seq-len 2048 --batch-size 2 --qlora
```

Shows load, forward, backward, optimizer, and peak memory phases.

## 4. Cost Estimator

```bash
peft-doctor estimate-cost --model llama-3-8b --dataset-size 8000 --offer "L4:0.80:750" --offer "A100:1.90:1800"
```

Calculates time and cost from hourly rates and tokens-per-second measurements supplied by the user.

## 5. Hyperparameter Advisor

```bash
peft-doctor advise-hparams --model llama-3-8b --dataset-size 8000 --gpu-vram 24
```

Provides a rule-based LoRA rank, alpha, and dropout starting point for validation.

## 6. Training Health Monitor

```bash
peft-doctor monitor trainer.log
```

Reads training logs and reports parsed loss direction and known runtime failures.
When PyTorch and CUDA are available, it also reports the current GPU name, allocated memory, reserved memory, total VRAM, and usage percentage.

## 7. Memory-Oriented Adjustment

```bash
peft-doctor auto-tune --model llama-3-8b --batch-size 4 --grad-accum 1 --target-vram 16
```

Suggests smaller batch size, adjusted gradient accumulation, and safer sequence length.

## 8. Fine-Tuning Score

```bash
peft-doctor score train.py --dataset data.jsonl --gpu T4
```

Summarizes checked errors and warnings with the documented formula `100 - 25 per error - 7 per warning`.

## 9. Dataset Intelligence

```bash
peft-doctor dataset-intel data.jsonl
```

Counts duplicated conversations, empty assistant replies, assistant-only rows, phrase-based review markers, malformed rows, and rows without a matched issue.

## 10. LoRA Coverage Review

```bash
peft-doctor lora-efficiency --model llama-3-8b --rank 32 --dataset-size 8000
```

Reports rank and target coverage. It does not infer adapter size or model quality from rank alone.

## 11. Adapter Comparison

```bash
peft-doctor compare-adapters ./adapter-r16 ./adapter-r64
```

Compares rank, alpha, target modules, actual local file size, and exact safetensors parameter counts.

## 12. Automatic Upgrade Suggestions

```bash
peft-doctor upgrade-suggestions
```

Compares installed package versions with PEFT Doctor's declared minimum versions.

## 13. GPU Fingerprinting

```bash
peft-doctor gpu-fingerprint "RTX 3060"
```

Reports a built-in reference VRAM value and local CUDA details when available.

## 14. Dataset Visualizer

```bash
peft-doctor dataset-report data.jsonl --output dataset-report.html
```

Creates an HTML report with length histogram, token histogram, duplicate clusters, role distribution, character-set buckets, response length, and outliers.

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

Matches the question with local dataset/log checks and bundled troubleshooting rules. If a dataset is provided, it includes a short preview of the first matched row.

## 18. Project Optimizer

```bash
peft-doctor optimize . --html-report optimize-report.html
peft-doctor optimize . --write
```

Combines the supported fixer, dataset checks, rule score, and an HTML report. Default mode is dry-run.

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

## 21. Training Plan Calculator

```bash
peft-doctor training-plan --dataset-size 8000 --batch-size 1 --grad-accum 8 --epochs 3
```

Calculates effective batch size, optimizer steps, warmup, and checkpoint count from explicit inputs.

## 22. Disk Budget Check

```bash
peft-doctor disk-check ./outputs --checkpoint-gb 1.2 --save-total-limit 2 --reserve-gb 5
```

Combines real filesystem free space with the storage budget you provide.

## 23. Checkpoint Completeness

```bash
peft-doctor checkpoint-check ./outputs/checkpoint-500
```

Checks model/adapter weights plus Trainer, optimizer, scheduler, and RNG state without loading pickle files.

## 24. Resume Compatibility

```bash
peft-doctor resume-check ./outputs/checkpoint-500 --config config.json
```

Finds material differences between the saved run and current training settings.

## 25. Reproducibility Manifest

```bash
peft-doctor manifest . --output run-manifest.json
```

Records streamed SHA-256 hashes and environment versions for a run.

## 26. Manifest Verification

```bash
peft-doctor verify-manifest run-manifest.json
```

Checks recorded file identities and important package versions for drift without loading model weights.

## 27. Token-Length Profile

```bash
peft-doctor token-lengths data.jsonl --model Qwen/Qwen2.5-7B --max-length 2048
```

Reports length percentiles, outliers, and static-padding waste with exact tokenizer counts when available.

## 28. Distributed Configuration Check

```bash
peft-doctor distributed-check accelerate.yaml --training-args training.json
```

Checks Accelerate and DeepSpeed precision, process, ZeRO, offload, accumulation, and device-map settings.

## 29. Generation Configuration Check

```bash
peft-doctor generation-check ./merged-model
```

Checks EOS/padding ids, length controls, sampling numbers, and beam-return compatibility.

## 30. Adapter Integrity

```bash
peft-doctor adapter-integrity ./adapter --write-checksums adapter_checksums.json
```

Validates adapter structure and safetensors metadata, then creates or verifies SHA-256 digests without loading weight tensors.

## 31. TRL SFT Check

```bash
peft-doctor sft-check sft.yaml --dataset data.jsonl --chat-template template.jinja
```

Checks selective loss modes, packing, padding-free attention requirements, dataset shape, and assistant generation markers.

## 32. FSDP Check

```bash
peft-doctor fsdp-check accelerate-fsdp.yaml --training-args training.json
```

Checks state-dict type, synchronized CPU-efficient loading, process count, auto-wrap policy, compile settings, and duplicate checkpointing.

## 33. LoRA Hotswap Check

```bash
peft-doctor hotswap-check ./loaded ./incoming --compiled --prepared-max-rank 64
```

Checks adapter type, base model, target-layer subset, reserved rank, raw target parameters, and `modules_to_save` differences.

## 34. Current LoraConfig Check

```bash
peft-doctor lora-config-check adapter_config.json
```

Checks trainable tokens, weight tying, aLoRA, QALoRA, initialization modes, all-linear targeting, and risky output/embedding targets.

## 35. Checkpoint Inventory

```bash
peft-doctor checkpoint-list ./outputs
```

Checks every checkpoint and identifies the latest directory without a structural error.

## 36. Train/Eval Leakage Check

```bash
peft-doctor leakage-check train.jsonl eval.jsonl --threshold 0.9
```

Reports exact overlap and bounded token-set Jaccard matches with source row numbers.

## 37. Adapter Tensor Inventory

```bash
peft-doctor adapter-tensors ./adapter --output json
```

Reads exact tensor counts, parameter counts, dtypes, tensor ranks, and bytes from safetensors headers.

## 38. Chat-Template Check

```bash
peft-doctor chat-template-check tokenizer_config.json --dataset chat.jsonl --assistant-only-loss
```

Checks the messages variable, role names, and balanced generation markers.

## 39. Label-Mask Audit

```bash
peft-doctor label-audit tokenized.jsonl --pad-token-id 0
```

Measures supervision and finds length mismatches, all-ignored labels, and pad positions that still contribute to loss.

## 40. Quantization Check

```bash
peft-doctor quantization-check bnb.json --gpu T4
```

Checks bitsandbytes mode conflicts, NF4, compute dtype, T4/bf16 plans, and FSDP storage dtype.
