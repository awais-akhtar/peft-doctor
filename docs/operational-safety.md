# Operational Safety Guide

These commands cover the part of fine-tuning that is easy to overlook: whether a run plan is internally consistent, a checkpoint can really resume, an adapter survived copying, and a result can be reproduced later.

File and configuration checks operate on local paths. Commands that resolve remote model identifiers or explicitly upload merged artifacts may use network access. No hosted PEFT Doctor analysis service is used.

## Recommended Workflow

Before training:

```bash
peft-doctor training-plan --dataset-size 8000 --batch-size 1 --grad-accum 8 --epochs 3
peft-doctor disk-check ./outputs --checkpoint-gb 1.2 --save-total-limit 2 --reserve-gb 5
peft-doctor token-lengths data.jsonl --model Qwen/Qwen2.5-7B --max-length 2048
peft-doctor distributed-check accelerate.yaml --training-args training.json
peft-doctor manifest . --output run-manifest.json
```

After an interrupted run:

```bash
peft-doctor checkpoint-check ./outputs
peft-doctor resume-check ./outputs/checkpoint-500 --config config.json
```

Before evaluation, merge, or upload:

```bash
peft-doctor generation-check ./merged-model
peft-doctor adapter-integrity ./adapter --write-checksums adapter_checksums.json
peft-doctor verify-manifest run-manifest.json
```

## What Each Command Proves

| Command | It verifies | It does not prove |
| --- | --- | --- |
| `training-plan` | Step counts are mathematically consistent with the supplied batch/world-size/epoch settings | Wall-clock speed or final quality |
| `disk-check` | Current free space exceeds the explicit budget | Future cloud-volume growth or an unknown checkpoint size |
| `checkpoint-check` | Expected local state files exist and readable JSON is valid | Pickle contents are safe or semantically correct |
| `resume-check` | Shared saved/current settings do not contain known material mismatches | Every custom trainer field is compatible |
| `manifest` | Selected project files have recorded hashes and environment metadata | Dataset/model files skipped by the size limit are unchanged |
| `verify-manifest` | Recorded hashes and selected package versions still match | Identical GPU kernels or bit-for-bit training across platforms |
| `token-lengths` | Sampled length distribution and over-limit counts | The best sequence length for model quality |
| `distributed-check` | Common Accelerate/DeepSpeed conflicts are absent | A multi-node cluster is reachable or correctly networked |
| `generation-check` | Generation values and token ids are internally consistent | Responses are accurate, safe, or useful |
| `adapter-integrity` | Adapter structure, safetensors metadata, and checksums are valid | The adapter improves quality or matches an unrecorded base-model revision |

## Security Boundaries

- `.bin`, optimizer, scheduler, and RNG pickle files are never deserialized by these checks.
- Safetensors inspection reads only the bounded JSON header and validates tensor offsets against file size.
- SHA-256 hashing is streamed in 1 MB chunks, so large adapters do not have to fit in RAM.
- Manifests contain paths, file sizes, hashes, versions, platform details, and an optional Git commit. Review them before sharing.
- A hash verifies file identity, not whether the original file was trustworthy.

## Upstream Contracts

The checks are based on documented upstream behavior:

- [Transformers Trainer checkpoint and resume behavior](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Accelerate save and load state](https://huggingface.co/docs/accelerate/main/usage_guides/checkpoint)
- [Accelerate with DeepSpeed](https://huggingface.co/docs/accelerate/main/usage_guides/deepspeed)
- [PEFT checkpoint format](https://huggingface.co/docs/peft/main/developer_guides/checkpoint)
- [TRL SFTTrainer dataset and packing behavior](https://huggingface.co/docs/trl/en/sft_trainer)
- [Transformers generation configuration](https://huggingface.co/docs/transformers/en/main_classes/text_generation)
- [Safetensors metadata parsing](https://huggingface.co/docs/safetensors/metadata_parsing)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)

Upstream libraries evolve. Use `peft-doctor env` and `peft-doctor upgrade-suggestions`, pin a working environment, and keep a manifest with every important run.
