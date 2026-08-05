# Beginner Command Guide

This guide is for developers who are new to PEFT, LoRA, QLoRA, TRL, or Colab fine-tuning. It explains what each `peft-doctor` command is for, gives a copy-paste example, and tells you what to do with the output.

PEFT Doctor is local by default. It reads the files you point it at and prints reports. It does not upload your scripts, datasets, logs, adapters, or tokens.

## Start Here

Install:

```bash
python -m pip install -U "peft-doctor[ml]"
```

In Colab:

```python
%pip install -U "peft-doctor[ml]"
!peft-doctor env
```

The simplest beginner flow is:

```bash
peft-doctor env
peft-doctor dataset-doctor data.jsonl
peft-doctor check train.py --explain
peft-doctor fix --dry-run train.py
peft-doctor optimize . --html-report peft-doctor-report.html
```

If any command prints warnings, fix the highest-risk items first, then run the same command again.

## Main Workflow Commands

| Command | Use it when | Copy-paste example | What to do after |
| --- | --- | --- | --- |
| `env` | You want to check Python, CUDA, PyTorch, Transformers, PEFT, bitsandbytes, and Colab runtime setup. | `peft-doctor env` | If CUDA or key packages are missing, fix the environment before loading a model. |
| `init` | You want a starter project and do not want to write training files by hand. | `peft-doctor init --model llama3 --gpu T4 --dataset-type chat --target-vram 16 --output-dir my-run` | Open `my-run/README.md`, replace the model and dataset values, then run `validate-recipe`. |
| `recipe` | You want a complete runnable example for a common setup. | `peft-doctor recipe llama3-qlora-colab --copy ./my-run` | Review `./my-run/train.py`, add your model access if needed, and validate the project. |
| `validate-recipe` | You copied or generated a recipe and want to know if required files are present. | `peft-doctor validate-recipe ./my-run` | If a required file is missing, copy the recipe again or restore that file. |
| `check` | You want a pre-flight check before training. This is the command to run most often. | `peft-doctor check train.py --dataset data.jsonl --model llama-3-8b --explain` | Fix high-risk findings first: OOM risk, bad labels, wrong target modules, missing pad token, and precision conflicts. |
| `fix` | You want PEFT Doctor to preview supported patches for common script, config, or dataset mistakes. | `peft-doctor fix --dry-run train.py` | Read the patch list. If it looks right, write a copy with `--output train.fixed.py` or apply with `--write`. |
| `diagnose` | You want a plain-English explanation of why a run may fail. | `peft-doctor diagnose train.py --dataset data.jsonl --model llama-3-8b --gpu T4` | Apply the recommended fixes in order, then run `simulate` or `check` again. |
| `simulate` | You want to check a training plan without loading a model. | `peft-doctor simulate --model llama-3-8b --dataset data.jsonl --gpu L4 --seq-len 2048 --batch-size 2` | Review the formula-based VRAM and evaluation warnings; supply measured throughput or checkpoint size only when you have them. |
| `optimize` | You want one command that checks config, dataset, tokenizer, LoRA, memory, and project score. | `peft-doctor optimize . --html-report optimize-report.html` | Open the HTML report and fix the warnings before using `--write`. |
| `score` | You want one number summarizing checked errors and warnings. | `peft-doctor score train.py --dataset data.jsonl --model llama-3-8b --gpu T4` | Fix errors first. The score covers only rules that had enough input to run. |
| `training-plan` | You need the real effective batch, optimizer-step, warmup, and save counts from your settings. | `peft-doctor training-plan --dataset-size 8000 --batch-size 1 --grad-accum 8 --epochs 3` | Compare the values with Trainer startup logs before a long run. |
| `checkpoint-check` | Training stopped and you need to know whether the checkpoint can restore full state. | `peft-doctor checkpoint-check ./outputs/checkpoint-500` | If optimizer, scheduler, or RNG state is missing, use an earlier full checkpoint or restart deliberately. |
| `checkpoint-list` | You have several checkpoints and need the newest structurally valid one. | `peft-doctor checkpoint-list ./outputs` | Run `resume-check` on the selected path before resuming. |
| `resume-check` | You changed config files after a stopped run. | `peft-doctor resume-check ./outputs/checkpoint-500 --config config.json` | Do not resume when the base model, LoRA rank, or target modules differ. |

## Memory, Speed, And Cost Commands

| Command | Use it when | Copy-paste example | What to do after |
| --- | --- | --- | --- |
| `estimate` | You want a rough VRAM estimate before loading a model. | `peft-doctor estimate --model llama-3-8b --seq-len 2048 --batch-size 2 --qlora --target-vram 16` | If estimated VRAM is above your GPU, lower batch size or sequence length and enable QLoRA/checkpointing. |
| `memory-timeline` | You want to see where memory spikes: load, forward, backward, optimizer, or peak. | `peft-doctor memory-timeline --model llama-3-8b --seq-len 4096 --batch-size 1 --qlora` | If backward or optimizer memory is too high, reduce sequence length or effective batch size. |
| `estimate-cost` | You have current provider rates and measured throughput. | `peft-doctor estimate-cost --model llama-3-8b --dataset-size 8000 --offer "L4:0.80:750" --offer "A100:1.90:1800"` | Compare the calculated totals. Replace the example numbers with your provider rate and short-run tokens per second. |
| `auto-tune` | Your batch size is too large but you want to keep the same effective batch. | `peft-doctor auto-tune --model llama-3-8b --batch-size 4 --grad-accum 1 --target-vram 16` | Use the suggested smaller batch size and higher gradient accumulation. |
| `gpu-fingerprint` | You want GPU-specific advice for precision and attention settings. | `peft-doctor gpu-fingerprint "RTX 3060"` | Follow the precision warning, especially on consumer GPUs with fp16 instability. |
| `monitor` | You already have a trainer log and want a failure summary. | `peft-doctor monitor trainer.log` | Review NaN/Inf, gradient, and runtime findings plus the parsed loss direction. |
| `disk-check` | You want to avoid a disk-full failure during checkpoint saving. | `peft-doctor disk-check ./outputs --checkpoint-gb 1.2 --save-total-limit 2 --reserve-gb 5` | Reduce checkpoint retention or move outputs if the remaining margin is small. |

## Dataset Commands

| Command | Use it when | Copy-paste example | What to do after |
| --- | --- | --- | --- |
| `dataset-doctor` | You want to check dataset rows before training. | `peft-doctor dataset-doctor data.jsonl --sequence-length 2048` | Remove empty assistant responses, malformed rows, duplicates, and examples longer than your sequence length. |
| `inspect-dataset` | You want a quick format check for `.json`, `.jsonl`, `.csv`, or `.txt` data. | `peft-doctor inspect-dataset data.jsonl` | Make sure the dataset uses chat messages, instruction/response columns, prompt/completion columns, or pretokenized fields. |
| `dataset-intel` | You want rule-match counts and the share of rows without a flagged issue. | `peft-doctor dataset-intel data.jsonl` | Review phrase matches manually, clean confirmed issues, then rerun the command. |
| `dataset-report` | You want an HTML report with histograms and duplicate clusters. | `peft-doctor dataset-report data.jsonl --output dataset-report.html` | Open the HTML file locally and inspect outliers before training. |
| `token-lengths` | You need model-specific length percentiles before choosing `max_length`. | `peft-doctor token-lengths data.jsonl --model Qwen/Qwen2.5-7B --max-length 2048` | Filter unusual outliers or choose truncation/packing intentionally. |
| `leakage-check` | You want to find exact or high-overlap train/eval examples. | `peft-doctor leakage-check train.jsonl eval.jsonl --threshold 0.9` | Review the reported row pairs and rebuild the split when overlap is unintended. |
| `label-audit` | Your tokenized dataset already contains `input_ids` and `labels`. | `peft-doctor label-audit tokenized.jsonl --pad-token-id 0` | Fix length mismatches, all-ignored rows, and unmasked pad positions. |
| `sft-check` | You use TRL SFTTrainer selective loss or packing. | `peft-doctor sft-check sft.yaml --dataset data.jsonl --chat-template template.jinja` | Align loss mode, dataset shape, template markers, and attention backend. |
| `chat-template-check` | You need to verify role handling or assistant-only masking. | `peft-doctor chat-template-check tokenizer_config.json --dataset chat.jsonl --assistant-only-loss` | Fix unbalanced generation blocks and review unmatched role aliases. |

## Adapter And Merge Commands

| Command | Use it when | Copy-paste example | What to do after |
| --- | --- | --- | --- |
| `inspect-adapter` | You have a saved LoRA adapter and want to check it before upload or merge. | `peft-doctor inspect-adapter ./adapter` | Confirm `adapter_config.json` and adapter weights exist. |
| `adapter-check` | You want to check a merge plan without loading the full model. | `peft-doctor adapter-check --base-model meta-llama/Llama-2-7b-hf --adapter your-user/your-lora-adapter --output-dir merged-model` | Fix missing base model, tokenizer, or quantized-merge warnings before merging. |
| `merge-adapter` | You are ready to merge a LoRA adapter into the base model. | `peft-doctor merge-adapter --base-model meta-llama/Llama-2-7b-hf --adapter your-user/your-lora-adapter --output-dir merged-model --dtype fp16` | Test the merged model locally before uploading. Do not paste tokens into the command. |
| `compare-adapters` | You need a factual config and file comparison for two adapters. | `peft-doctor compare-adapters ./adapter-r16 ./adapter-r64` | Use held-out evaluation metrics to choose; metadata alone does not establish quality. |
| `lora-efficiency` | You want to review rank and target coverage. | `peft-doctor lora-efficiency --model llama-3-8b --rank 32 --dataset-size 8000` | Run `adapter-tensors` after saving and compare held-out metrics between runs. |
| `adapter-integrity` | You need to verify a local adapter copy before upload or merge. | `peft-doctor adapter-integrity ./adapter --write-checksums adapter_checksums.json` | Keep the checksum file with the adapter and recheck after copying/downloading it. |
| `adapter-tensors` | You need exact safetensors tensor and parameter counts. | `peft-doctor adapter-tensors ./adapter --output json` | Compare actual file metadata, then use evaluation metrics for quality decisions. |
| `hotswap-check` | You plan to swap LoRA adapters, possibly on a compiled model. | `peft-doctor hotswap-check ./loaded ./incoming --compiled --prepared-max-rank 64` | Fix base, target-layer, or prepared-rank conflicts before serving. |

## Configuration And Model Commands

| Command | Use it when | Copy-paste example | What to do after |
| --- | --- | --- | --- |
| `profiles` | You want the built-in profile for a model family. | `peft-doctor profiles qwen` | Use the profile to choose target modules and prompt format. |
| `targets` | You need LoRA `target_modules`. | `peft-doctor targets --model meta-llama/Llama-3-8B` | Copy the recommended list into your `LoraConfig`. |
| `safe-config` | You want starter LoRA, QLoRA, and training settings. | `peft-doctor safe-config --family llama` | Copy the config into your training script and adjust only after a short validation run. |
| `advise-hparams` | You are unsure about a first LoRA rank, alpha, and dropout trial. | `peft-doctor advise-hparams --model llama-3-8b --dataset-size 8000 --gpu-vram 24` | Compare more than one rank on held-out data. |
| `upgrade-suggestions` | You want to compare installed packages with the project's declared minimums. | `peft-doctor upgrade-suggestions` | Test upgrades in a clean environment before changing a working training stack. |
| `distributed-check` | You use Accelerate or DeepSpeed and want to catch launch/config conflicts. | `peft-doctor distributed-check accelerate.yaml --training-args training.json` | Fix precision, process, ZeRO, offload, and accumulation errors before launching. |
| `fsdp-check` | You use Accelerate FSDP. | `peft-doctor fsdp-check accelerate-fsdp.yaml --training-args training.json` | Fix state, sync, wrap, process, compile, and checkpointing conflicts before launch. |
| `lora-config-check` | You use current LoRA variants or trainable tokens. | `peft-doctor lora-config-check adapter_config.json` | Review variant-specific warnings before constructing the model. |
| `quantization-check` | You use bitsandbytes 4-bit or 8-bit loading. | `peft-doctor quantization-check bnb.json --gpu T4` | Fix mode and dtype conflicts before loading model weights. |
| `generation-check` | The fine-tuned model repeats, never stops, or ignores sampling settings. | `peft-doctor generation-check ./merged-model` | Align EOS/pad ids and test corrected generation settings on fixed prompts. |

## Logs, Notebooks, And Colab Commands

| Command | Use it when | Copy-paste example | What to do after |
| --- | --- | --- | --- |
| `scan-log` | Training failed and you have a log file. | `peft-doctor scan-log trainer.log` | Apply the fix for the first severe runtime error. |
| `analyze-log` | Same as log scanning, with a beginner-friendly command name. | `peft-doctor analyze-log trainer.log` | Use the explanation to decide whether the problem is memory, data, precision, disk, or shape mismatch. |
| `scan-notebook` | You want to scan a Colab/Jupyter notebook for common mistakes. | `peft-doctor scan-notebook notebook.ipynb` | Remove pasted tokens, fix runtime setup, and rerun the notebook from the top. |
| `notebook-check` | Same notebook check with a clearer alias. | `peft-doctor notebook-check notebook.ipynb` | Use it before sharing a notebook or running a long Colab job. |
| `colab` | You want a Colab setup cell. | `peft-doctor colab` | Paste the printed cell into Colab, then run `peft-doctor env`. |

## Reports, Validation, And Team Commands

| Command | Use it when | Copy-paste example | What to do after |
| --- | --- | --- | --- |
| `validate` | You want a markdown validation report for a model/dataset pair. | `peft-doctor validate --model qwen --dataset sample.jsonl --report report.md` | Attach the report to a GitHub issue or project notes. |
| `benchmark` | You want to run a packaged recipe pre-flight scenario. | `peft-doctor benchmark --recipe llama3-qlora-colab` | Treat the result as fixture validation, not a timed GPU benchmark. |
| `audit` | Your team has standards for max sequence length, precision, or dataset validation. | `peft-doctor audit . --policy peft-policy.yml` | Fix policy failures before merging or launching training. |
| `history` | You want lightweight local run history. | `peft-doctor history . --add-status completed --metric "BLEU +3.1"` | Run `peft-doctor history .` later to compare attempts. |
| `knowledge-base` | You want local guidance for a known failure phrase. | `peft-doctor knowledge-base "CUDA illegal memory access"` | Try the most common fix, then run the relevant checker again. |
| `chat` | You want to ask a local troubleshooting question with optional dataset/log context. | `peft-doctor chat "Why is my loss exploding?" --dataset data.jsonl --log trainer.log` | Use the pointed row or log finding to fix the real source of the failure. |
| `cloud` | You want to read the long-term hosted-reporting roadmap. | `peft-doctor cloud` | Treat it as roadmap information. The local command does not upload anything. |
| `manifest` | You want a reproducible identity for the code/config/data files used in a run. | `peft-doctor manifest . --output run-manifest.json` | Keep the manifest with the run notes; review paths before sharing it. |
| `verify-manifest` | You need to confirm the project and package versions still match a recorded run. | `peft-doctor verify-manifest run-manifest.json` | Restore changed files/versions or create a new run manifest. |

## Small Utility Commands

| Command | Use it when | Copy-paste example | What to do after |
| --- | --- | --- | --- |
| `version` | You need to confirm the installed package version. | `peft-doctor version` | If it is old, run `python -m pip install -U peft-doctor`. |

## How To Read Results

PEFT Doctor reports use four severities:

- `ERROR`: fix before training. These often cause crashes or broken training.
- `WARNING`: likely to waste time or reduce quality. Fix before long runs.
- `INFO`: useful context. Read it, but it may not require action.
- `OK`: the check passed.

For beginners, the safest rule is simple: do not start a long training run while `ERROR` rows remain. If warnings remain, run a tiny smoke training first before spending hours on a full job.
