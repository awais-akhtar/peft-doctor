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
| `fix` | You want PEFT Doctor to preview safe patches for common script, config, or dataset mistakes. | `peft-doctor fix --dry-run train.py` | Read the patch list. If it looks right, write a copy with `--output train.fixed.py` or apply with `--write`. |
| `diagnose` | You want a plain-English explanation of why a run may fail. | `peft-doctor diagnose train.py --dataset data.jsonl --model llama-3-8b --gpu T4` | Apply the recommended fixes in order, then run `simulate` or `check` again. |
| `simulate` | You want to predict whether training will start without actually training. | `peft-doctor simulate --model llama-3-8b --dataset data.jsonl --gpu L4 --seq-len 2048 --batch-size 2` | If it warns about eval OOM or disk usage, reduce eval batch size or checkpoint frequency before training. |
| `optimize` | You want one command that checks config, dataset, tokenizer, LoRA, memory, and project score. | `peft-doctor optimize . --html-report optimize-report.html` | Open the HTML report and fix the warnings before using `--write`. |
| `score` | You want a quick readiness score for a training project. | `peft-doctor score train.py --dataset data.jsonl --gpu T4` | Improve the lowest sub-score, usually dataset or configuration. |

## Memory, Speed, And Cost Commands

| Command | Use it when | Copy-paste example | What to do after |
| --- | --- | --- | --- |
| `estimate` | You want a rough VRAM estimate before loading a model. | `peft-doctor estimate --model llama-3-8b --seq-len 2048 --batch-size 2 --qlora --target-vram 16` | If estimated VRAM is above your GPU, lower batch size or sequence length and enable QLoRA/checkpointing. |
| `memory-timeline` | You want to see where memory spikes: load, forward, backward, optimizer, or peak. | `peft-doctor memory-timeline --model llama-3-8b --seq-len 4096 --batch-size 1 --qlora` | If backward or optimizer memory is too high, reduce sequence length or effective batch size. |
| `estimate-cost` | You are choosing between cloud GPUs. | `peft-doctor estimate-cost --model llama-3-8b --dataset-size 8000 --gpu L4 --gpu A100` | Pick the GPU with the best cost/time tradeoff for your budget. These are planning estimates, not provider quotes. |
| `auto-tune` | Your batch size is too large but you want to keep the same effective batch. | `peft-doctor auto-tune --model llama-3-8b --batch-size 4 --grad-accum 1 --target-vram 16` | Use the suggested smaller batch size and higher gradient accumulation. |
| `gpu-fingerprint` | You want GPU-specific advice for precision and attention settings. | `peft-doctor gpu-fingerprint "RTX 3060"` | Follow the precision warning, especially on consumer GPUs with fp16 instability. |
| `monitor` | You already have a trainer log and want training health feedback. | `peft-doctor monitor trainer.log` | If NaN chance or loss spikes are high, lower LR, inspect labels, and prefer bf16 where supported. |

## Dataset Commands

| Command | Use it when | Copy-paste example | What to do after |
| --- | --- | --- | --- |
| `dataset-doctor` | You want to check dataset rows before training. | `peft-doctor dataset-doctor data.jsonl --sequence-length 2048` | Remove empty assistant responses, malformed rows, duplicates, and examples longer than your sequence length. |
| `inspect-dataset` | You want a quick format check for `.json`, `.jsonl`, `.csv`, or `.txt` data. | `peft-doctor inspect-dataset data.jsonl` | Make sure the dataset uses chat messages, instruction/response columns, prompt/completion columns, or pretokenized fields. |
| `dataset-intel` | You want a quality score and issue counts. | `peft-doctor dataset-intel data.jsonl` | Clean the highest-count issue first, then rerun the command. |
| `dataset-report` | You want an HTML report with histograms and duplicate clusters. | `peft-doctor dataset-report data.jsonl --output dataset-report.html` | Open the HTML file locally and inspect outliers before training. |

## Adapter And Merge Commands

| Command | Use it when | Copy-paste example | What to do after |
| --- | --- | --- | --- |
| `inspect-adapter` | You have a saved LoRA adapter and want to check it before upload or merge. | `peft-doctor inspect-adapter ./adapter` | Confirm `adapter_config.json` and adapter weights exist. |
| `adapter-check` | You want to check a merge plan without loading the full model. | `peft-doctor adapter-check --base-model meta-llama/Llama-2-7b-hf --adapter your-user/your-lora-adapter --output-dir merged-model` | Fix missing base model, tokenizer, or quantized-merge warnings before merging. |
| `merge-adapter` | You are ready to merge a LoRA adapter into the base model. | `peft-doctor merge-adapter --base-model meta-llama/Llama-2-7b-hf --adapter your-user/your-lora-adapter --output-dir merged-model --dtype fp16` | Test the merged model locally before uploading. Do not paste tokens into the command. |
| `compare-adapters` | You have two adapters and want to choose one. | `peft-doctor compare-adapters ./adapter-r16 ./adapter-r64` | Prefer the smaller adapter when eval quality is close; use the larger one only if it improves validation. |
| `lora-efficiency` | You want to estimate adapter size and expected tradeoffs before training. | `peft-doctor lora-efficiency --model llama-3-8b --rank 32 --dataset-size 8000` | Use it for planning, then confirm quality with a held-out eval set. |

## Configuration And Model Commands

| Command | Use it when | Copy-paste example | What to do after |
| --- | --- | --- | --- |
| `profiles` | You want the built-in profile for a model family. | `peft-doctor profiles qwen` | Use the profile to choose target modules and prompt format. |
| `targets` | You need LoRA `target_modules`. | `peft-doctor targets --model meta-llama/Llama-3-8B` | Copy the recommended list into your `LoraConfig`. |
| `safe-config` | You want starter LoRA, QLoRA, and training settings. | `peft-doctor safe-config --family llama` | Copy the config into your training script and adjust only after a short validation run. |
| `advise-hparams` | You are unsure about LoRA rank, alpha, and dropout. | `peft-doctor advise-hparams --model llama-3-8b --dataset-size 8000 --gpu-vram 24` | Start with the recommended values, then tune rank based on validation quality and overfitting. |
| `upgrade-suggestions` | You want to know if your fine-tuning packages are old. | `peft-doctor upgrade-suggestions` | Upgrade packages carefully in a fresh environment if the report shows known-risk versions. |

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
| `benchmark` | You want to print a known validation-matrix style recipe entry. | `peft-doctor benchmark --recipe llama3-qlora-colab` | Use it as proof text for documentation, not as a live benchmark run. |
| `audit` | Your team has standards for max sequence length, precision, or dataset validation. | `peft-doctor audit . --policy peft-policy.yml` | Fix policy failures before merging or launching training. |
| `history` | You want lightweight local run history. | `peft-doctor history . --add-status completed --metric "BLEU +3.1"` | Run `peft-doctor history .` later to compare attempts. |
| `knowledge-base` | You want local guidance for a known failure phrase. | `peft-doctor knowledge-base "CUDA illegal memory access"` | Try the most common fix, then run the relevant checker again. |
| `chat` | You want to ask a local troubleshooting question with optional dataset/log context. | `peft-doctor chat "Why is my loss exploding?" --dataset data.jsonl --log trainer.log` | Use the pointed row or log finding to fix the real source of the failure. |
| `cloud` | You want to read the long-term hosted-reporting roadmap. | `peft-doctor cloud` | Treat it as roadmap information. The local command does not upload anything. |

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
