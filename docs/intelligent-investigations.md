# Investigation Commands

These commands inspect supplied files and report observed evidence. Findings are rule-based or metadata-based; they do not prove model quality or runtime compatibility. They do not use a hosted PEFT Doctor service. Reports can include excerpts from private data, so inspect them before sharing.

## Investigate a run

Place `train.py`, a JSON/JSONL/CSV dataset, and any `.log` files in one project directory:

```bash
peft-doctor investigate ./run --output markdown
```

The investigator connects findings such as empty answers, fully ignored labels, cache settings, precision flags, and known log errors. It reports filenames, row numbers, and causal chains. It does not load a model or execute training.

## Inspect the training objective

```bash
peft-doctor inspect-objective ./run --row 42
peft-doctor inspect-objective data.jsonl --row 42 --tokenizer ./tokenizer
```

The command shows stored text fields, chat turns, and stored `input_ids`/`labels`. With a local tokenizer, it also renders the chat template when the selected row contains messages. A supplied tokenizer must be local. Trainer truncation, packing, collator behavior, and the final loss mask are not reproduced; use an instrumented training batch when those transformations must be observed exactly.

## Inspect gradient flow

`gradients` accepts a JSON summary rather than a Python pickle checkpoint. Example `gradient-summary.json`:

```json
{
  "optimizer_parameters": ["base.layers.0.q_proj.lora_A.default.weight"],
  "parameters": {
    "base.layers.0.q_proj.lora_A.default.weight": {
      "trainable": true,
      "grad_norm": 0.004,
      "update_norm": 0.0002,
      "parameter_norm": 12.4
    },
    "base.layers.0.v_proj.lora_A.default.weight": {
      "trainable": true,
      "grad_norm": 0,
      "update_norm": 0
    }
  }
}
```

```bash
peft-doctor gradients gradient-summary.json
```

Create a snapshot from a live training step with the Python API:

```python
import json
from peft_doctor import collect_gradient_snapshot, snapshot_trainable_parameters

before = snapshot_trainable_parameters(model)
loss.backward()
optimizer.step()
snapshot = collect_gradient_snapshot(model, optimizer, before)
with open("gradient-summary.json", "w", encoding="utf-8") as handle:
    json.dump(snapshot, handle, indent=2)
optimizer.zero_grad(set_to_none=True)
```

Capture `before` before the forward/backward pass, retain gradients through `optimizer.step()`, then collect the snapshot before calling `zero_grad()`. For gradient accumulation, capture around one complete optimizer update rather than an individual micro-batch.

It flags missing or nonfinite gradient measurements, zero gradients/updates, updates with a relative norm at or below `1e-8`, and trainable parameters absent from the recorded optimizer list. The relative-update threshold is a screening heuristic, not a universal training target. It can only assess values captured by the snapshot.

## Minimize a reproducible failure

Create a JSON specification containing an argument array and a local dataset path:

```json
{
  "dataset": "data.jsonl",
  "argv": ["python", "reproduce.py", "--data", "{dataset}"],
  "failure_exit_code": 1
}
```

First review the command. It runs as a subprocess with `shell=False`, but the selected executable can still perform any action available to your account. Then explicitly permit a bounded run:

```bash
peft-doctor minimize failure.json --allow-execution --budget-runs 10
```

The minimizer writes `failure.minimized.jsonl` beside the specification. It stops at the budget or when no smaller subset reproduces the configured exit code. Review the output for sensitive content.

## Find possible data conflicts

```bash
peft-doctor conflicts data.jsonl --threshold 0.82 --limit 5000
```

Candidate rows have similar prompt text and different answers. Similarity is a string heuristic; every pair needs human review.

## Plan task coverage

Create `tasks.yaml`:

```yaml
tasks:
  - name: tool use
    keywords: [tool, function, call]
  - name: multilingual support
    keywords: [Spanish, French, Arabic]
languages:
  - name: Spanish
    keywords: [hola, gracias, buenos días]
conversation_patterns:
  - name: tool interaction
    roles: [user, assistant, tool, assistant]
```

```bash
peft-doctor coverage data.jsonl --tasks tasks.yaml
```

This checks whether supplied task and language markers appear in dataset text, and reports observed chat roles, exact role sequences, tool-call rows, and character-script hints. It does not identify languages from text or measure task competence. Include varied representative markers for each task and language.

## Review privacy patterns

```bash
peft-doctor privacy data.jsonl
peft-doctor privacy data.jsonl --redacted-output data.redacted.jsonl
```

The current local rules flag common email, phone, Hugging Face/PyPI/API-token-shaped strings, private-key headers, and values under field names such as `email`, `address`, `password`, and `api_key`. Redaction can miss sensitive material and can alter examples; inspect the output and apply your organization's policy before training or sharing.

## Choose an approach

Create `requirements.yaml`:

```yaml
facts_change_often: true
consistent_behavior: false
representative_examples: 0
private_data: true
```

```bash
peft-doctor strategy requirements.yaml
```

The report explains why retrieval may suit frequently changing facts and why stable behavior may justify an evaluated fine-tuning trial. It makes no quality prediction.

## Plan a hardware budget

```bash
peft-doctor plan --model llama-3-8b --vram 16 --hours 4 --seq-len 2048 --batch-size 1 --gpu-count 2 --disk-gb 80 --required-disk-gb 60 --dataset-tokens 5000000 --measured-tokens-per-second 500
```

The command compares a formula-based QLoRA VRAM estimate with supplied per-GPU VRAM, compares optional supplied disk budgets, and calculates throughput needed for the time limit. Measured throughput is used only when supplied and should come from a comparable run. The command does not discover provider prices or benchmark your hardware.

## Compare adapter composition inputs

```bash
peft-doctor compose-plan ./adapter-a ./adapter-b
```

The planner compares local adapter configs, base identifiers, recorded revisions, target modules, and hashes for matching tokenizer files. When the metadata aligns for ordinary LoRA adapters, it names weighted composition as a candidate to evaluate. It does not merge weights or prove compatibility; verify pinned base revisions and evaluate the result.

## Check serving metadata

```bash
peft-doctor serving-check ./adapter --engine vllm
```

The command flags metadata that may need attention, including `modules_to_save`, `target_parameters`, token-specific weights, base model, and tokenizer files. It cannot confirm a particular engine/version combination without loading and serving the model.

## Choose the next experiment

```bash
peft-doctor history ./my-run --add-status oom --note "Reduced sequence length from 4096 to 2048"
peft-doctor next-experiment ./my-run
```

This reads `.peft-doctor/history.jsonl`, cites recorded runs, and explains what comparable evidence is missing. It does not declare a winner unless comparable configuration and evaluation data have been recorded.

Add run notes with `history`; `next-experiment` reads the resulting `.peft-doctor/history.jsonl` file. Its suggestions are based only on the recorded evidence.

## Investigate distributed logs

Place per-rank text logs in a directory:

```bash
peft-doctor investigate-distributed ./rank-logs --output markdown
```

It reports common OOM, NCCL/collective, timeout, uneven-input, and device failure signatures. Line numbers across separate files do not establish event order; synchronized timestamps are needed for that.

## Ask about local evidence

```bash
peft-doctor assistant ./my-run --question "Why did training stop?"
```

This searches local scripts, common log names, and a bounded set of local JSON/JSONL files for lines matching the question. It returns file and line references and does not contact a remote model. It is evidence retrieval, not free-form model reasoning.
