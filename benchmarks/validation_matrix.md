# Validation Matrix

These rows are packaged pre-flight validation scenarios for PEFT Doctor recipes. They are designed to show the issue class, the safe auto-fix, and the saved failed run before users spend GPU time.

| Model | Dataset | GPU | Issue found | Auto-fix worked | Time saved |
| --- | --- | --- | --- | --- | --- |
| Llama-3-8B | alpaca sample | T4 | OOM risk | yes | avoided failed run |
| Qwen2.5 | chat data | L4 | bad EOS | yes | fixed generation stop |
| Mistral | completion data | A100 | wrong masking | yes | loss started working |
| Gemma | tiny chat sample | T4 | OOM risk | yes | kept first run small |
| Mistral | adapter export | A100 | risky merge path | yes | avoided broken export |

Run a packaged validation row:

```bash
peft-doctor benchmark --recipe llama3-qlora-colab
peft-doctor benchmark --recipe qwen2-qlora-colab
peft-doctor benchmark --recipe completion-only-sft
```

Create a local validation report:

```bash
peft-doctor validate --model qwen --dataset sample_data.jsonl --report report.md
```

