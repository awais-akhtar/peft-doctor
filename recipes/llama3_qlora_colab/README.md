# Llama 3 QLoRA Colab

Low-memory starter recipe for Llama 3 style QLoRA runs in Google Colab.

## Run

```bash
python -m pip install -r requirements.txt
python train.py --dry-run
python train.py --max-steps 10
```

Use a GPU runtime before loading the model. For gated models, log in with `huggingface-cli login` or an `HF_TOKEN` environment secret. Do not paste tokens into the notebook or this repo.

