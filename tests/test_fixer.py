import json

from peft_doctor.fixer import repair_config_file, repair_dataset_file, repair_python_source


def test_repair_python_source_adds_safe_fixes():
    source = """
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig

tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForCausalLM.from_pretrained("model")
peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["lm_head"])
args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=4,
    max_seq_length=8192,
    bf16=True,
    fp16=True,
    gradient_checkpointing=True,
)
"""
    fixed, report = repair_python_source(source, model_family="llama")
    assert "tokenizer.pad_token = tokenizer.eos_token" in fixed
    assert "model.config.use_cache = False" in fixed
    assert "fp16=False" in fixed
    assert "per_device_train_batch_size=1" in fixed
    assert "max_seq_length=2048" in fixed
    assert "warmup_ratio=0.03" in fixed
    assert "logging_steps=10" in fixed
    assert "save_strategy=\"steps\"" in fixed
    assert "lm_head" not in fixed
    assert report.metadata["safe_fixes"] >= 7


def test_repair_config_file_writes_output(tmp_path):
    source = tmp_path / "config.json"
    output = tmp_path / "config.fixed.json"
    source.write_text(
        json.dumps(
            {
                "bf16": True,
                "fp16": True,
                "gradient_checkpointing": True,
                "use_cache": True,
                "per_device_train_batch_size": 8,
                "target_modules": ["lm_head"],
            }
        ),
        encoding="utf-8",
    )
    report = repair_config_file(source, output=output, model_family="llama")
    fixed = json.loads(output.read_text(encoding="utf-8"))
    assert fixed["fp16"] is False
    assert fixed["use_cache"] is False
    assert fixed["per_device_train_batch_size"] == 1
    assert "q_proj" in fixed["target_modules"]
    assert report.metadata["written_to"] == str(output)


def test_repair_dataset_file_masks_pad_labels(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"input_ids":[1,0],"labels":[1,0]}\n', encoding="utf-8")
    report = repair_dataset_file(path, pad_token_id=0, write=True)
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["labels"] == [1, -100]
    assert report.metadata["safe_fixes"] == 1
