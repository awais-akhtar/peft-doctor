from peft_doctor import diagnose_peft


class Tokenizer:
    pad_token = None
    eos_token = "</s>"
    model_max_length = 2048


def test_diagnose_peft_returns_report():
    report = diagnose_peft(
        tokenizer=Tokenizer(),
        peft_config={"target_modules": ["q_proj"], "task_type": "CAUSAL_LM"},
        training_args={
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "gradient_checkpointing": True,
            "bf16": True,
        },
        train_dataset=[{"instruction": "Say hi", "response": "Hi"}],
        model_name="meta-llama/Llama-3-8B",
    )

    assert report.issues
    assert any(issue.code == "tokenizer.pad_token_missing" for issue in report.issues)
    assert any(issue.code == "dataset.instruction_columns" for issue in report.issues)
