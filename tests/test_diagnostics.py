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


class NoEosLeftPadTokenizer:
    pad_token = "<pad>"
    eos_token = None
    padding_side = "left"
    model_max_length = 2048

    def __len__(self):
        return 12


class Config:
    model_type = "llama"
    use_cache = True


class Weight:
    shape = (10, 4)


class Embedding:
    weight = Weight()


class Param:
    def __init__(self, count, requires_grad):
        self.count = count
        self.requires_grad = requires_grad

    def numel(self):
        return self.count


class Model:
    config = Config()

    def parameters(self):
        return iter([Param(100, False), Param(0, False)])

    def get_input_embeddings(self):
        return Embedding()


def test_diagnose_peft_advanced_checks():
    report = diagnose_peft(
        model=Model(),
        tokenizer=NoEosLeftPadTokenizer(),
        peft_config={
            "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
            "r": 256,
            "lora_alpha": 4,
            "lora_dropout": 0.3,
        },
        training_args={
            "learning_rate": 2e-4,
            "gradient_checkpointing": True,
            "load_in_4bit": True,
            "save_steps": 100,
            "dataloader_num_workers": 0,
            "max_steps": 10,
            "num_train_epochs": 3,
        },
        train_dataset=[{"instruction": "Say hi", "response": "Hi"}],
        sequence_length=2048,
        model_name="meta-llama/Llama-3-8B",
    )

    codes = {issue.code for issue in report.issues}
    assert "tokenizer.eos_token_missing" in codes
    assert "tokenizer.left_padding_training" in codes
    assert "model.use_cache_with_checkpointing" in codes
    assert "model.embedding_resize_needed" in codes
    assert "model.no_trainable_params" in codes
    assert "model.flash_attention_recommended" in codes
    assert "peft.rank_high" in codes
    assert "peft.alpha_low" in codes
    assert "peft.dropout_high" in codes
    assert "training.optimizer_missing" in codes
    assert "training.warmup_missing" in codes
    assert "training.scheduler_missing" in codes
    assert "training.save_total_limit_missing" in codes
    assert "training.seed_missing" in codes
    assert "training.dataloader_workers_zero" in codes
    assert "training.max_steps_overrides_epochs" in codes
