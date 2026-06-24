from peft_doctor import create_safe_bnb_config, create_safe_lora_config, create_safe_training_args


def test_safe_lora_config_dict():
    config = create_safe_lora_config(model_family="llama", as_dict=True)
    assert config["r"] == 16
    assert "q_proj" in config["target_modules"]
    assert config["task_type"] == "CAUSAL_LM"


def test_safe_bnb_config_dict():
    config = create_safe_bnb_config(as_dict=True)
    assert config["load_in_4bit"] is True
    assert config["bnb_4bit_quant_type"] == "nf4"


def test_safe_training_args():
    args = create_safe_training_args()
    assert args["per_device_train_batch_size"] == 1
    assert args["gradient_checkpointing"] is True
    assert args["warmup_ratio"] == 0.03
    assert args["lr_scheduler_type"] == "cosine"
    assert args["max_grad_norm"] == 1.0
    assert args["save_total_limit"] == 2
    assert args["seed"] == 42
