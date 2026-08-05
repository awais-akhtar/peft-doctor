from peft_doctor import (
    create_safe_bnb_config,
    create_safe_lora_config,
    create_safe_training_args,
    create_training_recipe,
)
from peft_doctor.recipes import copy_recipe_project, validate_recipe_project


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


def test_training_recipe_completion_only():
    recipe = create_training_recipe(kind="completion-only", model_family="llama")
    assert recipe["recipe"] == "completion-only"
    assert recipe["training_args"]["completion_only_loss"] is True
    assert recipe["training_args"]["response_template"] == "### Response:"


def test_training_recipe_adapter_merge():
    recipe = create_training_recipe(kind="adapter-merge")
    assert recipe["recipe"] == "adapter-merge"
    assert any("adapter-check" in command for command in recipe["commands"])


def test_copy_and_validate_recipe_project(tmp_path):
    destination = tmp_path / "run"
    copy_report = copy_recipe_project("llama3-qlora-colab", destination)
    assert not copy_report.has_errors
    assert (destination / "train.py").exists()
    assert (destination / "sample_data.jsonl").exists()

    validate_report = validate_recipe_project(destination)
    assert not validate_report.has_errors
