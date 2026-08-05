import json

from typer.main import get_command

from peft_doctor import (
    audit_label_masking,
    check_chat_template,
    check_dataset_leakage,
    check_fsdp_config,
    check_hotswap_compatibility,
    check_lora_config,
    check_quantization_config,
    check_sft_config,
    inspect_adapter_tensors,
    inventory_checkpoints,
)
from peft_doctor.cli import app


def _write_safetensors(path, tensors=None):
    tensors = tensors or {"layer.lora_A.weight": {"dtype": "F32", "shape": [2, 3]}}
    offset = 0
    header = {}
    for name, info in tensors.items():
        size = 4
        for dimension in info["shape"]:
            size *= dimension
        header[name] = {**info, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(offset))


def _write_adapter(path, *, rank=16, targets=None, base="org/model"):
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": base,
                "r": rank,
                "lora_alpha": rank * 2,
                "target_modules": targets or ["q_proj", "v_proj"],
            }
        ),
        encoding="utf-8",
    )
    _write_safetensors(path / "adapter_model.safetensors")


def test_cli_exposes_current_stack_checks():
    commands = set(get_command(app).commands)
    assert {
        "sft-check",
        "fsdp-check",
        "hotswap-check",
        "lora-config-check",
        "checkpoint-list",
        "leakage-check",
        "adapter-tensors",
        "chat-template-check",
        "label-audit",
        "quantization-check",
    }.issubset(commands)


def test_sft_check_matches_loss_mode_to_dataset_and_template(tmp_path):
    config = tmp_path / "sft.json"
    config.write_text(json.dumps({"assistant_only_loss": True, "packing": True}), encoding="utf-8")
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({"text": "not conversational"}) + "\n", encoding="utf-8")
    template = tmp_path / "template.jinja"
    template.write_text("{% for message in messages %}{{ message.content }}{% endfor %}", encoding="utf-8")

    report = check_sft_config(config, dataset_path=dataset, chat_template_path=template)

    assert report.has_errors
    codes = {issue.code for issue in report.issues}
    assert "sft.assistant_loss_dataset" in codes
    assert "sft.generation_markers_missing" in codes


def test_fsdp_check_detects_loading_and_compile_conflicts(tmp_path):
    config = tmp_path / "accelerate.yml"
    config.write_text(
        "\n".join(
            [
                "distributed_type: FSDP",
                "num_processes: 1",
                "fsdp_config:",
                "  fsdp_cpu_ram_efficient_loading: true",
                "  fsdp_sync_module_states: false",
                "  fsdp_state_dict_type: FULL_STATE_DICT",
                "  fsdp_use_orig_params: false",
                "dynamo_config:",
                "  dynamo_backend: INDUCTOR",
            ]
        ),
        encoding="utf-8",
    )

    report = check_fsdp_config(config)

    assert report.has_errors
    assert any(issue.code == "fsdp.cpu_loading_sync" for issue in report.issues)
    assert any(issue.code == "fsdp.compile_orig_params" for issue in report.issues)


def test_hotswap_check_enforces_targets_and_prepared_rank(tmp_path):
    loaded = tmp_path / "loaded"
    incoming = tmp_path / "incoming"
    _write_adapter(loaded, rank=16, targets=["q_proj"])
    _write_adapter(incoming, rank=64, targets=["q_proj", "v_proj"])

    report = check_hotswap_compatibility(loaded, incoming, compiled=True, prepared_max_rank=32)

    assert report.has_errors
    codes = {issue.code for issue in report.issues}
    assert "hotswap.target_modules" in codes
    assert "hotswap.rank_exceeds_prepared" in codes


def test_lora_config_check_covers_trainable_tokens_and_alora(tmp_path):
    config = tmp_path / "adapter.json"
    config.write_text(
        json.dumps(
            {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": "all-linear",
                "trainable_token_indices": [100, 101],
                "alora_invocation_tokens": [42],
            }
        ),
        encoding="utf-8",
    )

    report = check_lora_config(config)

    codes = {issue.code for issue in report.issues}
    assert "lora.trainable_tokens" in codes
    assert "lora.weight_tying_review" in codes
    assert "lora.alora_merge" in codes


def test_checkpoint_inventory_selects_latest_valid(tmp_path):
    valid = tmp_path / "checkpoint-10"
    invalid = tmp_path / "checkpoint-20"
    _write_adapter(valid)
    (valid / "trainer_state.json").write_text(json.dumps({"global_step": 10}), encoding="utf-8")
    invalid.mkdir()

    report = inventory_checkpoints(tmp_path)

    assert report.metadata["latest_valid"] == str(valid)
    assert any(issue.code == "checkpoint_list.checkpoint-20" and issue.severity == "error" for issue in report.issues)


def test_leakage_check_finds_exact_and_near_overlap(tmp_path):
    train = tmp_path / "train.jsonl"
    eval_data = tmp_path / "eval.jsonl"
    train.write_text(
        "\n".join([json.dumps({"text": "alpha beta gamma"}), json.dumps({"text": "one two three four"})]),
        encoding="utf-8",
    )
    eval_data.write_text(
        "\n".join([json.dumps({"text": "alpha beta gamma"}), json.dumps({"text": "one two three"})]),
        encoding="utf-8",
    )

    report = check_dataset_leakage(train, eval_data, threshold=0.7)

    codes = {issue.code for issue in report.issues}
    assert "leakage.exact" in codes
    assert "leakage.near" in codes


def test_adapter_tensor_report_uses_exact_header_metadata(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    _write_safetensors(
        adapter / "adapter_model.safetensors",
        {
            "a": {"dtype": "F32", "shape": [2, 3]},
            "b": {"dtype": "F32", "shape": [4]},
        },
    )

    report = inspect_adapter_tensors(adapter)

    assert report.metadata["tensor_count"] == 2
    assert report.metadata["parameter_count"] == 10
    assert report.metadata["dtype_tensor_counts"] == {"F32": 2}


def test_chat_template_check_requires_generation_markers(tmp_path):
    template = tmp_path / "template.jinja"
    template.write_text("{% for message in messages %}{{ message.content }}{% endfor %}", encoding="utf-8")

    report = check_chat_template(template, assistant_only_loss=True)

    assert report.has_errors
    assert any(issue.code == "chat_template.generation_missing" for issue in report.issues)


def test_label_audit_finds_unmasked_padding_and_empty_supervision(tmp_path):
    dataset = tmp_path / "tokens.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"input_ids": [1, 0], "labels": [1, 0]}),
                json.dumps({"input_ids": [2, 3], "labels": [-100, -100]}),
            ]
        ),
        encoding="utf-8",
    )

    report = audit_label_masking(dataset, pad_token_id=0)

    assert report.has_errors
    assert any(issue.code == "labels.pad_unmasked" for issue in report.issues)
    assert any(issue.code == "labels.all_ignored" for issue in report.issues)


def test_quantization_check_detects_mode_and_t4_dtype_conflicts(tmp_path):
    config = tmp_path / "bnb.json"
    config.write_text(
        json.dumps(
            {
                "load_in_4bit": True,
                "load_in_8bit": True,
                "bnb_4bit_quant_type": "fp4",
                "bnb_4bit_compute_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )

    report = check_quantization_config(config, gpu="T4")

    assert report.has_errors
    codes = {issue.code for issue in report.issues}
    assert "quantization.mode_conflict" in codes
    assert "quantization.nf4" in codes
    assert "quantization.t4_bf16" in codes
