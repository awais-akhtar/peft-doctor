import json

from typer.main import get_command

from peft_doctor import (
    analyze_token_lengths,
    calculate_training_plan,
    check_adapter_integrity,
    check_disk_budget,
    check_distributed_config,
    check_generation_settings,
    check_resume_compatibility,
    create_run_manifest,
    inspect_trainer_checkpoint,
    verify_run_manifest,
)
from peft_doctor.cli import app


def _write_safetensors(path):
    header = json.dumps(
        {
            "weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00\x00\x00\x00")


def _write_checkpoint(root):
    checkpoint = root / "checkpoint-20"
    checkpoint.mkdir()
    _write_safetensors(checkpoint / "adapter_model.safetensors")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "org/model",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 20}),
        encoding="utf-8",
    )
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(b"state")
    return checkpoint


def test_cli_exposes_all_operational_commands():
    commands = set(get_command(app).commands)

    assert {
        "training-plan",
        "disk-check",
        "checkpoint-check",
        "resume-check",
        "manifest",
        "verify-manifest",
        "token-lengths",
        "distributed-check",
        "generation-check",
        "adapter-integrity",
    }.issubset(commands)


def test_calculate_training_plan_reports_effective_batch_and_steps():
    report = calculate_training_plan(
        dataset_size=1000,
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
        world_size=2,
        num_train_epochs=3,
        save_steps=50,
    )

    assert report.metadata["effective_batch_size"] == 16
    assert report.metadata["steps_per_epoch"] == 63
    assert report.metadata["total_optimizer_steps"] == 189
    assert not report.has_errors


def test_disk_budget_uses_real_free_space(tmp_path):
    report = check_disk_budget(
        tmp_path,
        expected_checkpoint_gb=0,
        save_total_limit=0,
        reserve_gb=0,
    )

    assert report.metadata["free_gb"] > 0
    assert not report.has_errors


def test_checkpoint_check_finds_full_resume_state(tmp_path):
    checkpoint = _write_checkpoint(tmp_path)

    report = inspect_trainer_checkpoint(tmp_path)

    assert report.metadata["checkpoint"] == str(checkpoint)
    assert report.metadata["global_step"] == 20
    assert not report.has_errors
    assert any(issue.code == "checkpoint.optimizer_state" for issue in report.issues)


def test_resume_check_detects_lora_rank_change(tmp_path):
    checkpoint = _write_checkpoint(tmp_path)
    current = tmp_path / "current.json"
    current.write_text(
        json.dumps(
            {
                "base_model_name_or_path": "org/model",
                "r": 64,
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
        encoding="utf-8",
    )

    report = check_resume_compatibility(checkpoint, current_config=current)

    assert report.has_errors
    assert any(issue.code == "resume.mismatch.r" for issue in report.issues)


def test_manifest_detects_file_change(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    train = project / "train.py"
    train.write_text("learning_rate = 2e-4\n", encoding="utf-8")
    manifest = tmp_path / "run.json"

    created = create_run_manifest(project, manifest)
    assert not created.has_errors
    clean = verify_run_manifest(manifest)
    assert any(issue.code == "manifest.verified" for issue in clean.issues)

    train.write_text("learning_rate = 1e-4\n# changed\n", encoding="utf-8")
    changed = verify_run_manifest(manifest)
    assert any(issue.code == "manifest.file_size_changed" for issue in changed.issues)


def test_manifest_rejects_invalid_schema(tmp_path):
    manifest = tmp_path / "run.json"
    manifest.write_text(json.dumps({"schema_version": 99, "files": []}), encoding="utf-8")

    report = verify_run_manifest(manifest)

    assert report.has_errors
    assert any(issue.code == "manifest.schema_unsupported" for issue in report.issues)


def test_token_lengths_profiles_outliers(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"input_ids": [1, 2, 3]}),
                json.dumps({"input_ids": list(range(20))}),
            ]
        ),
        encoding="utf-8",
    )

    report = analyze_token_lengths(dataset, max_length=8)

    assert report.metadata["rows"] == 2
    assert report.metadata["p95"] == 20
    assert report.metadata["over_limit"] == 1


def test_distributed_check_detects_precision_and_offload_conflicts(tmp_path):
    config = tmp_path / "deepspeed.json"
    config.write_text(
        json.dumps(
            {
                "fp16": {"enabled": True},
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": 1,
                    "offload_optimizer": {"device": "cpu"},
                },
            }
        ),
        encoding="utf-8",
    )

    report = check_distributed_config(config)

    assert report.has_errors
    codes = {issue.code for issue in report.issues}
    assert "distributed.precision_conflict" in codes
    assert "distributed.optimizer_offload_stage" in codes


def test_generation_check_handles_invalid_numbers_without_crashing(tmp_path):
    config = tmp_path / "generation_config.json"
    config.write_text(
        json.dumps(
            {
                "do_sample": True,
                "temperature": "hot",
                "top_p": 2,
                "repetition_penalty": "bad",
                "max_new_tokens": 0,
                "min_new_tokens": 4,
                "top_k": -1,
                "num_return_sequences": 0,
            }
        ),
        encoding="utf-8",
    )

    report = check_generation_settings(config)

    assert report.has_errors
    codes = {issue.code for issue in report.issues}
    assert "generation.temperature_invalid" in codes
    assert "generation.top_p_invalid" in codes
    assert "generation.repetition_penalty_invalid" in codes
    assert "generation.top_k_invalid" in codes
    assert "generation.return_sequences_invalid" in codes


def test_adapter_integrity_writes_and_verifies_checksums(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "org/model",
            }
        ),
        encoding="utf-8",
    )
    _write_safetensors(adapter / "adapter_model.safetensors")

    report = check_adapter_integrity(adapter, write_checksums="adapter_checksums.json")

    assert not report.has_errors
    assert report.metadata["tensor_count"] == 1
    assert (adapter / "adapter_checksums.json").exists()

    with (adapter / "adapter_model.safetensors").open("ab") as handle:
        handle.write(b"changed")
    changed = check_adapter_integrity(adapter)
    assert any(issue.code == "integrity.checksum_mismatch" for issue in changed.issues)


def test_adapter_integrity_rejects_corrupted_safetensors(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "org/model"}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"not safetensors")

    report = check_adapter_integrity(adapter)

    assert report.has_errors
    assert any(issue.code == "integrity.safetensors_invalid" for issue in report.issues)


def test_adapter_integrity_rejects_overlapping_tensor_offsets(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "org/model"}),
        encoding="utf-8",
    )
    header = json.dumps(
        {
            "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "b": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"\x00\x00\x00\x00"
    )

    report = check_adapter_integrity(adapter)

    assert report.has_errors
    assert any(issue.code == "integrity.safetensors_invalid" for issue in report.issues)
