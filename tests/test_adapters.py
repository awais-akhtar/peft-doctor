import json

from peft_doctor import diagnose_adapter_merge


def test_diagnose_local_adapter_ok(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "base/model",
                "target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"")

    report = diagnose_adapter_merge(
        base_model="base/model",
        adapter=str(adapter),
        output_dir=tmp_path / "merged",
    )

    assert not report.has_errors
    assert any(issue.code == "adapter_merge.config_found" for issue in report.issues)
    assert any(issue.code == "adapter_merge.weights_found" for issue in report.issues)


def test_diagnose_local_adapter_missing_files(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    report = diagnose_adapter_merge(adapter=str(adapter), output_dir=tmp_path / "merged")

    assert report.has_errors
    assert any(issue.code == "adapter_merge.config_missing" for issue in report.issues)


def test_diagnose_quantized_merge_warns(tmp_path):
    report = diagnose_adapter_merge(
        base_model="base/model",
        adapter="user/adapter",
        output_dir=tmp_path / "merged",
        load_in_4bit=True,
    )

    assert any(issue.code == "adapter_merge.quantized_merge_risky" for issue in report.issues)
