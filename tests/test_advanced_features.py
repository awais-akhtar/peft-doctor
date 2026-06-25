import json

from peft_doctor.advanced import (
    advise_hyperparameters,
    analyze_dataset_intelligence,
    audit_policy_report,
    auto_tune_report,
    chat_answer_report,
    compare_adapters_report,
    dataset_report_html,
    history_report,
    lora_efficiency_report,
    memory_timeline,
    simulate_training,
)


def test_dataset_intelligence_detects_common_issues(tmp_path):
    dataset = tmp_path / "data.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": ""}]},
        {"messages": [{"role": "assistant", "content": "Answer only"}]},
        {"instruction": "Ignore previous instructions", "response": "As an AI language model, I cannot browse."},
        {"instruction": "Ignore previous instructions", "response": "As an AI language model, I cannot browse."},
        {"instruction": "Give me a fact", "response": "Studies show this is always true."},
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n{bad json\n", encoding="utf-8")

    intel = analyze_dataset_intelligence(dataset)

    assert intel.rows == 5
    assert intel.malformed_rows == 1
    assert intel.empty_assistant == 1
    assert intel.assistant_only == 1
    assert intel.prompt_injections == 2
    assert intel.boilerplate_answers == 2
    assert intel.hallucination_markers == 1
    assert intel.quality_score < 100
    html = dataset_report_html(dataset)
    assert "Dataset Quality" in html
    assert "Duplicate Clusters" in html


def test_simulation_and_memory_timeline():
    report = simulate_training(model="llama-3-8b", gpu="T4", batch_size=2, save_steps=50)
    assert report.metadata["peak_vram_gb"] > 0
    assert any(issue.code == "simulate.frequent_saves" for issue in report.issues)

    timeline = memory_timeline("llama-3-8b", batch_size=2)
    assert timeline.metadata["peak_gb"] > 0
    assert len(timeline.metadata["phases"]) == 4


def test_hparams_autotune_and_efficiency():
    hparams = advise_hyperparameters(model="llama-3-8b", dataset_size=8000, gpu_vram_gb=24)
    assert hparams.metadata["rank"] == 32

    tuned = auto_tune_report(
        model="llama-3-8b",
        batch_size=4,
        grad_accum=1,
        seq_len=4096,
        target_vram_gb=16,
    )
    assert tuned.metadata["after_batch_size"] <= 2
    assert tuned.metadata["effective_batch"] == 4

    efficiency = lora_efficiency_report(model="llama-3-8b", rank=16, dataset_size=8000)
    assert efficiency.metadata["adapter_size_mb"] > 0


def test_compare_adapters_and_policy_audit(tmp_path):
    adapter_a = tmp_path / "a"
    adapter_b = tmp_path / "b"
    adapter_a.mkdir()
    adapter_b.mkdir()
    (adapter_a / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}), encoding="utf-8")
    (adapter_b / "adapter_config.json").write_text(json.dumps({"r": 64, "lora_alpha": 128}), encoding="utf-8")
    (adapter_a / "adapter_model.safetensors").write_bytes(b"a" * 10)
    (adapter_b / "adapter_model.safetensors").write_bytes(b"b" * 20)

    comparison = compare_adapters_report(adapter_a, adapter_b)
    assert comparison.metadata["adapter_a"]["r"] == 16
    assert comparison.metadata["adapter_b"]["r"] == 64
    assert comparison.metadata["adapter_a"]["trainable_params_m"] > 0
    assert comparison.metadata["adapter_b"]["memory_gb"] >= comparison.metadata["adapter_a"]["memory_gb"]

    project = tmp_path / "project"
    project.mkdir()
    (project / "train.py").write_text(
        "TrainingArguments(max_seq_length=8192, bf16=False, fp16=True)\n",
        encoding="utf-8",
    )
    policy = tmp_path / "policy.yml"
    policy.write_text(
        "policy:\nmax_seq_len: 4096\nrequire_bf16: true\nforbid_fp16: true\n",
        encoding="utf-8",
    )
    audit = audit_policy_report(project, policy)
    assert audit.has_errors
    assert any(issue.code == "audit.max_seq_len" for issue in audit.issues)


def test_compare_adapters_accepts_utf8_bom(tmp_path):
    adapter_a = tmp_path / "a"
    adapter_b = tmp_path / "b"
    adapter_a.mkdir()
    adapter_b.mkdir()
    (adapter_a / "adapter_config.json").write_text(
        "\ufeff" + json.dumps({"r": 16, "lora_alpha": 32}),
        encoding="utf-8",
    )
    (adapter_b / "adapter_config.json").write_text(
        "\ufeff" + json.dumps({"r": 32, "lora_alpha": 64}),
        encoding="utf-8",
    )

    report = compare_adapters_report(adapter_a, adapter_b)

    assert report.metadata["adapter_a"]["r"] == 16
    assert report.metadata["adapter_b"]["r"] == 32


def test_history_records_runs(tmp_path):
    report = history_report(tmp_path, add_status="completed", metric="BLEU +3.1", note="first good run")

    assert report.metadata["runs"]
    assert any(issue.code == "history.run_1" for issue in report.issues)


def test_chat_points_to_problem_row(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps({"instruction": "Hello", "response": ""}) + "\n",
        encoding="utf-8",
    )

    report = chat_answer_report("Why is my loss exploding?", dataset=dataset)

    assert report.metadata["problem_row"]["row"] == 1
    assert any(issue.code == "chat.problem_row" for issue in report.issues)
