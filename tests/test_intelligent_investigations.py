import json
import math
import sys
from pathlib import Path

from typer.testing import CliRunner

from peft_doctor.cli import app
from peft_doctor.intelligence import (
    advise_strategy,
    answer_with_evidence,
    collect_gradient_snapshot,
    find_dataset_conflicts,
    inspect_gradient_snapshot,
    inspect_objective,
    inspect_privacy,
    inspect_serving,
    investigate_distributed_logs,
    investigate_project,
    minimize_failure,
    plan_adapter_composition,
    plan_hardware,
    plan_training_coverage,
    recommend_next_experiment,
    snapshot_trainable_parameters,
)


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _issue_codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_investigator_connects_dataset_script_and_log(tmp_path):
    _jsonl(tmp_path / "data.jsonl", [{"prompt": "Question", "answer": ""}, {"labels": [-100, -100]}])
    (tmp_path / "train.py").write_text("gradient_checkpointing=True\nfp16=True\nbf16=True\n", encoding="utf-8")
    (tmp_path / "trainer.log").write_text("loss = nan\n", encoding="utf-8")

    report = investigate_project(tmp_path)

    assert {"investigate.empty_to_loss", "investigate.masked_to_loss", "investigate.cache_checkpointing", "investigate.precision_conflict"} <= _issue_codes(report)
    assert report.metadata["rows_inspected"] == 2


def test_objective_inspector_reports_text_messages_and_masks(tmp_path):
    dataset = _jsonl(tmp_path / "data.jsonl", [{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}], "input_ids": [1, 2], "labels": [-100, 2]}])

    report = inspect_objective(dataset, 1)

    assert {"objective.message", "objective.label_mask", "objective.token_ids"} <= _issue_codes(report)


def test_objective_inspector_resolves_dataset_inside_project(tmp_path):
    _jsonl(tmp_path / "train.jsonl", [{"prompt": "Question", "answer": "Answer"}])

    report = inspect_objective(tmp_path, 1)

    assert report.metadata["dataset"].endswith("train.jsonl")
    assert "objective.text_field" in _issue_codes(report)


def test_gradient_snapshot_collects_optimizer_membership_and_updates():
    class Parameter:
        requires_grad = True
        grad = None
        device = "cpu"

        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def clone(self):
            return Parameter(self.value)

        def to(self, _device):
            return self

        def float(self):
            return self

        def norm(self):
            return self

        def item(self):
            return abs(self.value)

        def __sub__(self, other):
            return Parameter(self.value - other.value)

    class Model:
        weight = Parameter(1.0)

        def named_parameters(self):
            return iter([("adapter.weight", self.weight)])

    model = Model()
    before = snapshot_trainable_parameters(model)
    model.weight.value = 0.9
    model.weight.grad = Parameter(0.2)
    optimizer = type("Optimizer", (), {"param_groups": [{"params": [model.weight]}]})()

    summary = collect_gradient_snapshot(model, optimizer, before)

    assert math.isclose(summary["parameters"]["adapter.weight"]["update_norm"], 0.1)
    assert summary["optimizer_parameters"] == ["adapter.weight"]


def test_gradient_inspector_detects_nonfinite_zero_and_excluded(tmp_path):
    snapshot = tmp_path / "gradients.json"
    snapshot.write_text(json.dumps({"optimizer_parameters": [], "parameters": {"a": {"trainable": True, "grad_norm": 0}, "b": {"trainable": True, "grad_norm": float("inf")}}}), encoding="utf-8")

    codes = _issue_codes(inspect_gradient_snapshot(snapshot))

    assert {"gradients.zero", "gradients.nonfinite", "gradients.not_in_optimizer"} <= codes


def test_failure_minimizer_requires_opt_in_and_minimizes(tmp_path):
    data = _jsonl(tmp_path / "data.jsonl", [{"keep": False}, {"keep": True}, {"keep": False}])
    script = tmp_path / "repro.py"
    script.write_text("import json,sys\nrows=[json.loads(x) for x in open(sys.argv[1])]\nsys.exit(1 if any(r['keep'] for r in rows) else 0)\n", encoding="utf-8")
    spec = tmp_path / "failure.json"
    spec.write_text(json.dumps({"dataset": data.name, "argv": [sys.executable, str(script), "{dataset}"], "failure_exit_code": 1}), encoding="utf-8")

    denied = minimize_failure(spec, allow_execution=False, budget_runs=4)
    allowed = minimize_failure(spec, allow_execution=True, budget_runs=10)

    assert "minimize.execution_required" in _issue_codes(denied)
    assert allowed.metadata["remaining_rows"] == 1
    assert Path(allowed.metadata["output"]).exists()


def test_failure_minimizer_rejects_inline_command(tmp_path):
    data = _jsonl(tmp_path / "data.jsonl", [{"x": 1}])
    spec = tmp_path / "failure.json"
    spec.write_text(json.dumps({"dataset": data.name, "argv": [sys.executable, "-c", "pass {dataset}"]}), encoding="utf-8")

    report = minimize_failure(spec, allow_execution=True, budget_runs=1)

    assert "minimize.spec_invalid" in _issue_codes(report)


def test_conflict_finder_flags_candidate_pairs_for_review(tmp_path):
    dataset = _jsonl(tmp_path / "data.jsonl", [{"prompt": "What is the capital of France?", "answer": "Paris"}, {"prompt": "What is the capital of France", "answer": "Lyon"}])

    report = find_dataset_conflicts(dataset)

    assert "conflicts.similar_prompt_different_answer" in _issue_codes(report)
    assert report.issues[0].details["row_a"] == 1


def test_coverage_reports_configured_tasks_roles_and_markers(tmp_path):
    dataset = _jsonl(tmp_path / "data.jsonl", [{"messages": [{"role": "user", "content": "Hola tool"}, {"role": "assistant", "content": "Done"}, {"role": "tool", "content": "ok"}]}])
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text("tasks:\n  - name: tools\n    keywords: [tool]\nlanguages:\n  - name: Spanish\n    keywords: [Hola]\nconversation_patterns:\n  - name: tool flow\n    roles: [user, assistant, tool]\n", encoding="utf-8")

    report = plan_training_coverage(dataset, tasks)

    assert report.metadata["rows_with_tool_calls_or_tool_role"] == 1
    assert "coverage.task" in _issue_codes(report)
    assert "coverage.language_markers" in _issue_codes(report)
    assert "coverage.conversation_pattern" in _issue_codes(report)


def test_privacy_scan_redacts_field_and_inline_patterns(tmp_path):
    dataset = _jsonl(tmp_path / "data.jsonl", [{"email": "person@example.com", "text": "contact person@example.com"}])
    output = tmp_path / "redacted.jsonl"

    report = inspect_privacy(dataset, redacted_output=output)

    redacted = output.read_text(encoding="utf-8")
    assert "person@example.com" not in redacted
    assert "[REDACTED_SENSITIVE_FIELD]" in redacted
    assert "privacy.redacted_copy" in _issue_codes(report)


def test_privacy_refuses_to_overwrite_source(tmp_path):
    dataset = _jsonl(tmp_path / "data.jsonl", [{"email": "person@example.com"}])

    report = inspect_privacy(dataset, redacted_output=dataset)

    assert "privacy.output_overwrite" in _issue_codes(report)
    assert "person@example.com" in dataset.read_text(encoding="utf-8")


def test_strategy_advisor_uses_requirements_without_quality_claims(tmp_path):
    requirements = tmp_path / "requirements.yaml"
    requirements.write_text("facts_change_often: true\nconsistent_behavior: true\nrepresentative_examples: 200\nprivate_data: true\n", encoding="utf-8")

    codes = _issue_codes(advise_strategy(requirements))

    assert {"strategy.retrieval", "strategy.finetune", "strategy.privacy", "strategy.evaluation"} <= codes


def test_hardware_plan_checks_memory_and_measured_time():
    report = plan_hardware("llama-3-8b", 16, 2, dataset_tokens=100000, measured_tokens_per_second=1)

    assert "plan.memory_fit" in _issue_codes(report) or "plan.memory_adjustment" in _issue_codes(report) or "plan.memory_no_fit" in _issue_codes(report)
    assert "plan.throughput_needed" in _issue_codes(report)
    assert "plan.measured_duration" in _issue_codes(report)


def test_hardware_plan_rejects_invalid_budget_before_estimation():
    report = plan_hardware("unknown-model", 0, 2)

    assert "plan.invalid_budget" in _issue_codes(report)


def test_adapter_composition_flags_base_and_target_mismatch(tmp_path):
    paths = [tmp_path / "a", tmp_path / "b"]
    for path, base, targets in zip(paths, ("base-a", "base-b"), (["q_proj"], ["v_proj"])):
        path.mkdir()
        (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base, "peft_type": "LORA", "target_modules": targets}), encoding="utf-8")

    codes = _issue_codes(plan_adapter_composition(paths))

    assert {"compose.base_mismatch", "compose.target_mismatch"} <= codes


def test_serving_check_reports_metadata_gaps(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "org/model", "target_parameters": ["layers.0"], "modules_to_save": ["embed_tokens"]}), encoding="utf-8")

    codes = _issue_codes(inspect_serving(adapter, "vllm"))

    assert {"serving.base_model", "serving.modules_to_save", "serving.target_parameters", "serving.tokenizer_files"} <= codes


def test_next_experiment_reads_history_command_output_path(tmp_path):
    history = tmp_path / ".peft-doctor"
    history.mkdir()
    (history / "history.jsonl").write_text(json.dumps({"status": "oom", "note": "long sequence"}) + "\n", encoding="utf-8")

    report = recommend_next_experiment(tmp_path)

    assert report.metadata["records_read"] == 1
    assert "next_experiment.recommendation" in _issue_codes(report)


def test_distributed_investigator_reports_file_and_line(tmp_path):
    (tmp_path / "rank0.log").write_text("startup\nCUDA out of memory\n", encoding="utf-8")

    report = investigate_distributed_logs(tmp_path)

    issue = next(issue for issue in report.issues if issue.code == "distributed_logs.cuda_oom")
    assert issue.details["line"] == 2
    assert "rank0.log:2" in issue.message


def test_local_evidence_assistant_cites_matching_lines(tmp_path):
    (tmp_path / "train.py").write_text("learning_rate = 0.01\n", encoding="utf-8")

    report = answer_with_evidence(tmp_path, "learning rate")

    assert "assistant.evidence" in _issue_codes(report)
    assert report.issues[0].details["line"] == 1


def test_new_cli_commands_are_registered_and_invokable():
    runner = CliRunner()
    commands = ["investigate", "inspect-objective", "gradients", "minimize", "conflicts", "coverage", "privacy", "strategy", "plan", "compose-plan", "serving-check", "next-experiment", "investigate-distributed", "assistant"]

    for command in commands:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, (command, result.output, result.exception)


def test_next_experiment_history_location_matches_history_writer(tmp_path):
    runner = CliRunner()

    recorded = runner.invoke(app, ["history", str(tmp_path), "--add-status", "oom", "--note", "shorten context"])
    report = recommend_next_experiment(tmp_path)

    assert recorded.exit_code == 0, recorded.output
    assert report.metadata["records_read"] == 1
    assert "sequence length" in next(issue.message for issue in report.issues if issue.code == "next_experiment.recommendation")


def test_pyproject_readme_includes_detailed_new_feature_guide():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert 'readme = "README.md"' in pyproject
    assert "https://github.com/awais-akhtar/peft-doctor/blob/main/docs/intelligent-investigations.md" in readme
