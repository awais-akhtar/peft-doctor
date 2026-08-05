from peft_doctor.datasets import check_dataset, has_instruction_markers, load_dataset_records
from peft_doctor.report import DiagnosisReport


def test_instruction_markers():
    assert has_instruction_markers("### Instruction:\nDo it\n### Response:\nDone")


def test_dataset_instruction_columns():
    report = DiagnosisReport()
    check_dataset(report, train_dataset=[{"instruction": "Say hi", "response": "Hi"}])
    assert any(issue.code == "dataset.instruction_columns" for issue in report.issues)


def test_load_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"instruction": "A", "response": "B"}\n', encoding="utf-8")
    rows = load_dataset_records(path)
    assert rows == [{"instruction": "A", "response": "B"}]


def test_dataset_detects_duplicates_and_short_response():
    report = DiagnosisReport()
    check_dataset(
        report,
        train_dataset=[
            {"instruction": "Say hi", "response": ""},
            {"instruction": "Say hi", "response": ""},
        ],
    )

    assert any(issue.code == "dataset.duplicates" for issue in report.issues)
    assert any(issue.code == "dataset.short_responses" for issue in report.issues)


def test_dataset_detects_bad_labels_and_overlap():
    report = DiagnosisReport()
    check_dataset(
        report,
        train_dataset=[
            {"text": "same row", "input_ids": [1, 2], "labels": [-100, -100]},
            {"text": "bad lengths", "input_ids": [1, 2, 3], "labels": [1]},
        ],
        eval_dataset=[{"text": "same row"}],
    )

    assert any(issue.code == "dataset.labels_all_ignored" for issue in report.issues)
    assert any(issue.code == "dataset.label_length_mismatch" for issue in report.issues)
    assert any(issue.code == "dataset.train_eval_overlap" for issue in report.issues)


def test_dataset_detects_long_rows_and_missing_assistant():
    report = DiagnosisReport()
    check_dataset(
        report,
        train_dataset=[
            {"text": "word " * 100},
            {"messages": [{"role": "user", "content": "hello"}]},
        ],
        sequence_length=8,
    )

    assert any(issue.code == "dataset.long_rows" for issue in report.issues)
    assert any(issue.code == "dataset.chat_missing_assistant" for issue in report.issues)


class TemplateTokenizer:
    eos_token = "</s>"
    pad_token_id = 0


def test_dataset_detects_advanced_collator_risks():
    report = DiagnosisReport()
    check_dataset(
        report,
        train_dataset=[
            {"text": "short", "input_ids": [1, 2, 0], "labels": [1, 2, 0]},
            {"text": "� <html> lorem ipsum " + ("long " * 100)},
        ],
        tokenizer=TemplateTokenizer(),
        training_args={
            "packing": True,
            "response_template": "### Response:",
            "group_by_length": False,
        },
        sequence_length=2048,
    )

    codes = {issue.code for issue in report.issues}
    assert "dataset.pad_token_in_labels" in codes
    assert "dataset.response_template_missing" in codes
    assert "dataset.packing_without_eos" in codes
    assert "dataset.length_variance_high" in codes
    assert "dataset.bad_text_artifacts" in codes
    assert "dataset.too_small" in codes


def test_dataset_detects_modern_sft_schema_risks():
    report = DiagnosisReport()
    check_dataset(
        report,
        train_dataset=[
            {
                "messages": [
                    {"role": "user", "content": "Use a tool"},
                    {"role": "assistant", "content": "", "tool_calls": [{"name": "search"}]},
                    {"role": "critic", "content": "wrong role"},
                    {"role": "assistant", "content": "final"},
                ],
                "image": "sample.png",
            },
            {"instruction": "Explain LoRA", "response": "LoRA adapts small matrices."},
        ],
        training_args={"completion_only_loss": True},
    )

    codes = {issue.code for issue in report.issues}
    assert "dataset.empty_assistant_messages" in codes
    assert "dataset.unknown_chat_roles" in codes
    assert "dataset.tool_calls_missing_tools" in codes
    assert "dataset.vision_columns_detected" in codes
    assert "dataset.mixed_prompt_schemas" in codes
    assert "dataset.multi_turn_without_assistant_loss" in codes


def test_dataset_detects_loss_mode_mismatch():
    report = DiagnosisReport()
    check_dataset(
        report,
        train_dataset=[{"instruction": "Say hi", "response": "Hi"}],
        training_args={"assistant_only_loss": True},
    )

    assert any(issue.code == "dataset.assistant_loss_without_chat" for issue in report.issues)
