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
