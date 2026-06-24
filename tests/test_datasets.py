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
