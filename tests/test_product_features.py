from peft_doctor.estimator import estimate_vram_gb, infer_params_billion
from peft_doctor.explain import explanation_text, report_to_html, report_to_pdf_bytes, risk_summary
from peft_doctor.profiles import list_model_profiles, profile_for
from peft_doctor.report import DiagnosisReport


def test_infer_params_billion():
    assert infer_params_billion("llama-3-8b") == 8
    assert infer_params_billion("qwen2.5-14B") == 14


def test_estimate_vram_report():
    report = estimate_vram_gb("llama-3-8b", seq_len=2048, batch_size=2, qlora=True, target_vram_gb=16)
    assert report.metadata["estimated_total_gb"] > 0
    assert any(issue.code == "estimate.total_vram" for issue in report.issues)


def test_no_lora_estimate_has_no_adapter_memory():
    report = estimate_vram_gb("llama-3-8b", qlora=False, lora=False)
    issue = next(issue for issue in report.issues if issue.code == "estimate.total_vram")
    assert issue.details["adapter_gb"] == 0.0


def test_profiles_exist():
    names = {profile.name for profile in list_model_profiles()}
    assert {"llama", "qwen", "mistral", "gemma", "phi", "falcon"}.issubset(names)
    assert profile_for("Qwen/Qwen2.5-7B-Instruct").name == "qwen"


def test_explain_and_html_report():
    report = DiagnosisReport()
    report.add("x.error", "Broken", "error", "Something is wrong.", "Fix it.")
    risk = risk_summary(report)
    assert risk["level"] == "HIGH"
    assert "Run risk: HIGH" in explanation_text(report)
    assert "<html" in report_to_html(report)
    assert report_to_pdf_bytes(report).startswith(b"%PDF")
