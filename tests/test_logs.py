from peft_doctor import NanLossGuard, scan_training_log


def test_nan_loss_guard_detects_nan():
    guard = NanLossGuard()
    issues = guard.update({"loss": "nan"})
    assert issues
    assert issues[0].code == "loss.nan"


def test_loss_jump_warning():
    guard = NanLossGuard()
    assert guard.update({"loss": 1.0}) == []
    issues = guard.update({"loss": 5.0})
    assert issues[0].code == "loss.jump"


def test_scan_training_log_text():
    issues = scan_training_log(["step=1 loss=nan"])
    assert any(issue.code == "loss.nan" for issue in issues)
