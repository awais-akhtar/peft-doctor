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


def test_scan_training_log_runtime_failures():
    issues = scan_training_log(
        [
            "RuntimeError: No space left on device",
            "Expected all tensors to be on the same device",
            "mat1 and mat2 shapes cannot be multiplied",
            "token indices sequence length is longer than the specified maximum",
            "step=2 loss=1.0 grad_norm=150",
        ]
    )
    codes = {issue.code for issue in issues}
    assert "log.disk_full" in codes
    assert "log.device_mismatch" in codes
    assert "log.shape_mismatch" in codes
    assert "log.sequence_too_long" in codes
    assert "grad_norm.spike" in codes
