from peft_doctor.environment import PackageStatus, collect_environment


def test_package_status_old_version():
    status = PackageStatus(name="x", installed=True, version="1.0", minimum="2.0")
    assert status.is_old is True


def test_collect_environment_shape():
    env = collect_environment()
    assert "python" in env
    assert "packages" in env
    assert "cuda" in env
