import json

from peft_doctor.notebooks import scan_notebook


def test_scan_notebook_detects_token(tmp_path):
    path = tmp_path / "bad.ipynb"
    fake_token = "hf_" + ("a" * 32)
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [f'leaked_secret = "{fake_token}"'],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = scan_notebook(path)
    assert report.has_errors
    assert any(issue.code == "notebook.hf_token" for issue in report.issues)


def test_scan_notebook_detects_merge_flow(tmp_path):
    path = tmp_path / "merge.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["model = PeftModel.from_pretrained(model, adapter)\nmodel.merge_and_unload()"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = scan_notebook(path)
    assert any(issue.code == "notebook.merge_flow" for issue in report.issues)
