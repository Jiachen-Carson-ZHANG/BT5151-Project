import json

from bt5151_credit_risk.codegen_audit import record_codegen_attempt


def test_record_codegen_attempt_writes_expected_files_and_redacts_obvious_secrets(tmp_path):
    attempt_path = record_codegen_attempt(
        log_root=tmp_path,
        run_id="20260425_120000",
        family="preprocessing",
        attempt_label="01_generate_preprocessing_code",
        generated_code={
            "code": "def run_preprocessing(raw_df, workspace_path):\n    return raw_df\n",
            "entrypoint": "run_preprocessing",
        },
        prompt_payload={
            "api_key": "sk-secret",
            "nested": {"Authorization": "Bearer top-secret"},
            "safe_value": "keep-me",
        },
        response_payload={
            "code": "def run_preprocessing(raw_df, workspace_path):\n    return raw_df\n",
            "entrypoint": "run_preprocessing",
            "_internal_only": {"prompt_tokens": 123},
        },
        metadata={"entrypoint": "run_preprocessing", "caller": "generate-preprocessing-code"},
    )

    assert attempt_path == tmp_path / "20260425_120000" / "preprocessing" / "01_generate_preprocessing_code"
    assert (attempt_path / "generated.py").read_text(encoding="utf-8").startswith("def run_preprocessing")

    prompt_payload = json.loads((attempt_path / "prompt_payload.json").read_text(encoding="utf-8"))
    assert prompt_payload["api_key"] == "<redacted>"
    assert prompt_payload["nested"]["Authorization"] == "<redacted>"
    assert prompt_payload["safe_value"] == "keep-me"

    response_payload = json.loads((attempt_path / "response.json").read_text(encoding="utf-8"))
    assert response_payload["entrypoint"] == "run_preprocessing"
    assert "_internal_only" not in response_payload


def test_record_codegen_attempt_merges_metadata_and_persists_reports(tmp_path):
    attempt_path = record_codegen_attempt(
        log_root=tmp_path,
        run_id="20260425_120001",
        family="feature_engineering",
        attempt_label="02_repair_feature_engineering_code",
        metadata={"entrypoint": "engineer_features"},
    )

    record_codegen_attempt(
        log_root=tmp_path,
        run_id="20260425_120001",
        family="feature_engineering",
        attempt_label="02_repair_feature_engineering_code",
        metadata={"caller": "repair-feature-engineering-code"},
        execution_log={"returncode": 0, "timed_out": False},
        validation_report={"passed": True},
        audit_report={"verdict": "pass"},
    )

    metadata = json.loads((attempt_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == {
        "caller": "repair-feature-engineering-code",
        "entrypoint": "engineer_features",
    }
    assert json.loads((attempt_path / "execution_log.json").read_text(encoding="utf-8"))["returncode"] == 0
    assert json.loads((attempt_path / "validation_report.json").read_text(encoding="utf-8"))["passed"] is True
    assert json.loads((attempt_path / "audit_report.json").read_text(encoding="utf-8"))["verdict"] == "pass"
