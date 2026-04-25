import json
from pathlib import Path

import pandas as pd
import pytest

from bt5151_credit_risk.preprocess import execute_generated_preprocessing
from bt5151_credit_risk.preprocess import generate_column_transform_spec
from bt5151_credit_risk.preprocess import generate_dataset_policy_spec
from bt5151_credit_risk.preprocess import generate_preprocessing_code
from bt5151_credit_risk.preprocess import inspect_preprocessing_code
from bt5151_credit_risk.preprocess import normalize_preprocessing_artifacts
from bt5151_credit_risk.preprocess import repair_preprocessing_code
from bt5151_credit_risk.preprocess import validate_preprocessing_output
from bt5151_credit_risk.preprocess import validate_semantic_invariants
from bt5151_credit_risk.preprocess import validate_semantic_roles


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        [
            {
                "ID": "0x1",
                "Customer_ID": "CUS_A",
                "Name": "Alice",
                "SSN": "111-11-1111",
                "Credit_Score": "Good",
                "Age": 25,
                "Outstanding_Debt": 1000.0,
                "Occupation": "Engineer",
            },
            {
                "ID": "0x2",
                "Customer_ID": "CUS_A",
                "Name": "Alice",
                "SSN": "111-11-1111",
                "Credit_Score": "Standard",
                "Age": 26,
                "Outstanding_Debt": 1500.0,
                "Occupation": "Engineer",
            },
            {
                "ID": "0x3",
                "Customer_ID": "CUS_B",
                "Name": "Bob",
                "SSN": "222-22-2222",
                "Credit_Score": "Poor",
                "Age": 40,
                "Outstanding_Debt": 3000.0,
                "Occupation": "Analyst",
            },
            {
                "ID": "0x4",
                "Customer_ID": "CUS_B",
                "Name": "Bob",
                "SSN": "222-22-2222",
                "Credit_Score": "Poor",
                "Age": 41,
                "Outstanding_Debt": 3200.0,
                "Occupation": "Analyst",
            },
            {
                "ID": "0x5",
                "Customer_ID": "CUS_C",
                "Name": "Cara",
                "SSN": "333-33-3333",
                "Credit_Score": "Good",
                "Age": 31,
                "Outstanding_Debt": 800.0,
                "Occupation": "Manager",
            },
            {
                "ID": "0x6",
                "Customer_ID": "CUS_D",
                "Name": "Dan",
                "SSN": "444-44-4444",
                "Credit_Score": "Standard",
                "Age": 29,
                "Outstanding_Debt": 2200.0,
                "Occupation": "Teacher",
            },
        ]
    )


def test_generate_dataset_policy_spec_uses_runtime_skill_prompt(sample_frame, monkeypatch):
    captured = {}

    def fake_load_skill_prompt(skill_name):
        assert skill_name == "dataset-policy-spec"
        return "runtime dataset policy prompt"

    def fake_agent(system_prompt, payload, **kwargs):
        captured["system_prompt"] = system_prompt
        captured["payload"] = payload
        return {
            "task_type": "multiclass_classification",
            "target_column": "Credit_Score",
            "group_column": "Customer_ID",
            "identifier_columns": ["ID", "Name", "SSN"],
            "split_strategy": {"type": "grouped_holdout", "test_size": 0.2},
            "validation_policy": {
                "type": "grouped_entity",
                "group_column": "Customer_ID",
                "time_column": None,
                "stratify_target": True,
            },
            "leakage_rules": {"drop_columns": ["ID", "Name", "SSN"]},
            "imbalance_strategy": {"method": "none"},
            "feature_policy": {"categorical_encoding": "one_hot"},
        }

    monkeypatch.setattr("bt5151_credit_risk.preprocess.load_skill_prompt", fake_load_skill_prompt)
    monkeypatch.setattr("bt5151_credit_risk.preprocess._call_preprocess_agent", fake_agent)
    profile = {
        "row_count": len(sample_frame),
        "target_distribution": sample_frame["Credit_Score"].value_counts().to_dict(),
        "missing_counts": sample_frame.isna().sum().to_dict(),
    }

    policy = generate_dataset_policy_spec(sample_frame, profile)

    assert captured["system_prompt"] == "runtime dataset policy prompt"
    assert captured["payload"]["columns"] == sample_frame.columns.tolist()
    assert captured["payload"]["sample_rows"] == sample_frame.head(5).to_dict(orient="records")
    assert captured["payload"]["dataset_profile"] == profile
    assert policy["target_column"] == "Credit_Score"
    assert policy["group_column"] == "Customer_ID"
    assert policy["split_strategy"]["type"] == "grouped_holdout"
    assert policy["validation_policy"]["type"] == "grouped_entity"


def test_generate_column_transform_spec_uses_runtime_skill_prompt(sample_frame, monkeypatch):
    captured = {}

    def fake_load_skill_prompt(skill_name):
        assert skill_name == "column-transform-spec"
        return "runtime column transform prompt"

    def fake_agent(system_prompt, payload, **kwargs):
        captured["system_prompt"] = system_prompt
        captured["payload"] = payload
        return {
            "transforms": {
                "Age": {
                    "action": "keep",
                    "semantic_role": "numeric_continuous",
                    "imputation": "median",
                    "representation_intent": "standardized",
                },
                "Outstanding_Debt": {
                    "action": "keep",
                    "semantic_role": "numeric_continuous",
                    "imputation": "median",
                    "representation_intent": "standardized",
                },
                "Occupation": {
                    "action": "keep",
                    "semantic_role": "unordered_categorical",
                    "encoding": "one_hot",
                    "imputation": "mode",
                    "representation_intent": "one_hot",
                },
                "ID": {"action": "drop", "semantic_role": "identifier"},
                "Name": {"action": "drop", "semantic_role": "identifier"},
                "SSN": {"action": "drop", "semantic_role": "identifier"},
            }
        }

    monkeypatch.setattr("bt5151_credit_risk.preprocess.load_skill_prompt", fake_load_skill_prompt)
    monkeypatch.setattr("bt5151_credit_risk.preprocess._call_preprocess_agent", fake_agent)
    policy = {
        "target_column": "Credit_Score",
        "group_column": "Customer_ID",
        "identifier_columns": ["ID", "Name", "SSN"],
    }

    spec = generate_column_transform_spec(sample_frame, policy)

    assert captured["system_prompt"] == "runtime column transform prompt"
    assert captured["payload"]["columns"] == sample_frame.columns.tolist()
    assert captured["payload"]["sample_rows"] == sample_frame.head(10).to_dict(orient="records")
    assert "column_profiles" in captured["payload"]
    assert captured["payload"]["dataset_policy_spec"] == policy
    assert "allowed_cleaning_primitives" in captured["payload"]
    assert "parse_age_series" in captured["payload"]["allowed_cleaning_primitives"]
    assert spec["transforms"]["Age"]["imputation"] == "median"
    assert spec["transforms"]["Occupation"]["encoding"] == "one_hot"


def test_generate_preprocessing_code_uses_runtime_prompt_and_payload_context(sample_frame, monkeypatch):
    captured = {}

    def fake_load_skill_prompt(skill_name):
        assert skill_name == "generate-preprocessing-code"
        return "runtime system prompt"

    def fake_codegen_agent(system_prompt, payload, **kwargs):
        captured["system_prompt"] = system_prompt
        captured["payload"] = payload
        return {
            "code": "def run_preprocessing(raw_df, workspace_path):\n    return raw_df",
            "entrypoint": "run_preprocessing",
        }

    monkeypatch.setattr("bt5151_credit_risk.preprocess.load_skill_prompt", fake_load_skill_prompt)
    monkeypatch.setattr("bt5151_credit_risk.preprocess._call_preprocess_codegen_agent", fake_codegen_agent)

    dataset_profile = {
        "row_count": len(sample_frame),
        "target_distribution": sample_frame["Credit_Score"].value_counts().to_dict(),
        "missing_counts": sample_frame.isna().sum().to_dict(),
    }
    dataset_policy_spec = _sample_policy_spec()
    column_transform_spec = _sample_column_transform_spec()

    result = generate_preprocessing_code(
        sample_frame,
        dataset_profile,
        dataset_policy_spec,
        column_transform_spec,
    )

    assert result["code"] == "def run_preprocessing(raw_df, workspace_path):\n    return raw_df"
    assert result["entrypoint"] == "run_preprocessing"
    assert captured["system_prompt"] == "runtime system prompt"
    assert captured["payload"]["dataset_profile"] == dataset_profile
    assert captured["payload"]["dataset_policy_spec"] == dataset_policy_spec
    assert captured["payload"]["column_transform_spec"] == column_transform_spec
    assert captured["payload"]["columns"] == sample_frame.columns.tolist()
    assert captured["payload"]["sample_rows"] == sample_frame.head(10).to_dict(orient="records")
    assert "column_profiles" in captured["payload"]
    assert "allowed_cleaning_primitives" in captured["payload"]
    assert "parse_age_series" in captured["payload"]["allowed_cleaning_primitives"]
    assert "parse_duration_months" in captured["payload"]["allowed_cleaning_primitives"]


def test_repair_preprocessing_code_uses_runtime_prompt_and_failure_context(sample_frame, monkeypatch):
    captured = {}

    def fake_load_skill_prompt(skill_name):
        assert skill_name == "repair-preprocessing-code"
        return "runtime repair prompt"

    def fake_codegen_agent(system_prompt, payload, **kwargs):
        captured["system_prompt"] = system_prompt
        captured["payload"] = payload
        return {
            "code": "def run_preprocessing(raw_df, workspace_path):\n    return raw_df",
            "entrypoint": "run_preprocessing",
            "repair_strategy": "patched",
        }

    monkeypatch.setattr("bt5151_credit_risk.preprocess.load_skill_prompt", fake_load_skill_prompt)
    monkeypatch.setattr("bt5151_credit_risk.preprocess._call_preprocess_codegen_agent", fake_codegen_agent)

    previous_generated_code = {
        "code": "def run_preprocessing(raw_df, workspace_path):\n    return raw_df.copy()",
        "entrypoint": "run_preprocessing",
    }
    code_review = {
        "passed": False,
        "issues": [{"rule": "forbidden_import", "message": "Importing subprocess is not allowed."}],
    }
    execution_log = {
        "returncode": 1,
        "stdout": "",
        "stderr": "Traceback (most recent call last): ...",
        "timed_out": False,
    }
    validation_report = {
        "passed": False,
        "errors": [{"rule": "target_excluded", "message": "Target column is still present."}],
    }
    dataset_profile = {
        "row_count": len(sample_frame),
        "target_distribution": sample_frame["Credit_Score"].value_counts().to_dict(),
        "missing_counts": sample_frame.isna().sum().to_dict(),
    }
    dataset_policy_spec = _sample_policy_spec()
    column_transform_spec = _sample_column_transform_spec()

    result = repair_preprocessing_code(
        previous_generated_code=previous_generated_code,
        code_review=code_review,
        execution_log=execution_log,
        validation_report=validation_report,
        dataset_profile=dataset_profile,
        dataset_policy_spec=dataset_policy_spec,
        column_transform_spec=column_transform_spec,
    )

    assert result["code"] == "def run_preprocessing(raw_df, workspace_path):\n    return raw_df"
    assert result["entrypoint"] == "run_preprocessing"
    assert result["repair_strategy"] == "patched"
    assert captured["system_prompt"] == "runtime repair prompt"
    assert captured["payload"]["previous_generated_code"] == previous_generated_code
    assert captured["payload"]["code_review"] == code_review
    assert captured["payload"]["execution_log"] == execution_log
    assert captured["payload"]["validation_report"] == validation_report
    assert captured["payload"]["dataset_profile"] == dataset_profile
    assert captured["payload"]["dataset_policy_spec"] == dataset_policy_spec
    assert captured["payload"]["column_transform_spec"] == column_transform_spec
    assert "allowed_cleaning_primitives" in captured["payload"]
    assert "multi_hot_membership" in captured["payload"]["allowed_cleaning_primitives"]


def test_inspect_preprocessing_code_rejects_subprocess_import():
    result = inspect_preprocessing_code(
        {
            "code": "import subprocess\n\ndef run_preprocessing(df):\n    return df",
            "entrypoint": "run_preprocessing",
        }
    )

    assert result["passed"] is False
    assert any(issue["rule"] == "forbidden_import" for issue in result["issues"])


def test_inspect_preprocessing_code_rejects_os_system_call():
    result = inspect_preprocessing_code(
        {
            "code": "import os\n\ndef run_preprocessing(df):\n    os.system('echo unsafe')\n    return df",
            "entrypoint": "run_preprocessing",
        }
    )

    assert result["passed"] is False
    assert any(issue["rule"] == "forbidden_call" for issue in result["issues"])


def test_inspect_preprocessing_code_rejects_missing_entrypoint():
    result = inspect_preprocessing_code(
        {
            "code": "def other_entrypoint(df):\n    return df",
            "entrypoint": "run_preprocessing",
        }
    )

    assert result["passed"] is False
    assert any(issue["rule"] == "missing_entrypoint" for issue in result["issues"])


def test_inspect_preprocessing_code_accepts_safe_entrypoint():
    result = inspect_preprocessing_code(
        {
            "code": "def run_preprocessing(df):\n    return df.copy()",
            "entrypoint": "run_preprocessing",
        }
    )

    assert result["passed"] is True
    assert result["entrypoint"] == "run_preprocessing"
    assert result["issues"] == []


@pytest.mark.parametrize(
    "code_snippet,expected_rule",
    [
        ("eval('1+1')\n\ndef run_preprocessing(df):\n    return df", "forbidden_call"),
        ("exec('x=1')\n\ndef run_preprocessing(df):\n    return df", "forbidden_call"),
        ("__import__('os')\n\ndef run_preprocessing(df):\n    return df", "forbidden_call"),
        ("import socket\n\ndef run_preprocessing(df):\n    return df", "forbidden_import"),
        ("from urllib import request\n\ndef run_preprocessing(df):\n    return df", "forbidden_import"),
        ("import os\n\ndef run_preprocessing(df):\n    os.popen('ls')\n    return df", "forbidden_call"),
        ("import importlib\n\ndef run_preprocessing(df):\n    importlib.import_module('subprocess')\n    return df", "forbidden_call"),
    ],
    ids=["eval", "exec", "__import__", "socket", "urllib", "os.popen", "importlib.import_module"],
)
def test_inspect_preprocessing_code_rejects_additional_forbidden_patterns(code_snippet, expected_rule):
    result = inspect_preprocessing_code(
        {"code": code_snippet, "entrypoint": "run_preprocessing"}
    )
    assert result["passed"] is False
    assert any(issue["rule"] == expected_rule for issue in result["issues"])


def _sample_policy_spec():
    return {
        "task_type": "multiclass_classification",
        "target_column": "Credit_Score",
        "group_column": "Customer_ID",
        "identifier_columns": ["ID", "Name", "SSN"],
        "split_strategy": {"type": "grouped_holdout", "test_size": 0.34},
        "validation_policy": {
            "type": "grouped_entity",
            "group_column": "Customer_ID",
            "time_column": None,
            "stratify_target": True,
        },
        "leakage_rules": {"drop_columns": ["ID", "Name", "SSN"]},
        "imbalance_strategy": {"method": "none"},
        "feature_policy": {"categorical_encoding": "one_hot"},
    }


def _sample_column_transform_spec():
    return {
        "transforms": {
            "ID": {"action": "drop", "semantic_role": "identifier"},
            "Customer_ID": {"action": "drop", "semantic_role": "group_identifier"},
            "Name": {"action": "drop", "semantic_role": "identifier"},
            "SSN": {"action": "drop", "semantic_role": "identifier"},
            "Credit_Score": {"action": "drop", "semantic_role": "target"},
            "Age": {
                "action": "keep",
                "semantic_role": "numeric_continuous",
                "imputation": "median",
                "representation_intent": "standardized",
            },
            "Outstanding_Debt": {
                "action": "keep",
                "semantic_role": "numeric_continuous",
                "imputation": "median",
                "representation_intent": "standardized",
            },
            "Occupation": {
                "action": "keep",
                "semantic_role": "unordered_categorical",
                "imputation": "mode",
                "representation_intent": "deferred",
            },
        }
    }


def _generated_preprocessing_code(feature_frame_expression: str, split_manifest: dict) -> dict:
    return {
        "code": (
            "from pathlib import Path\n"
            "import json\n"
            "\n"
            "def run_preprocessing(raw_df, workspace_path):\n"
            "    workspace = Path(workspace_path)\n"
            "    cleaned = raw_df.copy()\n"
            f"    feature_frame = {feature_frame_expression}\n"
            "    target = cleaned['Credit_Score']\n"
            "    cleaned.to_csv(workspace / 'cleaned_frame.csv', index=False)\n"
            "    feature_frame.to_csv(workspace / 'feature_frame.csv', index=False)\n"
            "    target.to_frame(name='Credit_Score').to_csv(workspace / 'target.csv', index=False)\n"
            f"    (workspace / 'split_manifest.json').write_text(json.dumps({split_manifest!r}))\n"
            "    (workspace / 'preprocessing_report.json').write_text(json.dumps({'status': 'ok'}))\n"
            "    return {'status': 'ok'}\n"
        ),
        "entrypoint": "run_preprocessing",
    }


def test_validate_preprocessing_output_passes_on_expected_artifacts(sample_frame, tmp_path):
    generated_code = _generated_preprocessing_code(
        "cleaned.drop(columns=['Credit_Score', 'ID', 'Customer_ID', 'Name', 'SSN'])",
        {"train_indices": [0, 1, 2, 3], "test_indices": [4, 5]},
    )

    execution_result = execute_generated_preprocessing(sample_frame, generated_code, tmp_path)
    result = validate_preprocessing_output(
        execution_result,
        _sample_policy_spec(),
        _sample_column_transform_spec(),
    )

    assert result["passed"] is True
    assert result["checks"]["feature_frame_non_empty"] is True
    assert result["checks"]["split_manifest_exists"] is True
    assert result["checks"]["target_file_exists"] is True
    assert result["checks"]["target_excluded"] is True
    assert result["checks"]["group_overlap_zero"] is True
    assert result["errors"] == []


def test_validate_preprocessing_output_reports_missing_split_manifest(sample_frame, tmp_path):
    generated_code = _generated_preprocessing_code(
        "cleaned.drop(columns=['Credit_Score'])",
        {"train_indices": [0, 1, 2, 3], "test_indices": [4, 5]},
    )

    execution_result = execute_generated_preprocessing(sample_frame, generated_code, tmp_path)
    Path(execution_result["artifacts"]["split_manifest.json"]).unlink()

    result = validate_preprocessing_output(
        execution_result,
        _sample_policy_spec(),
        _sample_column_transform_spec(),
    )

    assert result["passed"] is False
    assert result["checks"]["split_manifest_exists"] is False
    assert any(error["rule"] == "split_manifest_exists" for error in result["errors"])


def test_validate_preprocessing_output_reports_target_in_feature_frame(sample_frame, tmp_path):
    generated_code = _generated_preprocessing_code(
        "cleaned.copy()",
        {"train_indices": [0, 1, 2, 3], "test_indices": [4, 5]},
    )

    execution_result = execute_generated_preprocessing(sample_frame, generated_code, tmp_path)
    result = validate_preprocessing_output(
        execution_result,
        _sample_policy_spec(),
        _sample_column_transform_spec(),
    )

    assert result["passed"] is False
    assert result["checks"]["target_excluded"] is False
    assert any(error["rule"] == "target_excluded" for error in result["errors"])


def test_validate_preprocessing_output_reports_group_overlap_for_grouped_split(sample_frame, tmp_path):
    generated_code = _generated_preprocessing_code(
        "cleaned.drop(columns=['Credit_Score'])",
        {"train_indices": [0, 1, 2, 3], "test_indices": [3, 4, 5]},
    )

    execution_result = execute_generated_preprocessing(sample_frame, generated_code, tmp_path)
    result = validate_preprocessing_output(
        execution_result,
        _sample_policy_spec(),
        _sample_column_transform_spec(),
    )

    assert result["passed"] is False
    assert result["checks"]["group_overlap_zero"] is False
    assert any(error["rule"] == "group_overlap_zero" for error in result["errors"])


def test_validate_preprocessing_output_reports_empty_feature_frame(sample_frame, tmp_path):
    generated_code = _generated_preprocessing_code(
        "cleaned.head(0).drop(columns=['Credit_Score'])",
        {"train_indices": [0, 1, 2, 3], "test_indices": [4, 5]},
    )

    execution_result = execute_generated_preprocessing(sample_frame, generated_code, tmp_path)
    result = validate_preprocessing_output(
        execution_result,
        _sample_policy_spec(),
        _sample_column_transform_spec(),
    )

    assert result["passed"] is False
    assert result["checks"]["feature_frame_non_empty"] is False
    assert any(error["rule"] == "feature_frame_non_empty" for error in result["errors"])


def test_validate_preprocessing_output_reports_missing_target_file(sample_frame, tmp_path):
    generated_code = _generated_preprocessing_code(
        "cleaned.drop(columns=['Credit_Score'])",
        {"train_indices": [0, 1, 2, 3], "test_indices": [4, 5]},
    )

    execution_result = execute_generated_preprocessing(sample_frame, generated_code, tmp_path)
    Path(execution_result["artifacts"]["target.csv"]).unlink()

    result = validate_preprocessing_output(
        execution_result,
        _sample_policy_spec(),
        _sample_column_transform_spec(),
    )

    assert result["passed"] is False
    assert result["checks"]["target_file_exists"] is False
    assert any(error["rule"] == "target_file_exists" for error in result["errors"])


def test_validate_preprocessing_output_reports_mangled_duplicate_columns(sample_frame, tmp_path):
    generated_code = _generated_preprocessing_code(
        "cleaned.drop(columns=['Credit_Score'])",
        {"train_indices": [0, 1, 2, 3], "test_indices": [4, 5]},
    )

    execution_result = execute_generated_preprocessing(sample_frame, generated_code, tmp_path)
    feature_frame_path = Path(execution_result["artifacts"]["feature_frame.csv"])
    feature_frame_path.write_text("A,A\n1,0\n0,1\n1,0\n0,1\n1,0\n0,1\n", encoding="utf-8")

    result = validate_preprocessing_output(
        execution_result,
        _sample_policy_spec(),
        _sample_column_transform_spec(),
    )

    assert result["passed"] is False
    assert result["checks"]["no_mangled_duplicate_columns"] is False
    assert any(error["rule"] == "no_mangled_duplicate_columns" for error in result["errors"])


def test_execute_generated_preprocessing_creates_workspace_and_artifacts(sample_frame, tmp_path):
    generated_code = _generated_preprocessing_code(
        "cleaned.drop(columns=['Credit_Score'])",
        {"train_indices": [0, 1, 2, 3], "test_indices": [4, 5]},
    )

    result = execute_generated_preprocessing(sample_frame, generated_code, tmp_path)

    workspace_path = Path(result["workspace_path"])
    assert workspace_path.is_dir()
    assert workspace_path.parent == tmp_path
    assert Path(result["code_path"]).is_file()

    for artifact_name in [
        "cleaned_frame.csv",
        "feature_frame.csv",
        "target.csv",
        "split_manifest.json",
        "preprocessing_report.json",
    ]:
        assert Path(result["artifacts"][artifact_name]).is_file()

    execution_log = result["execution_log"]
    assert execution_log["returncode"] == 0
    assert execution_log["timeout_seconds"] == 300
    assert "stdout" in execution_log
    assert "stderr" in execution_log


def test_execute_generated_preprocessing_honors_declared_entrypoint(sample_frame, tmp_path):
    generated_code = {
        "code": (
            "from pathlib import Path\n"
            "import json\n"
            "\n"
            "def run_preprocessing(raw_df, workspace_path):\n"
            "    raise AssertionError('wrong entrypoint was called')\n"
            "\n"
            "def custom_preprocessing(raw_df, workspace_path):\n"
            "    workspace = Path(workspace_path)\n"
            "    cleaned = raw_df.copy()\n"
            "    feature_frame = cleaned.drop(columns=['Credit_Score'])\n"
            "    target = cleaned['Credit_Score']\n"
            "    cleaned.to_csv(workspace / 'cleaned_frame.csv', index=False)\n"
            "    feature_frame.to_csv(workspace / 'feature_frame.csv', index=False)\n"
            "    target.to_frame(name='Credit_Score').to_csv(workspace / 'target.csv', index=False)\n"
            "    (workspace / 'split_manifest.json').write_text(json.dumps({'train_indices': [0, 1, 2, 3], 'test_indices': [4, 5]}))\n"
            "    (workspace / 'preprocessing_report.json').write_text(json.dumps({'status': 'ok', 'entrypoint': 'custom_preprocessing'}))\n"
            "    return {'entrypoint': 'custom_preprocessing'}\n"
        ),
        "entrypoint": "custom_preprocessing",
    }

    result = execute_generated_preprocessing(sample_frame, generated_code, tmp_path)

    assert result["execution_log"]["returncode"] == 0
    assert "custom_preprocessing" in result["execution_log"]["stdout"]
    assert Path(result["artifacts"]["cleaned_frame.csv"]).is_file()
    assert Path(result["artifacts"]["preprocessing_report.json"]).is_file()


def test_normalize_preprocessing_artifacts_repairs_credit_history_and_multivalue_missingness(tmp_path):
    raw_frame = pd.DataFrame(
        {
            "Customer_ID": ["C1", "C1", "C2", "C2"],
            "Credit_Score": ["Good", "Good", "Poor", "Poor"],
            "Age": ["21", "21", "45", "45"],
            "Credit_History_Age": [
                "30 Years and 8 Months",
                "8 Years and 9 Months",
                "19 Years and 11 Months",
                "20 Years and 0 Months",
            ],
            "Type_of_Loan": [
                "Not Specified",
                None,
                "Home Loan, Personal Loan",
                "Home Loan",
            ],
        }
    )
    cleaned_frame = pd.DataFrame(
        {
            "Age": [21.0, 21.0, 45.0, 45.0],
            "Credit_History_Age": [87.0, 87.0, 87.0, 87.0],
        }
    )
    feature_frame = pd.DataFrame(
        {
            "Age": [21.0, 21.0, 45.0, 45.0],
            "Credit_History_Age": [87.0, 87.0, 87.0, 87.0],
            "Type_of_Loan_missing": [1, 0, 0, 0],
            "Type_of_Loan_Not Specified": [1, 1, 0, 0],
            "Type_of_Loan_Home Loan": [0, 1, 1, 1],
            "Type_of_Loan_Personal Loan": [0, 0, 1, 0],
        }
    )

    raw_frame_path = tmp_path / "raw_frame.csv"
    cleaned_frame_path = tmp_path / "cleaned_frame.csv"
    feature_frame_path = tmp_path / "feature_frame.csv"
    target_path = tmp_path / "target.csv"
    split_manifest_path = tmp_path / "split_manifest.json"
    report_path = tmp_path / "preprocessing_report.json"
    raw_frame.to_csv(raw_frame_path, index=False)
    cleaned_frame.to_csv(cleaned_frame_path, index=False)
    feature_frame.to_csv(feature_frame_path, index=False)
    raw_frame["Credit_Score"].to_frame().to_csv(target_path, index=False)
    split_manifest_path.write_text(json.dumps({"train_indices": [0, 1], "test_indices": [2, 3]}), encoding="utf-8")
    report_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    execution_result = {
        "workspace_path": str(tmp_path),
        "raw_frame_path": str(raw_frame_path),
        "artifacts": {
            "cleaned_frame.csv": str(cleaned_frame_path),
            "feature_frame.csv": str(feature_frame_path),
            "target.csv": str(target_path),
            "split_manifest.json": str(split_manifest_path),
            "preprocessing_report.json": str(report_path),
        },
    }
    column_transform_spec = {
        "transforms": {
            "Customer_ID": {"action": "drop", "semantic_role": "group_identifier"},
            "Credit_Score": {"action": "drop", "semantic_role": "target"},
            "Age": {"action": "keep", "semantic_role": "numeric_continuous"},
            "Credit_History_Age": {
                "action": "keep",
                "semantic_role": "numeric_continuous",
                "cleaning": "extract '(\\d+) Years and (\\d+) Months', compute years*12+months, mark values >1200 as NaN",
            },
            "Type_of_Loan": {
                "action": "keep",
                "semantic_role": "multi_value_set",
                "representation_intent": "binary_membership",
                "cleaning": "create Type_of_Loan_missing when original is NaN or 'Not Specified', then multi_hot",
            },
        }
    }

    report = normalize_preprocessing_artifacts(execution_result, column_transform_spec)

    normalized_feature = pd.read_csv(feature_frame_path)
    normalized_cleaned = pd.read_csv(cleaned_frame_path)

    assert report["applied"] is True
    assert "Credit_History_Age" in report["normalized_columns"]
    assert "Type_of_Loan" in report["normalized_columns"]
    assert "Type_of_Loan_Not Specified" not in normalized_feature.columns
    assert normalized_feature["Type_of_Loan_missing"].tolist()[:2] == [1, 1]
    assert normalized_feature.loc[0, "Type_of_Loan_Home Loan"] == 0
    assert normalized_feature.loc[1, "Type_of_Loan_Home Loan"] == 0
    assert normalized_feature["Credit_History_Age"].tolist() == [39.0, 39.0, 239.0, 240.0]
    assert normalized_cleaned["Credit_History_Age"].tolist() == [39.0, 39.0, 239.0, 240.0]


def test_normalize_preprocessing_artifacts_repairs_age_before_credit_history_cap(tmp_path):
    raw_frame = pd.DataFrame(
        {
            "Customer_ID": ["C1", "C1", "C2"],
            "Credit_Score": ["Good", "Good", "Poor"],
            "Age": ["23", "-500", "45_"],
            "Credit_History_Age": [
                "22 Years and 1 Months",
                None,
                "19 Years and 11 Months",
            ],
            "Type_of_Loan": ["Auto Loan", "Not Specified", "Home Loan"],
        }
    )
    cleaned_frame = pd.DataFrame(
        {
            "Age": [23.0, 118.0, 45.0],
            "Credit_History_Age": [265.0, 265.0, 239.0],
        }
    )
    feature_frame = cleaned_frame.copy()
    feature_frame["Type_of_Loan_missing"] = [0, 0, 0]
    feature_frame["Type_of_Loan_Not Specified"] = [0, 1, 0]
    feature_frame["Type_of_Loan_Auto Loan"] = [1, 0, 0]
    feature_frame["Type_of_Loan_Home Loan"] = [0, 0, 1]

    raw_frame_path = tmp_path / "raw_frame.csv"
    cleaned_frame_path = tmp_path / "cleaned_frame.csv"
    feature_frame_path = tmp_path / "feature_frame.csv"
    raw_frame.to_csv(raw_frame_path, index=False)
    cleaned_frame.to_csv(cleaned_frame_path, index=False)
    feature_frame.to_csv(feature_frame_path, index=False)

    execution_result = {
        "workspace_path": str(tmp_path),
        "raw_frame_path": str(raw_frame_path),
        "artifacts": {
            "cleaned_frame.csv": str(cleaned_frame_path),
            "feature_frame.csv": str(feature_frame_path),
        },
    }
    column_transform_spec = {
        "transforms": {
            "Customer_ID": {"action": "drop", "semantic_role": "group_identifier"},
            "Age": {"action": "keep", "semantic_role": "numeric_continuous"},
            "Credit_History_Age": {"action": "keep", "semantic_role": "numeric_continuous"},
            "Type_of_Loan": {
                "action": "keep",
                "semantic_role": "multi_value_set",
                "representation_intent": "binary_membership",
                "cleaning": "create Type_of_Loan_missing for Not Specified",
            },
        }
    }

    report = normalize_preprocessing_artifacts(execution_result, column_transform_spec)
    normalized = pd.read_csv(feature_frame_path)

    assert report["applied"] is True
    assert "Age" in report["normalized_columns"]
    assert normalized["Age"].tolist() == [23.0, 23.0, 45.0]
    max_months = (normalized["Age"] - 18).clip(lower=0) * 12 * 1.1
    assert (normalized["Credit_History_Age"] <= max_months).all()
    assert "Type_of_Loan_Not Specified" not in normalized.columns
    assert normalized.loc[1, "Type_of_Loan_missing"] == 1
    assert normalized.loc[1, "Type_of_Loan_Auto Loan"] == 0


def test_normalize_preprocessing_artifacts_handles_fractional_group_median_without_crashing(tmp_path):
    raw_frame = pd.DataFrame(
        {
            "Customer_ID": ["C1", "C1", "C1", "C2"],
            "Credit_Score": ["Good", "Good", "Good", "Poor"],
            "Age": ["40", "40", "40", "50"],
            "Credit_History_Age": [
                "1 Years and 0 Months",
                "2 Years and 1 Months",
                None,
                "10 Years and 0 Months",
            ],
        }
    )
    feature_frame = pd.DataFrame(
        {
            "Age": [40.0, 40.0, 40.0, 50.0],
            "Credit_History_Age": [12.0, 25.0, 120.0, 120.0],
        }
    )
    raw_frame_path = tmp_path / "raw_frame.csv"
    feature_frame_path = tmp_path / "feature_frame.csv"
    raw_frame.to_csv(raw_frame_path, index=False)
    feature_frame.to_csv(feature_frame_path, index=False)

    execution_result = {
        "workspace_path": str(tmp_path),
        "raw_frame_path": str(raw_frame_path),
        "artifacts": {
            "feature_frame.csv": str(feature_frame_path),
        },
    }
    column_transform_spec = {
        "transforms": {
            "Customer_ID": {"action": "drop", "semantic_role": "group_identifier"},
            "Age": {"action": "keep", "semantic_role": "numeric_continuous"},
            "Credit_History_Age": {"action": "keep", "semantic_role": "numeric_continuous"},
        }
    }

    report = normalize_preprocessing_artifacts(execution_result, column_transform_spec)
    normalized = pd.read_csv(feature_frame_path)

    assert report["applied"] is True
    assert report["skipped"] == []
    assert normalized.loc[2, "Credit_History_Age"] == 18.0


def test_normalize_preprocessing_artifacts_enforces_declared_numeric_bounds(tmp_path):
    raw_frame = pd.DataFrame(
        {
            "Customer_ID": ["C1", "C1", "C2"],
            "Credit_Score": ["Good", "Good", "Poor"],
            "Monthly_Inhand_Salary": [1500.0, 36160.17, 8000.0],
            "Total_EMI_per_month": [500.0, 82331.0, 750.0],
            "Num_Credit_Inquiries": [2.0, 1000.0, 4.0],
        }
    )
    cleaned_frame = raw_frame.drop(columns=["Customer_ID", "Credit_Score"]).copy()
    feature_frame = cleaned_frame.copy()

    raw_frame_path = tmp_path / "raw_frame.csv"
    cleaned_frame_path = tmp_path / "cleaned_frame.csv"
    feature_frame_path = tmp_path / "feature_frame.csv"
    raw_frame.to_csv(raw_frame_path, index=False)
    cleaned_frame.to_csv(cleaned_frame_path, index=False)
    feature_frame.to_csv(feature_frame_path, index=False)

    execution_result = {
        "workspace_path": str(tmp_path),
        "raw_frame_path": str(raw_frame_path),
        "artifacts": {
            "cleaned_frame.csv": str(cleaned_frame_path),
            "feature_frame.csv": str(feature_frame_path),
        },
    }
    column_transform_spec = {
        "transforms": {
            "Customer_ID": {"action": "drop", "semantic_role": "group_identifier"},
            "Monthly_Inhand_Salary": {
                "action": "keep",
                "semantic_role": "numeric_continuous",
                "primitive": "parse_dirty_numeric",
                "primitive_params": {"bounds": [0, 20000]},
                "cleaning": "clip to [0,20000]",
            },
            "Total_EMI_per_month": {
                "action": "keep",
                "semantic_role": "numeric_continuous",
                "primitive": "parse_dirty_numeric",
                "primitive_params": {"bounds": [0, 50000]},
                "cleaning": "clip to [0,50000]",
            },
            "Num_Credit_Inquiries": {
                "action": "keep",
                "semantic_role": "numeric_count",
                "primitive": "parse_dirty_numeric",
                "primitive_params": {"lower_bound": 0, "upper_bound": 100},
                "cleaning": "mark values <0 or >100 as NaN",
            },
        }
    }

    report = normalize_preprocessing_artifacts(execution_result, column_transform_spec)

    normalized_feature = pd.read_csv(feature_frame_path)
    normalized_cleaned = pd.read_csv(cleaned_frame_path)

    assert report["applied"] is True
    assert {
        "Monthly_Inhand_Salary",
        "Total_EMI_per_month",
        "Num_Credit_Inquiries",
    }.issubset(set(report["normalized_columns"]))
    assert normalized_feature["Monthly_Inhand_Salary"].tolist() == [1500.0, 20000.0, 8000.0]
    assert normalized_feature["Total_EMI_per_month"].tolist() == [500.0, 50000.0, 750.0]
    assert normalized_feature["Num_Credit_Inquiries"].tolist() == [2.0, 100.0, 4.0]
    assert normalized_cleaned.equals(normalized_feature)


def test_validate_preprocessing_output_auto_normalizes_deterministic_columns(tmp_path):
    raw_frame = pd.DataFrame(
        {
            "Customer_ID": ["C1", "C1", "C2", "C2"],
            "Credit_Score": ["Good", "Good", "Poor", "Poor"],
            "Age": ["21", "21", "45", "45"],
            "Credit_History_Age": [
                "30 Years and 8 Months",
                "8 Years and 9 Months",
                "19 Years and 11 Months",
                "20 Years and 0 Months",
            ],
            "Type_of_Loan": [
                "Not Specified",
                None,
                "Home Loan, Personal Loan",
                "Home Loan",
            ],
        }
    )
    cleaned_frame = pd.DataFrame(
        {
            "Age": [21.0, 21.0, 45.0, 45.0],
            "Credit_History_Age": [87.0, 87.0, 87.0, 87.0],
        }
    )
    feature_frame = pd.DataFrame(
        {
            "Age": [21.0, 21.0, 45.0, 45.0],
            "Credit_History_Age": [87.0, 87.0, 87.0, 87.0],
            "Type_of_Loan_missing": [1, 0, 0, 0],
            "Type_of_Loan_Not Specified": [1, 1, 0, 0],
            "Type_of_Loan_Home Loan": [0, 1, 1, 1],
            "Type_of_Loan_Personal Loan": [0, 0, 1, 0],
        }
    )

    raw_frame_path = tmp_path / "raw_frame.csv"
    cleaned_frame_path = tmp_path / "cleaned_frame.csv"
    feature_frame_path = tmp_path / "feature_frame.csv"
    target_path = tmp_path / "target.csv"
    split_manifest_path = tmp_path / "split_manifest.json"
    report_path = tmp_path / "preprocessing_report.json"
    raw_frame.to_csv(raw_frame_path, index=False)
    cleaned_frame.to_csv(cleaned_frame_path, index=False)
    feature_frame.to_csv(feature_frame_path, index=False)
    raw_frame["Credit_Score"].to_frame().to_csv(target_path, index=False)
    split_manifest_path.write_text(json.dumps({"train_indices": [0, 1], "test_indices": [2, 3]}), encoding="utf-8")
    report_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    execution_result = {
        "workspace_path": str(tmp_path),
        "raw_frame_path": str(raw_frame_path),
        "artifacts": {
            "cleaned_frame.csv": str(cleaned_frame_path),
            "feature_frame.csv": str(feature_frame_path),
            "target.csv": str(target_path),
            "split_manifest.json": str(split_manifest_path),
            "preprocessing_report.json": str(report_path),
        },
        "execution_log": {"returncode": 0, "timed_out": False},
    }
    dataset_policy_spec = {
        "target_column": "Credit_Score",
        "group_column": "Customer_ID",
        "split_strategy": {"type": "grouped_holdout", "test_size": 0.5},
    }
    column_transform_spec = {
        "transforms": {
            "Customer_ID": {"action": "drop", "semantic_role": "group_identifier"},
            "Credit_Score": {"action": "drop", "semantic_role": "target"},
            "Age": {"action": "keep", "semantic_role": "numeric_continuous"},
            "Credit_History_Age": {
                "action": "keep",
                "semantic_role": "numeric_continuous",
                "cleaning": "extract '(\\d+) Years and (\\d+) Months', compute years*12+months, mark values >1200 as NaN",
            },
            "Type_of_Loan": {
                "action": "keep",
                "semantic_role": "multi_value_set",
                "representation_intent": "binary_membership",
                "cleaning": "create Type_of_Loan_missing when original is NaN or 'Not Specified', then multi_hot",
            },
        }
    }

    result = validate_preprocessing_output(
        execution_result,
        dataset_policy_spec,
        column_transform_spec,
    )

    assert result["passed"] is True, result["errors"] + result["role_violations"] + result["cross_field_violations"]
    assert result["deterministic_normalization"]["applied"] is True
    assert set(result["deterministic_normalization"]["normalized_columns"]) == {"Age", "Credit_History_Age", "Type_of_Loan"}


# ---------------------------------------------------------------------------
# Semantic role validator tests
# ---------------------------------------------------------------------------


def test_validate_semantic_roles_catches_old_schema_without_roles():
    """Old-shape spec {'columns': {...}} with no semantic_role fields triggers
    schema_missing_semantic_roles — the validator must NOT silently become a no-op."""
    feature_frame = pd.DataFrame({"Age": [25, 30], "Debt": [1000, 2000]})
    old_spec = {
        "columns": {
            "Age": {"action": "keep", "imputation": "median"},
            "Debt": {"action": "keep", "imputation": "median"},
        }
    }
    violations = validate_semantic_roles(feature_frame, old_spec)
    assert len(violations) == 1
    assert violations[0]["violation"] == "schema_missing_semantic_roles"


def test_validate_semantic_roles_catches_mixed_schema_partial_regression():
    """If some columns declare semantic_role but others don't, the missing ones
    get a missing_semantic_role violation — not silently skipped."""
    feature_frame = pd.DataFrame({"Age": [25.0, 30.0], "Income": [50000.0, 60000.0]})
    mixed_spec = {
        "transforms": {
            "Age": {
                "action": "keep",
                "semantic_role": "numeric_continuous",
                "representation_intent": "standardized",
            },
            "Income": {
                "action": "keep",
                "imputation": "median",
                # no semantic_role — partial regression
            },
        }
    }
    violations = validate_semantic_roles(feature_frame, mixed_spec)
    assert any(
        v["violation"] == "missing_semantic_role" and v["column"] == "Income"
        for v in violations
    )


def test_validate_semantic_roles_no_violations_on_correct_output():
    """A well-formed feature frame matching the declared roles produces no violations."""
    feature_frame = pd.DataFrame({
        "Age": [25.0, 30.0, 40.0],
        "Outstanding_Debt": [1000.0, 2000.0, 3000.0],
        "Occupation_Engineer": [1, 0, 0],
        "Occupation_Analyst": [0, 1, 0],
        "Occupation_Manager": [0, 0, 1],
    })
    spec = {
        "transforms": {
            "ID": {"action": "drop", "semantic_role": "identifier"},
            "Credit_Score": {"action": "drop", "semantic_role": "target"},
            "Age": {
                "action": "keep",
                "semantic_role": "numeric_continuous",
                "representation_intent": "standardized",
            },
            "Outstanding_Debt": {
                "action": "keep",
                "semantic_role": "numeric_continuous",
                "representation_intent": "standardized",
            },
            "Occupation": {
                "action": "keep",
                "semantic_role": "unordered_categorical",
                "representation_intent": "one_hot",
            },
        }
    }
    violations = validate_semantic_roles(feature_frame, spec)
    assert violations == []


def test_validate_semantic_roles_catches_leaked_identifier():
    """Identifier column still in feature frame is a must_be_absent violation."""
    feature_frame = pd.DataFrame({"ID": ["x1", "x2"], "Age": [25.0, 30.0]})
    spec = {
        "transforms": {
            "ID": {"action": "drop", "semantic_role": "identifier"},
            "Age": {
                "action": "keep",
                "semantic_role": "numeric_continuous",
            },
        }
    }
    violations = validate_semantic_roles(feature_frame, spec)
    assert any(v["violation"] == "must_be_absent" and v["column"] == "ID" for v in violations)


def test_validate_semantic_roles_catches_binary_flag_violation():
    """Binary flag with values outside {0,1} triggers not_binary."""
    feature_frame = pd.DataFrame({"Has_Mortgage": [0, 1, 2]})
    spec = {
        "transforms": {
            "Has_Mortgage": {
                "action": "keep",
                "semantic_role": "binary_flag",
            },
        }
    }
    violations = validate_semantic_roles(feature_frame, spec)
    assert any(v["violation"] == "not_binary" and v["column"] == "Has_Mortgage" for v in violations)


def test_validate_semantic_roles_catches_non_contiguous_ordinal():
    """Ordered categorical with gaps (e.g. {0,2,5}) or 1-based ({1,2,3}) is rejected."""
    import numpy as np

    # 1-based: should be 0,1,2
    frame_1based = pd.DataFrame({"Risk_Level": [1, 2, 3, 1]})
    spec = {
        "transforms": {
            "Risk_Level": {
                "action": "keep",
                "semantic_role": "ordered_categorical",
            },
        }
    }
    violations = validate_semantic_roles(frame_1based, spec)
    assert any(v["violation"] == "ordinal_not_contiguous" and v["column"] == "Risk_Level" for v in violations)

    # Gaps: {0,2,5}
    frame_gaps = pd.DataFrame({"Risk_Level": [0, 2, 5, 0]})
    violations = validate_semantic_roles(frame_gaps, spec)
    assert any(v["violation"] == "ordinal_not_contiguous" and v["column"] == "Risk_Level" for v in violations)

    # Correct: {0,1,2}
    frame_ok = pd.DataFrame({"Risk_Level": [0, 1, 2, 0]})
    violations = validate_semantic_roles(frame_ok, spec)
    assert not any(v["column"] == "Risk_Level" for v in violations)


def test_validate_semantic_roles_catches_inf_in_numeric_continuous():
    """numeric_continuous with inf/-inf values triggers has_inf."""
    import numpy as np

    frame = pd.DataFrame({"Income": [1000.0, np.inf, 3000.0, -np.inf]})
    spec = {
        "transforms": {
            "Income": {
                "action": "keep",
                "semantic_role": "numeric_continuous",
            },
        }
    }
    violations = validate_semantic_roles(frame, spec)
    assert any(v["violation"] == "has_inf" and v["column"] == "Income" for v in violations)


def test_escalation_caller_on_repeated_violation(monkeypatch):
    """When the same (column, violation) pair repeats across attempts,
    repair_preprocessing_code is called with escalate=True which sets
    caller='repair-preprocessing-code-escalated'."""
    captured = {}

    def fake_load_skill_prompt(skill_name):
        return "repair prompt"

    def fake_codegen_agent(system_prompt, payload, caller="repair-preprocessing-code"):
        captured["caller"] = caller
        captured["payload"] = payload
        return {
            "code": "def run_preprocessing(raw_df, workspace_path):\n    return raw_df",
            "entrypoint": "run_preprocessing",
        }

    monkeypatch.setattr("bt5151_credit_risk.preprocess.load_skill_prompt", fake_load_skill_prompt)
    monkeypatch.setattr("bt5151_credit_risk.preprocess._call_preprocess_codegen_agent", fake_codegen_agent)

    # Non-escalated call
    repair_preprocessing_code(
        previous_generated_code={"code": "...", "entrypoint": "run_preprocessing"},
        code_review={},
        execution_log={},
        validation_report={},
        dataset_profile={},
        dataset_policy_spec={},
        column_transform_spec={},
        escalate=False,
    )
    assert captured["caller"] == "repair-preprocessing-code"
    assert "escalation_notice" not in captured["payload"]

    # Escalated call
    repair_preprocessing_code(
        previous_generated_code={"code": "...", "entrypoint": "run_preprocessing"},
        code_review={},
        execution_log={},
        validation_report={},
        dataset_profile={},
        dataset_policy_spec={},
        column_transform_spec={},
        escalate=True,
    )
    assert captured["caller"] == "repair-preprocessing-code-escalated"
    assert "escalation_notice" in captured["payload"]


def test_validate_semantic_roles_deferred_column_stays_as_string():
    """Deferred unordered_categorical must remain object-dtype in feature_frame — no violation."""
    feature_frame = pd.DataFrame({
        "Occupation": ["Engineer", "Doctor", "Engineer"],  # object dtype, deferred
        "Age": [25, 30, 35],
    })
    spec = {
        "transforms": {
            "Occupation": {
                "semantic_role": "unordered_categorical",
                "action": "keep",
                "representation_intent": "deferred",
            },
            "Age": {
                "semantic_role": "numeric_continuous",
                "action": "keep",
            },
        }
    }
    violations = validate_semantic_roles(feature_frame, spec)
    deferred_violations = [v for v in violations if v["column"] == "Occupation"]
    assert deferred_violations == [], (
        f"Deferred column should have no violations, got: {deferred_violations}"
    )


def test_validate_semantic_roles_rejects_deferred_column_that_was_encoded():
    """If preprocessing encoded a deferred column to numeric, that is a violation."""
    feature_frame = pd.DataFrame({
        "Occupation": [0.1, 0.3, 0.1],  # frequency-encoded — wrong for deferred
        "Age": [25, 30, 35],
    })
    spec = {
        "transforms": {
            "Occupation": {
                "semantic_role": "unordered_categorical",
                "action": "keep",
                "representation_intent": "deferred",
            },
            "Age": {
                "semantic_role": "numeric_continuous",
                "action": "keep",
            },
        }
    }
    violations = validate_semantic_roles(feature_frame, spec)
    occ_violations = [v for v in violations if v["column"] == "Occupation"]
    assert any(v["violation"] == "deferred_column_encoded_prematurely" for v in occ_violations), (
        f"Expected 'deferred_column_encoded_prematurely' violation, got: {occ_violations}"
    )


def test_validate_semantic_roles_rejects_non_deferred_string_column():
    """Non-deferred unordered_categorical must be encoded — object dtype is a violation."""
    feature_frame = pd.DataFrame({
        "Occupation": ["Engineer", "Doctor", "Engineer"],  # object dtype, but intent=one_hot
        "Age": [25, 30, 35],
    })
    spec = {
        "transforms": {
            "Occupation": {
                "semantic_role": "unordered_categorical",
                "action": "keep",
                "representation_intent": "one_hot",
            },
            "Age": {
                "semantic_role": "numeric_continuous",
                "action": "keep",
            },
        }
    }
    violations = validate_semantic_roles(feature_frame, spec)
    occ_violations = [v for v in violations if v["column"] == "Occupation"]
    assert any(v["violation"] == "unordered_categorical_not_encoded" for v in occ_violations), (
        f"Expected 'unordered_categorical_not_encoded' violation, got: {occ_violations}"
    )


def test_numeric_role_rejects_object_dtype():
    """Phase 1.1: numeric_count / numeric_continuous with object dtype must emit
    not_numeric_dtype — the old one-sided `if numeric and ...` guard let this slip."""
    frame = pd.DataFrame({"Age": ["25", "bad", "40", None]})
    spec = {
        "transforms": {
            "Age": {"semantic_role": "numeric_count", "action": "keep"},
        }
    }
    violations = validate_semantic_roles(frame, spec)
    assert any(
        v["violation"] == "not_numeric_dtype" and v["column"] == "Age"
        for v in violations
    )


def test_numeric_continuous_object_dtype_rejected():
    """Same gate applies to numeric_continuous."""
    frame = pd.DataFrame({"Income": ["1000", "2000", "bad"]})
    spec = {
        "transforms": {
            "Income": {"semantic_role": "numeric_continuous", "action": "keep"},
        }
    }
    violations = validate_semantic_roles(frame, spec)
    assert any(
        v["violation"] == "not_numeric_dtype" and v["column"] == "Income"
        for v in violations
    )


def test_invariant_age_out_of_range():
    """Phase 1.2: Age outside [18, 100] triggers age_out_of_range."""
    frame = pd.DataFrame({"Age": [25, 30, -5, 200, 45]})
    violations = validate_semantic_invariants(frame, raw_frame=None, column_transform_spec={})
    assert any(v["violation"] == "age_out_of_range" for v in violations)


def test_invariant_age_low_cardinality_single_value():
    """Phase 1.2: Age collapsed to a single value on a production-sized frame
    triggers age_low_cardinality (threshold is frame_len >= 100)."""
    frame = pd.DataFrame({"Age": [33] * 150})
    violations = validate_semantic_invariants(frame, raw_frame=None, column_transform_spec={})
    assert any(v["violation"] == "age_low_cardinality" for v in violations)


def test_invariant_age_low_cardinality_skipped_on_small_frame():
    """Phase 1.2: cardinality check must not fire on toy/test frames where
    nunique is bounded by row count."""
    frame = pd.DataFrame({"Age": list(range(25, 31))})
    violations = validate_semantic_invariants(frame, raw_frame=None, column_transform_spec={})
    assert not any(v["violation"] == "age_low_cardinality" for v in violations)


def test_invariant_credit_history_age_year_only_flagged():
    """Phase 1.2: if every Credit_History_Age value is a multiple of 12, months
    were discarded — credit_history_age_year_only must fire."""
    frame = pd.DataFrame({"Credit_History_Age": [12, 24, 36, 48, 60, 72, 12, 24]})
    violations = validate_semantic_invariants(frame, raw_frame=None, column_transform_spec={})
    assert any(v["violation"] == "credit_history_age_year_only" for v in violations)


def test_invariant_credit_history_age_low_cardinality_vs_raw():
    """Phase 1.2: output cardinality much smaller than raw triggers
    credit_history_age_low_cardinality."""
    raw = pd.DataFrame({
        "Credit_History_Age": [f"{y} Years and {m} Months" for y in range(1, 21) for m in range(12)]
    })
    out = pd.DataFrame({"Credit_History_Age": [12.0, 24.0, 36.0, 48.0, 60.0] * 48})
    violations = validate_semantic_invariants(out, raw_frame=raw, column_transform_spec={})
    assert any(
        v["violation"] == "credit_history_age_low_cardinality" for v in violations
    )


def test_invariant_missing_sentinel_collision():
    """Phase 1.2: a row with <col>_missing=1 and <col>_<value>=1 at the same
    time is a double-encoded missingness bug."""
    frame = pd.DataFrame({
        "Type_of_Loan_missing": [1, 0, 0, 1, 0],
        "Type_of_Loan_Home Loan": [1, 1, 0, 0, 0],
        "Type_of_Loan_Auto Loan": [0, 0, 1, 0, 1],
    })
    violations = validate_semantic_invariants(frame, raw_frame=None, column_transform_spec={})
    assert any(v["violation"] == "missing_sentinel_collision" for v in violations)


def test_invariant_duplicate_missingness_encoding_not_specified():
    """Phase 1.2: a <col>_Not Specified dummy alongside <col>_missing is forbidden."""
    frame = pd.DataFrame({
        "Occupation_missing": [1, 0, 1, 0],
        "Occupation_Not Specified": [1, 0, 1, 0],
        "Occupation_Engineer": [0, 1, 0, 1],
    })
    violations = validate_semantic_invariants(frame, raw_frame=None, column_transform_spec={})
    assert any(v["violation"] == "duplicate_missingness_encoding" for v in violations)


def test_invariant_passes_on_clean_frame():
    """Phase 1.2: well-formed frame should emit zero invariant violations."""
    raw = pd.DataFrame({
        "Credit_History_Age": [
            f"{y} Years and {m} Months" for y in range(1, 11) for m in range(0, 6)
        ]  # 60 unique raw strings
    })
    # Output covers months with enough non-year values (>5%) and cardinality
    # ratio stays >= 0.3 against the raw 60 unique strings.
    ages = list(range(18, 78))
    cha = [12 * y + m for y in range(1, 11) for m in range(0, 6)]
    frame = pd.DataFrame({
        "Age": ages,
        "Credit_History_Age": cha,
    })
    violations = validate_semantic_invariants(frame, raw_frame=raw, column_transform_spec={})
    assert violations == [], violations
