from pathlib import Path

import pandas as pd
import pytest

from bt5151_credit_risk.preprocess import execute_generated_preprocessing
from bt5151_credit_risk.preprocess import generate_column_transform_spec
from bt5151_credit_risk.preprocess import generate_dataset_policy_spec
from bt5151_credit_risk.preprocess import generate_preprocessing_code
from bt5151_credit_risk.preprocess import inspect_preprocessing_code
from bt5151_credit_risk.preprocess import repair_preprocessing_code
from bt5151_credit_risk.preprocess import validate_preprocessing_output
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
                "encoding": "one_hot",
                "imputation": "mode",
                "representation_intent": "one_hot",
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
