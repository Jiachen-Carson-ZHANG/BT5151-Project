import pandas as pd
from langgraph.graph import StateGraph

import bt5151_credit_risk.graph as graph_module
from bt5151_credit_risk.graph import build_graph
from bt5151_credit_risk.graph import compile_graph


class _DeterministicSplitter:
    def __init__(self, *args, **kwargs):
        pass

    def split(self, feature_frame, target, groups):
        train_idx = [0, 1, 2, 3, 4, 5]
        test_idx = [6, 7, 8]
        yield train_idx, test_idx


def _generated_preprocessing_code(
    feature_frame_expression: str,
    *,
    include_forbidden_import: bool,
) -> dict:
    lines = [
        "from pathlib import Path\n",
        "import json\n",
    ]
    if include_forbidden_import:
        lines.append("import subprocess\n")
    lines.extend(
        [
            "\n",
            "def run_preprocessing(raw_df, workspace_path):\n",
            "    workspace = Path(workspace_path)\n",
            "    cleaned = raw_df.copy()\n",
            f"    feature_frame = {feature_frame_expression}\n",
            "    target = cleaned['Credit_Score']\n",
            "    cleaned.to_csv(workspace / 'cleaned_frame.csv', index=False)\n",
            "    feature_frame.to_csv(workspace / 'feature_frame.csv', index=False)\n",
            "    target.to_frame(name='Credit_Score').to_csv(workspace / 'target.csv', index=False)\n",
            "    (workspace / 'split_manifest.json').write_text(json.dumps({'train_indices': [0, 1, 2, 3, 4, 5], 'test_indices': [6, 7, 8]}))\n",
            "    (workspace / 'preprocessing_report.json').write_text(json.dumps({'status': 'ok'}))\n",
            "    return {'status': 'ok'}\n",
        ]
    )
    return {
        "code": "".join(lines),
        "entrypoint": "run_preprocessing",
    }


def _sample_policy_spec():
    return {
        "task_type": "multiclass_classification",
        "target_column": "Credit_Score",
        "group_column": "Customer_ID",
        "identifier_columns": ["ID", "Name", "SSN"],
        "split_strategy": {"type": "grouped_holdout", "test_size": 0.34},
        "leakage_rules": {"drop_columns": ["ID", "Name", "SSN"]},
        "imbalance_strategy": {"method": "none"},
        "feature_policy": {"categorical_encoding": "one_hot"},
    }


def _sample_column_transform_spec():
    return {
        "columns": {
            "ID": {"action": "drop"},
            "Customer_ID": {"action": "drop"},
            "Name": {"action": "drop"},
            "SSN": {"action": "drop"},
            "Credit_Score": {"action": "drop"},
            "Age": {"action": "keep", "imputation": "median"},
            "Outstanding_Debt": {"action": "keep", "imputation": "median"},
        }
    }


def test_graph_contains_required_nodes():
    graph = build_graph()
    assert isinstance(graph, StateGraph)
    expected_nodes = {
        "dataset-policy-spec",
        "column-transform-spec",
        "generate-preprocessing-code",
        "inspect-preprocessing-code",
        "execute-generated-preprocessing",
        "validate-preprocessing-output",
        "repair-preprocessing-code",
        "train-models",
        "evaluate-models",
        "select-model",
        "run-inference",
        "explain-risk",
        "recommend-action",
    }
    assert expected_nodes.issubset(set(graph.nodes.keys()))
    assert "execute-preprocessing" not in graph.nodes
    assert "audit-preprocessing" not in graph.nodes
    assert hasattr(graph.compile(), "invoke")


def test_compiled_graph_runs_new_preprocessing_loop_end_to_end(tmp_path, monkeypatch):
    sample_frame = pd.DataFrame(
        [
            {"ID": "0x1", "Customer_ID": "CUS_A", "Name": "Alice", "SSN": "111", "Credit_Score": "Good", "Age": 25, "Outstanding_Debt": 1000.0},
            {"ID": "0x2", "Customer_ID": "CUS_A", "Name": "Alice", "SSN": "111", "Credit_Score": "Good", "Age": 26, "Outstanding_Debt": 900.0},
            {"ID": "0x3", "Customer_ID": "CUS_B", "Name": "Bob", "SSN": "222", "Credit_Score": "Standard", "Age": 33, "Outstanding_Debt": 1800.0},
            {"ID": "0x4", "Customer_ID": "CUS_B", "Name": "Bob", "SSN": "222", "Credit_Score": "Standard", "Age": 34, "Outstanding_Debt": 1750.0},
            {"ID": "0x5", "Customer_ID": "CUS_C", "Name": "Cara", "SSN": "333", "Credit_Score": "Poor", "Age": 45, "Outstanding_Debt": 4200.0},
            {"ID": "0x6", "Customer_ID": "CUS_C", "Name": "Cara", "SSN": "333", "Credit_Score": "Poor", "Age": 46, "Outstanding_Debt": 4300.0},
            {"ID": "0x7", "Customer_ID": "CUS_D", "Name": "Drew", "SSN": "444", "Credit_Score": "Good", "Age": 28, "Outstanding_Debt": 950.0},
            {"ID": "0x8", "Customer_ID": "CUS_E", "Name": "Elle", "SSN": "555", "Credit_Score": "Standard", "Age": 38, "Outstanding_Debt": 2100.0},
            {"ID": "0x9", "Customer_ID": "CUS_F", "Name": "Finn", "SSN": "666", "Credit_Score": "Poor", "Age": 52, "Outstanding_Debt": 5100.0},
        ]
    )
    dataset_path = tmp_path / "sample_credit.csv"
    sample_frame.to_csv(dataset_path, index=False)

    call_sequence = []
    repair_attempts = {"count": 0}

    def fake_generate_dataset_policy_spec(raw_frame, dataset_profile):
        call_sequence.append("dataset-policy-spec")
        return _sample_policy_spec()

    def fake_generate_column_transform_spec(raw_frame, dataset_policy_spec):
        call_sequence.append("column-transform-spec")
        return _sample_column_transform_spec()

    def fake_generate_preprocessing_code(raw_frame, dataset_profile, dataset_policy_spec, column_transform_spec):
        call_sequence.append("generate-preprocessing-code")
        return _generated_preprocessing_code(
            "cleaned[['Credit_Score', 'Age', 'Outstanding_Debt']]",
            include_forbidden_import=True,
        )

    def fake_repair_preprocessing_code(**kwargs):
        repair_attempts["count"] += 1
        call_sequence.append("repair-preprocessing-code")
        if repair_attempts["count"] == 1:
            return _generated_preprocessing_code(
                "cleaned[['Credit_Score', 'Age', 'Outstanding_Debt']]",
                include_forbidden_import=False,
            )
        return _generated_preprocessing_code(
            "cleaned[['Age', 'Outstanding_Debt']]",
            include_forbidden_import=False,
        )

    def fake_explain_risk(predicted_label, probabilities, **kwargs):
        call_sequence.append("explain-risk")
        return {
            "predicted_label": predicted_label,
            "risk_level": "high" if predicted_label == "Poor" else "moderate",
            "confidence_band": "medium",
            "summary": "Synthetic explanation for graph test.",
        }

    def fake_recommend_action(risk_explanation, prediction_output=None):
        call_sequence.append("recommend-action")
        return {
            "action": "manual_review",
            "reason": "The predicted risk should be reviewed by a human analyst.",
        }

    monkeypatch.setattr(graph_module, "generate_dataset_policy_spec", fake_generate_dataset_policy_spec)
    monkeypatch.setattr(graph_module, "generate_column_transform_spec", fake_generate_column_transform_spec)
    monkeypatch.setattr(graph_module, "generate_preprocessing_code", fake_generate_preprocessing_code)
    monkeypatch.setattr(graph_module, "repair_preprocessing_code", fake_repair_preprocessing_code)
    monkeypatch.setattr(graph_module, "explain_risk", fake_explain_risk)
    monkeypatch.setattr(graph_module, "recommend_action", fake_recommend_action)

    graph = compile_graph()
    result = graph.invoke(
        {
            "raw_dataset_path": str(dataset_path),
            "inference_input": {"row_index": 6},
        }
    )

    assert repair_attempts["count"] == 2
    assert call_sequence.count("repair-preprocessing-code") == 2
    assert result["dataset_policy_spec"]["target_column"] == "Credit_Score"
    assert result["preprocessing_validation_report"]["passed"] is True
    assert result["full_feature_frame"].equals(pd.read_csv(result["preprocessing_artifacts"]["feature_frame.csv"]))
    assert result["train_frame"].shape[0] > 0
    assert result["test_frame"].shape[0] > 0
    assert result["feature_columns"] == ["Age", "Outstanding_Debt"]
    assert result["class_names"] == ["Good", "Poor", "Standard"]
    assert result["label_to_id"] == {"Good": 0, "Poor": 1, "Standard": 2}
    assert result["id_to_label"] == {0: "Good", 1: "Poor", 2: "Standard"}
    assert result["selected_model_name"] in {"logistic_regression", "random_forest"}
    assert result["prediction_output"]["predicted_label"] in {"Good", "Standard", "Poor"}
    assert result["risk_explanation"]["summary"] == "Synthetic explanation for graph test."
    assert result["recommended_action"]["action"] == "manual_review"
