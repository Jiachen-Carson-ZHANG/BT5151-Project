import pandas as pd
from langgraph.graph import StateGraph

import bt5151_credit_risk.business as business
import bt5151_credit_risk.preprocess as preprocess
from bt5151_credit_risk.graph import build_graph
from bt5151_credit_risk.graph import compile_graph


class _DeterministicSplitter:
    def __init__(self, *args, **kwargs):
        pass

    def split(self, feature_frame, target, groups):
        train_idx = [0, 1, 2, 3, 4, 5]
        test_idx = [6, 7, 8]
        yield train_idx, test_idx


def test_graph_contains_required_nodes():
    graph = build_graph()
    assert isinstance(graph, StateGraph)
    expected_nodes = {
        "dataset-policy-spec",
        "column-transform-spec",
        "execute-preprocessing",
        "audit-preprocessing",
        "train-models",
        "evaluate-models",
        "select-model",
        "run-inference",
        "explain-risk",
        "recommend-action",
    }
    assert expected_nodes.issubset(set(graph.nodes.keys()))
    assert hasattr(graph.compile(), "invoke")


def test_compiled_graph_runs_end_to_end(tmp_path, monkeypatch):
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

    def fake_business_agent(system_prompt, payload):
        if "keys: action, reason" in system_prompt:
            return {
                "action": "manual_review",
                "reason": "The predicted risk should be reviewed by a human analyst.",
            }
        return {
            "predicted_label": payload["predicted_label"],
            "risk_level": "high" if payload["predicted_label"] == "Poor" else "moderate",
            "confidence_band": "medium",
            "summary": "Synthetic explanation for graph test.",
        }

    def fake_preprocess_agent(system_prompt, payload):
        if "task_type, target_column, group_column" in system_prompt:
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

    monkeypatch.setattr(preprocess, "GroupShuffleSplit", _DeterministicSplitter)
    monkeypatch.setattr(preprocess, "_call_preprocess_agent", fake_preprocess_agent)
    monkeypatch.setattr(business, "_call_json_agent", fake_business_agent)

    graph = compile_graph()
    result = graph.invoke(
        {
            "raw_dataset_path": str(dataset_path),
            "inference_input": {"row_index": 6},
        }
    )

    assert result["dataset_policy_spec"]["target_column"] == "Credit_Score"
    assert result["preprocessing_audit_report"]["passed"] is True
    assert result["selected_model_name"] in {"logistic_regression", "random_forest"}
    assert result["prediction_output"]["predicted_label"] in {"Good", "Standard", "Poor"}
    assert result["risk_explanation"]["summary"] == "Synthetic explanation for graph test."
    assert result["recommended_action"]["action"] == "manual_review"
