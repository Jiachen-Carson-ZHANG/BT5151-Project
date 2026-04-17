import numpy as np
import pandas as pd
from langgraph.graph import StateGraph
from sklearn.ensemble import RandomForestClassifier
from types import SimpleNamespace

import bt5151_credit_risk.graph as graph_module
from bt5151_credit_risk.graph import (
    build_graph,
    compile_graph,
    evaluate_models_node,
    package_analysis_bundle_node,
    run_inference_node,
    select_model_node,
    training_diagnostics_node,
)
from bt5151_credit_risk.state import CreditRiskState


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
        }
    }


def test_graph_contains_required_nodes():
    graph = build_graph()
    assert isinstance(graph, StateGraph)
    expected_nodes = {
        "dataset-policy-spec",
        "exploratory-data-analysis",
        "generate-eda-hypotheses",
        "column-transform-spec",
        "generate-preprocessing-code",
        "inspect-preprocessing-code",
        "execute-generated-preprocessing",
        "validate-preprocessing-output",
        "review-preprocessing-quality",
        "repair-preprocessing-code",
        "generate-feature-engineering-code",
        "inspect-feature-engineering-code",
        "execute-feature-engineering",
        "validate-feature-engineering",
        "repair-feature-engineering-code",
        "train-models",
        "evaluate-models",
        "training-diagnostics",
        "select-model",
        "global-xai",
        "local-xai",
        "shortcut-feature-audit",
        "interpret-global-xai",
        "interpret-local-xai",
        "package-analysis-bundle",
        "run-inference",
        "explain-risk",
    }
    assert expected_nodes.issubset(set(graph.nodes.keys()))
    assert "execute-preprocessing" not in graph.nodes
    assert "audit-preprocessing" not in graph.nodes
    assert hasattr(graph.compile(), "invoke")


def test_generate_feature_engineering_defaults_to_deterministic_safe_path(monkeypatch):
    """Production FE should not spend repair loops on LLM codegen unless explicitly enabled."""
    monkeypatch.setattr(graph_module, "FEATURE_ENGINEERING_MODE", "deterministic")

    def fail_if_llm_codegen_called(*args, **kwargs):
        raise AssertionError("LLM feature engineering codegen should not be called in deterministic mode")

    monkeypatch.setattr(graph_module, "generate_feature_engineering_code", fail_if_llm_codegen_called)

    state = SimpleNamespace(
        train_frame=pd.DataFrame({"Age": [30, 40], "Occupation": ["Engineer", "Doctor"]}),
        test_frame=pd.DataFrame({"Age": [35], "Occupation": ["Engineer"]}),
        feature_columns=["Age", "Occupation"],
        dataset_profile={"row_count": 3},
        eda_report={},
        eda_hypotheses={},
        deferred_categorical_columns={"Occupation": 2},
    )

    result = graph_module.generate_feature_engineering_code_node(state)

    assert result["feature_engineering_attempt_count"] == 1
    assert "code" in result["feature_engineering_code"]
    assert result["feature_engineering_hypothesis"]["interactions_rationale"].startswith(
        "Deterministic feature engineering"
    )


def test_route_after_quality_review_only_accepts_minor_issues_after_retries():
    state = SimpleNamespace(
        preprocessing_validation_report={"structural_passed": True, "passed": False},
        preprocessing_audit_report={
            "verdict": "needs_repair",
            "issues": [{"severity": "major", "category": "distribution_sanity"}],
        },
        preprocessing_attempt_count=3,
    )

    assert graph_module._route_after_quality_review(state) == "repair-preprocessing-code"

    state.preprocessing_audit_report = {
        "verdict": "needs_repair",
        "issues": [{"severity": "minor", "category": "feature_engineering"}],
    }
    assert graph_module._route_after_quality_review(state) == "generate-feature-engineering-code"


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

    def fake_generate_column_transform_spec(raw_frame, dataset_policy_spec, eda_report=None):
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
            "recommended_action": {
                "action": "manual_review",
                "urgency": "routine",
                "rationale": "Synthetic rationale for graph test.",
            },
        }

    def fake_review_preprocessing_quality(execution_result, dataset_policy_spec, column_transform_spec, previous_audit_report=None):
        call_sequence.append("review-preprocessing-quality")
        return {
            "verdict": "pass",
            "issues": [],
            "summary": "Synthetic quality review for graph test.",
        }

    def fake_generate_feature_engineering_code(
        train_frame,
        test_frame,
        feature_columns,
        dataset_profile,
        eda_report=None,
        eda_hypotheses=None,
        deferred_categorical_columns=None,
    ):
        call_sequence.append("generate-feature-engineering-code")
        # Identity transform — writes train/test through unchanged.
        code = (
            "import json\n"
            "import pandas as pd\n"
            "from pathlib import Path\n\n"
            "def engineer_features(train_df, test_df, workspace_path):\n"
            "    workspace_path = Path(workspace_path)\n"
            "    train_df.to_csv(workspace_path / 'engineered_train.csv', index=False)\n"
            "    test_df.to_csv(workspace_path / 'engineered_test.csv', index=False)\n"
            "    report = {'dropped': [], 'transformed': [], 'added': []}\n"
            "    (workspace_path / 'feature_engineering_report.json').write_text(json.dumps(report))\n"
            "    lineage = {\n"
            "        'derived_features': [],\n"
            "        'dropped_features': [],\n"
            "        'passthrough_features': list(train_df.columns),\n"
            "    }\n"
            "    (workspace_path / 'feature_lineage.json').write_text(json.dumps(lineage))\n"
        )
        return {"code": code, "entrypoint": "engineer_features"}

    def fake_repair_feature_engineering_code(**kwargs):
        call_sequence.append("repair-feature-engineering-code")
        return fake_generate_feature_engineering_code(None, None, None, None)

    monkeypatch.setattr(graph_module, "generate_dataset_policy_spec", fake_generate_dataset_policy_spec)
    monkeypatch.setattr(graph_module, "generate_column_transform_spec", fake_generate_column_transform_spec)
    monkeypatch.setattr(graph_module, "generate_preprocessing_code", fake_generate_preprocessing_code)
    monkeypatch.setattr(graph_module, "repair_preprocessing_code", fake_repair_preprocessing_code)
    monkeypatch.setattr(graph_module, "review_preprocessing_quality", fake_review_preprocessing_quality)
    monkeypatch.setattr(graph_module, "generate_feature_engineering_code", fake_generate_feature_engineering_code)
    monkeypatch.setattr(graph_module, "repair_feature_engineering_code", fake_repair_feature_engineering_code)
    def fake_reason_hyperparameter_grids(model_names, train_rows, feature_count, class_distribution, current_metrics):
        call_sequence.append("reason-hyperparameter-grid")
        return {"grids": {}, "reasoning": "Test: skip tuning"}

    def fake_tune_models(
        models,
        grids,
        train_frame,
        train_target,
        sample_weights=None,
        validation_policy=None,
        train_group_values=None,
        train_time_values=None,
    ):
        # Just fit models with defaults, no actual CV.
        from sklearn.utils.class_weight import compute_sample_weight
        sw = compute_sample_weight("balanced", train_target) if sample_weights is not None else None
        assert validation_policy is not None
        assert validation_policy["type"] == "grouped_entity"
        assert train_group_values is not None
        assert len(train_group_values) == len(train_frame)
        for name, model in models.items():
            if name == "xgboost" and sw is not None:
                model.fit(train_frame, train_target, sample_weight=sw)
            else:
                model.fit(train_frame, train_target)
        return models, {}, {}

    def fake_reason_model_selection(evaluation_results, **kwargs):
        call_sequence.append("reason-model-selection")
        best_name = max(evaluation_results, key=lambda n: evaluation_results[n]["macro_f1"])
        return {
            "model_name": best_name,
            "justification": "Test: best macro_f1",
            "hypothesis_validation": "Test: no hypothesis to validate",
        }

    def fake_extract_learning_curves(model, model_name):
        return None

    def fake_build_eda_report(df, target_column):
        call_sequence.append("exploratory-data-analysis")
        return {
            "correlations": {"high_pairs": [], "matrix_shape": [2, 2]},
            "class_separability": {"class_means": {}, "anova_top_features": []},
            "skewness": {"all": {}, "highly_skewed": {}},
            "missing_patterns": {"missing_pct": {}, "mnar_suspects": []},
            "cardinality": {"all": {}, "high_cardinality": []},
            "top_discriminative_features": [],
        }

    def fake_generate_eda_hypotheses(eda_report, dataset_profile):
        call_sequence.append("generate-eda-hypotheses")
        return {
            "tested_predictions": [{"hypothesis": "test", "basis": "test", "testable_at": "evaluate-models"}],
            "supported_conjectures": [],
            "exploratory_leads": [],
            "model_selection_prediction": "XGBoost will win",
            "class_struggle_prediction": "Standard will struggle",
        }

    def fake_generate_training_diagnostics(**kwargs):
        call_sequence.append("training-diagnostics")
        return {
            "per_class_analysis": {},
            "capacity_analysis": {},
            "confidence_analysis": "Synthetic",
            "hypothesis_validation": {"tested": [], "supported": []},
            "new_hypotheses": {"tested_predictions": [], "supported_conjectures": [], "exploratory_leads": []},
        }

    def fake_compute_global_shap(model, model_name, test_frame, feature_columns, class_names, **kwargs):
        return {"importance": [], "beeswarm_data": {}, "dependence_data": {}}

    def fake_compute_permutation_importance(model, model_name, test_frame, test_target, feature_columns, **kwargs):
        return {"raw": [], "grouped": []}

    def fake_compute_partial_dependence(*args, **kwargs):
        return {}

    def fake_compute_ale(*args, **kwargs):
        return {}

    def fake_compute_shap_contributions_for_case(model, model_name, input_frame, predicted_class_idx, feature_names, **kwargs):
        return {
            "predicted_class_waterfall": {
                "class": "Good",
                "base_value": 0.33,
                "top_features": [{"feature": "Age", "shap_value": 0.1, "direction": "toward"}],
            },
            "all_classes": {},
            "base_values": {},
        }

    def fake_select_classification_cases(test_frame, test_target, model, class_names, id_to_label):
        return [
            {"row_index": int(test_frame.index[0]), "true_label": "Good", "predicted_label": "Good",
             "probabilities": {"Good": 0.8, "Poor": 0.1, "Standard": 0.1}, "case_type": "representative",
             "confused_with_class": None},
        ]

    def fake_interpret_global_xai(**kwargs):
        call_sequence.append("interpret-global-xai")
        return {
            "observations": ["Fake global observation"],
            "insights": ["Fake global insight"],
            "feature_importance_consensus": {"agreement": [], "interpretation": "Synthetic"},
            "cross_layer_validation": {},
            "hypotheses": {"tested_predictions": [], "supported_conjectures": [], "exploratory_leads": []},
        }

    def fake_interpret_local_xai(**kwargs):
        call_sequence.append("interpret-local-xai")
        return {
            "per_class_stories": {"Good": {"representative_profile": "fake", "borderline_story": "fake", "worst_misclassification_story": "fake"}},
            "confusion_patterns": {"dominant_direction": "fake"},
            "global_vs_local_consistency": {},
            "decision_boundary_analysis": {"thinnest_boundary": "fake"},
            "hypotheses": {"tested_predictions": [], "supported_conjectures": [], "exploratory_leads": []},
        }

    monkeypatch.setattr(graph_module, "reason_hyperparameter_grids", fake_reason_hyperparameter_grids)
    monkeypatch.setattr(graph_module, "tune_models", fake_tune_models)
    monkeypatch.setattr(graph_module, "extract_learning_curves", fake_extract_learning_curves)
    monkeypatch.setattr(graph_module, "reason_model_selection", fake_reason_model_selection)
    monkeypatch.setattr(graph_module, "build_eda_report", fake_build_eda_report)
    monkeypatch.setattr(graph_module, "explain_risk", fake_explain_risk)
    monkeypatch.setattr(graph_module, "generate_eda_hypotheses", fake_generate_eda_hypotheses)
    monkeypatch.setattr(graph_module, "generate_training_diagnostics", fake_generate_training_diagnostics)
    monkeypatch.setattr(graph_module, "compute_global_shap", fake_compute_global_shap)
    monkeypatch.setattr(graph_module, "compute_permutation_importance", fake_compute_permutation_importance)
    monkeypatch.setattr(graph_module, "compute_partial_dependence", fake_compute_partial_dependence)
    monkeypatch.setattr(graph_module, "compute_ale", fake_compute_ale)
    monkeypatch.setattr(graph_module, "compute_shap_contributions_for_case", fake_compute_shap_contributions_for_case)
    monkeypatch.setattr(graph_module, "select_classification_cases", fake_select_classification_cases)
    monkeypatch.setattr(graph_module, "interpret_global_xai", fake_interpret_global_xai)
    monkeypatch.setattr(graph_module, "interpret_local_xai", fake_interpret_local_xai)

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
    assert result["train_group_values"] is not None
    assert result["test_group_values"] is not None
    assert len(result["train_group_values"]) == len(result["train_frame"])
    assert len(result["test_group_values"]) == len(result["test_frame"])
    assert result["feature_columns"] == ["Age", "Outstanding_Debt"]
    assert result["class_names"] == ["Good", "Poor", "Standard"]
    assert result["label_to_id"] == {"Good": 0, "Poor": 1, "Standard": 2}
    assert result["id_to_label"] == {0: "Good", 1: "Poor", 2: "Standard"}
    assert result["selected_model_name"] in {"logistic_regression", "random_forest", "xgboost"}
    assert result["prediction_output"]["predicted_label"] in {"Good", "Standard", "Poor"}
    assert result["risk_explanation"]["summary"] == "Synthetic explanation for graph test."
    assert result["recommended_action"]["action"] == "manual_review"
    # New XAI overhaul assertions
    assert result["eda_hypotheses"] is not None
    assert result["eda_hypotheses"]["model_selection_prediction"] == "XGBoost will win"
    assert result["training_diagnostics"] is not None
    assert result["global_xai_results"] is not None
    assert "methods_used" in result["global_xai_results"]
    assert result["local_xai_cases"] is not None
    assert len(result["local_xai_cases"]) >= 1
    assert result["analysis_bundle"] is not None
    assert result["analysis_bundle_summary"] is not None


def test_validate_feature_engineering_node_filters_identifier_top_mi_features(monkeypatch):
    """Identifier/group/target/drop-policy columns must not be enforced as FE survivors."""
    captured: dict[str, object] = {}

    def fake_validate_feature_engineering_output(*args, **kwargs):
        captured["top_mi_features"] = kwargs.get("top_mi_features")
        return {"passed": False, "checks": {}, "errors": []}

    monkeypatch.setattr(graph_module, "validate_feature_engineering_output", fake_validate_feature_engineering_output)

    state = SimpleNamespace(
        feature_engineering_execution_log={"artifacts": {}},
        train_frame=pd.DataFrame({"Annual_Income": [50000.0, 60000.0], "Outstanding_Debt": [1000.0, 2000.0]}),
        test_frame=pd.DataFrame({"Annual_Income": [55000.0], "Outstanding_Debt": [1500.0]}),
        feature_columns=["Annual_Income", "Outstanding_Debt"],
        deferred_categorical_columns={},
        eda_report={
            "top_discriminative_features": [
                {"column": "Customer_ID"},
                {"column": "SSN"},
                {"column": "ID"},
                {"column": "Annual_Income"},
                {"column": "Outstanding_Debt"},
                {"column": "Credit_Mix"},
            ]
        },
        dataset_policy_spec={
            "identifier_columns": ["ID", "SSN"],
            "group_column": "Customer_ID",
            "target_column": "Credit_Score",
        },
        column_transform_spec={
            "transforms": {
                "Customer_ID": {"action": "drop"},
                "SSN": {"action": "drop"},
                "ID": {"action": "drop"},
                "Credit_Score": {"action": "drop"},
                "Annual_Income": {"action": "keep"},
                "Outstanding_Debt": {"action": "keep"},
                "Credit_Mix": {"action": "keep"},
            }
        },
    )

    graph_module.validate_feature_engineering_node(state)

    assert captured["top_mi_features"] == ["Annual_Income", "Outstanding_Debt", "Credit_Mix"]


# ---------------------------------------------------------------------------
# Focused tests for select_model_node SHAP recomputation
# ---------------------------------------------------------------------------

def _build_select_model_state(trained_models, test_frame, test_target, feature_columns, class_names):
    """Build a minimal CreditRiskState for testing select_model_node."""
    return CreditRiskState(
        raw_dataset_path="dummy.csv",
        trained_models=trained_models,
        test_frame=test_frame,
        test_target=test_target,
        feature_columns=feature_columns,
        class_names=class_names,
        id_to_label={i: c for i, c in enumerate(class_names)},
        label_to_id={c: i for i, c in enumerate(class_names)},
        evaluation_results={
            name: {"macro_f1": 0.8 - i * 0.05, "weighted_f1": 0.8 - i * 0.05, "accuracy": 0.8}
            for i, name in enumerate(trained_models)
        },
    )


def test_select_model_ignores_llm_override_and_keeps_metric_best(monkeypatch):
    """The metric-best model must remain authoritative even if the LLM disagrees."""
    rng = np.random.RandomState(42)
    n = 50
    X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n)})
    y = pd.Series(np.where(X["f1"] > 0, 1, 0))

    rf = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf2 = RandomForestClassifier(n_estimators=5, random_state=99, n_jobs=-1)
    rf2.fit(X, y)

    state = _build_select_model_state(
        trained_models={"random_forest": rf, "xgboost": rf2},
        test_frame=X, test_target=y,
        feature_columns=["f1", "f2"], class_names=["c0", "c1"],
    )

    # LLM tries to select xgboost (not the metric-best random_forest)
    def fake_reason_model_selection(**kwargs):
        return {"model_name": "xgboost", "justification": "test override"}

    recompute_calls = []
    original_compute_global_shap = graph_module.compute_global_shap

    def tracking_compute_global_shap(model, model_name, *args, **kwargs):
        recompute_calls.append(model_name)
        return original_compute_global_shap(model, model_name, *args, **kwargs)

    monkeypatch.setattr(graph_module, "reason_model_selection", fake_reason_model_selection)
    monkeypatch.setattr(graph_module, "compute_global_shap", tracking_compute_global_shap)

    result = select_model_node(state)

    assert result["selected_model_name"] == "random_forest"
    assert "test override" in result["selection_justification"]
    assert "xgboost" not in recompute_calls, (
        f"LLM override should not control SHAP recomputation, got calls for: {recompute_calls}"
    )
    assert "random_forest" in recompute_calls
    assert len(result["global_shap_importance"]) > 0


def test_select_model_retries_shap_when_initial_fails(monkeypatch):
    """When initial SHAP fails for metric-best, it must retry for the selected model."""
    rng = np.random.RandomState(42)
    n = 50
    X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n)})
    y = pd.Series(np.where(X["f1"] > 0, 1, 0))

    rf = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    state = _build_select_model_state(
        trained_models={"random_forest": rf},
        test_frame=X, test_target=y,
        feature_columns=["f1", "f2"], class_names=["c0", "c1"],
    )

    def fake_reason_model_selection(**kwargs):
        return {"model_name": "random_forest", "justification": "only model"}

    recompute_calls = []
    original_compute_global_shap = graph_module.compute_global_shap

    def tracking_compute_global_shap(model, model_name, *args, **kwargs):
        recompute_calls.append(model_name)
        return original_compute_global_shap(model, model_name, *args, **kwargs)

    # Force initial SHAP (the inline try block) to fail by breaking shap import
    import shap as shap_module
    original_tree = shap_module.TreeExplainer

    def broken_tree(*args, **kwargs):
        raise RuntimeError("Simulated SHAP failure")

    monkeypatch.setattr(shap_module, "TreeExplainer", broken_tree)
    monkeypatch.setattr(graph_module, "reason_model_selection", fake_reason_model_selection)
    monkeypatch.setattr(graph_module, "compute_global_shap", tracking_compute_global_shap)

    result = select_model_node(state)

    assert result["selected_model_name"] == "random_forest"
    # The retry path must have been triggered (compute_global_shap called)
    assert "random_forest" in recompute_calls, (
        f"Expected SHAP retry for 'random_forest', got calls for: {recompute_calls}"
    )


class _ColumnCheckingModel:
    def __init__(self, expected_columns, predicted_class=0):
        self.expected_columns = expected_columns
        self.predicted_class = predicted_class

    def predict(self, frame):
        assert list(frame.columns) == self.expected_columns
        return np.array([self.predicted_class] * len(frame))

    def predict_proba(self, frame):
        assert list(frame.columns) == self.expected_columns
        rows = len(frame)
        return np.tile(np.array([[0.8, 0.1, 0.1]]), (rows, 1))


def test_evaluate_models_uses_model_specific_test_views():
    state = SimpleNamespace(
        trained_models={
            "logistic_regression": _ColumnCheckingModel(["lin_a", "lin_b"], predicted_class=0),
            "xgboost": _ColumnCheckingModel(["tree_x"], predicted_class=0),
        },
        test_frame=pd.DataFrame({"tree_x": [1.0, 2.0]}),  # legacy/default frame
        test_views={
            "linear_view": pd.DataFrame({"lin_a": [0.1, 0.2], "lin_b": [1.0, 1.1]}),
            "tree_view": pd.DataFrame({"tree_x": [1.0, 2.0]}),
        },
        model_view_map={
            "logistic_regression": "linear_view",
            "xgboost": "tree_view",
        },
        test_target=pd.Series([0, 0]),
        class_names=["c0", "c1", "c2"],
    )

    result = evaluate_models_node(state)
    assert set(result["evaluation_results"]) == {"logistic_regression", "xgboost"}


def test_run_inference_uses_selected_model_full_view(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "compute_shap_contributions_for_case",
        lambda model, model_name, input_frame, predicted_code, feature_names, top_n=10, class_names=None: {
            "predicted_class_waterfall": {
                "class": "Good",
                "base_value": 0.33,
                "top_features": [{"feature": feature_names[0], "shap_value": 0.4, "direction": "toward"}],
            },
            "all_classes": {},
            "base_values": {},
        },
    )

    state = SimpleNamespace(
        inference_input={"row_index": 7},
        selected_model_name="logistic_regression",
        trained_models={"logistic_regression": _ColumnCheckingModel(["lin_a", "lin_b"], predicted_class=0)},
        full_feature_frame=pd.DataFrame({"tree_x": [9.0]}, index=[7]),  # legacy/default frame
        full_feature_frames_by_view={
            "linear_view": pd.DataFrame({"lin_a": [0.5], "lin_b": [1.5]}, index=[7]),
            "tree_view": pd.DataFrame({"tree_x": [9.0]}, index=[7]),
        },
        model_view_map={"logistic_regression": "linear_view"},
        feature_columns=["tree_x"],
        feature_columns_by_view={"linear_view": ["lin_a", "lin_b"], "tree_view": ["tree_x"]},
        id_to_label={0: "Good", 1: "Poor", 2: "Standard"},
        class_names=["Good", "Poor", "Standard"],
        evaluation_results={"logistic_regression": {"macro_f1": 0.7}},
        raw_frame=pd.DataFrame({"Customer_ID": ["CUS_A"]}, index=[7]),
        global_xai_results=None,
        training_diagnostics=None,
        local_xai_cases=None,
    )

    result = run_inference_node(state)
    assert result["prediction_output"]["predicted_label"] == "Good"
    waterfall = result["prediction_output"]["shap_waterfall"]
    assert waterfall["predicted_class_waterfall"]["top_features"][0]["feature"] == "lin_a"


def test_run_inference_uses_structured_confidence_and_casebook_similarity(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "compute_shap_contributions_for_case",
        lambda model, model_name, input_frame, predicted_code, feature_names, top_n=10, class_names=None: {
            "predicted_class_waterfall": {
                "class": "Good",
                "base_value": 0.33,
                "top_features": [
                    {"feature": "lin_a", "shap_value": 0.4, "direction": "toward"},
                    {"feature": "lin_b", "shap_value": -0.2, "direction": "away_from"},
                ],
            },
            "all_classes": {},
            "base_values": {},
        },
    )

    state = SimpleNamespace(
        inference_input={"row_index": 7},
        selected_model_name="logistic_regression",
        trained_models={"logistic_regression": _ColumnCheckingModel(["lin_a", "lin_b"], predicted_class=0)},
        full_feature_frame=pd.DataFrame({"tree_x": [9.0]}, index=[7]),
        full_feature_frames_by_view={
            "linear_view": pd.DataFrame({"lin_a": [0.5], "lin_b": [1.5]}, index=[7]),
            "tree_view": pd.DataFrame({"tree_x": [9.0]}, index=[7]),
        },
        model_view_map={"logistic_regression": "linear_view"},
        feature_columns=["tree_x"],
        feature_columns_by_view={"linear_view": ["lin_a", "lin_b"], "tree_view": ["tree_x"]},
        id_to_label={0: "Good", 1: "Poor", 2: "Standard"},
        class_names=["Good", "Poor", "Standard"],
        evaluation_results={"logistic_regression": {"macro_f1": 0.7}},
        raw_frame=pd.DataFrame({"Customer_ID": ["CUS_A"]}, index=[7]),
        global_xai_results=None,
        training_diagnostics={
            "per_class_analysis": {"Good": {"struggle_level": "low"}},
            "confidence_analysis": {
                "summary": "Synthetic confidence summary",
                "by_model": {
                    "logistic_regression": {
                        "correct_mean_confidence": 0.81,
                        "per_class_correct_mean_confidence": {"Good": 0.79},
                    }
                },
            },
        },
        local_xai_cases=[
            {
                "row_index": 11,
                "case_type": "representative",
                "true_label": "Good",
                "predicted_label": "Good",
                "probabilities": {"Good": 0.9, "Poor": 0.05, "Standard": 0.05},
                "shap_contributions": {
                    "predicted_class_waterfall": {
                        "class": "Good",
                        "top_features": [
                            {"feature": "lin_a", "shap_value": 0.4, "direction": "toward"},
                            {"feature": "lin_b", "shap_value": -0.2, "direction": "away_from"},
                        ],
                    },
                    "all_classes": {},
                    "base_values": {},
                },
            }
        ],
    )

    result = run_inference_node(state)
    confidence_diagnosis = result["prediction_output"]["confidence_diagnosis"]
    nearest_case = result["prediction_output"]["nearest_casebook_case"]

    assert confidence_diagnosis["typical_correct_confidence"] == 0.79
    assert nearest_case["row_index"] == 11
    assert nearest_case["case_type"] == "representative"


def test_training_diagnostics_node_wraps_machine_readable_confidence_stats(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "generate_training_diagnostics",
        lambda **kwargs: {
            "per_class_analysis": {},
            "capacity_analysis": {},
            "confidence_analysis": "Narrative confidence summary",
            "hypothesis_validation": {"tested": [], "supported": []},
            "new_hypotheses": {"tested_predictions": [], "supported_conjectures": [], "exploratory_leads": []},
        },
    )

    state = SimpleNamespace(
        trained_models={"logistic_regression": _ColumnCheckingModel(["lin_a", "lin_b"], predicted_class=0)},
        test_frame=pd.DataFrame({"tree_x": [9.0, 8.0]}),
        test_views={"linear_view": pd.DataFrame({"lin_a": [0.5, 0.7], "lin_b": [1.5, 1.7]})},
        model_view_map={"logistic_regression": "linear_view"},
        test_target=pd.Series([0, 0]),
        candidate_model_specs={},
        evaluation_results={"logistic_regression": {"macro_f1": 0.7}},
        learning_curves=None,
        eda_hypotheses=None,
        feature_engineering_hypothesis=None,
        class_names=["Good", "Poor", "Standard"],
    )

    result = training_diagnostics_node(state)
    confidence_analysis = result["training_diagnostics"]["confidence_analysis"]

    assert confidence_analysis["summary"] == "Narrative confidence summary"
    assert confidence_analysis["by_model"]["logistic_regression"]["correct_mean_confidence"] == 0.8
    assert confidence_analysis["by_model"]["logistic_regression"]["per_class_correct_mean_confidence"]["Good"] == 0.8


def test_package_analysis_bundle_uses_run_id_for_bundle_filename(tmp_path):
    dataset_path = tmp_path / "train.csv"
    dataset_path.write_text("x\n1\n", encoding="utf-8")

    state = SimpleNamespace(
        raw_dataset_path=str(dataset_path),
        run_id="20260415_131136",
        selected_model_name="xgboost",
        class_names=["Good", "Poor", "Standard"],
        feature_columns=["tree_x"],
        feature_columns_by_view={"tree_view": ["tree_x"]},
        model_view_map={"xgboost": "tree_view"},
        eda_hypotheses={},
        training_diagnostics={},
        global_xai_interpretation={},
        local_xai_interpretation={},
        local_xai_cases=[],
        feature_engineering_hypothesis={},
        selection_justification="synthetic",
        global_xai_results={"methods_used": ["shap"], "shap": {"importance": [], "dependence_data": {}}},
    )

    result = package_analysis_bundle_node(state)

    expected_path = tmp_path / "lab" / "logs" / "analysis_bundle_20260415_131136.json"
    assert expected_path.is_file()
    assert result["analysis_bundle"]["metadata"]["run_id"] == "20260415_131136"
