import json
from pathlib import Path

import pandas as pd
from langgraph.graph import END, START, StateGraph

from bt5151_credit_risk.business import explain_risk, recommend_action
from bt5151_credit_risk.evaluate import choose_best_model, compute_multiclass_metrics
from bt5151_credit_risk.preprocess import cleanup_old_workspaces
from bt5151_credit_risk.preprocess import execute_generated_preprocessing
from bt5151_credit_risk.preprocess import generate_column_transform_spec
from bt5151_credit_risk.preprocess import generate_dataset_policy_spec
from bt5151_credit_risk.preprocess import generate_preprocessing_code
from bt5151_credit_risk.preprocess import inspect_preprocessing_code
from bt5151_credit_risk.preprocess import repair_preprocessing_code
from bt5151_credit_risk.preprocess import validate_preprocessing_output
from bt5151_credit_risk.profile import build_dataset_profile
from bt5151_credit_risk.state import CreditRiskState
from bt5151_credit_risk.train import build_candidate_models


MAX_REPAIR_ATTEMPTS = 3


# Load the raw dataset once and get the high-level preprocessing policy.
def dataset_policy_spec_node(state: CreditRiskState):
    raw_frame = pd.read_csv(state.raw_dataset_path, low_memory=False)
    dataset_profile = build_dataset_profile(raw_frame)
    dataset_policy_spec = generate_dataset_policy_spec(raw_frame, dataset_profile)

    return {
        "raw_frame": raw_frame,
        "dataset_profile": dataset_profile,
        "dataset_policy_spec": dataset_policy_spec,
    }


# Ask for the detailed column-level transformation rules.
def column_transform_spec_node(state: CreditRiskState):
    column_transform_spec = generate_column_transform_spec(
        state.raw_frame,
        state.dataset_policy_spec,
    )
    return {"column_transform_spec": column_transform_spec}


# Generate preprocessing code from the two specs.
def generate_preprocessing_code_node(state: CreditRiskState):
    generated_code = generate_preprocessing_code(
        state.raw_frame,
        state.dataset_profile,
        state.dataset_policy_spec,
        state.column_transform_spec,
    )
    return {
        "preprocessing_code": generated_code,
        "preprocessing_codegen_metadata": {
            "entrypoint": generated_code.get("entrypoint"),
        },
        "preprocessing_attempt_count": 1,
    }


# Inspect generated code before we try to run it.
def inspect_preprocessing_code_node(state: CreditRiskState):
    code_review = inspect_preprocessing_code(state.preprocessing_code or {})
    return {
        "preprocessing_code_review": code_review,
    }


# Execute the generated preprocessing code and save its artifact metadata.
def execute_generated_preprocessing_node(state: CreditRiskState):
    run_root = Path(state.raw_dataset_path).resolve().parent / "generated_preprocessing_runs"
    execution_result = execute_generated_preprocessing(
        state.raw_frame,
        state.preprocessing_code,
        run_root,
    )
    return {
        "preprocessing_workspace": execution_result["workspace_path"],
        "preprocessing_raw_frame_path": execution_result["raw_frame_path"],
        "preprocessing_artifacts": execution_result["artifacts"],
        "preprocessing_execution_log": execution_result["execution_log"],
        "preprocessing_execution_report": execution_result,
    }


# Validate artifacts, then rebuild train/test data for deterministic modeling.
def validate_preprocessing_output_node(state: CreditRiskState):
    execution_result = {
        "workspace_path": state.preprocessing_workspace,
        "artifacts": state.preprocessing_artifacts,
        "execution_log": state.preprocessing_execution_log or {},
        "raw_frame_path": state.preprocessing_raw_frame_path,
    }
    validation_report = validate_preprocessing_output(
        execution_result,
        state.dataset_policy_spec,
        state.column_transform_spec,
    )
    if not validation_report["passed"]:
        return {"preprocessing_validation_report": validation_report}

    artifacts = state.preprocessing_artifacts or {}
    feature_frame = pd.read_csv(artifacts["feature_frame.csv"])
    target_frame = pd.read_csv(artifacts["target.csv"])
    target_series = target_frame.iloc[:, 0]

    # Sort labels so the encoded class mapping stays stable across runs.
    class_names = sorted(pd.Series(target_series.dropna()).astype(str).unique())
    label_to_id = {label: idx for idx, label in enumerate(class_names)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    encoded_target = target_series.astype(str).map(label_to_id)
    split_manifest = json.loads(Path(artifacts["split_manifest.json"]).read_text(encoding="utf-8"))
    train_indices = split_manifest["train_indices"]
    test_indices = split_manifest["test_indices"]

    # Clean up older run folders only after we have rebuilt everything we still need from this run.
    run_root = Path(state.preprocessing_workspace).parent
    cleanup_old_workspaces(run_root, keep_latest=1)

    return {
        "preprocessing_validation_report": validation_report,
        "full_feature_frame": feature_frame,
        "train_frame": feature_frame.iloc[train_indices],
        "test_frame": feature_frame.iloc[test_indices],
        "train_target": encoded_target.iloc[train_indices],
        "test_target": encoded_target.iloc[test_indices],
        "feature_columns": feature_frame.columns.tolist(),
        "class_names": class_names,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
    }


# Ask the LLM to repair preprocessing code after a failed attempt.
def repair_preprocessing_code_node(state: CreditRiskState):
    attempt_count = state.preprocessing_attempt_count or 0
    # attempt_count includes the initial generation, so 3 means initial try plus up to 2 repairs.
    if attempt_count >= MAX_REPAIR_ATTEMPTS:
        raise RuntimeError(
            f"Preprocessing repair failed after {attempt_count} attempts."
        )

    repaired_code = repair_preprocessing_code(
        previous_generated_code=state.preprocessing_code,
        code_review=state.preprocessing_code_review or {},
        execution_log=state.preprocessing_execution_log or {},
        validation_report=state.preprocessing_validation_report or {},
        dataset_profile=state.dataset_profile,
        dataset_policy_spec=state.dataset_policy_spec,
        column_transform_spec=state.column_transform_spec,
    )
    return {
        "preprocessing_code": repaired_code,
        "preprocessing_codegen_metadata": {
            "entrypoint": repaired_code.get("entrypoint"),
        },
        "preprocessing_attempt_count": attempt_count + 1,
    }


# Fit the candidate sklearn models on the generated training data.
def train_models_node(state: CreditRiskState):
    models = build_candidate_models()
    trained_models = {}
    candidate_model_specs = {}
    for model_name, model in models.items():
        trained_models[model_name] = model.fit(state.train_frame, state.train_target)
        candidate_model_specs[model_name] = {
            "estimator": type(model).__name__,
        }
    return {
        "candidate_model_specs": candidate_model_specs,
        "trained_models": trained_models,
    }


# Score every candidate model on the held-out test split.
def evaluate_models_node(state: CreditRiskState):
    results = {}
    for model_name, model in state.trained_models.items():
        predictions = model.predict(state.test_frame)
        results[model_name] = compute_multiclass_metrics(
            state.test_target,
            predictions,
            state.class_names,
        )
    return {"evaluation_results": results}


# Choose the final model from the evaluation results.
def select_model_node(state: CreditRiskState):
    selection = choose_best_model(state.evaluation_results)
    return {
        "selected_model_name": selection["model_name"],
        "selection_justification": selection["justification"],
    }


# Run one inference example with the selected model.
def run_inference_node(state: CreditRiskState):
    if not state.inference_input or "row_index" not in state.inference_input:
        raise ValueError("inference_input with 'row_index' is required.")

    row_index = int(state.inference_input["row_index"])
    input_frame = state.full_feature_frame.loc[[row_index]]
    model = state.trained_models[state.selected_model_name]
    probabilities = model.predict_proba(input_frame)[0]
    predicted_code = int(model.predict(input_frame)[0])
    probability_map = {
        state.id_to_label[idx]: float(score)
        for idx, score in enumerate(probabilities)
    }
    return {
        "prediction_output": {
            "row_index": row_index,
            "predicted_label": state.id_to_label[predicted_code],
            "probabilities": probability_map,
            "confidence": max(probability_map.values()),
            "selected_model_name": state.selected_model_name,
            "evaluation_metrics": state.evaluation_results[state.selected_model_name],
            "source_record": state.raw_frame.loc[row_index].to_dict(),
        }
    }


# Convert the prediction into business-friendly language.
def explain_risk_node(state: CreditRiskState):
    prediction_output = state.prediction_output
    explanation = explain_risk(
        prediction_output["predicted_label"],
        prediction_output["probabilities"],
        selected_model_name=prediction_output["selected_model_name"],
        evaluation_metrics=prediction_output["evaluation_metrics"],
        source_record=prediction_output["source_record"],
    )
    return {"risk_explanation": explanation}


# Convert the explanation into an action recommendation.
def recommend_action_node(state: CreditRiskState):
    action = recommend_action(state.risk_explanation, prediction_output=state.prediction_output)
    return {"recommended_action": action}


# Good code goes to execution; failed inspection goes to repair.
def _route_after_inspection(state: CreditRiskState):
    if state.preprocessing_code_review and state.preprocessing_code_review.get("passed"):
        return "execute-generated-preprocessing"
    return "repair-preprocessing-code"


# Valid artifacts go to training; failed validation goes to repair.
def _route_after_validation(state: CreditRiskState):
    if state.preprocessing_validation_report and state.preprocessing_validation_report.get("passed"):
        return "train-models"
    return "repair-preprocessing-code"


# Define the end-to-end LangGraph pipeline.
def build_graph():
    graph = StateGraph(CreditRiskState)
    graph.add_node("dataset-policy-spec", dataset_policy_spec_node)
    graph.add_node("column-transform-spec", column_transform_spec_node)
    graph.add_node("generate-preprocessing-code", generate_preprocessing_code_node)
    graph.add_node("inspect-preprocessing-code", inspect_preprocessing_code_node)
    graph.add_node("execute-generated-preprocessing", execute_generated_preprocessing_node)
    graph.add_node("validate-preprocessing-output", validate_preprocessing_output_node)
    graph.add_node("repair-preprocessing-code", repair_preprocessing_code_node)
    graph.add_node("train-models", train_models_node)
    graph.add_node("evaluate-models", evaluate_models_node)
    graph.add_node("select-model", select_model_node)
    graph.add_node("run-inference", run_inference_node)
    graph.add_node("explain-risk", explain_risk_node)
    graph.add_node("recommend-action", recommend_action_node)

    graph.add_edge(START, "dataset-policy-spec")
    graph.add_edge("dataset-policy-spec", "column-transform-spec")
    graph.add_edge("column-transform-spec", "generate-preprocessing-code")
    graph.add_edge("generate-preprocessing-code", "inspect-preprocessing-code")
    graph.add_conditional_edges(
        "inspect-preprocessing-code",
        _route_after_inspection,
        {
            "execute-generated-preprocessing": "execute-generated-preprocessing",
            "repair-preprocessing-code": "repair-preprocessing-code",
        },
    )
    graph.add_edge("execute-generated-preprocessing", "validate-preprocessing-output")
    graph.add_conditional_edges(
        "validate-preprocessing-output",
        _route_after_validation,
        {
            "train-models": "train-models",
            "repair-preprocessing-code": "repair-preprocessing-code",
        },
    )
    graph.add_edge("repair-preprocessing-code", "inspect-preprocessing-code")
    graph.add_edge("train-models", "evaluate-models")
    graph.add_edge("evaluate-models", "select-model")
    graph.add_edge("select-model", "run-inference")
    graph.add_edge("run-inference", "explain-risk")
    graph.add_edge("explain-risk", "recommend-action")
    graph.add_edge("recommend-action", END)
    return graph


# Compile the graph into a runnable pipeline object.
def compile_graph():
    return build_graph().compile()
