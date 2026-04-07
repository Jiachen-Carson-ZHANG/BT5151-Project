import pandas as pd
from langgraph.graph import END, START, StateGraph

from bt5151_credit_risk.business import explain_risk, recommend_action
from bt5151_credit_risk.evaluate import choose_best_model, compute_multiclass_metrics
from bt5151_credit_risk.preprocess import audit_preprocessing_output
from bt5151_credit_risk.preprocess import execute_preprocessing
from bt5151_credit_risk.preprocess import generate_column_transform_spec
from bt5151_credit_risk.preprocess import generate_dataset_policy_spec
from bt5151_credit_risk.profile import build_dataset_profile
from bt5151_credit_risk.state import CreditRiskState
from bt5151_credit_risk.train import build_candidate_models


def dataset_policy_spec_node(state: CreditRiskState):
    raw_frame = pd.read_csv(state.raw_dataset_path, low_memory=False)
    dataset_profile = build_dataset_profile(raw_frame)
    dataset_policy_spec = generate_dataset_policy_spec(raw_frame, dataset_profile)

    return {
        "raw_frame": raw_frame,
        "dataset_profile": dataset_profile,
        "dataset_policy_spec": dataset_policy_spec,
    }


def column_transform_spec_node(state: CreditRiskState):
    column_transform_spec = generate_column_transform_spec(
        state.raw_frame,
        state.dataset_policy_spec,
    )
    return {"column_transform_spec": column_transform_spec}


def execute_preprocessing_node(state: CreditRiskState):
    preprocess_result = execute_preprocessing(
        state.raw_frame,
        state.dataset_policy_spec,
        state.column_transform_spec,
    )
    class_names = list(pd.Series(preprocess_result.target.dropna()).astype(str).unique())
    label_to_id = {label: idx for idx, label in enumerate(class_names)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    encoded_target = preprocess_result.target.map(label_to_id)
    feature_frame = preprocess_result.feature_frame

    return {
        "preprocessing_rules": {
            "dataset_policy_spec": state.dataset_policy_spec,
            "column_transform_spec": state.column_transform_spec,
        },
        "preprocessing_execution_report": preprocess_result.execution_report,
        "feature_columns": feature_frame.columns.tolist(),
        "class_names": class_names,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "full_feature_frame": feature_frame,
        "train_frame": feature_frame.iloc[preprocess_result.train_indices],
        "test_frame": feature_frame.iloc[preprocess_result.test_indices],
        "train_target": encoded_target.iloc[preprocess_result.train_indices],
        "test_target": encoded_target.iloc[preprocess_result.test_indices],
    }


def audit_preprocessing_node(state: CreditRiskState):
    preprocess_result = execute_preprocessing(
        state.raw_frame,
        state.dataset_policy_spec,
        state.column_transform_spec,
    )
    audit_report = audit_preprocessing_output(
        state.raw_frame,
        preprocess_result,
        state.dataset_policy_spec,
        state.column_transform_spec,
    )
    if not audit_report["passed"]:
        raise ValueError(f"Preprocessing audit failed: {audit_report}")
    return {"preprocessing_audit_report": audit_report}


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


def select_model_node(state: CreditRiskState):
    selection = choose_best_model(state.evaluation_results)
    return {
        "selected_model_name": selection["model_name"],
        "selection_justification": selection["justification"],
    }


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


def recommend_action_node(state: CreditRiskState):
    action = recommend_action(state.risk_explanation, prediction_output=state.prediction_output)
    return {"recommended_action": action}


def build_graph():
    graph = StateGraph(CreditRiskState)
    graph.add_node("dataset-policy-spec", dataset_policy_spec_node)
    graph.add_node("column-transform-spec", column_transform_spec_node)
    graph.add_node("execute-preprocessing", execute_preprocessing_node)
    graph.add_node("audit-preprocessing", audit_preprocessing_node)
    graph.add_node("train-models", train_models_node)
    graph.add_node("evaluate-models", evaluate_models_node)
    graph.add_node("select-model", select_model_node)
    graph.add_node("run-inference", run_inference_node)
    graph.add_node("explain-risk", explain_risk_node)
    graph.add_node("recommend-action", recommend_action_node)

    graph.add_edge(START, "dataset-policy-spec")
    graph.add_edge("dataset-policy-spec", "column-transform-spec")
    graph.add_edge("column-transform-spec", "execute-preprocessing")
    graph.add_edge("execute-preprocessing", "audit-preprocessing")
    graph.add_edge("audit-preprocessing", "train-models")
    graph.add_edge("train-models", "evaluate-models")
    graph.add_edge("evaluate-models", "select-model")
    graph.add_edge("select-model", "run-inference")
    graph.add_edge("run-inference", "explain-risk")
    graph.add_edge("explain-risk", "recommend-action")
    graph.add_edge("recommend-action", END)
    return graph


def compile_graph():
    return build_graph().compile()
