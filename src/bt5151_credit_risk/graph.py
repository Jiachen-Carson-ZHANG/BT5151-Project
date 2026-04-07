import pandas as pd
from langgraph.graph import END, START, StateGraph
from sklearn.model_selection import GroupShuffleSplit

from bt5151_credit_risk.business import explain_risk, recommend_action
from bt5151_credit_risk.config import GROUP_COLUMN, RANDOM_SEED, TEST_SIZE
from bt5151_credit_risk.evaluate import choose_best_model, compute_multiclass_metrics
from bt5151_credit_risk.preprocess import preprocess_credit_data
from bt5151_credit_risk.profile import build_dataset_profile
from bt5151_credit_risk.state import CreditRiskState
from bt5151_credit_risk.train import build_candidate_models

LABEL_TO_ID = {"Poor": 0, "Standard": 1, "Good": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
CLASS_NAMES = ["Poor", "Standard", "Good"]


def preprocess_data_node(state: CreditRiskState):
    raw_frame = pd.read_csv(state.raw_dataset_path, low_memory=False)
    dataset_profile = build_dataset_profile(raw_frame)
    preprocess_result = preprocess_credit_data(raw_frame)
    feature_frame = preprocess_result.feature_frame.select_dtypes(include=["number"]).fillna(0)
    target = preprocess_result.target.map(LABEL_TO_ID)

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(feature_frame, target, preprocess_result.groups))

    return {
        "raw_frame": raw_frame,
        "dataset_profile": dataset_profile,
        "preprocessing_rules": {
            "numeric_only": True,
            "fillna": 0,
            "target_map": LABEL_TO_ID,
        },
        "feature_columns": feature_frame.columns.tolist(),
        "full_feature_frame": feature_frame,
        "train_frame": feature_frame.iloc[train_idx],
        "test_frame": feature_frame.iloc[test_idx],
        "train_target": target.iloc[train_idx],
        "test_target": target.iloc[test_idx],
    }


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
            CLASS_NAMES,
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
        ID_TO_LABEL[idx]: float(score)
        for idx, score in enumerate(probabilities)
    }
    return {
        "prediction_output": {
            "row_index": row_index,
            "predicted_label": ID_TO_LABEL[predicted_code],
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
    graph.add_node("preprocess-data", preprocess_data_node)
    graph.add_node("train-models", train_models_node)
    graph.add_node("evaluate-models", evaluate_models_node)
    graph.add_node("select-model", select_model_node)
    graph.add_node("run-inference", run_inference_node)
    graph.add_node("explain-risk", explain_risk_node)
    graph.add_node("recommend-action", recommend_action_node)

    graph.add_edge(START, "preprocess-data")
    graph.add_edge("preprocess-data", "train-models")
    graph.add_edge("train-models", "evaluate-models")
    graph.add_edge("evaluate-models", "select-model")
    graph.add_edge("select-model", "run-inference")
    graph.add_edge("run-inference", "explain-risk")
    graph.add_edge("explain-risk", "recommend-action")
    graph.add_edge("recommend-action", END)
    return graph


def compile_graph():
    return build_graph().compile()
