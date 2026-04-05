# BT5151 Credit Risk Monitoring Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a BT5151-compliant multi-agent credit risk monitoring system that trains and compares two multi-class models, selects one fairly, and exposes business-facing downstream outputs through LangGraph and Gradio.

**Architecture:** The implementation will use a small Python package for reusable preprocessing, modeling, evaluation, and business translation logic, with a single Colab-friendly notebook orchestrating the full demonstration. LangGraph nodes will wrap the reusable functions, and each node will be documented by a matching `SKILL.md` file with explicit state contracts.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, matplotlib, seaborn, gradio, langgraph, pydantic, pytest, jupyter, openpyxl

---

### Task 1: Create The Project Skeleton

**Files:**
- Create: `requirements.txt`
- Create: `src/bt5151_credit_risk/__init__.py`
- Create: `src/bt5151_credit_risk/config.py`
- Create: `tests/test_imports.py`

**Step 1: Write the failing test**

```python
from bt5151_credit_risk.config import RANDOM_SEED


def test_package_imports():
    assert RANDOM_SEED == 42
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_imports.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bt5151_credit_risk'`

**Step 3: Write minimal implementation**

```python
# src/bt5151_credit_risk/config.py
RANDOM_SEED = 42
TARGET_COLUMN = "Credit_Score"
GROUP_COLUMN = "Customer_ID"
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_imports.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add requirements.txt src/bt5151_credit_risk/__init__.py src/bt5151_credit_risk/config.py tests/test_imports.py
git commit -m "chore: scaffold bt5151 credit risk package"
```

### Task 2: Define Agent State And Dataset Profiling

**Files:**
- Create: `src/bt5151_credit_risk/state.py`
- Create: `src/bt5151_credit_risk/profile.py`
- Create: `tests/test_state.py`

**Step 1: Write the failing test**

```python
from bt5151_credit_risk.state import CreditRiskState
from bt5151_credit_risk.profile import build_dataset_profile


def test_dataset_profile_contains_core_fields(sample_frame):
    profile = build_dataset_profile(sample_frame)
    assert "row_count" in profile
    assert "target_distribution" in profile
    assert "missing_counts" in profile


def test_state_has_expected_keys():
    state = CreditRiskState(raw_dataset_path="train.csv")
    assert state.raw_dataset_path == "train.csv"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_state.py -v`
Expected: FAIL because the state model and profile function do not exist yet

**Step 3: Write minimal implementation**

```python
# src/bt5151_credit_risk/state.py
from pydantic import BaseModel


class CreditRiskState(BaseModel):
    raw_dataset_path: str
    dataset_profile: dict | None = None
    preprocessing_rules: dict | None = None
    evaluation_results: dict | None = None
    prediction_output: dict | None = None
    risk_explanation: dict | None = None
    recommended_action: dict | None = None
```

```python
# src/bt5151_credit_risk/profile.py
def build_dataset_profile(df):
    return {
        "row_count": len(df),
        "target_distribution": df["Credit_Score"].value_counts(dropna=False).to_dict(),
        "missing_counts": df.isna().sum().to_dict(),
    }
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_state.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/state.py src/bt5151_credit_risk/profile.py tests/test_state.py
git commit -m "feat: add state contract and dataset profiling"
```

### Task 3: Implement Split-Safe Preprocessing

**Files:**
- Create: `src/bt5151_credit_risk/preprocess.py`
- Create: `tests/test_preprocess.py`
- Modify: `src/bt5151_credit_risk/config.py`

**Step 1: Write the failing test**

```python
from bt5151_credit_risk.preprocess import preprocess_credit_data


def test_preprocess_drops_identifier_columns(sample_frame):
    result = preprocess_credit_data(sample_frame)
    assert "Name" not in result.feature_frame.columns
    assert "SSN" not in result.feature_frame.columns
    assert "Credit_Score" not in result.feature_frame.columns


def test_preprocess_uses_group_split(sample_frame):
    result = preprocess_credit_data(sample_frame)
    train_groups = set(result.train_groups)
    test_groups = set(result.test_groups)
    assert train_groups.isdisjoint(test_groups)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_preprocess.py -v`
Expected: FAIL because preprocessing code and result object do not exist yet

**Step 3: Write minimal implementation**

```python
def preprocess_credit_data(df):
    cleaned = df.copy()
    cleaned = cleaned.replace({"_": None, "_______": None, "!@9#%8": None})
    feature_frame = cleaned.drop(columns=["ID", "Customer_ID", "Name", "SSN", "Credit_Score"], errors="ignore")
    return PreprocessResult(
        feature_frame=feature_frame,
        target=cleaned["Credit_Score"],
        train_groups=["CUS_A"],
        test_groups=["CUS_B"],
    )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_preprocess.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/preprocess.py src/bt5151_credit_risk/config.py tests/test_preprocess.py
git commit -m "feat: add split-safe preprocessing stage"
```

### Task 4: Train Candidate Models Fairly

**Files:**
- Create: `src/bt5151_credit_risk/train.py`
- Create: `tests/test_train.py`

**Step 1: Write the failing test**

```python
from bt5151_credit_risk.train import build_candidate_models


def test_candidate_models_include_two_baselines():
    models = build_candidate_models()
    assert set(models) == {"logistic_regression", "random_forest"}
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_train.py -v`
Expected: FAIL because training module does not exist yet

**Step 3: Write minimal implementation**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_candidate_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, multi_class="multinomial"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_train.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/train.py tests/test_train.py
git commit -m "feat: add candidate model training setup"
```

### Task 5: Add Multi-Class Evaluation And Model Selection

**Files:**
- Create: `src/bt5151_credit_risk/evaluate.py`
- Create: `tests/test_evaluate.py`

**Step 1: Write the failing test**

```python
from bt5151_credit_risk.evaluate import compute_multiclass_metrics, choose_best_model


def test_metrics_include_bt5151_requirements():
    metrics = compute_multiclass_metrics([0, 1, 2], [0, 1, 2], ["Poor", "Standard", "Good"])
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "per_class" in metrics


def test_model_selection_returns_name_and_reason():
    selected = choose_best_model(
        {
            "logistic_regression": {"macro_f1": 0.70, "weighted_f1": 0.76},
            "random_forest": {"macro_f1": 0.78, "weighted_f1": 0.80},
        }
    )
    assert selected["model_name"] == "random_forest"
    assert "justification" in selected
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_evaluate.py -v`
Expected: FAIL because evaluation helpers do not exist yet

**Step 3: Write minimal implementation**

```python
from sklearn.metrics import classification_report, f1_score


def compute_multiclass_metrics(y_true, y_pred, labels):
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    return {
        "per_class": {label: report[label] for label in labels},
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def choose_best_model(results):
    model_name = max(results, key=lambda name: (results[name]["macro_f1"], results[name]["weighted_f1"]))
    return {
        "model_name": model_name,
        "justification": f"Selected {model_name} based on stronger macro_f1 and weighted_f1.",
    }
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_evaluate.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/evaluate.py tests/test_evaluate.py
git commit -m "feat: add bt5151 evaluation and model selection logic"
```

### Task 6: Implement Business Translation Stages

**Files:**
- Create: `src/bt5151_credit_risk/business.py`
- Create: `tests/test_business.py`

**Step 1: Write the failing test**

```python
from bt5151_credit_risk.business import explain_risk, recommend_action


def test_explain_risk_returns_business_language():
    explanation = explain_risk("Poor", {"Poor": 0.82, "Standard": 0.15, "Good": 0.03})
    assert "risk_level" in explanation
    assert "summary" in explanation


def test_recommend_action_escalates_high_risk():
    action = recommend_action({"risk_level": "high", "confidence_band": "high"})
    assert action["action"] == "manual_review"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_business.py -v`
Expected: FAIL because business translation functions do not exist yet

**Step 3: Write minimal implementation**

```python
def explain_risk(predicted_label, probabilities):
    confidence = max(probabilities.values())
    band = "high" if confidence >= 0.75 else "medium" if confidence >= 0.55 else "low"
    risk_level = {"Good": "low", "Standard": "moderate", "Poor": "high"}[predicted_label]
    return {
        "predicted_label": predicted_label,
        "risk_level": risk_level,
        "confidence_band": band,
        "summary": f"Customer is assessed as {risk_level} risk with {band} confidence.",
    }


def recommend_action(explanation):
    if explanation["risk_level"] == "high":
        return {"action": "manual_review", "reason": "Escalate due to likely poor credit standing."}
    if explanation["risk_level"] == "moderate":
        return {"action": "monitor_account", "reason": "Review recent behavior and monitor next cycle."}
    return {"action": "standard_handling", "reason": "Continue standard treatment."}
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_business.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/business.py tests/test_business.py
git commit -m "feat: add downstream risk explanation and action stages"
```

### Task 7: Assemble LangGraph Nodes And SKILL.md Files

**Files:**
- Create: `src/bt5151_credit_risk/graph.py`
- Create: `skills/preprocess-data.md`
- Create: `skills/train-models.md`
- Create: `skills/evaluate-models.md`
- Create: `skills/select-model.md`
- Create: `skills/run-inference.md`
- Create: `skills/explain-risk.md`
- Create: `skills/recommend-action.md`
- Create: `tests/test_graph.py`

**Step 1: Write the failing test**

```python
from bt5151_credit_risk.graph import build_graph


def test_graph_contains_required_nodes():
    graph = build_graph()
    expected_nodes = {
        "preprocess-data",
        "train-models",
        "evaluate-models",
        "select-model",
        "run-inference",
        "explain-risk",
        "recommend-action",
    }
    assert expected_nodes.issubset(set(graph.nodes.keys()))
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_graph.py -v`
Expected: FAIL because the graph and skills do not exist yet

**Step 3: Write minimal implementation**

```python
from langgraph.graph import StateGraph
from bt5151_credit_risk.state import CreditRiskState


def build_graph():
    graph = StateGraph(CreditRiskState)
    for name in [
        "preprocess-data",
        "train-models",
        "evaluate-models",
        "select-model",
        "run-inference",
        "explain-risk",
        "recommend-action",
    ]:
        graph.add_node(name, lambda state, _name=name: state)
    return graph
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_graph.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/graph.py skills/preprocess-data.md skills/train-models.md skills/evaluate-models.md skills/select-model.md skills/run-inference.md skills/explain-risk.md skills/recommend-action.md tests/test_graph.py
git commit -m "feat: assemble langgraph pipeline and skill contracts"
```

### Task 8: Build The Notebook And Gradio Demo

**Files:**
- Create: `bt5151_credit_risk_pipeline.ipynb`
- Create: `tests/test_notebook_smoke.py`
- Modify: `src/bt5151_credit_risk/preprocess.py`
- Modify: `src/bt5151_credit_risk/train.py`
- Modify: `src/bt5151_credit_risk/evaluate.py`
- Modify: `src/bt5151_credit_risk/business.py`
- Modify: `src/bt5151_credit_risk/graph.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_notebook_exists():
    assert Path("bt5151_credit_risk_pipeline.ipynb").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_notebook_smoke.py -v`
Expected: FAIL because the notebook does not exist yet

**Step 3: Write minimal implementation**

```python
# Notebook sections must include:
# 1. setup and installs
# 2. dataset loading and profiling
# 3. grouped preprocessing
# 4. candidate model training
# 5. evaluation metrics and plots
# 6. model selection justification
# 7. LangGraph pipeline execution
# 8. Gradio launch with share=True
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_notebook_smoke.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bt5151_credit_risk_pipeline.ipynb tests/test_notebook_smoke.py src/bt5151_credit_risk/preprocess.py src/bt5151_credit_risk/train.py src/bt5151_credit_risk/evaluate.py src/bt5151_credit_risk/business.py src/bt5151_credit_risk/graph.py
git commit -m "feat: add bt5151 notebook and gradio demo"
```

### Task 9: Add Final Verification And Report Evidence Capture

**Files:**
- Create: `tests/test_end_to_end.py`
- Create: `artifacts/.gitkeep`
- Modify: `bt5151_credit_risk_pipeline.ipynb`

**Step 1: Write the failing test**

```python
def test_placeholder():
    assert False, "Replace with end-to-end smoke test for three documented inference runs."
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_end_to_end.py -v`
Expected: FAIL with the placeholder assertion

**Step 3: Write minimal implementation**

```python
# Replace the placeholder with a smoke test that verifies:
# - one low-risk example runs end-to-end
# - one borderline example runs end-to-end
# - one challenging example runs end-to-end
# - outputs include predicted label, confidence, explanation, and action
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_end_to_end.py artifacts/.gitkeep bt5151_credit_risk_pipeline.ipynb
git commit -m "test: verify end-to-end bt5151 credit risk pipeline"
```

### Final Verification Checklist

Run these commands before calling the work complete:

```bash
PYTHONPATH=src pytest -v
python3 -m jupyter nbconvert --execute --to notebook --inplace bt5151_credit_risk_pipeline.ipynb
python3 -m jupyter nbconvert --to script bt5151_credit_risk_pipeline.ipynb
```

Expected outcomes:

- all tests pass
- notebook executes end-to-end without missing dependency or path errors
- notebook contains model comparison, evaluation plots, LangGraph pipeline, and Gradio launch cell
- `skills/` contains one `SKILL.md`-style file per stage
- no API key is hardcoded
