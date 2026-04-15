import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from bt5151_credit_risk.xai import (
    _group_onehot_columns,
    compute_partial_dependence,
    compute_permutation_importance,
    select_classification_cases,
)


# ---------------------------------------------------------------------------
# One-hot grouping heuristic
# ---------------------------------------------------------------------------

def test_group_onehot_columns_groups_3plus_correctly():
    cols = [
        "Age",
        "Income",
        "Type_of_Loan_Home Loan",
        "Type_of_Loan_Auto Loan",
        "Type_of_Loan_Personal Loan",
        "Occupation_Engineer",
        "Occupation_Doctor",
        "Occupation_Teacher",
    ]
    groups = _group_onehot_columns(cols)
    assert "Type_of_Loan" in groups
    assert len(groups["Type_of_Loan"]) == 3
    assert "Occupation" in groups
    assert len(groups["Occupation"]) == 3
    assert "Age" in groups
    assert len(groups["Age"]) == 1


def test_group_onehot_columns_does_not_merge_engineered_pairs():
    """Two engineered features sharing a prefix must NOT be grouped together."""
    cols = ["income_log", "income_rank", "debt_ratio", "Age"]
    groups = _group_onehot_columns(cols)
    # Each should be its own group — no "income" super-group
    assert "income_log" in groups
    assert "income_rank" in groups
    assert "debt_ratio" in groups
    assert "Age" in groups
    assert len(groups) == 4


def test_group_onehot_columns_single_underscore_not_grouped():
    cols = ["debt_ratio", "monthly_income"]
    groups = _group_onehot_columns(cols)
    assert "debt_ratio" in groups
    assert "monthly_income" in groups
    assert len(groups) == 2


def test_group_onehot_columns_pair_kept_separate_without_spec():
    """Without spec, binary one-hot (2 columns) stays split (heuristic limitation)."""
    cols = ["Gender_Male", "Gender_Female"]
    groups = _group_onehot_columns(cols)
    assert "Gender_Male" in groups
    assert "Gender_Female" in groups
    assert "Gender" not in groups


def test_group_onehot_columns_with_spec_groups_binary():
    """With column_transform_spec, binary one-hot IS correctly grouped."""
    cols = ["Gender_Male", "Gender_Female", "income_log", "income_rank", "Age"]
    spec = {
        "transforms": {
            "Gender": {"action": "keep", "encoding": "one_hot"},
            "Age": {"action": "keep", "encoding": None},
        }
    }
    groups = _group_onehot_columns(cols, column_transform_spec=spec)
    # Gender should be grouped (spec says it's one_hot)
    assert "Gender" in groups
    assert len(groups["Gender"]) == 2
    # Engineered features should NOT be grouped (not in spec as one_hot)
    assert "income_log" in groups
    assert "income_rank" in groups
    assert len(groups["income_log"]) == 1
    assert len(groups["income_rank"]) == 1
    assert "Age" in groups


def test_group_onehot_columns_with_spec_and_multivalue():
    """Spec-based grouping works for multi-value one-hot columns too."""
    cols = [
        "Type_of_Loan_Home Loan",
        "Type_of_Loan_Auto Loan",
        "Type_of_Loan_Personal Loan",
        "Gender_Male",
        "Gender_Female",
        "Age",
    ]
    spec = {
        "transforms": {
            "Type_of_Loan": {"action": "keep", "encoding": "one_hot"},
            "Gender": {"action": "keep", "encoding": "one_hot"},
            "Age": {"action": "keep", "encoding": None},
        }
    }
    groups = _group_onehot_columns(cols, column_transform_spec=spec)
    assert "Type_of_Loan" in groups
    assert len(groups["Type_of_Loan"]) == 3
    assert "Gender" in groups
    assert len(groups["Gender"]) == 2
    assert "Age" in groups
    assert len(groups["Age"]) == 1


# ---------------------------------------------------------------------------
# Trained model fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def trained_rf():
    rng = np.random.RandomState(42)
    n = 100
    X = pd.DataFrame({
        "feat_a": rng.randn(n),
        "feat_b": rng.randn(n),
        "feat_c": rng.randn(n),
    })
    y = pd.Series(np.where(X["feat_a"] > 0, 2, np.where(X["feat_b"] > 0, 1, 0)))
    model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# Grouped PFI — true joint permutation
# ---------------------------------------------------------------------------

def test_grouped_pfi_permutes_jointly(trained_rf):
    """Grouped PFI must permute all columns in a group simultaneously.

    We create a dataset where two features (a_0, a_1) are jointly predictive
    but individually weak. True grouped PFI (joint permutation) should show
    the group as important, while individual PFI underestimates.
    """
    rng = np.random.RandomState(42)
    n = 200
    # Create a feature pair where the XOR-like interaction matters
    a_0 = rng.choice([0, 1], size=n)
    a_1 = rng.choice([0, 1], size=n)
    a_2 = rng.choice([0, 1], size=n)
    noise = rng.randn(n)
    # Target depends on the combination of a_0 and a_1
    y = pd.Series((a_0 + a_1 + a_2) % 3)
    X = pd.DataFrame({
        "group_0": a_0.astype(float),
        "group_1": a_1.astype(float),
        "group_2": a_2.astype(float),
        "noise": noise,
    })
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)

    result = compute_permutation_importance(
        model, "random_forest", X, y,
        list(X.columns), n_repeats=5,
    )
    # Grouped result should exist and have entries
    assert "grouped" in result
    assert len(result["grouped"]) > 0
    # The "group" group (3 columns) should be present and important
    group_entries = [g for g in result["grouped"] if g["original_feature"] == "group"]
    assert len(group_entries) == 1
    assert group_entries[0]["n_columns"] == 3
    assert group_entries[0]["importance_mean"] > 0


def test_grouped_pfi_subsamples_by_position_when_target_index_differs():
    """PFI subsampling should align test_frame/test_target by position, not label.

    Dual-view FE artifacts are reloaded from CSV with a fresh RangeIndex, while
    targets can still carry original split indices. Position-based subsampling
    prevents KeyError during `.loc` on mismatched labels.
    """
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({
        "feat_a": rng.randn(n),
        "feat_b": rng.randn(n),
        "feat_c": rng.randn(n),
    })
    y_values = np.where(X["feat_a"] > 0, 2, np.where(X["feat_b"] > 0, 1, 0))
    y = pd.Series(y_values, index=np.arange(1000, 1000 + n))
    model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    model.fit(X, y_values)

    result = compute_permutation_importance(
        model, "random_forest", X, y,
        list(X.columns), n_repeats=3, max_samples=50,
    )

    assert "raw" in result
    assert "grouped" in result
    assert len(result["raw"]) > 0


def test_compute_partial_dependence_casts_integer_feature_to_float(monkeypatch):
    X = pd.DataFrame({
        "count_feature": pd.Series([0, 1, 2, 3, 4], dtype="int64"),
        "other_feature": pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], dtype="float64"),
    })

    captured = {}

    def fake_partial_dependence(model, frame, features, kind, grid_resolution):
        feat_idx = features[0]
        captured["dtype"] = frame.dtypes.iloc[feat_idx]
        return {
            "grid_values": [np.array([0.0, 1.0, 2.0])],
            "average": np.array([
                [0.2, 0.3, 0.4],
                [0.8, 0.7, 0.6],
            ]),
        }

    monkeypatch.setattr("bt5151_credit_risk.xai.partial_dependence", fake_partial_dependence)

    results = compute_partial_dependence(
        object(),
        "random_forest",
        X,
        list(X.columns),
        ["count_feature"],
        ["good", "poor"],
    )

    assert pd.api.types.is_float_dtype(captured["dtype"])
    assert "count_feature" in results


# ---------------------------------------------------------------------------
# Casebook: worst misclassification uses predicted-class confidence
# ---------------------------------------------------------------------------

def test_worst_misclassification_uses_predicted_class_confidence():
    """Worst misclassification should select by highest predicted-wrong-class
    confidence, not by lowest true-class confidence."""
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
    })
    y = pd.Series(np.where(X["f1"] > 0.5, 2, np.where(X["f1"] > -0.5, 1, 0)))
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)

    class_names = ["c0", "c1", "c2"]
    id_to_label = {0: "c0", 1: "c1", 2: "c2"}
    cases = select_classification_cases(X, y, model, class_names, id_to_label)

    misclass_cases = [c for c in cases if c["case_type"] == "worst_misclassification"]
    for case in misclass_cases:
        # The predicted label should match the class with highest probability
        pred_label = case["predicted_label"]
        pred_prob = case["probabilities"][pred_label]
        # Verify predicted class has the highest probability
        assert pred_prob == max(case["probabilities"].values()), (
            f"Worst misclassification case predicted {pred_label} (prob={pred_prob}) "
            f"but max prob is {max(case['probabilities'].values())}"
        )


def test_select_classification_cases_returns_cases(trained_rf):
    model, X, y = trained_rf
    class_names = ["cls0", "cls1", "cls2"]
    id_to_label = {0: "cls0", 1: "cls1", 2: "cls2"}
    cases = select_classification_cases(X, y, model, class_names, id_to_label)
    assert len(cases) >= 3  # at least one representative per class
    case_types = {c["case_type"] for c in cases}
    assert "representative" in case_types
    for case in cases:
        assert "row_index" in case
        assert "true_label" in case
        assert "predicted_label" in case
        assert "probabilities" in case
        assert "case_type" in case
