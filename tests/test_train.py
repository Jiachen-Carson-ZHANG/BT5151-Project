import numpy as np
import pandas as pd

from bt5151_credit_risk.train import _build_holdout_indices
from bt5151_credit_risk.train import _normalize_validation_policy
from bt5151_credit_risk.train import _prepare_policy_aligned_data
from bt5151_credit_risk.train import build_candidate_models


def test_candidate_models_include_three_candidates():
    models = build_candidate_models()
    assert set(models) == {"logistic_regression", "random_forest", "xgboost"}


def test_grouped_validation_holdout_keeps_groups_disjoint():
    target = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    groups = ["A", "A", "B", "B", "C", "C", "D", "D"]

    policy = _normalize_validation_policy({"type": "grouped_entity"}, group_values=groups)
    train_idx, val_idx = _build_holdout_indices(target, policy, group_values=groups, test_size=0.25)

    train_groups = set(np.asarray(groups)[train_idx])
    val_groups = set(np.asarray(groups)[val_idx])
    assert train_groups.isdisjoint(val_groups)


def test_temporal_policy_sorts_rows_before_validation_split():
    frame = pd.DataFrame({"feature": [30, 10, 40, 20]})
    target = pd.Series([0, 1, 0, 1], index=[10, 11, 12, 13])
    time_values = ["2024-03-01", "2024-01-01", "2024-04-01", "2024-02-01"]

    aligned_frame, aligned_target, _, _, aligned_times, policy = _prepare_policy_aligned_data(
        frame,
        target,
        time_values=time_values,
        validation_policy={"type": "temporal", "time_column": "event_time"},
    )

    assert list(aligned_frame["feature"]) == [10, 20, 30, 40]
    assert list(aligned_target) == [1, 1, 0, 0]
    assert list(aligned_times) == ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]

    train_idx, val_idx = _build_holdout_indices(aligned_target, policy, test_size=0.25)
    assert list(train_idx) == [0, 1, 2]
    assert list(val_idx) == [3]
