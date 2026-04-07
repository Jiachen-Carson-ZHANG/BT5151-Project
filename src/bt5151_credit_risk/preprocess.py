from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from bt5151_credit_risk.config import GROUP_COLUMN, RANDOM_SEED, TARGET_COLUMN, TEST_SIZE
from bt5151_credit_risk.llm import call_json_response
from bt5151_credit_risk.skill_prompts import load_skill_prompt


@dataclass
class PreprocessResult:
    cleaned_frame: pd.DataFrame
    feature_frame: pd.DataFrame
    target: pd.Series
    groups: pd.Series
    train_indices: list[int]
    test_indices: list[int]
    train_groups: list[str]
    test_groups: list[str]
    execution_report: dict | None = None


PLACEHOLDER_VALUES = {"_": pd.NA, "_______": pd.NA, "!@9#%8": pd.NA}


def _call_preprocess_agent(system_prompt, payload):
    return call_json_response(system_prompt, payload)


def _call_preprocess_codegen_agent(system_prompt, payload):
    return call_json_response(system_prompt, payload)


def generate_dataset_policy_spec(df: pd.DataFrame, dataset_profile: dict) -> dict:
    system_prompt = (
        "You design preprocessing policy for labeled tabular machine learning. "
        "Return only valid JSON with keys: task_type, target_column, group_column, "
        "identifier_columns, split_strategy, leakage_rules, imbalance_strategy, feature_policy."
    )
    payload = {
        "columns": df.columns.tolist(),
        "sample_rows": df.head(5).to_dict(orient="records"),
        "dataset_profile": dataset_profile,
    }
    return _call_preprocess_agent(system_prompt, payload)


def generate_column_transform_spec(df: pd.DataFrame, dataset_policy_spec: dict) -> dict:
    system_prompt = (
        "You design per-column transformations for labeled tabular machine learning. "
        "Return only valid JSON with key 'columns'. "
        "Each column rule should include an action and optional imputation, encoding, or fill_value."
    )
    payload = {
        "columns": df.columns.tolist(),
        "sample_rows": df.head(5).to_dict(orient="records"),
        "dataset_policy_spec": dataset_policy_spec,
    }
    return _call_preprocess_agent(system_prompt, payload)


def generate_preprocessing_code(
    raw_df: pd.DataFrame,
    dataset_profile: dict,
    dataset_policy_spec: dict,
    column_transform_spec: dict,
) -> dict:
    system_prompt = load_skill_prompt("generate-preprocessing-code")
    payload = {
        "columns": raw_df.columns.tolist(),
        "sample_rows": raw_df.head(5).to_dict(orient="records"),
        "dataset_profile": dataset_profile,
        "dataset_policy_spec": dataset_policy_spec,
        "column_transform_spec": column_transform_spec,
    }
    return _call_preprocess_codegen_agent(system_prompt, payload)


def _default_dataset_policy_spec(df: pd.DataFrame) -> dict:
    identifier_columns = [column for column in ["ID", "Name", "SSN"] if column in df.columns]
    return {
        "task_type": "multiclass_classification",
        "target_column": TARGET_COLUMN if TARGET_COLUMN in df.columns else df.columns[-1],
        "group_column": GROUP_COLUMN if GROUP_COLUMN in df.columns else None,
        "identifier_columns": identifier_columns,
        "split_strategy": {"type": "grouped_holdout", "test_size": TEST_SIZE},
        "leakage_rules": {"drop_columns": identifier_columns},
        "imbalance_strategy": {"method": "none"},
        "feature_policy": {"categorical_encoding": "one_hot"},
    }


def _default_column_transform_spec(df: pd.DataFrame, dataset_policy_spec: dict) -> dict:
    columns = {}
    target_column = dataset_policy_spec["target_column"]
    group_column = dataset_policy_spec.get("group_column")
    identifier_columns = set(dataset_policy_spec.get("identifier_columns", []))
    for column in df.columns:
        if column == target_column or column == group_column or column in identifier_columns:
            columns[column] = {"action": "drop"}
        elif pd.api.types.is_numeric_dtype(df[column]):
            columns[column] = {"action": "keep", "imputation": "median"}
        else:
            columns[column] = {"action": "keep", "imputation": "mode", "encoding": "one_hot"}
    return {"columns": columns}


def execute_preprocessing(
    df: pd.DataFrame,
    dataset_policy_spec: dict,
    column_transform_spec: dict,
) -> PreprocessResult:
    cleaned = df.copy().replace(PLACEHOLDER_VALUES)

    target_column = dataset_policy_spec.get("target_column", TARGET_COLUMN)
    group_column = dataset_policy_spec.get("group_column", GROUP_COLUMN)
    identifier_columns = set(dataset_policy_spec.get("identifier_columns", []))
    forbidden_columns = identifier_columns | set(
        dataset_policy_spec.get("leakage_rules", {}).get("drop_columns", [])
    )
    column_rules = column_transform_spec.get("columns", {})

    for column, rule in column_rules.items():
        if column not in cleaned.columns:
            continue
        if rule.get("action") == "drop":
            continue
        imputation = rule.get("imputation")
        if imputation == "median":
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
        elif imputation == "mean":
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
            cleaned[column] = cleaned[column].fillna(cleaned[column].mean())
        elif imputation == "mode":
            mode = cleaned[column].mode(dropna=True)
            if not mode.empty:
                cleaned[column] = cleaned[column].fillna(mode.iloc[0])
        elif imputation == "constant":
            cleaned[column] = cleaned[column].fillna(rule.get("fill_value"))

    drop_columns = set()
    for column, rule in column_rules.items():
        if rule.get("action") == "drop":
            drop_columns.add(column)
    drop_columns |= forbidden_columns
    drop_columns.add(target_column)
    if group_column:
        drop_columns.add(group_column)

    raw_feature_frame = cleaned.drop(columns=list(drop_columns), errors="ignore")
    feature_frame = pd.get_dummies(raw_feature_frame, dummy_na=False)
    target = cleaned[target_column].copy()
    if group_column and group_column in cleaned.columns:
        groups = cleaned[group_column].copy()
    else:
        groups = pd.Series(range(len(cleaned)), index=cleaned.index, name="row_group")

    split_strategy = dataset_policy_spec.get("split_strategy", {})
    test_size = split_strategy.get("test_size", TEST_SIZE)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(feature_frame, target, groups))

    return PreprocessResult(
        cleaned_frame=cleaned,
        feature_frame=feature_frame,
        target=target,
        groups=groups,
        train_indices=list(train_idx),
        test_indices=list(test_idx),
        train_groups=groups.iloc[train_idx].drop_duplicates().tolist(),
        test_groups=groups.iloc[test_idx].drop_duplicates().tolist(),
        execution_report={
            "target_column": target_column,
            "group_column": group_column,
            "dropped_columns": sorted(drop_columns),
            "feature_count": int(feature_frame.shape[1]),
        },
    )


def audit_preprocessing_output(
    raw_df: pd.DataFrame,
    preprocess_result: PreprocessResult,
    dataset_policy_spec: dict,
    column_transform_spec: dict,
) -> dict:
    target_column = dataset_policy_spec.get("target_column", TARGET_COLUMN)
    group_column = dataset_policy_spec.get("group_column", GROUP_COLUMN)
    forbidden_columns = set(dataset_policy_spec.get("identifier_columns", []))
    forbidden_columns |= set(dataset_policy_spec.get("leakage_rules", {}).get("drop_columns", []))
    forbidden_columns |= {
        column for column, rule in column_transform_spec.get("columns", {}).items()
        if rule.get("action") == "drop"
    }

    checks = {
        "target_excluded": target_column not in preprocess_result.feature_frame.columns,
        "group_leakage_free": set(preprocess_result.train_groups).isdisjoint(set(preprocess_result.test_groups)),
        "forbidden_columns_removed": all(
            column not in preprocess_result.feature_frame.columns for column in forbidden_columns
        ),
        "feature_frame_non_empty": preprocess_result.feature_frame.shape[1] > 0,
        "row_count_preserved": len(raw_df) == len(preprocess_result.cleaned_frame),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
    }


def preprocess_credit_data(df: pd.DataFrame) -> PreprocessResult:
    dataset_policy_spec = _default_dataset_policy_spec(df)
    column_transform_spec = _default_column_transform_spec(df, dataset_policy_spec)
    return execute_preprocessing(df, dataset_policy_spec, column_transform_spec)
