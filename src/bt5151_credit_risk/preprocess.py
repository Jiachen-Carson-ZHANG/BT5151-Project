from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from bt5151_credit_risk.config import GROUP_COLUMN, RANDOM_SEED, TARGET_COLUMN, TEST_SIZE


@dataclass
class PreprocessResult:
    cleaned_frame: pd.DataFrame
    feature_frame: pd.DataFrame
    target: pd.Series
    groups: pd.Series
    train_groups: list[str]
    test_groups: list[str]


def preprocess_credit_data(df: pd.DataFrame) -> PreprocessResult:
    cleaned = df.copy().replace({"_": pd.NA, "_______": pd.NA, "!@9#%8": pd.NA})
    target = cleaned[TARGET_COLUMN].copy()
    groups = cleaned[GROUP_COLUMN].copy()
    feature_frame = cleaned.drop(
        columns=["ID", GROUP_COLUMN, "Name", "SSN", TARGET_COLUMN],
        errors="ignore",
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(feature_frame, target, groups))

    return PreprocessResult(
        cleaned_frame=cleaned,
        feature_frame=feature_frame,
        target=target,
        groups=groups,
        train_groups=groups.iloc[train_idx].drop_duplicates().tolist(),
        test_groups=groups.iloc[test_idx].drop_duplicates().tolist(),
    )
