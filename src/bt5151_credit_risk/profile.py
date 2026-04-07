from bt5151_credit_risk.config import TARGET_COLUMN


# Build a small dataset summary to give the pipeline basic context.
def build_dataset_profile(df, target_column=None):
    target_column = target_column or TARGET_COLUMN
    profile = {
        "row_count": len(df),
        "missing_counts": df.isna().sum().to_dict(),
    }
    if target_column in df.columns:
        profile["target_distribution"] = df[target_column].value_counts(dropna=False).to_dict()
    return profile
