def build_dataset_profile(df, target_column: str | None = None):
    profile = {
        "row_count": len(df),
        "missing_counts": df.isna().sum().to_dict(),
    }
    if target_column and target_column in df.columns:
        profile["target_distribution"] = df[target_column].value_counts(dropna=False).to_dict()
    return profile
