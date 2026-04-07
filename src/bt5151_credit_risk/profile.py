def build_dataset_profile(df):
    return {
        "row_count": len(df),
        "target_distribution": df["Credit_Score"].value_counts(dropna=False).to_dict(),
        "missing_counts": df.isna().sum().to_dict(),
    }
