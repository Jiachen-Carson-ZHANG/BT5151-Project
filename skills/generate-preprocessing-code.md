---
name: generate-preprocessing-code
description: Generate executable preprocessing code from dataset and column-level specs.
---

## When to use

Use this skill after dataset policy and column transform specs are available and you need codegen for preprocessing.

## How to execute

1. Read the raw dataframe shape, columns, and sample rows.
2. Read the dataset profile, dataset policy spec, and column transform spec.
3. Return JSON with executable preprocessing code and the entrypoint name.
4. Make the generated code implement `run_preprocessing(raw_df, workspace_path)`.
5. Have the code write the required preprocessing artifacts into `workspace_path`.

## Output format

Return JSON with `code` and `entrypoint`.

## Runtime contract

The generated code must be compatible with the isolated executor. Use the exact entrypoint
signature `run_preprocessing(raw_df, workspace_path)`.

Inside that function, write these artifacts into `workspace_path`:

- `cleaned_frame.csv`
- `feature_frame.csv`
- `target.csv`
- `split_manifest.json`
- `preprocessing_report.json`

## Notes

Keep the output focused on code generation only. Deterministic execution and validation happen in later stages.
