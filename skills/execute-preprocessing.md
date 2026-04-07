---
name: execute-preprocessing
description: Deterministically apply preprocessing specs and build model-ready train/test data.
---

## When to use

Use this skill after dataset-level and column-level preprocessing specs have been generated.

## How to execute

1. Apply placeholder replacement and basic cleaning.
2. Apply column-level imputation and drop rules.
3. Encode remaining categorical features.
4. Build feature frame and target.
5. Split data according to the dataset policy.

## Inputs from agent state

- `raw_frame`
- `dataset_policy_spec`
- `column_transform_spec`

## Outputs to agent state

- `preprocessing_rules`
- `preprocessing_execution_report`
- `feature_columns`
- `class_names`
- `label_to_id`
- `id_to_label`
- `full_feature_frame`
- `train_frame`
- `test_frame`
- `train_target`
- `test_target`

## Output format

Return deterministic preprocessing outputs and execution metadata.

## Notes

This stage should be reproducible and should not rely on LLM reasoning once the specs are fixed.
