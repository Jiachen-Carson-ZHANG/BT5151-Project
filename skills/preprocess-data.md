---
name: preprocess-data
description: Prepare raw credit monitoring data for safe model training and inference.
---

## When to use

Use this skill when raw customer-month credit data must be profiled, cleaned, split safely, and transformed into model-ready inputs.

## How to execute

1. Load the raw dataset.
2. Build a dataset profile and inspect target balance.
3. Replace invalid placeholders and malformed values.
4. Remove identifier columns from modeling inputs.
5. Create split-safe feature and target outputs using grouped customer logic.

## Inputs from agent state

- `raw_dataset_path`
- `dataset_profile` if already available

## Outputs to agent state

- `dataset_profile`
- `preprocessing_rules`
- `feature_columns`
- `split_metadata`
- `train_frame`
- `validation_frame`
- `test_frame`

## Output format

Return structured preprocessing metadata, cleaned feature frames, and grouped split information.

## Notes

Prevent target leakage and customer-level leakage. Do not let the target or direct identifiers enter the feature frame.
