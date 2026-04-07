---
name: audit-preprocessing
description: Verify that preprocessing output is safe, leakage-aware, and usable for training.
---

## When to use

Use this skill immediately after preprocessing execution and before model training.

## How to execute

1. Verify target exclusion from feature columns.
2. Verify no overlap between training and test groups when grouped splitting is used.
3. Verify forbidden columns were removed.
4. Verify feature output is non-empty and structurally usable.

## Inputs from agent state

- `raw_frame`
- `dataset_policy_spec`
- `column_transform_spec`
- `preprocessing_execution_report`
- `full_feature_frame`
- `train_frame`
- `test_frame`

## Outputs to agent state

- `preprocessing_audit_report`

## Output format

Return structured audit checks plus an overall pass/fail flag.

## Notes

This stage is deterministic and should fail loudly if preprocessing output is unsafe.
