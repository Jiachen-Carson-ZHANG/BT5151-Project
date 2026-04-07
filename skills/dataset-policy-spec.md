---
name: dataset-policy-spec
description: Generate dataset-level preprocessing policy for a labelled tabular ML task.
---

## When to use

Use this skill at the start of preprocessing after the raw dataset has been loaded and profiled.

## How to execute

1. Inspect dataset profile, columns, and sample rows.
2. Decide task type, target column, group column, identifier columns, split strategy, leakage policy, and imbalance policy.
3. Return a structured dataset-level preprocessing spec.

## Inputs from agent state

- `raw_frame`
- `dataset_profile`

## Outputs to agent state

- `dataset_policy_spec`

## Output format

Return structured JSON describing dataset-level preprocessing policy.

## Notes

This is a reasoning stage. It should propose policy, not directly mutate the dataset.
