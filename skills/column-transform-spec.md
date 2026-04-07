---
name: column-transform-spec
description: Generate per-column cleaning, imputation, drop, and encoding rules.
---

## When to use

Use this skill after dataset-level preprocessing policy is available.

## How to execute

1. Read the dataset-level policy and sample rows.
2. Decide per-column keep/drop rules.
3. Decide column-level cleaning, imputation, and encoding rules.
4. Return a structured column transform spec.

## Inputs from agent state

- `raw_frame`
- `dataset_policy_spec`

## Outputs to agent state

- `column_transform_spec`

## Output format

Return structured JSON with one rule set per column.

## Notes

This stage should stay at the rule-definition level. Deterministic code will execute the rules later.
