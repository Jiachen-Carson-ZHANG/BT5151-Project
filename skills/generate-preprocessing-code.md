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

## Output format

Return JSON with `code` and `entrypoint`.

## Notes

Keep the output focused on code generation only. Deterministic execution and validation happen in later stages.
