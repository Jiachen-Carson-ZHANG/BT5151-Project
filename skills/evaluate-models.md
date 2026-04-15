---
name: evaluate-models
description: Evaluate candidate models with task-appropriate multi-class metrics.
---

## When to use

Use this skill after all candidate models have been trained and held-out evaluation data is ready.

## How to execute

1. Score every candidate model on the same held-out data.
2. Compute per-class precision, recall, and F1.
3. Compute macro F1 and weighted F1.
4. Save evaluation tables and plot artifacts.

## Inputs from agent state

- `trained_models`
- `test_frame`
- `feature_columns`

## Outputs to agent state

- `evaluation_results`
- `evaluation_visual_paths`

## Output format

Return structured metric dictionaries and references to saved evaluation visualizations.

## Notes

Interpret results in business terms later, but keep the raw metrics accurate and reproducible here.
