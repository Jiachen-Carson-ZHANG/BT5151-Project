---
name: select-model
description: Choose the final model using evaluation evidence and business-aware reasoning.
---

## When to use

Use this skill once candidate evaluation metrics are available for all models.

## How to execute

1. Compare macro F1 and weighted F1 across candidates.
2. Inspect per-class behavior and important error trade-offs.
3. Select the final model.
4. Record a written justification.

## Inputs from agent state

- `evaluation_results`
- `candidate_model_specs`

## Outputs to agent state

- `selected_model_name`
- `selection_justification`

## Output format

Return the chosen model name and a concise written justification.

## Notes

Do not choose a model only because it is more complex. Prefer evidence and business impact.
