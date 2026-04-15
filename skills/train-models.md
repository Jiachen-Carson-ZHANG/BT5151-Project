---
name: train-models
description: Train and record candidate multi-class models on the prepared dataset.
---

## When to use

Use this skill after preprocessing is complete and grouped train/test splits are available.

## How to execute

1. Read the prepared training data from agent state.
2. Initialize the candidate models.
3. Fit each candidate on the same training data.
4. Store fitted models and training metadata.

## Inputs from agent state

- `train_frame`
- `feature_columns`
- `preprocessing_rules`

## Outputs to agent state

- `candidate_model_specs`
- `trained_models`

## Output format

Return a dictionary of model names to fitted model objects and model configuration summaries.

## Notes

Keep the model comparison fair. Use the same processed inputs for every candidate.
