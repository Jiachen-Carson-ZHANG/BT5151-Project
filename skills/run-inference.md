---
name: run-inference
description: Score a new customer-month record with the selected credit risk model.
---

## When to use

Use this skill after a final model has been selected and a new inference input is available.

## How to execute

1. Read the selected model and preprocessing rules.
2. Transform the new input with the same preprocessing logic.
3. Produce the predicted class and class probabilities.
4. Write the prediction output to agent state.

## Inputs from agent state

- `selected_model_name`
- `trained_models`
- `preprocessing_rules`
- `inference_input`

## Outputs to agent state

- `prediction_output`

## Output format

Return predicted label, class probabilities, and confidence-related details.

## Notes

Inference must use the same fitted preprocessing assumptions as training.
