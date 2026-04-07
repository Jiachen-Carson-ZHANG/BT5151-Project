---
name: explain-risk
description: Translate model predictions into business-readable credit risk language.
---

## When to use

Use this skill after inference has produced a predicted class and confidence values.

## How to execute

1. Read the model prediction and confidence values.
2. Map the predicted class into a business-friendly risk level.
3. Summarize the result in plain language for a non-technical user.

## Inputs from agent state

- `prediction_output`

## Outputs to agent state

- `risk_explanation`

## Output format

Return predicted class, risk level, confidence band, and a human-readable summary.

## Notes

This stage should bridge technical outputs and business understanding.
