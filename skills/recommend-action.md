---
name: recommend-action
description: Convert the business-readable risk explanation into an operational next step.
---

## When to use

Use this skill after a business-readable risk explanation is available.

## How to execute

1. Read the risk level and confidence band.
2. Map the case to a business action.
3. Return the action with a short reason.

## Inputs from agent state

- `risk_explanation`

## Outputs to agent state

- `recommended_action`

## Output format

Return an action code and a business-facing reason statement.

## Notes

High-risk outputs should escalate. Moderate-risk outputs should monitor. Low-risk outputs can continue standard handling.
