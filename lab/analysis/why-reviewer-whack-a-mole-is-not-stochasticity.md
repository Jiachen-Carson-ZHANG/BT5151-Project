# Why the reviewer whack-a-mole was a state problem, not a stochasticity problem

Date: 2026-04-09

## The question

LLM reviews often produce findings in multiple rounds rather than all at once. This is a well-known pattern:

| Round | Typical focus |
|---|---|
| 1 | Obvious bugs, syntax, formatting |
| 2 | Edge cases, architectural flaws |
| 3+ | Fine-tuning, verification of previous fixes |

This happens because of the stochastic (probabilistic) nature of language models — each pass samples a different reasoning path, surfacing different issues. A natural hypothesis: our quality reviewer kept finding new issues each round because of this inherent stochasticity.

## Why that diagnosis was wrong for our case

Our quality reviewer is **more constrained** than a general-purpose code reviewer:

1. **Fixed checklist** — 7 explicit reasoning steps (completeness, spec compliance, information loss, encoding quality, distribution sanity, target alignment, feature engineering)
2. **Structured output** — JSON with specific categories and severity levels
3. **Concrete spec to check against** — the `column_transform_spec` provides ground truth

In a constrained review like this, the LLM should find most issues on the first pass because it follows the same checklist every time. Stochastic variation would cause minor differences in wording or severity, not fundamentally different issue sets.

## The actual root cause: state misalignment

The reviewer was confused because **it saw the wrong data**:

1. **cleaned_frame in the payload**: The cleaned frame legitimately contained identifier columns (ID, Customer_ID, Name, SSN) that were intentionally kept until encoding. The reviewer saw these and flagged them as "identifier columns still present" — a valid concern for the feature frame, but wrong for the cleaned frame.

2. **No preprocessing_report**: The reviewer didn't know what the code intentionally did. When the code clipped Annual_Income to 250k, the reviewer couldn't tell if that was an intentional decision or a missed outlier.

3. **No memory of previous issues**: Each review was a fresh comprehensive scan. Even if the reviewer found the same issues, stochastic sampling meant it also found *different* issues — and the repair loop treated all of them as new work.

## The fix

1. Remove cleaned_frame from the payload — reviewer only sees feature_frame (the final artifact)
2. Add preprocessing_report — reviewer knows what was intentional
3. Pass previous_audit_report — follow-up reviews focus on whether previous issues were fixed, not on finding new ones

## The lesson

When an LLM-in-a-loop produces inconsistent results, the first question should be: **is the LLM seeing the right information?** Not: **is the LLM being stochastic?**

Stochasticity is a real issue for open-ended tasks (creative writing, brainstorming). But for structured review with a checklist and ground truth, inconsistency usually means the input is ambiguous or incomplete, not that the model is randomly sampling different answers.

This distinction matters for designing reliable agentic systems: fixing the prompt or reducing temperature treats a symptom. Fixing the payload treats the cause.
