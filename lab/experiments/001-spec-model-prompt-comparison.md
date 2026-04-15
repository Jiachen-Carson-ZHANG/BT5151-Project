# Experiment 001: Spec Stage — Model & Prompt Comparison

Date: 2026-04-07
Stage: `run_stage.py spec` (dataset-policy-spec + column-transform-spec)
Dataset: train.csv (100,000 rows x 28 columns)

## Objective

Determine whether spec-stage quality is driven by model capability (gpt-4o vs gpt-4o-mini) or prompt engineering (CoT, few-shot examples, richer context).

## Variables

| Variable | Description |
|----------|-------------|
| **Model** | gpt-4o-mini vs gpt-4o |
| **Prompt version** | V1 (vague instructions, no schema, no examples, no column summaries) vs V2 (CoT reasoning steps, few-shot example, explicit JSON schema, column summaries with dtype/cardinality/unique values) |

## Runs

| Run | Model | Prompt | Log file |
|-----|-------|--------|----------|
| A | gpt-4o-mini | V1 | `stage_spec_20260407_181001.log` |
| B | gpt-4o | V2 | `stage_spec_20260407_184231.log` |
| C | gpt-4o-mini | V2 | `stage_spec_20260407_185143.log` |

## Results: Policy Spec Correctness

| Decision | Ground truth | Run A (mini+V1) | Run B (4o+V2) | Run C (mini+V2) |
|----------|-------------|-----------------|---------------|-----------------|
| task_type | multiclass_classification | regression | multiclass_classification | multiclass_classification |
| target_column | Credit_Score | Credit_Score | Credit_Score | Credit_Score |
| group_column | Customer_ID | null | Customer_ID | Customer_ID |
| split_strategy | grouped_holdout | holdout (no stratify) | grouped_holdout | grouped_holdout |
| identifier_columns | ID, Customer_ID, SSN, Name | ID, Customer_ID | ID, Customer_ID, SSN, Name | ID, Customer_ID, SSN, Name |

## Results: Column Transform Spec Correctness

| Decision | Expected | Run A (mini+V1) | Run B (4o+V2) | Run C (mini+V2) |
|----------|----------|-----------------|---------------|-----------------|
| Credit_Score action | drop (target) | N/A (crashed) | drop | keep (label encoding) — **leakage bug** |
| Type_of_Loan encoding | label (high cardinality) | N/A | label | one_hot — **feature explosion risk** |
| Changed_Credit_Limit cleaning | handle `_` character | N/A | "replace '_' with NaN, convert to float" | "convert to float" — **misses `_` cleanup** |
| Credit_History_Age cleaning | extract months | N/A | "extract total months" | "convert to float, extract years and months" |

## Results: Token Usage & Cost

| Metric | Run A (mini+V1) | Run B (4o+V2) | Run C (mini+V2) |
|--------|-----------------|---------------|-----------------|
| LLM calls | 6 (3+3 retries) | 2 (no retries) | 2 (no retries) |
| Input tokens | ~15,500 | 6,988 | 6,992 |
| Output tokens | ~4,970 | 1,255 | 1,059 |
| Total tokens | ~20,460 | 8,243 | 8,051 |
| Duration | 120s+ (crashed on col spec) | 11.7s | 49.8s |
| Est. cost | ~$0.005 (wasted — wrong answer) | ~$0.030 | ~$0.002 |

Note: 4o-mini V1 input tokens were lower per call (2,433) because V1 did not include column_summaries. But 3 retries per node tripled the total.

## Results: Reliability

| Metric | Run A (mini+V1) | Run B (4o+V2) | Run C (mini+V2) |
|--------|-----------------|---------------|-----------------|
| JSON valid on 1st try | No (0/6) | Yes (2/2) | Yes (2/2) |
| Column spec completed | No (crashed after 3 retries) | Yes | Yes |
| Errors in policy spec | 3 (task type, group col, identifiers) | 0 | 0 |
| Errors in column spec | N/A | 0 | 1 (target leakage) |

## Analysis

### Prompt engineering was the dominant factor

The V2 prompts (CoT + few-shot + column summaries + explicit JSON schema) fixed the majority of issues for both models:

- **JSON reliability**: V1 produced invalid JSON on every attempt. V2 produced valid JSON on the first try for both models. The explicit "no markdown fences" instruction and the JSON schema were sufficient.
- **Task type detection**: V1 misidentified the task as regression because the LLM only saw 5 rows and the column name "Credit_Score" (sounds numeric). V2 included column_summaries showing `Credit_Score` has dtype=object, nunique=3, unique_values=["Good","Standard","Poor"] — both models correctly identified multiclass classification.
- **Group column**: V1 missed Customer_ID entirely. V2's CoT step ("if multiple rows share the same entity...") and the few-shot example (Patient_ID) guided both models to the correct answer.

### Model capability matters for downstream precision

While both models got the policy spec right with V2 prompts, gpt-4o-mini made a critical error in the column transform spec: it kept Credit_Score as a feature with label encoding instead of dropping it. This would cause target leakage — the model would achieve artificially high accuracy by reading the answer column.

gpt-4o avoided this because it cross-referenced the policy spec (which says to drop Credit_Score) with the column spec. gpt-4o-mini had the same information but failed to connect the dots.

### Cost-effectiveness

- gpt-4o is 15x more expensive per token but 4x faster.
- For spec nodes that run once per pipeline invocation, the absolute cost difference is $0.028 (~3 cents).
- The cost of a wrong spec (undetected target leakage, wrong task type) far exceeds the token cost.

## Decision

Use **gpt-4o** for correctness-critical nodes: dataset-policy-spec, column-transform-spec, generate-preprocessing-code, repair-preprocessing-code.

Use **gpt-4o-mini** for simple text generation nodes: explain-risk, recommend-action.

## Prompt Changes (V1 → V2)

1. Added "You are a senior data scientist" persona
2. Added numbered CoT reasoning steps with explicit decision criteria
3. Added few-shot example using a different domain (medical dataset) to demonstrate expected output
4. Added explicit JSON schema with all required keys
5. Added "Return raw JSON only. Do not wrap in markdown code fences" instruction
6. Enriched payload with `column_summaries` (dtype, nunique, unique_values for low-cardinality columns) — previously only sent column names + 5 sample rows
