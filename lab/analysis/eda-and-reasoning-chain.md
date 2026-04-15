# EDA and Reasoning Chain: Why Every Analytical Node Needs Reasoning

**Date:** 2026-04-10

## The problem

After Run 008, the pipeline produced a prediction with SHAP explanation, but there was no way to trace *why* the model was built the way it was. The column-transform-spec silently decided to one-hot encode Type_of_Loan (exploding cardinality), the FE node silently added 3 interaction features without explaining why, and model selection was `max(macro_f1)` with a generic justification.

The SHAP explanation at the end was grounded in model reasoning (good), but disconnected from data understanding (bad). If SHAP says "Outstanding_Debt is the top feature," we can't check whether the EDA also ranked it highly, whether the FE hypothesis expected this, or whether the model selection reasoning anticipated it.

## The insight: hypothesis chain

Each analytical node should produce a hypothesis that flows downstream:

```
EDA                    → "Outstanding_Debt has highest MI with target (0.12)"
column-transform-spec  → "Keep Outstanding_Debt, clip outliers at [p1, p99] per EDA skewness"
FE                     → "Created Debt_to_Income ratio because EDA shows both have high MI individually"
model selection        → "RF chosen: handles non-linear interactions in debt/income features better than LR"
explain-risk           → "SHAP confirms Debt_to_Income is top feature — FE hypothesis validated"
```

Without this chain, each node is a black box. With it, we have a traceable argument from data understanding to final explanation.

## What this requires from each node

### EDA (new, programmatic)
No LLM needed. Computes: correlation matrix, ANOVA F-stat, mutual information, skewness, missing patterns (MNAR detection), cardinality. Output is structured data passed to all downstream LLM nodes.

### column-transform-spec (enhanced)
**Input**: EDA insights (discriminative features, correlation pairs, skewness, cardinality, MNAR suspects).
**Output**: Adds `reasoning` dict — per-column explanation referencing EDA when relevant.

### FE (enhanced)
**Input**: EDA insights.
**Output**: Adds `hypothesis` — why specific interactions were created, which features were dropped and why, expected impact on model performance. This is the key testable claim.

### Model selection (new LLM skill)
**Input**: Evaluation results, tuning results, SHAP importance, EDA top features, FE hypothesis.
**Output**: Justification + `hypothesis_validation` — explicitly compares what was expected with what was observed.

### explain-risk (enhanced)
**Input**: Full hypothesis chain (EDA, FE, SHAP, selection reasoning).
**Output**: Adds `hypothesis_notes` — which hypotheses confirmed/refuted by this specific prediction's SHAP values.

## Why reasoning models matter here

The analytical nodes (column-transform-spec, FE, model selection, explain-risk) benefit from deeper reasoning. They need to:
- Cross-reference multiple inputs (EDA stats + column profiles + sample data)
- Make judgment calls (is this correlation worth dropping a feature?)
- Produce structured reasoning (not just decisions)

gpt-4o excels at instruction-following (codegen) but reasoning models (o4-mini) are better at multi-step analytical reasoning. The `.env.example` now supports per-node model overrides.

## Trade-offs

- **More LLM calls**: Model selection is now an LLM call (was hardcoded `max(macro_f1)`). Falls back to hardcoded on failure.
- **Larger payloads**: EDA insights add ~500-600 tokens to column-transform-spec and FE prompts (now includes ANOVA class separability).
- **Hypothesis may be wrong**: The FE hypothesis is a prediction, not a guarantee. That's the point — the chain lets us *verify* it against SHAP, not just assert it.
- **Reasoning models are slower**: o4-mini takes ~36s vs gpt-4o's ~12s for column-transform-spec. But the output is 3× richer (5,043 vs ~1,500 tokens) and dramatically more stable. If this reduces repair rounds, total pipeline time decreases.

## Validated (2026-04-10): Reasoning model A/B result

The column-transform-spec was run with gpt-4o (2 runs) and o4-mini (1 run) on the same dataset. o4-mini produced:
- Concrete domain-plausible clip bounds (rejected inflated p99 values)
- Stable encoding choices (ordinal for ordered categories, consistently)
- Explicit delimiter artifact handling (strip "and " prefix)
- Every reasoning entry cited actual numbers from column_profiles and EDA

See `lab/analysis/reasoning-model-for-analytical-nodes.md` for the full comparison.

## What this enables for the report

The BT5151 rubric values "justification of choices" and "critical analysis." The reasoning chain provides:
1. Data-driven feature decisions (not arbitrary)
2. Testable hypotheses (not post-hoc rationalization)
3. Hypothesis validation via SHAP (closing the loop)
4. Traceable argument from raw data to business explanation
