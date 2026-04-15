# o3 vs o4-mini on `column-transform-spec`

Date: 2026-04-15

## Why this note exists

We switched `column-transform-spec` from `o4-mini` to `o3` and the very next run changed pipeline behavior materially.

This note is not claiming a final model improvement yet.
The latest `o3` run failed in feature engineering before training completed.

But the run is still valuable because it shows what the stronger reasoning model changed upstream, and what kinds of downstream contract bugs that stronger reasoning exposed.

## Runs compared

- `o4-mini` spec run:
  - `logs/stage_full_20260415_030032.log`
- `o3` spec run:
  - `logs/stage_full_20260415_121006.log`

## Runtime comparison

The stronger reasoning model was not slower here.

From the logs:

- `o4-mini` `column-transform-spec`
  - input tokens: `12,136`
  - output tokens: `21,802`
  - duration: `253.34s`
- `o3` `column-transform-spec`
  - input tokens: `12,431`
  - output tokens: `8,606`
  - duration: `55.86s`

So in this case `o3` produced a more compressed, more decisive transform plan and finished much faster.

That matters because it weakens the argument that "stronger reasoning will necessarily slow the pipeline too much."
At least for this node, the stronger model appears to have reduced wandering and produced a more direct answer.

## Main observation

`o3` produced a **meaningfully better semantic preprocessing plan** than `o4-mini`.

This was not just different wording.
It changed the architecture direction of the generated pipeline:

- `Occupation` became explicitly **identity-significant** and was deferred instead of being prematurely frequency-encoded.
- `Payment_Behaviour` was also deferred instead of being hard-baked too early.
- `Monthly_Inhand_Salary` used a more realistic **fallback formula** (`Annual_Income / 12`) before group imputation.
- `Credit_History_Age` reasoning explicitly preserved both years and months.
- Numeric bounds were generally tighter and more realistic.

That is exactly the kind of judgment work that belongs in a stronger reasoning model.

## What improved with `o3`

### 1. Better semantic role / representation decisions

In the `o4-mini` run, `Occupation` was still described and implemented in a way that collapsed identity too early.

In the `o3` run, the reasoning explicitly recognized:

- small cardinality is not enough reason to encode early
- some categoricals are **identity-significant**
- those should usually be **deferred** to the FE dual-view stage

This is a first-principles improvement, not a cosmetic one.

### 2. Better use of domain-agnostic structural reasoning

The `o3` spec made stronger use of generalized patterns we wanted:

- fallback formulas
- group-aware imputation
- ordered categorical handling
- realistic numeric bounds
- preserving semantic structure for downstream model-specific views

This is closer to the intended architecture:

- spec node does the brain work
- worker node executes
- FE owns model-facing representation

### 3. It exposed the next real bottleneck faster

The `o3` run did not fail because the spec got worse.
It failed because the stronger spec pushed the downstream system into a more correct dual-view behavior, and the FE/runtime contract was not ready for it.

That is an important distinction.

## What got worse operationally

### 1. Preprocessing became more ambitious and the first attempt timed out

The first preprocessing attempt in the `o3` run hit the subprocess timeout and wrote no artifacts.

That does **not** yet justify increasing the timeout by itself.

Why:

- preprocessing timeout is currently `120s`
- the repaired preprocessing attempt later completed successfully in about `30s`

So the evidence does **not** say "the pipeline now fundamentally needs more time."
It says the first generated worker implementation was inefficient enough to hit the guardrail.

The right first fix is still:

- improve the worker prompt / repair behavior for performance-sensitive code paths
- keep the timeout guardrail unless good implementations repeatedly cluster near the limit

### 2. FE failed because the harness misclassified boolean dummies as non-numeric

The key FE failure in the `o3` run is easy to misread.

The log says views were "not fully numeric," but the final FE artifact actually contained:

- `21` boolean columns
- `0` object columns

So the immediate issue was **not** "the model forgot to encode deferred categoricals."

It was:

- FE correctly created boolean one-hot dummy columns
- the FE validator treated anything outside pandas `"number"` as non-numeric
- `bool` therefore failed the contract even though the downstream models can consume it

This is a downstream contract bug, not proof that `o3` made FE worse.

## Bottom line

`o3` looks better than `o4-mini` for `column-transform-spec`.

Not because it already improved final metrics.
We do not have that evidence yet.

It looks better because it produced a more architecture-aligned semantic plan:

- defer identity-significant categoricals
- preserve semantic structure longer
- use fallback formulas and more realistic bounds
- reason more clearly about canonical base-table responsibilities

The cost of that improvement is that the downstream pipeline now has less room to hide weak contracts.

In this case, `o3` exposed two things:

1. the worker prompt can still generate a preprocessing implementation that is too slow
2. the FE validator currently treats valid boolean dummy columns as "non-numeric"

## Interpretation

This is the important lesson:

**A stronger reasoning model does not just improve answers. It also surfaces architectural debt sooner.**

If the downstream codegen, validator, or repair loop is under-specified, `o3` will often reach those edge cases faster because it is actually trying to follow the higher-level design more faithfully.

So this run should not be read as:

- "`o3` failed, go back to `o4-mini`"

It should be read as:

- "`o3` is probably the right spec model, and it has now revealed the next harness bugs we need to fix."

## Recommended next step

1. Fix the FE numeric contract so boolean dummy columns are accepted or cast to `int8`.
2. Rerun with `o3` still on `column-transform-spec`.
3. Only then compare end-to-end performance and XAI quality against the last successful `o4-mini` run.

Until that rerun exists, the fairest conclusion is:

- `o3` improved the semantic planning layer
- end-to-end benefit is not yet measured
- the latest failure is downstream of the spec quality improvement, not evidence against it
