# Pipeline Diagnostics Q&A — Insights and Design Decisions

Captured from a live diagnostic session on 2026-04-15, reviewing runs 012–013 and the
XAI/preprocessing codegen behaviour. Five questions, each turning into a concrete finding
or a design decision.

---

## 1. Does training-diagnostics diagnose *why* class predictions differ?

**Short answer:** Yes, and the coverage is broad — but class imbalance is not a first-class
diagnostic, which is a gap.

### What it covers

The `generate-training-diagnostics` LLM node is explicitly asked to reason about:

- **Per-class struggle analysis** — for each class: struggle level (low/medium/high), WHY
  it's hard (feature overlap, class size, boundary ambiguity), which class it loses samples
  to, and whether the confusion is symmetric or asymmetric.
- **Confusion flow direction** — "Standard→Good at 2× the rate of Good→Standard" tells you
  which side of the boundary is thin, not just that confusion exists.
- **Capacity diagnosis per model** — LR's C value reveals whether it's underfitting (C=0.016
  = heavy regularisation = can't express complex boundary); RF's max_depth reveals
  interaction-heavy signal; XGB's best_n_trees vs cap reveals whether learning is complete.
- **Confidence gap analysis** — a model at 0.95 confidence on correct and 0.80 on wrong
  "knows when it knows"; one at 0.55 on both is guessing. The gap is diagnostic.
- **EDA hypothesis validation** — each tested prediction from EDA is confirmed or refuted
  with the actual metric value, not just asserted.

### The gap: class imbalance is implicit, not explicit

The skill mentions "class size?" as one possible per-class diagnosis, and `support` counts
are in the evaluation_results. But the prompt does not say: *"check whether low-recall
classes are minority classes, and distinguish imbalance from overlap — these are different
root causes requiring different fixes."*

Imbalance → needs resampling or class-weighted loss.
Overlap → needs better features or a different boundary shape.
Diagnosing one as the other sends the analysis in the wrong direction.

**Potential improvement:** add an explicit imbalance check bullet to the training-diagnostics
skill: "If a class has recall < 0.70 and support < 20% of total rows, flag it as a
potential imbalance case and distinguish from feature overlap."

---

## 2. Does any node diagnose the local XAI casebook?

**Short answer:** Yes — `interpret-local-xai` reads all 9 cases and produces a full
interpretation. It reaches explain-risk through two separate channels.

### What interpret-local-xai produces

The skill is explicitly framed around the casebook's three case types:

| Case type | What it reveals |
|---|---|
| Representative (most confident correct per class) | What the model "understands well" — the shared feature profile of easy cases |
| Borderline (least confident correct per class) | Decision boundary thinness — which features sit near a threshold |
| Worst misclassification (most confident wrong per class) | Systematic failure mode — which features pushed the wrong class, and how confident |

Outputs include: per-class stories with specific row citations and SHAP values, confusion
direction (asymmetric vs symmetric), global vs local consistency (does the top global SHAP
feature drive failures, or does the model abandon it on hard cases?), boundary candidate for
feature engineering ("one FE move that would thicken the weakest boundary"), and three-tier
hypotheses from case evidence.

### How it reaches explain-risk

Two channels, not one:

1. **`analysis_bundle_summary`** — `local_xai_interpretation` is in the full semantic bundle
   that package-analysis-bundle passes verbatim to explain-risk. The LLM gets the full
   per-class stories.
2. **`nearest_casebook_case`** — at inference time, `run_inference_node` computes cosine
   similarity between the current customer's SHAP vector and all 9 casebook cases. The
   nearest match (case type, true label, predicted label, confidence) is passed directly to
   explain-risk so it can say "this customer resembles a borderline Standard case where
   Credit_Mix was the swing feature."

The casebook is well-wired. The risk is whether explain-risk actually *uses* the local
interpretation rather than defaulting to generic language — that's a skill prompt and run
quality question, not a wiring question.

---

## 3. Are the top 2 features from the notebook still in the pipeline?

**Short answer:** Yes — same two features, same top tier. The apparent discrepancy is a
metric difference, not a missing feature.

### Notebook vs pipeline comparison (run 173718)

| Rank | Feature | Notebook MDI | Pipeline SHAP (mean\|SHAP\|) |
|------|---------|-------------|---------------------------|
| 1 (notebook) / 2 (pipeline) | `Outstanding_Debt` | 0.091 | 0.300 |
| 2 (notebook) / 1 (pipeline) | `Credit_Mix` | 0.082 | 0.498 |

Both features are present and dominant. The order flipped because:

- The **notebook uses MDI** (Mean Decrease in Impurity from a Random Forest) — a biased
  metric that over-weights high-cardinality features and tree-depth effects.
- The **pipeline uses mean |TreeSHAP|** from XGBoost — model-agnostic, theoretically sound
  attribution that correctly credits features across the full prediction surface.

The ratio features from the notebook (`Debt_to_Income`, `EMI_to_Salary`) also appear in the
pipeline as `Debt_Income_Ratio` (rank 12) and `EMI_Salary_Ratio` (rank 11). The FE LLM
generated 4 of the 8 canonical ratios — it selected based on MI parent scores rather than
exhaustively implementing all 8.

### Side finding: ALE failure in run 173718 is the pre-fix bug

Run 173718 showed ALE failing for all 5 features with "all-object dtype." Root cause: the
Payment_Behaviour bool one-hots mixed with float columns produce a numpy object array via
`.values` — which then reconstructs as all-object dtype when fed back to XGBoost inside the
ALE perturbation loop. The fix (bool→int8 cast + `.astype(float)` + `dtype=float` on
DataFrame reconstruction) was applied in this session and will take effect from the next run.

---

## 4. Why did preprocessing codegen get worse despite upgrading to o3 for column-transform-spec?

**The regression in one sentence:** o3 writes a richer spec, so gpt-4o generates more
operations, so the 180s timeout budget that was already marginal gets exceeded more often —
and the skill had no performance contract.

### Three compounding bugs

**Bug 1 — The runner itself reads without `low_memory=False` (hardcoded, not LLM-fixable)**

`preprocess.py` line 305 hardcodes `pd.read_csv(raw_frame_path)` in the subprocess wrapper.
The generated code never sees or controls this call. Every attempt prints:
`DtypeWarning: Columns (0: Monthly_Balance) have mixed types`

The LLM cannot fix a bug that is outside its scope. Fixed: `low_memory=False` added to all
four `pd.read_csv` calls in `preprocess.py`.

**Bug 2 — `.median()` on unconverted object series (pandas 3.x crash)**

`Num_of_Loan` contains stray underscores (`"3_"`, `"2-"`) and is stored as object dtype.
The generated code (attempt 2) did:
```python
df['Num_of_Loan'] = s.fillna(df['Num_of_Loan'].median())  # TypeError in pandas 3.x
```
The skill already warned about the chained-assignment pattern at line 198, but gpt-4o
violated it under a complex spec. The instruction was correct; the model didn't follow it.
Fixed: new gotcha bullet naming object-dtype median as a hard rule, `Num_of_Loan` and
`Monthly_Balance` cited by name.

**Bug 3 — No performance budget in the skill (timeouts on attempts 1 and 3)**

The 180s limit (now 300s) was already marginal. o3's richer spec → gpt-4o implements more
groupby steps → repair model (o4-mini) sometimes replaces a fast operation with a slow one
→ re-timeout. Fixed: explicit 200-second performance budget gotcha added to the skill, with
three specifically forbidden patterns (row-wise `.apply()`, `groupby().apply(lambda...)`,
looping over columns with separate `transform` calls) and the canonical fast
`groupby().transform('median')` pattern shown inline.

### The model mismatch insight

The real cause of the regression is a **model-complexity mismatch**:

| Node | Model | Spec complexity with o3 | Outcome |
|---|---|---|---|
| `column-transform-spec` | o3 | Produces richer, more prescriptive spec | ✓ Better spec quality |
| `generate-preprocessing-code` | gpt-4o | Receives richer spec, generates more operations | More timeouts, more instruction misses |
| `repair-preprocessing-code` | o4-mini | Receives error + bad code | Fixes it — but at extra cost |

The repair model (o4-mini) reliably fixed the chained-median bug. The right fix is to stop
making the repair model clean up after the weaker codegen model.

---

## 5. Should we hard-code column names in the skill to prevent the object-median bug?

**Decision: No. Upgrade the codegen model to o4-mini instead.**

### Why explicit column names are a band-aid

Naming `Num_of_Loan` in the skill patches this dataset. The same gpt-4o behaviour would
then hit `Delay_from_due_date` or `Changed_Credit_Limit` on the next dataset. The result
is a forever game of whack-a-mole that erodes the pipeline's dataset-agnosticism.

The instruction at skill line 198 already correctly describes the required pattern. The bug
is that gpt-4o doesn't follow it reliably under a complex spec. Naming columns treats the
symptom, not the cause.

### Why the model upgrade is the right fix

The evidence is already in the logs:

- Repair attempt 1 (o4-mini) received the chained-median crash and the current bad code.
- It produced a fix that passed on attempt 3 (attempt 3 timed out for an unrelated reason,
  but the median pattern was fixed).
- o4-mini follows the existing instruction. gpt-4o doesn't.

The cost argument also runs the wrong way: gpt-4o generates bad code → o4-mini repairs it →
this costs one extra LLM call (94s, ~11k tokens) plus two subprocess executions (3 min + 3
min). The net cost of using o4-mini for codegen from the start is lower than the repair tax.

**Applied:** `OPENAI_MODEL_GENERATE_PREPROCESSING_CODE=o4-mini` set in `.env` and
`.env.example`. The codegen and repair nodes now use the same model, which means the repair
loop becomes a genuine fallback rather than a routine cleanup pass.

### General principle derived

> When an instruction exists in the skill prompt and is clear, but a cheaper model
> consistently violates it under complex input, the fix is a model upgrade — not a more
> explicit instruction. More instructions make the prompt longer and increase the chance of
> other instructions being deprioritised. A reasoning model that follows instructions
> compactly is strictly better.

This aligns with the existing decision documented in
`lab/analysis/reasoning-model-for-analytical-nodes.md` — the same logic that pushed
analytical nodes to o4-mini applies to codegen nodes that must follow dense constraint sets.
