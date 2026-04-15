# Experiment 003: Audit Prompt & Training Fixes

**Date:** 2026-04-09
**Runs:** Run 4 (16:15, failed) and Run 5 (17:19, passed)

## What changed (from Run 3 / Experiment 002-C)

1. **Audit prompt expanded** — 3 new major-severity checks in `audit-preprocessing.md`:
   - Garbage category names (underscore padding, special character noise)
   - Delimiter split artifacts ("and " prefixed duplicate columns)
   - One-sided clipping (min clipped but max not)
2. **6 skill prompt leaks generalized** — removed all dataset-specific terms from skill descriptions and examples
3. **Accuracy logging fixed** — added `accuracy_score` to `evaluate.py`
4. **Model training improved** — `train.py` now uses `StandardScaler` pipeline for LR, `class_weight='balanced'` for both models
5. **Chained-assignment gotcha added** — both `generate-preprocessing-code.md` and `repair-preprocessing-code.md` now warn against referencing the original column inside chained assignments

## Hypothesis

- Expanded audit will catch data quality issues (garbage categories, outliers) that slipped through before
- StandardScaler will dramatically improve LR performance
- Chained-assignment warning will prevent the repeat crash pattern from Run 4

## Results

### Run 4 (16:15) — FAILED, 5 attempts exhausted

- Audit improvements validated: 10 issues caught on first review (vs 4 in Run 3)
- Repair 1 fixed 3 issues but left 7
- Repairs 2-4 all crashed on same bug: `pd.to_numeric(df['col'], errors='coerce').fillna(df['col'].median())` — median called on the still-string original column
- The chained-assignment gotcha was NOT yet in the skill prompts for this run

### Run 5 (17:19) — PASSED, 2 repair rounds

| Step | Attempt 1 | Attempt 2 | Attempt 3 |
|------|-----------|-----------|-----------|
| Execution | OK | OK | OK |
| Quality review | 8 issues | 2 issues | PASS (0 issues) |
| Features | 73 cols | 64 cols | 62 cols |

- Zero execution crashes (chained-assignment fix worked)
- Audit convergence: 8 → 2 → 0, clean progression
- Garbage columns progressively removed: `Occupation________`, `Credit_Mix__`, `Payment_Behaviour_!@9#%8`, `and Auto Loan` artifacts
- `Num_Credit_Inquiries` NaN issue (critical) fixed in repair 1

### Model metrics (Run 5 vs Run 3)

| Metric | Run 3 | Run 5 | Delta |
|--------|-------|-------|-------|
| LR accuracy | 0.554 | 0.621 | +6.7pp |
| LR macro_f1 | 0.39 | 0.617 | **+22.7pp** |
| RF accuracy | 0.696 | 0.694 | -0.2pp |
| RF macro_f1 | 0.67 | 0.667 | -0.3pp |

### Token usage

- Run 4: 74k tokens, 9 calls, 110s LLM time (failed)
- Run 5: 67k tokens, 10 calls, 78s LLM time (passed)

## Insights

1. **Chained-assignment is a systematic LLM codegen antipattern.** GPT-4o generated the same broken one-liner 3 times in a row in Run 4 — it never learned from the crash. Explicit skill prompt warnings fixed it immediately. This reinforces: prompt-level bans work for patterns the LLM consistently gets wrong.

2. **Expanded audit scope pays off.** Run 3 caught 4 issues; Run 5 caught 8 on the same kind of data. The garbage category and delimiter artifact checks are genuine improvements — these were real data quality problems that would have hurt model performance.

3. **StandardScaler is the single biggest LR improvement.** LR macro_f1 nearly doubled (0.39 → 0.617). RF was unaffected (scale-invariant). This confirms the previous diagnosis: unscaled features with wildly different magnitudes (Interest_Rate max=5797 vs Credit_Utilization_Ratio max=50) dominated LR's L2 regularization.

4. **RF is robust to preprocessing quality.** RF macro_f1 barely moved despite cleaner features (fewer garbage columns, better clipping). This suggests RF was already ignoring the noisy columns. LR benefited more from cleaner features.

5. **Two-mode audit convergence still works.** Follow-up reviews (195 tokens, 54 tokens) vs first review (674 tokens). Focused mode converges quickly.

## Next steps

- Try temperature=0 for codegen/audit calls to reduce stochastic variation
- Add GBM/XGBoost as third candidate model
- Investigate whether 62 features is optimal or if further pruning helps
- Run with different seeds to check stability
