# Experiment 004: Feature Engineering + XGBoost — First Run

**Date:** 2026-04-09
**Run:** Run 6 (21:43, failed)

## What changed (from Run 5 / Experiment 003)

1. **Feature engineering codegen node** added between preprocessing quality review and training
2. **XGBoost** added as third candidate model (LR + RF + XGBoost)
3. **New skill prompts**: `generate-feature-engineering-code.md`, `repair-feature-engineering-code.md`
4. **New module**: `feature_engineering.py` with generate, execute, validate, repair functions
5. **Graph rewired**: quality review pass → FE codegen → inspect → execute → validate → train

## Hypothesis

Feature engineering will improve model metrics by pruning noise and adding domain interactions. XGBoost will likely outperform RF. The FE codegen loop will work on first or second attempt since input is already clean.

## Results

### FAILED — never reached feature engineering or training

The pipeline exhausted all 5 preprocessing repair attempts. The audit reviewer kept finding new issues on follow-up rounds instead of converging.

### Preprocessing timeline

| Attempt | Outcome | Issues | Key problem |
|---------|---------|--------|-------------|
| 1 | Execution crash | - | `str.extract` 2-group → single column |
| 2 | Quality FAIL | 9 | NaNs, garbage categories, negative values, delimiter artifacts |
| 3 | Quality FAIL | 4 | Garbage→minor, but NEW majors: Annual_Income max=24M, Interest_Rate max=5797 |
| 4 | Quality FAIL | 5 | Annual_Income "resolved max but low cardinality" = NEW critical, Num_Bank_Accounts max=1798 |
| 5 | Quality FAIL | 3 | Same Annual_Income critical persists |

### Token usage

- 89.7k tokens, 11 LLM calls, 141s LLM time (all wasted — never reached training)

## Root cause analysis

**The reviewer is escalating on follow-up rounds instead of converging.** Specifically:

1. Round 3 flagged Annual_Income as `major/distribution_sanity` (max too high)
2. Round 4: repair clipped Annual_Income. Reviewer acknowledged "resolved issue with max" but then flagged it as `critical/information_loss` — a brand new category
3. Round 5: same `critical/information_loss` blocks the pass

This violates the audit skill's own follow-up rules ("only flag new issues if critical"). The reviewer is technically obeying by labeling it critical — but **inventing a new critical category** that doesn't exist in the defined list (which only includes: remaining NaNs, target leakage, identifier columns).

**Why sequential discovery happens:** Each repair changes the data distribution. Clipping Annual_Income changes its cardinality. Removing garbage categories changes column count. The reviewer sees genuinely new stats each round — but the follow-up rules should prevent it from escalating these into blocking issues.

**Temperature wouldn't help:** The reviewer is consistently and correctly identifying real characteristics — it's just misclassifying their severity on follow-up. This is a prompt discipline issue, not stochasticity.

## Fix applied

Patched `audit-preprocessing.md`:
1. **First review**: Added "anticipate repair side effects" — flag all columns in the same category (counts, rates, monetary) at once, not just the first one noticed
2. **Follow-up**: Tightened to "only flag new issues if they match DEFINED critical categories" — explicitly lists the categories and says don't promote others to critical on follow-up

## Next steps

- Rerun pipeline to test convergence fix
- If preprocessing passes, this will be the first run testing the FE codegen loop and XGBoost
