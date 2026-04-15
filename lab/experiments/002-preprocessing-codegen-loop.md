# Experiment 002: Preprocessing Codegen Loop — End-to-End

Date: 2026-04-09
Stage: Full pipeline (all nodes from dataset-policy-spec through recommend-action)
Dataset: train.csv (100,000 rows x 28 columns, 3-class credit risk)

## Hypothesis

The LLM-driven preprocessing codegen loop (generate → inspect → execute → validate → quality review → repair) can produce a clean feature frame sufficient for model training, if:
1. Skill prompts are well-structured with CoT, critical rules, and cross-domain examples
2. Static AST inspection catches dangerous patterns before execution
3. An LLM quality reviewer catches data quality issues the hardcoded validation misses
4. The repair skill fixes all issues in one pass (not whack-a-mole)

## Changes from baseline

This experiment involved multiple iterative changes across the session. Key changes in chronological order:

### Prompt improvements (before Run A)
- Added `inplace=True` ban to Critical Rules section of generate and repair skills
- Added step-by-step workflow (10 steps) to generate skill
- Added common gotchas section (defensive numeric conversion, multi-value delimiters, two-sided clipping, str.extract DataFrame return)
- Added "Scan for ALL issues, not just the first one" to repair skill
- All examples changed from credit-risk-specific to medical dataset (cross-domain)

### AST inspection (before Run A)
- Added `inplace=True` detection to `inspect_preprocessing_code()` — rejects code before execution

### Markdown fence stripping (before Run A)
- Added regex to `llm.py:call_json_response` to strip ```json fences from LLM output

### LLM quality reviewer (before Run B)
- Added `review_preprocessing_quality()` — calls audit-preprocessing skill
- Added `review-preprocessing-quality` node to graph between validate and train
- Routing: both structural validation AND quality review must pass to proceed to training

### Two-mode review & state alignment (before Run C)
- Removed `cleaned_frame` from audit payload (was confusing reviewer — saw identifier columns)
- Added `preprocessing_report.json` to audit payload (so reviewer knows what was intentional)
- Added `previous_audit_report` parameter (follow-up reviews focus on previous issues, not fresh scan)
- Rewrote audit skill with two modes: comprehensive first review vs focused follow-up
- Added "converge, don't escalate" principle

### MAX_REPAIR_ATTEMPTS increased from 3 to 5 (before Run B)

## Runs

| Run | Log file | Changes since previous | Outcome |
|-----|----------|----------------------|---------|
| A | `stage_full_20260409_110041.log` | Prompt improvements + AST inspection + fence stripping | Failed — 3 repair attempts exhausted |
| B | `stage_full_20260409_121751.log` | Added LLM quality reviewer, MAX_REPAIR=5 | Failed — 5 repair attempts exhausted (quality reviewer kept finding new issues each round) |
| C | `stage_full_20260409_131431.log` | Two-mode review, state alignment fixes | **Passed** — 1 initial + 2 repairs (1 execution fix + 1 quality fix) |

## Results: Run C (first successful full pipeline)

| Metric | Value |
|--------|-------|
| Total LLM calls | 9 |
| Total tokens | 58,436 (input: 49,843, output: 8,593) |
| Total LLM duration | 70s |
| Total pipeline duration | 328s |
| Repair rounds | 2 (1 execution crash + 1 quality review failure) |
| Feature frame shape | 100,000 rows x 72 columns |
| LR macro_f1 | 0.39 |
| RF macro_f1 | 0.67 |
| Selected model | random_forest |

### Preprocessing timeline (Run C)

| Step | What happened |
|------|--------------|
| Generate (attempt 1) | Passed inspection. Execution crashed: `str.extract` with 2 capture groups assigned to single column (Credit_History_Age). |
| Repair #1 (attempt 2) | Fixed str.extract. Execution succeeded. Structural validation passed. Quality review found 4 major issues: Annual_Income max=24M, Num_Bank_Accounts/Num_of_Loan/Num_of_Delayed_Payment negative minimums. |
| Repair #2 (attempt 3) | Fixed all 4 quality issues. Structural validation passed. Quality review passed (72-token focused follow-up confirming all previous issues resolved). |

### Quality review convergence (Run C)

| Review round | Mode | Token output | Issues found | Verdict |
|--------------|------|-------------|-------------|---------|
| 1 | Comprehensive (no previous audit) | 460 tokens | 4 major + 1 minor | needs_repair |
| 2 | Follow-up (with previous audit) | 72 tokens | 0 | pass |

This is exactly the convergence behavior we designed. The follow-up review was focused and terse.

## Observations

### What worked
1. **AST inspection for `inplace=True`** — gpt-4o ignored the prompt instruction for 3 consecutive runs in Run A, but AST rejection forced the repair loop to fix it. Hardcoded enforcement > prompt instruction for mechanical rules.
2. **Two-mode quality review** — Eliminated the whack-a-mole problem from Run B. The reviewer stopped inventing new issues each round.
3. **State alignment** — Removing cleaned_frame from the audit payload and adding preprocessing_report was the critical fix. The reviewer was confused because it saw identifier columns in cleaned_frame and flagged them, even though they were correctly dropped before feature_frame.
4. **Cross-domain examples** — No evidence of answer leakage from the medical dataset examples. The LLM correctly identified credit risk-specific patterns from the data, not from the prompt.
5. **str.get_dummies for Type_of_Loan** — Multi-value delimiter handling worked correctly (9 individual loan types + 9 "and X" artifacts).

### What broke or was missed
1. **One-sided clipping** — Num_Bank_Accounts (max=1798), Num_Credit_Card (max=1499), Interest_Rate (max=5797), Num_of_Loan (max=1496), Num_of_Delayed_Payment (max=4397), Num_Credit_Inquiries (max=2597). Only lower bounds clipped, upper bounds left uncapped. Quality reviewer caught some but not all.
2. **Garbage values encoded as categories** — `Occupation________` (underscore padding) and `Payment_Behaviour_!@9#%8` (noise string) became one-hot columns instead of being treated as NaN.
3. **"and X Loan" delimiter artifacts** — Type_of_Loan split created 9 extra columns like "and Auto Loan" because raw data has entries like "Auto Loan, and Credit-Builder Loan". The "and " prefix wasn't stripped.
4. **No feature scaling** — LR convergence warning, LR accuracy 0.55 (should be ~0.65+ with scaling).
5. **accuracy=0.0000 in logs** — Not a model bug. The `compute_multiclass_metrics()` function doesn't compute accuracy, but `graph.py` logs `metrics.get("accuracy", 0)` which returns the default 0.
6. **Quality reviewer scope too narrow** — Only caught 4 of 10+ data quality issues. Missed encoding artifacts, one-sided clipping, garbage categories.

### Run B failure analysis (quality reviewer whack-a-mole)

The quality reviewer ran 5 rounds and kept finding new issues each time because:
1. It saw `cleaned_frame` which legitimately had identifier columns → flagged them → repair removed them → broke other things
2. No memory of previous issues → fresh comprehensive review each round → found different things due to stochastic LLM output
3. Didn't see `preprocessing_report` → couldn't distinguish intentional decisions from bugs

Root cause: **state misalignment** between what the reviewer saw and what the code actually did. Fixed in Run C by aligning the payload to only feature_frame + preprocessing_report + previous audit.

## Insights

1. **Prompt engineering has diminishing returns for mechanical rules.** `inplace=True` ban in the prompt was ignored 3 times. AST enforcement fixed it instantly. Lesson: use hardcoded enforcement for binary rules, LLM reasoning for nuanced judgment.

2. **LLM reviewers need the same context as the code.** The reviewer saw a different artifact (cleaned_frame) than what the code produced (feature_frame). This mismatch caused confusion. Every LLM node must see the artifact it's judging, plus context about what produced it.

3. **Convergence requires memory.** Without the previous audit report, the reviewer did independent reviews that drifted stochastically. With it, the follow-up was focused and converged in one round.

4. **Data quality review is necessary but insufficient.** The reviewer caught 4 out of 10+ issues. It's better than nothing (structural validation caught 0 of these), but not comprehensive. The reviewer prioritizes obvious distribution issues and misses encoding artifacts.

5. **The repair skill's "fix all issues in one pass" instruction works for crash bugs but not for quality issues.** The repair got all 4 flagged quality issues in one pass. But it couldn't fix issues the reviewer didn't flag.

## Next steps

See `lab/backlog.md` for the full list. Highest priority:
1. Fix the accuracy logging bug (missing key in compute_multiclass_metrics)
2. Add feature scaling (StandardScaler in Pipeline) to training
3. Improve quality reviewer to catch encoding artifacts and one-sided clipping
4. Experiment with temperature=0 for audit calls to reduce stochastic variation
5. Test whether reasoning models (o1/o3) produce better preprocessing code
