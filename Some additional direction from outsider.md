# Where the codebase is getting in its own way:

The highest-risk engineering issue is generated-code execution. preprocess.py explicitly says its AST inspection is “a lightweight denylist check, not a full sandbox,” and the generated preprocessing / feature-engineering code is then written out and executed in a subprocess with timeouts. That is workable for a project prototype, but it is not a safe execution boundary and it is also fragile operationally.
The most important ML-correctness issue I found is a policy mismatch before tuning. In graph.py, the baseline metrics used to feed the LLM grid reasoner are computed with StratifiedKFold via cross_val_score, while train.py has explicit machinery for grouped / temporal validation policy. On a customer-level credit-risk task, that mismatch can distort the LLM’s view of what models and grids look promising.
Too many critical knobs are hardcoded in scattered files instead of centralized config. config.py only exposes the seed and default model, while repair-attempt limits, Optuna trial count, and execution timeouts live elsewhere as literals. That makes controlled experimentation harder than it needs to be.
There is at least one provenance bug in the bundle layer: the bundle metadata hashes the dataset path string rather than the dataset contents. That means two different CSV snapshots at the same path can look identical from the metadata’s point of view.

Precise diff report: what changed, why it hurt, how to fix it
1) The strongest, best-supported regression was Occupation semantics

In the 009-era preprocess run, Occupation was still treated as a real categorical variable: the spec kept it and label-encoded it after cleaning placeholders, so category identity was preserved. In the later problematic run audited in 012, the repo explicitly identifies “Occupation was frequency-encoded at preprocess, erasing category identity.” That same pattern is visible in the April 15 131136 run, where FE logs show Occupation one-hot for one view but frequency_encoding for the tree view. In the later 173718 rerun, the spec changes direction and explicitly says to defer encoding for Occupation; tree-view feature count rises from 37 to 39 and XGBoost macro-F1 improves slightly from 0.6862 to 0.6902. That is the clearest sign that the repo is moving back toward the right representation.

Why this hurt: frequency encoding collapses identity into prevalence. Two occupations with the same frequency become numerically indistinguishable, even if their risk profiles differ. That is fine for some high-cardinality cases, but here the repo itself says there are only about 16 real job titles plus garbage placeholders, which is small enough that preserving identity matters more than compactness.

How to fix it:
Use a stable, semantics-preserving split by view: keep raw cleaned Occupation through preprocess; in linear_view, one-hot it; in tree_view, use either one-hot as well or a safer categorical handling strategy that preserves category identity instead of collapsing it to frequency. Also add an explicit missing/placeholder flag for _______, because that missingness may itself carry signal.

2) The second strongest regression was Credit_History_Age precision collapse

The 012 audit is very explicit: a greedy regex in preprocessing collapsed Credit_History_Age to just 34 unique values, versus 405 unique values in Run 009. That is exactly the kind of silent representation damage that can wreck a model without breaking the pipeline. In the later 131136 and 173718 runs, the column spec now explicitly says to parse "X Years and Y Months" into total months, which is the correct direction. But in the currently committed latest logs, I do not see a post-transform uniqueness check proving the recovered feature actually kept high granularity.

Why this hurt: collapsing a duration field from hundreds of distinct month values down to coarse multiples of 12 destroys within-year variation. In credit behavior, “2 years 1 month” and “2 years 11 months” are no longer meaningfully different if both end up as the same bucket pattern. That is a direct information loss, not just noise.

How to fix it:
Make this deterministic and testable. Use an explicit regex with separate year and month capture groups, then add a validator that fails the run if Credit_History_Age uniqueness falls below a floor such as 100. Your own verification plan in 012 already points in this direction.

3) Type_of_Loan got structurally better, but you are still leaving signal on the table

In the 009-era preprocess run, Type_of_Loan was still handled as a cleaned label-encoded field. In the later runs, that changed to the right structural idea: split on commas and and, then multi-hot encode membership. That is a better first-principles representation for a set-valued field. But both 131136 and 173718 still explicitly note that the missingness indicator for Type_of_Loan was not implemented, even though EDA repeatedly flags that column as MNAR and hypotheses predict missingness should matter.

Why this hurt: the field has two signals, not one. The first is which loan types are present. The second is whether the loan field itself is absent or underspecified, and your own EDA says that absence is target-correlated. The latest pipeline captures the first signal much better than 009, but still drops the second.

How to fix it:
Keep the current multi-hot representation, but add Type_of_Loan_missing as a separate binary feature. That is a cheap change and is directly supported by your own hypotheses/logs.

4) The raw feature space got smaller, then slightly recovered — but the count itself is not the real issue

The 009-era preprocess log shows 49 feature columns before FE. The April 15 131136 run uses a 37-feature tree view, and 173718 rises to 39. So yes, the latest reruns are operating in a materially narrower tree feature space than the 009-era preprocess output. But that alone does not mean they are worse. The more important point is which semantics disappeared: the damage was not “too few columns,” it was “collapsed category identity” and “collapsed duration resolution.”

So the correct principle is not “go back to more features.” It is “only compress when the compression preserves the business meaning of the original variable.” On this repo’s evidence, the 173718 run is a partial recovery in that direction, but not all the way back.

5) A few other representation shifts are plausible secondary drags, but less strongly proven

In the 009-era preprocess run, Month was one-hot encoded. In later runs it became ordinal. Likewise, several later specs allow looser numeric ranges than the tighter 009-era bounds on fields like income and count variables. Those changes can plausibly hurt robustness by imposing fake ordinal distance on months or by leaving more corrupted tail values alive, but the repo evidence does not support them as strongly as the Occupation and Credit_History_Age regressions. I would treat these as benchmark-worthy suspects, not proven root causes.

6) What the newest completed rerun actually tells us

173718 is directionally better than 131136: it preserves more semantics, expands tree-view features from 37 to 39, and nudges XGBoost macro-F1 from 0.6862 to 0.6902. But it is still far below the repo’s Run 009 benchmark of about 0.8017–0.802. Also, the analysis bundle metadata regressed in one odd way: 131136 stores human-readable class names (Good, Poor, Standard), while 173718 stores 0/1/2. That does not directly hurt model quality, but it is a real explanation-layer regression because downstream artifacts become less legible.

Bottom-line diagnosis

The most-supported explanation for the gap from Run 009 to the latest reruns is:

Occupation semantics got compressed too aggressively.
Credit_History_Age suffered a serious parsing/precision collapse in the bad intermediate generation.
Type_of_Loan structure improved, but its MNAR missingness signal is still not fully modeled.
The latest reruns are partly repairing this, but the repairs are incomplete.


# Execution Plan

Phase 1 — Add comparability artifacts before changing logic

Right now, the project needs better mechanical observability.

Add two lightweight artifacts to every run:

feature_contract_report.json
column name
source column
role
dtype
null rate
unique count
top sample values
encoding applied
produced in which view (linear_view, tree_view, both)
validation_policy_report.json
splitter family
grouping column
whether leakage check passed
baseline preview split policy
tuning split policy

The reason is simple: your current problem is not “the model failed mysteriously.” The problem is that semantics drifted between runs. So every future run must tell you, in one file, exactly how each important column ended up being represented.

Definition of done: every full or evaluate-stage run leaves behind those two reports in lab/logs/.

Phase 2 — Fix the three highest-confidence semantic regressions

Do these in this order.

2.1 Occupation

Target behavior:

clean placeholders like _______
preserve raw category identity through preprocess
add Occupation_missing_flag
in linear_view: one-hot encode
in tree_view: preserve identity, do not frequency-encode

Why first: this is the clearest regression signal from the audit trail. The later rerun already moved toward deferring encoding, and performance improved slightly afterward.

Tests to add

placeholder cleanup test
missing-flag test
test that tree_view does not contain frequency-encoded Occupation
test that category count after cleaning stays plausible
2.2 Credit_History_Age

Target behavior:

deterministic parser for "X Years and Y Months" → total months
invalid formats become missing, not silently mangled
add uniqueness floor check

Guardrail:

fail validation if unique count is suspiciously low

Why second: this is the most dangerous “silent numeric collapse” bug.

Tests to add

parser unit tests for standard forms
parser unit tests for malformed forms
validation test for uniqueness floor
2.3 Type_of_Loan

Target behavior:

tokenize multiple loan types
multi-hot encode membership
add Type_of_Loan_missing_flag

Why third: the structural representation is already better than the old single-label treatment, but you are still dropping the missingness signal that your own EDA says matters.

Tests to add

tokenization test
multi-hot membership test
missing-flag test
Phase 3 — Make baseline preview use the same validation world as tuning

This is the most important training-layer fix.

Right now the repo says the true validation policy is grouped by Customer_ID. That is the rule the project claims as its leakage protection.

So the implementation rule should be:

the baseline preview shown to the LLM
the tuning split
the evaluation split

must all come from the same policy family.

What to change

In graph.py, remove the separate ad hoc baseline scoring path that uses StratifiedKFold/cross_val_score.

Instead:

expose one shared helper from train.py, something like compute_policy_aligned_baseline_metrics(...)
make that helper use the same split builder or the same grouped splitter family as tuning
log the splitter policy in the baseline section of the run log
pass those grouped baseline metrics into the LLM grid-selection prompt
Why this matters

Otherwise the LLM sees optimistic scores from an easier world, then recommends grids for a harder grouped-validation world. That creates bad upstream reasoning even before Optuna starts.

Definition of done

logs explicitly say baseline preview uses grouped policy
baseline preview scores become closer to tuned grouped-CV scores
no use of plain StratifiedKFold in the preview path for grouped tasks

Phase 5 — Run a narrow ablation ladder, not a giant rerun spree

After the fixes above, do not change five things at once.

Run this exact ladder:

current 173718 behavior as control
control + Occupation fix
control + Credit_History_Age fix
control + Type_of_Loan_missing_flag
control + all three fixes together

For each run, compare:

grouped CV
test macro-F1
class-wise recall
top SHAP features
whether the feature rankings are economically sensible

This tells you which fix is truly moving the needle and which one just sounds plausible.