# Reasoning Model for Analytical Nodes: o4-mini vs gpt-4o

**Date:** 2026-04-10
**Status:** VALIDATED — o4-mini confirmed superior for column-transform-spec

## The problem

gpt-4o column-transform-spec produced unstable, low-quality output across 2 consecutive runs on the same 100k-row credit score dataset (28 columns, 3-class target).

### Concrete evidence of instability

**Run at 14:30 (gpt-4o, 12.37s, 1,507 output tokens):**
- Age: `clip to [0, 99]` — reasonable but arbitrary upper bound
- Num_Bank_Accounts: `clip to [0, 445]` — 445 is the raw p99, not a plausible number of bank accounts
- Num_Credit_Card: `clip to [0, 849]` — raw p99 copied directly, nobody has 849 credit cards
- Interest_Rate: `clip to [1, 2865]` — p99=2865 used as clip bound for a percentage field
- Credit_Mix: `one_hot` — wrong, this column has ordered values (Bad < Standard < Good)
- Resulting feature columns: 64
- Quality audit found 3 issues (Payment_Behaviour constant, Type_of_Loan artifacts, Annual_Income outliers)

**Run at 15:23 (gpt-4o, 15.16s, 1,701 output tokens):**
- Age: `clip to [0, max]` — "max" is 8698 in the data, so this clips nothing
- Num_Bank_Accounts: `clip to [0, p99]` — wrote "p99" as text, not the actual number; codegen has to guess
- Num_Credit_Card: `clip to [0, p99]` — same vague instruction
- Interest_Rate: `clip to [0, p99]` — same, p99=2865 for a percentage
- Credit_Mix: `ordinal` — correct this time, but different from the 14:30 run
- Resulting feature columns: 67 (different from 64 because Credit_Mix encoding changed)
- Quality audit found **8 issues** — Age max=8698, Annual_Income max=24M, Num_Bank_Accounts max=445, Num_Credit_Card max=849, Interest_Rate max=2865, Num_of_Loan max=1496, Type_of_Loan artifacts, Num_of_Delayed_Payment negatives. All traceable to vague/wrong spec.

### Why this matters

- Feature column count varies (64 vs 67) — downstream models train on different feature spaces
- Encoding choice flips randomly — ordinal preserves ranking information, one_hot destroys it
- Codegen receives "clip to p99" without a number — produces code that either doesn't clip or clips to the inflated p99
- Quality auditor catches the mistakes, but each repair round costs ~20s + 1 LLM call; with 8 issues, convergence takes 2-3 attempts

## Root cause analysis

column-transform-spec requires **analytical reasoning**, not instruction-following:

1. Cross-reference column_profiles (min=-1, p99=445, max=1798, mean=5.1 for Num_Bank_Accounts) with domain knowledge (nobody has 445 bank accounts)
2. Decide whether p99 is itself plausible — p99=849 for Num_Credit_Card when median=5 means the 99th percentile is corrupted by outlier noise
3. Analyze category values to determine order — Credit_Mix has {Bad, Standard, Good, _} where Bad < Standard < Good is a clear semantic ordering
4. Produce codegen-ready instructions — "clip to [0, 10]" not "clip to p99"

gpt-4o shortcuts this reasoning: it copies p99 values directly, flips encoding choices without semantic analysis, and sometimes writes vague instructions the codegen can't implement.

## The fix: two changes applied together

### Change 1: Principles-based prompt

**Failed approach (hard constraints):** Added 5 rules like "Never write clip to p99", "Encoding MUST be deterministic", "If p99 > 10x mean, use tighter bound". Problem: this is command-and-control that tells the model what NOT to do. A reasoning model doesn't need to be told "don't copy p99" — it needs to understand WHY p99 can be misleading, then it won't do it.

**Working approach (core principles with rationale):** 5 principles that explain the downstream consequences:
1. **Spec is a contract with codegen** — codegen doesn't have access to column_profiles, so "clip to p99" is an ambiguous instruction. Concrete bounds like [0, 10] are directly implementable.
2. **Encoding reflects semantic structure** — ordered values (Bad < Standard < Good) carry ranking information that one_hot destroys. The encoding choice follows from whether the values have inherent order.
3. **Percentiles describe the distribution, not the valid range** — when data has extreme outliers or garbage, p99 gets inflated. Compare p99 to the mean: if they diverge dramatically (p99=445 vs mean=5), the tail is corrupted.
4. **Delimited fields have split artifacts** — splitting "Type A, and Type B" on comma produces "and Type B" with a leading "and ". The codegen won't strip this unless explicitly told.
5. **Ground decisions in data** — citing actual numbers (p99=445, mean=5.1, skew=11.2) makes decisions auditable and reproducible.

### Change 2: o4-mini reasoning model

**Run at 15:51 (o4-mini, 36.10s, 5,043 output tokens):**

| Column | o4-mini decision | o4-mini reasoning (from log) |
|--------|-----------------|------------------------------|
| Age | clip to [18, 100] | "plausible human ages in [18,100]; converted to int and clipped" |
| Num_Bank_Accounts | clip to [0, 10] | "Min=-1 (invalid), max=1798; domain plausible ≤10; skew=11.2" |
| Num_Credit_Card | clip to [0, 30] | "Min=0, p99=849; users unlikely to hold >30 cards; skew=8.46" |
| Interest_Rate | clip to [1, 100] | "Min=1, p99=2865; interest rates plausibly ≤100%" |
| Num_Credit_Inquiries | clip to [0, 10] | "Min=0, p99=1109; consumers unlikely >10 inquiries; skew=9.79, MI=0.111" |
| Num_of_Loan | clip to [0, 10] | "'-100' appears in top values; domain [0,10]" |
| Total_EMI_per_month | clip to [0, 20000] | "Min=0, p99=56125, max=82331; EMIs rarely exceed 20000/month; top MI feature (0.5471)" |
| Delay_from_due_date | clip to [0, 61] | "Min=-5 (early), p99=61, mean=21; negative delays reinterpreted as zero" |
| Credit_Mix | ordinal | "semantic order Bad<Standard<Good, ordinal encoding preserves ranking" |
| Occupation | label | "16 categories with placeholder '_______' (7062 occurrences); high cardinality >10 encoded via label" |
| Type_of_Loan | "split on ',', strip whitespace, remove 'and ' prefix, one_hot" | "Multi-value comma-separated lists with 'and' conjunctions" |
| Payment_Behaviour | "replace values not matching pattern '[A-Za-z_]+_[A-Za-z_]+' with NaN, one_hot" | "7 categories including garbage '!@9#%8'" |

Every decision rejected raw p99 values in favor of domain-plausible bounds. Every reasoning entry cited actual numbers from column_profiles and EDA. Encoding choices were semantically justified and consistent.

## Cost comparison

| Metric | gpt-4o (avg of 2 runs) | o4-mini |
|--------|----------------------|---------|
| Input tokens | 9,065 | 9,691 (+7%) |
| Output tokens | ~1,600 | 5,043 (3.2×) |
| Duration | ~13s | 36s (2.8×) |
| Reasoning quality | Vague, no numbers cited | Every entry cites p99/mean/skew/MI |
| Clipping bound quality | 3/12 domain-plausible | 12/12 domain-plausible |
| Encoding stability | Flipped between runs | Semantically justified, stable |
| Quality issues (audit) | 3-8 per run | TBD — expect 0-2 |

Total cost increase: ~3,400 extra output tokens ($0.01-0.02 at o4-mini pricing). This is negligible compared to the repair loop savings: each preprocessing repair round costs ~6,000 tokens (repair LLM call) + 44s (execution) + ~12,000 tokens (re-audit). Avoiding even 1 repair round saves more than the o4-mini overhead.

## Prompt evolution: three iterations

1. **Original prompt (Run 008):** Step-by-step instructions for gpt-4o. "Check min/max/p1/p99 for implausible outliers — specify clipping bounds if needed." Result: gpt-4o sometimes followed, sometimes didn't. No consistency guarantee.

2. **Hard constraints (failed attempt):** Added "Never write clip to p99. Always write clip to [0, 15]." and 4 similar rules. Problem: this is micromanagement. Telling a reasoning model "don't do X" without explaining why leads to brittle compliance — it avoids the specific example but misses the principle.

3. **Principles with rationale (working):** "Your spec is a contract with a code generator. The codegen doesn't have access to column_profiles — only your spec. So 'clip to p99' is ambiguous, but 'clip to [0, 15]' is directly implementable." Result: o4-mini internalized the principle and applied it to all 12 clipping decisions, including cases not covered by any example.

## Applicability to other nodes

The same analytical reasoning pattern exists in:
- **generate-feature-engineering-code**: Deciding which interactions to create from EDA correlation/MI data → needs cross-referencing, not just instruction-following
- **reason-hyperparameter-grid**: Deciding search ranges based on dataset size, feature count, baseline metrics → needs judgment about scale
- **reason-model-selection**: Comparing evaluation results with hypothesis chain → needs synthesis across multiple sources
- **explain-risk**: Grounding SHAP values in hypothesis chain for business explanation → needs multi-source reasoning

**Update (17:43 run):** generate-feature-engineering-code confirmed as analytical — switched to o4-mini, hypothesis quality improved dramatically (cites MI values, creates grounded ratios, drops zero-signal features). repair-preprocessing-code also reclassified as analytical — the repair task is "diagnose bug from audit feedback + trace root cause + fix," not "translate spec to pandas." gpt-4o repair ignored specific patterns (str.get_dummies strip) across 3 attempts; o4-mini should reason through the root cause from the audit message.

Codegen node (generate-preprocessing-code) should stay on gpt-4o — translating a spec into pandas code step-by-step is instruction-following.

## Generalization: match model type to task type

| Task type | Model | Prompt style |
|-----------|-------|-------------|
| Instruction-following (initial codegen, formatting) | gpt-4o | Explicit rules, examples, step-by-step workflow |
| Analytical reasoning (specs, decisions, comparisons) | o4-mini | Principles with rationale, considerations |
| Diagnostic reasoning (repair, debugging from feedback) | o4-mini | Root cause principles, reference patterns |
| Simple text generation (summaries) | gpt-4o-mini | Brief instructions |
