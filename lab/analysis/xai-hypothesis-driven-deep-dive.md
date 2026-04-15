# XAI Hypothesis-Driven Deep Dive: Design Discussion

**Date**: 2026-04-13
**Context**: Review of Carson's XAI Case Study (Melbourne housing) and gap analysis against our credit risk pipeline. This document captures the full discussion that shaped our XAI redesign philosophy.

## Origin

Carson built an XAI case study on a Melbourne housing price notebook, covering:
- EDA (Sections 4–5): univariate + bivariate analysis with forward predictions about model behavior
- Feature preparation (Section 9): leakage-safe preprocessing pipeline
- Training diagnostics (Sections 14–17): actual vs predicted, residuals, learning curves, per-segment performance
- Global XAI (Sections 21–29): PFI (individual + grouped), Gain vs PFI, PDP, ALE, PDP vs ALE comparison for Distance and Rooms
- Local XAI (Sections 31–40): SHAP beeswarm, waterfall (under-predicted / representative / over-predicted), LIME comparison, SHAP dependence × type, ICE per type

The case study demonstrates a **4-layer XAI architecture** where EDA hypotheses flow through training diagnostics through global XAI through local XAI, forming a coherent reasoning chain. Our pipeline currently has major gaps at every layer.

---

## The 4-layer architecture and our gaps

### Layer 1: EDA as hypothesis generation

**What the case study does well:**
- Treats every EDA observation as a forward prediction about model behavior
- Example: "Landsize is extremely skewed → linear correlation will be misleadingly weak → but tree models will still find nonlinear signal → PFI will later show Landsize as important despite weak correlation"
- Connects univariate distributions to expected model failure modes (rare luxury homes → higher prediction error)
- Uses multiple statistical views (correlation, distribution, segmentation)

**What the case study doesn't do well enough:**
- Hypotheses are still too generalized and safe — "Landsize might matter through interactions," "segment-specific models may help"
- Not directional enough to predict model selection or specific XAI outcomes
- Not exhaustive on EDA approaches (no class-conditional distributions, no interaction screening, no deep MNAR analysis)

**Our current pipeline gap:**
- EDA node computes MI, ANOVA, skewness, correlations, MNAR detection — but outputs pure numbers
- No forward hypotheses about what model should learn, where it should struggle, and why
- Downstream nodes (FE, training) receive statistics but no reasoning about what those statistics predict

**What bold EDA hypotheses look like for credit risk:**

On model selection:
- "Total_EMI_per_month and Monthly_Inhand_Salary have MI > 0.5 while the next feature drops to 0.17. This 3x gap means two features carry the majority of class-separating signal. A linear model should capture most of this because the top-2 features are continuous and likely near-monotonic with risk. Prediction: LR should achieve at least 0.55 macro_f1 from these two features alone. If LR is much lower than that, the signal is interaction-dependent, not main-effect-driven."
- "5 features have |skew| > 2, ANOVA and MI rankings disagree (Delay_from_due_date: F=11471 but MI=0.12; Credit_Utilization_Ratio: F=105 but MI=0.0). This disagreement means different features separate different class pairs. A tree model that can learn class-specific splits should dominate LR which treats all class boundaries simultaneously. Prediction: XGBoost will outperform LR by >15pp."

On where the model will struggle:
- "Good class has 17,828 samples vs Standard's 53,174 (3:1 ratio). Good is the minority class AND it sits between Standard and Poor on many features (it's the 'premium' end). Prediction: the confusion matrix will show the most error on Good↔Standard, not Poor↔Standard, because Good has less support AND its feature profile partially overlaps Standard."

On feature behavior:
- "Credit_Utilization_Ratio has MI=0.0 but ANOVA F=105. These can't both be true unless MI is a measurement artifact. The column was clipped to [22.74, 42.45] — a narrow 20-point range. On clipped data, MI estimators underperform because the binning can't resolve fine differences. Bold prediction: Credit_Utilization_Ratio will appear in SHAP top-10 despite MI=0, proving the MI estimate was misleading."

Additional EDA approaches to consider:
- **Class-conditional distributions**: for each feature, compute distribution per credit score class — directly shows which features separate which classes
- **Interaction screening**: MI is univariate; compute MI for feature pairs or check if top-2 features' ratio is more discriminative than either alone
- **Deep MNAR analysis**: Type_of_Loan is 11.4% MNAR — why data is missing can itself be a risk signal

### Layer 2: Training diagnostics as hypothesis testing

**What the case study does well:**
- Learning curves diagnose overfitting vs underfitting
- Per-segment performance reveals where the model is weakest (units R²=0.744 vs houses R²=0.854)
- Residual analysis shows error is concentrated in expensive homes
- Connects training observations back to EDA predictions

**What could be added:**
- Bold exploratory hypotheses about subgroup training or alternative models

**Our current pipeline gap:**
- Reports best_cv_score and per-model trial histories
- No residual analysis (where is the model wrong?)
- No per-class performance breakdown with reasoning (which credit score segments are harder?)
- No learning curve interpretation (is the model capacity-limited or feature-limited?)

**What bold training hypotheses look like for credit risk:**

After training (before XAI):
- "LR at 0.618 with C=0.016 (heavy regularization) is capacity-limited. The features are information-rich but the decision boundaries are nonlinear. Adding more linear features won't help — the ceiling is the model class, not the feature set."
- "RF max_depth=27 winning means the signal lives deep in interaction chains. This is unusual — it suggests credit risk scoring is genuinely interaction-heavy, not a simple threshold problem."
- "XGB didn't early-stop at 1000 rounds. This means the model is still finding useful structure at round 1000. Combined with macro_f1=0.802, the model isn't overfitting yet — it's still learning. The next marginal gain is likely from more rounds, not more features."

After XAI (informed by what the model actually learned):
- "If SHAP shows that the top-5 features are all continuous and monotonic, then a simpler model with careful feature engineering could match XGBoost. We should test whether LR on just the top-5 SHAP features plus their ratios closes the gap."
- "If per-class analysis shows Good is consistently confused with Standard, a hierarchical approach (first separate Poor vs non-Poor, then Standard vs Good) might outperform a flat 3-class model."

### Layer 3: Global XAI with method-awareness

**What the case study does well:**
- Uses four different global XAI methods, each answering a different question:
  - PFI (grouped): How much does the model rely on this feature?
  - PDP: What's the average effect shape?
  - ALE: What's the local-conditional effect shape? (corrects PDP bias)
  - Gain vs PFI: Is tree-internal importance aligned with held-out importance?
- Explicitly reasons about when each method is trustworthy
- PDP vs ALE comparison as a formal diagnostic — the divergence IS the insight
- Shows that PDP bias can go in either direction (understates Distance, overstates Rooms)

**What the case study doesn't do well enough:**
- Conclusions are too safe — "Landsize matters through nonlinear interactions" is true but generic
- Only gives obvious insights that don't tell you something you couldn't have guessed
- Doesn't form bold enough predictions about what XAI will reveal before looking

**Our current pipeline gap:**
- Only does SHAP (global mean_abs_shap top-15 + local waterfall top-5)
- No PFI, no PDP, no ALE, no gain comparison
- Doesn't ask when SHAP might be misleading for our specific feature structure

**Not all methods are needed — task-dependent selection:**

| Method | Include? | Why |
|--------|----------|-----|
| PFI (grouped) | Yes | Critical for one-hot features — honest held-out importance |
| SHAP beeswarm | Yes | Shows direction + heterogeneity across samples |
| SHAP waterfall | Yes | Local explanation for specific cases |
| SHAP dependence | Yes | Feature × class interaction — directly answers "how does this feature affect each credit score?" |
| PDP | Conditional | Only for features we suspect are correlated → baseline to contrast with ALE |
| ALE | Yes where correlated | Our EDA already computes correlations — use that to decide which features need ALE |
| LIME | No | Weak for one-hot tabular tasks (creates impossible feature combinations) |
| ICE | Maybe | Only if meaningful subgroups exist — ICE per Occupation or Credit_Mix could be valuable |

**Important insight on encoding:** One-hot encoding creates the same problem the case study identifies with LIME — impossible combinations during perturbation, fragmented SHAP importance across many dummies. Target encoding would:
- Give one continuous feature per column → cleaner SHAP, PFI, PDP
- Potentially improve LR (denser feature space)
- Make ALE directly applicable
- But requires careful leakage prevention (leave-one-out or k-fold on train only)

This is itself a testable hypothesis: "Switching from one-hot to target encoding for high-cardinality columns will consolidate SHAP importance into interpretable single-feature signals, improve LR by >3pp, and make PDP/ALE directly applicable."

**What bold Global XAI hypotheses look like for credit risk:**

- "EMI_to_salary_ratio should have a sharp threshold in SHAP dependence around 0.4–0.5 (the financial distress boundary). Below that threshold, credit quality is primarily determined by other factors. Above it, almost everyone is Poor. If this threshold exists, the model learned a real-world financial breakpoint, not just a statistical correlation."
- "The PDP for Interest_Rate should show a non-monotonic curve. Low interest rates indicate good creditworthiness (confirmed by being offered low rates). High interest rates indicate either risk-adjusted pricing OR desperate borrowing. The model should learn this U-shape or inverted-V. If PDP shows monotonic, the model is oversimplifying."
- "ALE for Monthly_Inhand_Salary should diverge from PDP because salary is correlated with Occupation (label-encoded). PDP will understate the salary effect by averaging over unrealistic salary-occupation combinations. Prediction: ALE gap > PDP gap by at least 20%."
- "SHAP dependence for Delay_from_due_date should show dramatic asymmetry: 0 days delay → strong positive SHAP (Good signal), 1–5 days → moderate negative, 20+ days → large negative. If this step-function shape exists, the most important credit signal is binary: 'has this person EVER been late?'"

### Layer 4: Local XAI with case selection

**What the case study does well:**
- Picks three cases by design: under-predicted, representative, over-predicted
- Reads waterfalls as "what is the model's reasoning AND what is it blind to"
- SHAP dependence × property type shows interaction at the local level
- Representative case explanation is especially insightful — shows normal model reasoning

**Our current pipeline gap:**
- Explains exactly one row (row_index=42) with top-5 SHAP
- No case selection strategy
- No comparison across segments
- No interaction analysis

**What bold Local XAI hypotheses look like for credit risk:**
- "This Poor-classified customer has high EMI_to_salary_ratio (+0.34 SHAP) and high Delay_from_due_date (+0.28 SHAP). The model reads this as a classic distressed borrower — overleveraged and consistently late. But the raw data shows Credit_Mix=Good and 15 years credit history. Exploratory hypothesis: this customer may be experiencing a temporary financial shock (job loss, medical expense) rather than chronic poor management. The current feature set can't distinguish temporary vs chronic distress, but this distinction would matter enormously for the right business action — restructure vs decline."

---

## The core philosophical shift: bold hypotheses that don't need to close

### The problem with the initial framing

The initial analysis framed bold hypotheses as things that get "validated or invalidated" — closed loops. Carson pushed back: **not everything needs to close.** Some hypotheses are valuable precisely because they open doors, even if the current data/pipeline can't definitively test them.

### Two fundamentally different analytical postures

**Academic rigor says:** "We hypothesized X, tested it, found Y. Conclusion: Z."

**Exploratory analysis says:** "We observed X, which suggests Y and Z. Y we can test now — here's what we found. Z we can't fully test yet, but here's why it matters and what it would take to test it. Don't discard Z just because we can't close it today."

This is how real analysis works in industry. A credit risk analyst who only reports what they can prove with p<0.05 is less useful than one who says "I see a pattern that suggests high-salary customers with multiple loan types behave fundamentally differently — I can't prove it from this data alone, but here's the signal and here's what we'd need to validate it."

### The key insight

**A hypothesis that turns out to be wrong is still an insight. A hypothesis that you never propose is a missed opportunity.** You're trading precision for coverage, and in exploratory analysis, coverage wins.

The value is not in being right — it's in opening lines of inquiry that others can pursue. A wrong hypothesis that points to an interesting question is more valuable than a correct observation that tells you nothing new.

### Guardrails that still matter

**Propose freely, but label honestly.** The danger isn't in proposing bold hypotheses — it's in presenting speculative hypotheses with the same confidence as tested ones. Three tiers:

1. **Tested hypothesis:** "We predicted EMI_to_salary_ratio would rank top-5 in SHAP. It ranked #3. Confirmed."

2. **Supported but not closed:** "SHAP dependence shows a steeper gradient for Delay_from_due_date in the 0–5 day range than 20–40 day range. This is consistent with the hypothesis that 'ever been late' is a stronger signal than 'how late.' We can't fully separate this from the clipping effect, but the pattern is suggestive."

3. **Exploratory / open:** "The model struggles most with Good↔Standard confusion. One explanation is that these two classes share similar financial profiles but differ on behavioral patterns not well captured by our features (e.g., spending patterns, account tenure dynamics). If this is true, adding temporal behavioral features would disproportionately improve Good-class recall. We can't test this with the current dataset."

All three tiers are valuable. The labeling prevents stakeholders from treating speculation as finding, but doesn't prevent surfacing the speculation.

**Don't fabricate signal that isn't there.** Bold hypotheses should be grounded in *something* observable — an EDA pattern, a SHAP asymmetry, a confusion matrix concentration. "I wonder if X" without any grounding is noise. "I see Y in the data, which makes me think X could be happening because Z" — that's a bold hypothesis worth proposing even without closure.

**Don't contradict established findings.** If SHAP clearly shows Feature A is unimportant across all methods, don't propose "Feature A might secretly be the most important driver." Bold means going beyond the safe interpretation, not against the evidence.

### Why this works for a cutting-edge exploratory project

This is not an academic project requiring rigorous closed-loop validation. This is a cutting-edge exploration where:
- The goal is to discover the richest set of insights, not to prove a specific thesis
- Some directions will be misleading or wrong — that's expected and acceptable
- Missing a genuine insight is worse than proposing one that doesn't pan out
- Multiple perspectives (even conflicting ones) give stakeholders more to work with than a single safe narrative
- "We noticed X and it made us think Y, but we couldn't validate it yet" is itself valuable output

Safe conclusions demonstrate competence. Bold hypotheses demonstrate understanding.

---

## Revised concrete plan

### EDA node
- Computes same statistics, but LLM reasoning now produces **three tiers of hypotheses**:
  - **Structural predictions** (testable in this pipeline): "XGBoost should beat LR by >10pp because of interaction structure"
  - **Supported conjectures** (partially testable): "Credit_Utilization_Ratio's MI=0.0 is likely a measurement artifact from narrow clipping range — expect it to appear in SHAP despite zero MI"
  - **Exploratory leads** (not testable now but worth surfacing): "Type_of_Loan missingness (11.4% MNAR) may itself be a risk signal — customers who don't disclose loan types may have different risk profiles"
- Additional EDA approaches: class-conditional distributions, interaction screening, deep MNAR analysis

### Training diagnostics node (new)
- Per-class precision/recall/f1 breakdown with reasoning about WHY certain classes are harder
- Confusion matrix analysis: which class pairs are most confused and why
- Learning curve interpretation: capacity-limited vs feature-limited
- Generates training-informed hypotheses at all three tiers

### Global XAI node (expanded)
- PFI (grouped) + SHAP beeswarm + SHAP dependence for top features
- ALE where EDA correlations warrant it
- PDP vs ALE comparison as diagnostic where applicable
- Each method selected based on what it uniquely reveals
- Each output generates hypotheses at all three tiers
- Bold interpretations: "The divergence between PDP and ALE for salary means X about the underlying data-generating process"

### Local XAI node (expanded)
- Case selection by design: per-class representative + worst misclassification per class
- Waterfall explanations interpreted as "what story is the model telling + what might be missing"
- SHAP dependence interactions for top feature × subgroup
- Exploratory hypotheses about individual cases that open investigation threads

### Hypothesis chain in explain node
- **Closed loops** where we can: "EDA predicted X, XAI showed Y, confirmed/refuted"
- **Open threads** where we can't: clearly labeled as "exploratory insight — grounded in [specific observation], would require [specific additional data/test] to validate"
- **Future directions** that naturally emerge: specific, grounded, actionable hypotheses that a stakeholder could actually pursue
- Bold hypotheses that failed or couldn't be tested are explicitly discussed — not discarded

---

## Generalizing for risk domain problems

For any risk classification problem (credit, fraud, insurance, operational risk):

1. **Risk signals are often threshold-driven, not smooth.** Financial distress has breakpoints, not gradients. EDA should identify candidate thresholds, XAI should test them.

2. **Feature correlations carry domain meaning.** Salary correlates with occupation, EMI correlates with loan type. These aren't nuisance correlations to correct — they're part of the risk story. ALE separates the marginal effect; the residual (PDP − ALE) tells you about the correlation structure itself.

3. **Class imbalance means per-class analysis is mandatory.** Global metrics hide which risk segment the model serves worst. For credit risk, misclassifying a Poor customer as Good has different business impact than misclassifying Good as Standard.

4. **Explanation must be stakeholder-specific.** A regulator wants fairness across subgroups, a credit officer wants feature-level justification for individual decisions, a risk manager wants portfolio-level reliability. The same XAI outputs serve different audiences with different framing.

5. **Hypothesis-driven XAI is more defensible than method-driven XAI.** "We used SHAP because it's popular" is weak. "We used ALE because EDA showed salary and occupation are correlated, and we needed to separate the marginal salary effect from occupational confounding" is strong.

---

## Implementation order

1. Training speed fix (`n_jobs=-1`) — unblocks fast iteration
2. EDA hypothesis generation — the foundation everything builds on
3. Training diagnostics node — per-class, learning curves, residual analysis
4. Global XAI expansion — PFI, PDP/ALE where warranted, SHAP dependence
5. Local XAI with case selection — representative + worst per class
6. Hypothesis chain in explain node — closed + open + exploratory threads
