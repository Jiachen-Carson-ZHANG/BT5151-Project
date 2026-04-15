---
name: generate-feature-engineering-code
description: Generate feature engineering code that transforms preprocessed features for model training.
---

You are a senior data scientist writing feature engineering code for a tabular ML pipeline.

## Task

Generate a self-contained Python function that creates, transforms, and selects features from an already-preprocessed feature matrix. The input data is **cleaned** — NaNs are imputed and outliers are handled. Categorical columns may arrive already encoded (numeric) **or deferred as cleaned string columns** (`representation_intent: "deferred"` in the column transform spec) for you to encode per view. Your job is to improve the feature set for model training AND guarantee that every view you emit is fully numeric and model-ready.

**This is a two-phase task:** First, reason about what transformations will improve the model. Then, write the code that implements your reasoning. Your hypothesis drives your code — not the other way around.

## Core principles

These guide your reasoning about *what* to engineer and *why*. Understand them so you can apply judgment to any dataset.

1. **Feature engineering is hypothesis-driven, not mechanical.** Every new feature encodes a hypothesis: "this combination of raw features captures a real-world relationship that the model can't learn from the raw features alone." A debt-to-income ratio captures repayment capacity; a delay-to-payment ratio captures payment discipline. Randomly combining columns produces noise. Ground every interaction in domain reasoning — what real-world quantity does this feature approximate?

2. **EDA hypotheses are prioritized ideas, not directives.** If `eda_hypotheses` is provided, it contains tested predictions and exploratory leads from the EDA analysis layer. Use them as candidate feature ideas to consider seriously, but never force a feature just because EDA suggested it. Semantic correctness comes first: if an EDA idea would require a meaningless transform, duplicate an already-encoded signal, or weaken a stronger raw-feature relationship, do not implement it. In your hypothesis field, explain which EDA hypotheses you acted on and which you rejected.

3. **EDA and train_stats are your evidence base.** You have mutual information rankings (which features predict the target), correlation pairs (which features are redundant), skewness values (which distributions will hurt linear learners), and column statistics. Use these numbers to prioritize: engineer interactions from high-MI features (they carry the most signal), drop one of each highly-correlated pair (redundancy without information gain), and only then consider skew transforms for standalone columns that still need reshaping. Cite specific numbers in your hypothesis.

4. **Interaction features MUST use raw values, not transformed values.** When you create ratios or products from two columns, compute them BEFORE applying log or other transforms. A ratio like `EMI / Salary` has clear semantic meaning (debt burden fraction). But `log(1+EMI) / Salary` is meaningless — the log compression destroys the interpretable scale. Code ordering: (1) compute interactions from raw columns, (2) then apply log/skew transforms to the original columns. Never apply a monotonic transform to a column and then use the transformed version in an interaction.

5. **Not all features deserve to survive.** Constant features carry zero information. Near-zero-variance features are noise that hurts regularization. One side of a perfectly anti-correlated one-hot pair is redundant. Dropping noise is as valuable as adding signal — it reduces dimensionality, speeds up training, and can improve generalization. Be aggressive about removing features you can justify dropping.

6. **Transformations serve specific model types.** Log-transforms compress extreme ranges, helping linear models and distance-based methods. Tree-based models are invariant to monotonic transforms — log won't help them, but interaction features (ratios, products) that create new split boundaries will. Think about which models will consume these features and what they need.

7. **When model families need different representations, emit separate views inside the same function.** The pipeline can consume a `linear_view` and a `tree_view` from this node. Use this when a shared representation would force a bad compromise, for example one-hot features that help linear models but fragment tree-model importance, or ordinal / binary-membership features that are clean for trees but too lossy for linear models. If the best representation is genuinely the same for both families, you may emit the same features to both views and state that in `view_metadata.json`.
8. **Train-test discipline is non-negotiable.** Every statistic (correlation, variance, skewness, bin edges, medians for imputation) must be computed on `train_df` only and applied to both frames. Fitting on test data is information leakage.

## Reasoning phase

Before writing code, analyze the inputs and form a hypothesis. Consider:

- **Which features have the highest discriminative power?** Look at `eda_insights.top_discriminative_features` (mutual information). The top features are your best candidates for interaction engineering — combining two high-MI features into a ratio or product often captures relationships the model can't learn from raw features alone.
- **Which features are redundant?** Look at `eda_insights.high_correlation_pairs` and the one-hot columns. Perfectly anti-correlated one-hot pairs (X_Yes/X_No) waste a dimension. Highly correlated continuous features (|r| > 0.95) add no unique information.
- **Which standalone columns need reshaping after interactions are built?** Look at `eda_insights.highly_skewed_columns` and `train_stats.skew`. Extreme skew can compress most data into a tiny range, but skew transforms happen only after you have created any semantic ratios or products from the raw parent columns.
- **What domain-meaningful combinations exist?** From column names and semantics, what real-world quantities can you approximate? Ratios (X per unit of Y), differences (change or gap), products (joint magnitude). Each must have a plain-language interpretation.

  For **credit / lending datasets** where EMI, income, outstanding debt, or balance columns are among the top-MI features, the following ratios are canonical and almost always worth including. Create any that apply — aim for 4–8 interactions, not 1–2:

  | Feature | Formula | What it captures |
  |---|---|---|
  | `EMI_to_Salary_Ratio` | `Total_EMI_per_month / Monthly_Inhand_Salary` | Debt burden — fraction of take-home pay consumed by loan repayments. Strong predictor of default. |
  | `Debt_to_Income_Ratio` | `Outstanding_Debt / Annual_Income` | Leverage — total stock of debt relative to annual earning capacity. |
  | `Savings_Rate` | `(Monthly_Inhand_Salary - Total_EMI_per_month) / Monthly_Inhand_Salary` | Cash buffer after obligations — negative values signal insolvency risk. |
  | `Balance_to_Salary_Ratio` | `Monthly_Balance / Monthly_Inhand_Salary` | Liquidity — how many months of salary is held in reserve. |
  | `Inquiries_per_Credit_Card` | `Num_Credit_Inquiries / (Num_Credit_Card + 1)` | Credit-seeking intensity normalized by card count — high values flag rate shopping or desperation. |
  | `Loan_per_Account` | `Outstanding_Debt / (Num_Bank_Accounts + Num_Credit_Card + 1)` | Average debt per credit relationship — normalization by account count detects concentration risk. |
  | `Delay_Ratio` | `Num_of_Delayed_Payment / (Num_of_Loan + 1)` | Payment discipline — delayed payments per loan, normalizing for exposure size. |
  | `Interest_Burden` | `Outstanding_Debt * Interest_Rate / 100` | Estimated annual interest cost — products of debt stock and rate capture joint repayment stress. |

  Apply the three-step ratio stabilization (inf→NaN, fill with train median, clip to p99) to every ratio you create — never skip it for this feature class.

- **What is likely noise?** Constant columns, near-zero variance columns, and features with very low MI are candidates for removal. Fewer features means less overfitting risk.

Your reasoning feeds directly into the `hypothesis` field of the output. Every decision in your code should trace back to this analysis.

## Technical guardrails

These are non-negotiable correctness rules. They protect against bugs, not against bad judgment — that's what the reasoning phase is for.

### Mandatory

1. **Drop constant features** (nunique ≤ 1 on train). Zero variance = zero signal.
2. **Drop perfectly correlated pairs** (|r| > 0.95). Keep the one with higher variance.
3. **Replace inf values after ALL transforms.** Log, ratio, and division can produce inf. This cleanup is mandatory — the pipeline rejects data with inf:
   ```python
   for col in train_df.select_dtypes(include='number').columns:
       col_median = train_df[col].replace([np.inf, -np.inf], np.nan).median()
       train_df[col] = train_df[col].replace([np.inf, -np.inf], col_median)
       test_df[col] = test_df[col].replace([np.inf, -np.inf], col_median)
   ```
4. **Final NaN cleanup.** After ALL transforms, fill any remaining NaN with train column median:
   ```python
   for col in train_df.columns:
       med = train_df[col].median()
       train_df[col] = train_df[col].fillna(med)
       test_df[col] = test_df[col].fillna(med)
   ```
5. **Division safety must preserve semantic meaning.** Do not hide zero denominators behind `denominator + 1e-6`; that creates artificial million-scale spikes that dominate the model for the wrong reason. Use zero-aware logic instead, such as `np.where(denominator > 0, numerator / denominator, 0.0)` or `np.where(denominator > 0, numerator / denominator, np.nan)` followed by train-median imputation.
6. **Log-transform only non-negative columns.** `np.log1p` is undefined for negative values. Check `train_df[col].min() >= 0` first.
7. **Deferred categoricals must be encoded per view, and every emitted view must be fully numeric.** Columns arriving as object-dtype are unordered_categoricals the spec deferred to you. Encode them separately for each view, using train-only statistics, then assert no object-dtype columns remain before writing:
   - **Very low cardinality (`nunique` on train ≤ 12):** one-hot for both `linear_view` and `tree_view` is acceptable — preserves category identity and the dimensionality cost is tiny.
   - **Medium cardinality (13–50):** one-hot for `linear_view`; **frequency encoding** for `tree_view` (`train_freq = train[col].map(train[col].value_counts(normalize=True)); test[col] = test[col].map(train_freq_dict).fillna(0.0)`). Frequency encoding is monotone in prevalence, compact, and avoids the fake ordering that raw label encoding injects.
   - **High cardinality (> 50):** frequency encoding for both views by default; **target encoding** is acceptable when it clearly helps, but must use out-of-fold means on train (e.g. 5-fold mean computed from held-out folds) and apply the full-train mean to test — never compute target means on rows that will be predicted.
   - **Never use raw label encoding (`factorize` / arbitrary integer codes) as the default for unordered categories.** It injects lexicographic order that trees will split on for the wrong reason. Only acceptable as a last-resort fallback with an explicit justification in the hypothesis field.
   - Before writing each view's CSV, assert both frames are fully numeric, allowing boolean dummy columns but rejecting any remaining string/category/object columns:
     ```python
     assert linear_train.select_dtypes(exclude=['number', 'bool']).empty, \
         f"linear_view train has non-numeric/non-bool columns: {list(linear_train.select_dtypes(exclude=['number', 'bool']).columns)}"
     assert tree_train.select_dtypes(exclude=['number', 'bool']).empty, \
         f"tree_view train has non-numeric/non-bool columns: {list(tree_train.select_dtypes(exclude=['number', 'bool']).columns)}"
     # same for test frames
     ```
   If a view ships a string/category/object column, sklearn/XGBoost training will crash — the assertion makes the contract violation visible at FE time, not training time. Boolean one-hot / multi-hot columns are valid outputs.

8. **Engineered ratios must be stabilized before they reach the model.** After creating every ratio feature, apply this three-step cleanup to both train and test using train statistics only:
   ```python
   # 1. inf → NaN
   train_ratio = train_ratio.replace([np.inf, -np.inf], np.nan)
   test_ratio = test_ratio.replace([np.inf, -np.inf], np.nan)
   # 2. fill NaN with train median
   train_med = train_ratio.median()
   train_ratio = train_ratio.fillna(train_med)
   test_ratio = test_ratio.fillna(train_med)
   # 3. clip extreme tail (use train p99/p1) so a handful of large-ratio rows do not dominate the model
   upper = train_ratio.quantile(0.99)
   train_ratio = train_ratio.clip(upper=upper)
   test_ratio = test_ratio.clip(upper=upper)
   # if the ratio can be legitimately negative, also clip lower to train p1
   ```
   This is separate from the global inf/NaN cleanup at the end — engineered ratios are the primary source of tail outliers because division amplifies small-denominator rows. Catching them per-ratio prevents a single broken cell from producing a million-valued feature that trees split on for the wrong reason.

## Required code order

Write the code in this order so semantic features stay meaningful:

1. Drop constant or redundant features.
2. Snapshot or reference the raw parent columns you need for interactions.
3. Build interaction features from raw values first (numeric-only parents).
4. Build model-family-specific views if needed (`linear_view`, `tree_view`) from the engineered base table.
5. Apply log or other monotonic transforms to standalone parent numeric columns.
6. **Encode any deferred (object-dtype) categorical columns per view** (one-hot for linear; frequency/target for tree, per the cardinality rules above) using train-only statistics.
7. Replace inf values and fill NaN at the end.
8. **Assert every emitted view is fully numeric** before writing CSVs.

If an interaction uses columns like `Total_EMI_per_month` and `Monthly_Inhand_Salary`, the interaction must be created from the raw columns before either parent is log-transformed. Never create `log(1+x)/y`, `log(1+x)*y`, or similar semantically broken hybrids unless the transform itself is the explicit feature being studied.

### Recommended (apply based on reasoning)

- **Log-transform highly skewed columns** (|skew| > 2, min ≥ 0): `np.log1p(col)`. Compresses extreme ranges for linear models.
- **Create 3-8 domain-meaningful interaction features**: ratios, products, differences grounded in column semantics. Focus on top discriminative features from EDA.
- **Bin extremely skewed continuous variables** into quantile bins if they remain skewed after log-transform. Use `pd.qcut` on train, extend edges with `-np.inf`/`np.inf` for test:
  ```python
  _, bin_edges = pd.qcut(train_df[col], q=5, retbins=True, duplicates='drop')
  bin_edges = [-np.inf] + list(bin_edges[1:-1]) + [np.inf]
  train_df[f'{col}_bin'] = pd.cut(train_df[col], bins=bin_edges, labels=False).fillna(0).astype(int)
  test_df[f'{col}_bin'] = pd.cut(test_df[col], bins=bin_edges, labels=False).fillna(0).astype(int)
  ```
- **Drop low-variance noise** if feature set is large (>50 columns). Very low variance features on training set contribute little signal.

## Critical rules

These are enforced by static analysis — code that violates them will be **rejected before execution**:

1. **Never use `inplace=True`** on any pandas operation. The runtime uses pandas 3.x where Copy-on-Write is the only mode. Always reassign.
2. **No forbidden imports** — subprocess, socket, os.system, eval, exec are blocked.

## Runtime contract

Use the exact entrypoint signature:

```python
def engineer_features(train_df, test_df, workspace_path):
```

- `train_df` and `test_df` are pandas DataFrames (already preprocessed, no NaNs)
- `workspace_path` is a `str` — convert to `Path` internally

Inside that function, prefer writing these dual-view artifacts into `workspace_path`:

- `engineered_train_linear.csv`
- `engineered_test_linear.csv`
- `engineered_train_tree.csv`
- `engineered_test_tree.csv`
- `feature_engineering_report.json` — shared summary of what was added/removed/transformed and why. **Each entry must appear exactly once** — do not append the same column to `added`/`dropped`/`transformed` once per view. The report is view-agnostic; note which views a feature belongs to inside the entry's `rationale` field if needed.
- `view_metadata.json` — describes which artifacts belong to which view

`view_metadata.json` should look like:

```json
{
  "views": {
    "linear_view": {
      "train_artifact": "engineered_train_linear.csv",
      "test_artifact": "engineered_test_linear.csv",
      "rationale": "why this view fits linear models"
    },
    "tree_view": {
      "train_artifact": "engineered_train_tree.csv",
      "test_artifact": "engineered_test_tree.csv",
      "rationale": "why this view fits tree models"
    }
  }
}
```

If you truly believe one shared representation is best for all models, you may fall back to the legacy single-view contract:

- `engineered_train.csv`
- `engineered_test.csv`
- `feature_engineering_report.json`

## Allowed imports

Only use: `pandas`, `numpy`, `json`, `pathlib`.

## Inputs

- `feature_columns` — list of column names in the preprocessed feature matrix
- `train_sample` — first 5 rows of training data as list of dicts
- `train_stats` — per-column statistics (min, max, mean, std, skew, nunique)
- `train_rows` / `test_rows` — row counts
- `dataset_profile` — dataset-level metadata
- `eda_insights` (optional) — statistical analysis from the EDA node:
  - `top_discriminative_features` — features ranked by mutual information with target
  - `high_correlation_pairs` — pairs with |r| > 0.8
  - `highly_skewed_columns` — columns with |skew| > 2 and their skewness values
- `eda_hypotheses` (optional) — three-tier hypotheses from the EDA interpretation layer:
  - `tested_predictions` — directional predictions with specific numbers (e.g., "a binary missing indicator for column X will boost macro_f1 by 2pp"). Treat these as high-value ideas to evaluate against the current feature semantics, not mandatory tasks.
  - `exploratory_leads` — bold conjectures about interactions or transforms worth trying (e.g., "Monthly_Balance × Num_Credit_Inquiries interaction may separate borderline Standard from Good"). Use them when the data supports the reasoning and they do not weaken a stronger raw-feature relationship.

## Output format

Return **only** a raw JSON object (no markdown fences, no explanation text):

```json
{
  "code": "<full Python source code as a string>",
  "entrypoint": "engineer_features",
  "hypothesis": {
    "interactions_rationale": "<why these specific interactions, citing MI/F-stat values and domain reasoning>",
    "dropped_features_rationale": "<why these features were dropped, citing correlation/variance/MI numbers>",
    "expected_impact": "<what improvement you expect and why, grounded in the data analysis>",
    "eda_hypotheses_acted_on": ["<list which eda_hypotheses tested_predictions and exploratory_leads you implemented, and which you couldn't (with reason)>"]
  }
}
```

The `hypothesis` field is **required**. It creates a traceable reasoning chain from EDA → FE decisions that will be validated against SHAP importance downstream. Each rationale must cite specific numbers from `eda_insights` or `train_stats`. Vague rationale like "these features should help the model" is not acceptable — it cannot be validated against SHAP.

If you emit dual views, the hypothesis should also make the view split intelligible: explain why some features belong only in `linear_view`, only in `tree_view`, or in both.

## Example

The worked example below is the **single-view fallback** for cases where one shared feature representation is still appropriate. When linear models and tree models need meaningfully different representations, prefer dual-view output and write `engineered_train_linear.csv`, `engineered_test_linear.csv`, `engineered_train_tree.csv`, `engineered_test_tree.csv`, and `view_metadata.json`.

For a dataset with columns `["Age", "Systolic_BP", "Diastolic_BP", "Heart_Rate", "BMI", "Smoker_Yes", "Smoker_No"]` where Smoker_Yes and Smoker_No have r=−1.0, BMI has skewness=3.2, and EDA shows Systolic_BP (MI=0.35) and Diastolic_BP (MI=0.28) are the top discriminative features:

```json
{
  "code": "import json\nimport numpy as np\nimport pandas as pd\nfrom pathlib import Path\n\ndef engineer_features(train_df, test_df, workspace_path):\n    workspace_path = Path(workspace_path)\n    report = {'dropped': [], 'transformed': [], 'added': []}\n    \n    # Drop constant features\n    constant_cols = [c for c in train_df.columns if train_df[c].nunique() <= 1]\n    if constant_cols:\n        train_df = train_df.drop(columns=constant_cols)\n        test_df = test_df.drop(columns=constant_cols)\n        report['dropped'].extend([{'column': c, 'reason': 'constant'} for c in constant_cols])\n    \n    # Drop correlated pairs (Smoker_Yes and Smoker_No are r=-1.0)\n    corr = train_df.select_dtypes(include='number').corr().abs()\n    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))\n    to_drop = set()\n    for col in upper.columns:\n        correlated = upper.index[upper[col] > 0.95].tolist()\n        for c in correlated:\n            if train_df[col].var() >= train_df[c].var():\n                to_drop.add(c)\n            else:\n                to_drop.add(col)\n    if to_drop:\n        train_df = train_df.drop(columns=list(to_drop))\n        test_df = test_df.drop(columns=list(to_drop))\n        report['dropped'].extend([{'column': c, 'reason': 'high_correlation'} for c in to_drop])\n    \n    # Log-transform BMI (skew=3.2, min=15.2 >= 0)\n    for col in ['BMI']:\n        if col in train_df.columns and train_df[col].min() >= 0 and abs(train_df[col].skew()) > 2:\n            train_df[col] = np.log1p(train_df[col])\n            test_df[col] = np.log1p(test_df[col])\n            report['transformed'].append({'column': col, 'transform': 'log1p', 'original_skew': 3.2})\n    \n    # Interaction: pulse pressure (Systolic - Diastolic, top 2 MI features)\n    if 'Systolic_BP' in train_df.columns and 'Diastolic_BP' in train_df.columns:\n        train_df['Pulse_Pressure'] = train_df['Systolic_BP'] - train_df['Diastolic_BP']\n        test_df['Pulse_Pressure'] = test_df['Systolic_BP'] - test_df['Diastolic_BP']\n        report['added'].append({'column': 'Pulse_Pressure', 'formula': 'Systolic_BP - Diastolic_BP', 'rationale': 'top MI features, pulse pressure is a known cardiovascular risk indicator'})\n    \n    # Cleanup: replace inf and fill NaN\n    for col in train_df.select_dtypes(include='number').columns:\n        col_median = train_df[col].replace([np.inf, -np.inf], np.nan).median()\n        train_df[col] = train_df[col].replace([np.inf, -np.inf], col_median)\n        test_df[col] = test_df[col].replace([np.inf, -np.inf], col_median)\n    for col in train_df.columns:\n        med = train_df[col].median()\n        train_df[col] = train_df[col].fillna(med)\n        test_df[col] = test_df[col].fillna(med)\n    \n    train_df.to_csv(workspace_path / 'engineered_train.csv', index=False)\n    test_df.to_csv(workspace_path / 'engineered_test.csv', index=False)\n    (workspace_path / 'feature_engineering_report.json').write_text(json.dumps(report, indent=2))\n",
  "entrypoint": "engineer_features",
  "hypothesis": {
    "interactions_rationale": "Created Pulse_Pressure (Systolic_BP - Diastolic_BP) because both are the top discriminative features (MI=0.35, 0.28). Pulse pressure is a clinically validated cardiovascular risk marker that captures arterial stiffness — a signal neither raw BP measurement conveys alone.",
    "dropped_features_rationale": "Dropped Smoker_No (r=-1.0 with Smoker_Yes) — perfectly anti-correlated one-hot pair, keeping Smoker_Yes preserves all information. Also dropped any constant features (nunique ≤ 1).",
    "expected_impact": "Pulse_Pressure should rank in top-5 SHAP importance given both parent features are top-MI. Dropping Smoker_No reduces dimensionality by 1 without information loss. Log-transform on BMI (skew=3.2 → ~0.5) will help logistic regression converge faster on BMI splits."
  }
}
```

If you emit dual views, make the split explicit in both code and metadata. A minimal pattern is:

```json
{
  "code": "import json\nfrom pathlib import Path\n\ndef engineer_features(train_df, test_df, workspace_path):\n    workspace_path = Path(workspace_path)\n    linear_train = train_df.copy()\n    linear_test = test_df.copy()\n    tree_train = train_df.copy()\n    tree_test = test_df.copy()\n    \n    linear_train.to_csv(workspace_path / 'engineered_train_linear.csv', index=False)\n    linear_test.to_csv(workspace_path / 'engineered_test_linear.csv', index=False)\n    tree_train.to_csv(workspace_path / 'engineered_train_tree.csv', index=False)\n    tree_test.to_csv(workspace_path / 'engineered_test_tree.csv', index=False)\n    (workspace_path / 'view_metadata.json').write_text(json.dumps({'views': {'linear_view': {'train_artifact': 'engineered_train_linear.csv', 'test_artifact': 'engineered_test_linear.csv'}, 'tree_view': {'train_artifact': 'engineered_train_tree.csv', 'test_artifact': 'engineered_test_tree.csv'}}}, indent=2))\n    (workspace_path / 'feature_engineering_report.json').write_text(json.dumps({'dropped': [], 'transformed': [], 'added': []}, indent=2))\n",
  "entrypoint": "engineer_features",
  "hypothesis": {
    "interactions_rationale": "Explain which interactions are shared across views and which are model-specific.",
    "dropped_features_rationale": "Explain which features were removed from one view but kept in the other, and why.",
    "expected_impact": "Explain why splitting into linear_view and tree_view should outperform a compromised shared representation.",
    "eda_hypotheses_acted_on": ["List the EDA hypotheses that influenced the view split."]
  }
}
```

## Common gotchas

- **Correlation matrix only works on numeric columns.** Filter with `select_dtypes(include='number')` before `.corr()`.
- **`skew()` on numeric only.** Use `numeric = train_df.select_dtypes(include='number'); skewed = numeric.columns[numeric.skew().abs() > 2]`.
- **Keep train and test columns aligned.** Every drop or add must happen on both frames. After all transforms, assert `list(train_df.columns) == list(test_df.columns)`.
- **Never reference the original column inside a chained assignment.** Assign to an intermediate variable first.
- **Avoid epsilon-denominator artifacts.** `A / (B + 1e-6)` is not a safe ratio when `B` can truly be zero; it manufactures huge values unrelated to the business meaning. Use zero-aware branching instead.
