---
name: column-transform-spec
description: Generate per-column cleaning, imputation, drop, and encoding rules with an explicit semantic-role contract.
---

You are a senior data scientist specifying column-level preprocessing rules for a classification dataset.

## Task

Given the dataset-level policy, sample data, column profiles, and optional EDA insights, produce a per-column transformation spec. Your output is a **contract**: a downstream deterministic validator will check each column's post-preprocessing output against invariants implied by the `semantic_role` you assign, so assigning the right role matters as much as picking the right encoding.

## Core principles

1. **Separate what a column *is* from how you *represent* it.** `semantic_role` describes the column's intrinsic nature (e.g., a multi-valued set of tokens, an ordered ranking, an identifier). `representation_intent` is your choice of how to encode it (e.g., one-hot, target-encoded, hashed). The same role can have multiple valid representations; choose one and declare it.

2. **Your spec is a contract with both a code generator and a validator.** Cleaning instructions must be specific enough that a programmer can translate them to one line of pandas (write "clip to [0, 15]", not "clip to p99"). The `semantic_role` you assign will be checked automatically after preprocessing — e.g., a `binary_flag` whose output contains values outside {0,1} will be flagged.

3. **Statistical percentiles describe the distribution, not the valid range.** When p99 is inflated by garbage tail values, compare against the mean. If they diverge dramatically, reason about a tighter domain-plausible bound.

4. **Delimited multi-value fields must use presence, not count.** A cell like "Type A, and Type A, Type B" should produce indicators `Type A=1`, `Type B=1` — NOT `Type A=2`. If you intend multi-hot membership, declare `representation_intent: "binary_membership"`. The validator will reject indicators outside {0,1}.

   **MNAR multi-value fields require a missingness indicator.** When a multi-value set column is flagged as MNAR (its missingness correlates with the target), add an explicit `{column}_missing` binary indicator (1 = originally missing/empty/placeholder, 0 = had at least one valid token) in your `cleaning` instruction. The missingness indicator must be created BEFORE the column is multi-hot expanded, because after expansion there is no way to distinguish "originally missing" from "none of the categories applied." For example: if `Type_of_Loan` is MNAR, instruct the generator to create `Type_of_Loan_missing = (original column is NaN or 'Not Specified')` as a binary column, then proceed with the multi-hot expansion. This is mandatory when `eda_insights.mnar_suspects` lists the column.

5. **Ground every decision in data you can see.** Use the actual min/max/p1/p99/mean for numerics and `top_10_values` for categoricals. Reference numbers in your reasoning.

6. **Preprocessing should preserve a compact canonical base table unless there is a strong reason not to.** This node defines the clean semantic representation that downstream feature engineering will build on. Prefer scalar `0/1` for `binary_flag`, ordinal integers for `ordered_categorical`, and other compact encodings that preserve meaning without fragmenting one concept into many columns. Do not default to one-hot just because it is familiar.

7. **Structured strings should preserve information, not collapse it.** If a duration like `"22 Years and 7 Months"` or a money/count field with separators is being parsed, choose cleaning instructions that preserve real variation. A parsing plan that would collapse most rows to one fallback value is worse than leaving the column raw.

8. **Exploit group structure when it exists.** If a `group_identifier` column is present and the same entity recurs across multiple rows, many per-entity-stable fields are better imputed from the group's own mode/median than a global statistic. Prefer `group_impute_by` for such fields; global median/mode is a fallback, not a default. `group_impute_by` may be a single column **or a list of columns** when joint conditioning makes sense (e.g. imputing by the intersection of two related counts).

9. **Outlier handling: clip vs NaN-then-impute.** Two valid strategies for implausible numeric values:
   - **Hard clip** (`clip_to_bounds`): squash values to `[low, high]`. Default when no group structure exists.
   - **Mark-and-refill** (`mark_out_of_range_nan`): set out-of-range values to NaN, then let the `imputation` strategy (group_median / fallback_formula / median) fill them. Preferred when group structure exists — a customer's own peer records restore signal that hard-clipping would flatten. Declare which strategy in `cleaning` (e.g. `"mark values < 0 or > 50000 as NaN"` vs `"clip to [0, 50000]"`).

10. **Derived fallbacks are valid imputation.** When a missing field is mechanically related to another column, declare a `fallback_formula` so the generator computes the derived value before falling back to a statistical fill. Formulas are pandas-evaluable expressions over other columns. **This is not optional when a mechanical relationship exists** — global median imputation on a column with 15% missingness gives 15% of rows the same constant, corrupting every downstream ratio feature that uses the imputed column (a column ranked #1 in mutual information can drop out of SHAP top-10 entirely this way).

    **Why this matters:** global median imputation on a column with 15% missingness gives 15% of rows the *same constant value*. Any ratio feature that uses the imputed column (e.g. EMI-to-Salary) will be corrupted for exactly those rows — the column that ranked #1 in mutual information can drop out of the SHAP top-10 entirely if its denominator is contaminated this way. A `fallback_formula` that uses an available peer column restores real variation instead of injecting a constant artifact.

    Named canonical patterns for financial / credit datasets:
    - **Monthly from annual:** `Annual_Income / 12` — when a `*_monthly` or `*_per_month` income field is missing and an annual equivalent exists, this is almost always the right formula, not the median.
    - **Balance from income and outflows:** `Monthly_Inhand_Salary - Total_EMI_per_month` — when a balance field is missing but both salary and EMI components are available.
    - **Derived payment from principal and rate:** `Outstanding_Debt * Interest_Rate / 1200` — when a monthly payment amount is missing but debt and rate are present.

    Decision rule: any time you see a `*_monthly` / `*_per_month` column with missingness > 5% **and** a corresponding annual or total column, ask "is there a mechanical formula?" If yes, use `imputation: "fallback_formula"` and set `fallback_formula` to that expression. Keep `imputation: "median"` only as the last-resort global backstop (i.e., still declare `fallback_formula` but the codegen will fall through to median when the formula's inputs are also NaN).

11. **Bucket-conditional imputation.** When a field's missing values correlate with a ranged/categorical peer (e.g. "delay severity" bucketed from another column), the spec can declare `group_impute_by` as a derived bucket via `bucket_spec` (bin edges + source column). This generalizes beyond raw group IDs to any conditioning structure.

12. **Numeric bounds are mandatory, not optional.** Every `numeric_continuous` or `numeric_count` column must carry a concrete `cleaning` range. If `max` is >3× `p99` or the field has a known domain bound, encode it. A missing bound lets corrupt tail values survive into the feature frame and break downstream scalers and models.

13. **Ordinal means ordinal.** If a column has inherent order, assign `ordered_categorical` + declare the explicit `ordinal_mapping`. Do not fall back to `unordered_categorical` + one-hot just because the string values look categorical — you lose ordering that tree splits and linear models can both exploit.

14. **Garbage and sentinel tokens are first-class cleaning targets.** Real-world data frequently contains placeholder strings that are not NaN but should be treated as missing: repeated underscores (`"_"`, `"_______"`), repeating sentinels (`"!@9#%8"`, long-repeat-digit strings like `"__-3333...__"`), or domain-specific "Not Specified" / "Unknown" tokens. Identify them from the column profile's `top_10_values` (high-frequency non-numeric values in a nominally numeric column, or values that break the expected schema) and declare them in `garbage_tokens`. The generator will replace these with NaN before any other cleaning so downstream imputation is not polluted.

15. **Non-numeric artifacts in numeric-looking columns.** When a numeric column's raw values carry currency symbols, commas, stray underscores, or trailing unit labels, request explicit pre-numeric stripping in `cleaning` (e.g. `"strip non-numeric characters, then to_numeric"`). A blind `to_numeric` will coerce these rows to NaN and inflate missingness.

16. **Target rows with missing labels must be dropped, not imputed.** If the target column has any NaN after cleaning, the generator should drop those rows — never impute a label. This is enforced after target extraction in the codegen step; the spec simply needs to mark the target correctly.

17. **Prefer additive primitive hints when the runtime exposes an approved cleaning library.** You may receive `allowed_cleaning_primitives`, which enumerates deterministic helpers already available to preprocessing codegen. When one fits the column semantics, declare it explicitly with `primitive` and `primitive_params` while still keeping the correct `semantic_role`. This extends the contract; it does not replace it.

## Semantic roles

Assign exactly one `semantic_role` to every column. The role drives the invariant that the deterministic validator enforces on the post-preprocessing output.

| Role | Meaning | Invariant after preprocessing |
|---|---|---|
| `identifier` | Row-unique ID, no predictive value | Must be absent from feature frame |
| `group_identifier` | Grouping key (e.g. customer ID) used by splitter only | Must be absent from feature frame; available for grouped splits |
| `target` | The label to predict | Must be absent from feature frame; kept in target file |
| `numeric_continuous` | Real-valued measurement | Finite, no NaN after imputation |
| `numeric_count` | Non-negative integer counts | Values ≥ 0 |
| `ordered_categorical` | Values have inherent order (e.g. Bad<Standard<Good) | Encoded as integer 0..K-1 preserving order |
| `unordered_categorical` | Categories with no inherent order | Encoding matches `representation_intent`; no false ordering |
| `binary_flag` | Two-valued yes/no, true/false, 0/1 | Values ⊆ {0, 1} |
| `multi_value_set` | Delimited list of tokens per cell | Row count unchanged; derived indicators ⊆ {0, 1}; no count artifacts; no combined-string dummies |
| `temporal_feature` | Dates or timestamps | Parseable; derived features (year, month, etc.) in valid ranges |
| `free_text` | Unstructured text | Encoded per declared intent with declared dimensionality |
| `leakage_risk_feature` | Feature suspected of target leakage | Dropped or quarantined with justification |

## representation_intent

Required when the role admits multiple valid encodings. Leave null when it doesn't apply.

| Role | Valid `representation_intent` values |
|---|---|
| `unordered_categorical` | `deferred`, `one_hot`, `ordinal_proxy`, `target_encoded`, `frequency_encoded`, `hash_encoded` |
| `multi_value_set` | `binary_membership` (presence only), `count_membership` (counts per token) |
| `numeric_continuous` | `raw`, `standardized`, `min_max_scaled`, `log_transformed` |
| `free_text` | `tfidf`, `embedding`, `drop` |
| `temporal_feature` | `decomposed` (year/month/dow as separate features), `epoch_seconds`, `drop` |

Cardinality is a *property* of the data that should drive your intent choice — a 50-category column with `unordered_categorical` role should typically use `target_encoded`, `frequency_encoded`, `hash_encoded`, or a deliberately compact `ordinal_proxy`, not `one_hot`. Do not invent a new role for high-cardinality columns.

For canonical preprocessing output, prefer these defaults unless the data clearly argues otherwise:

- `binary_flag` -> scalar 0/1, not one-hot.
- `ordered_categorical` -> ordinal integer 0..K-1.
- `unordered_categorical` -> **`deferred` is the preferred default when a dual-view feature engineering stage is available**, because different model families need different encodings (linear wants one-hot for low-card, tree-based prefers frequency/target for compact numeric representation without fake order). With `deferred`, preprocessing leaves the column as a cleaned string and the downstream FE node picks the encoding per view. Only commit to a global encoding at this layer when there is a specific reason to (e.g. the representation is identical for every model family, or a trusted domain encoding is known).
  - If you do commit here: `one_hot` only when cardinality is genuinely small and the expansion is an acceptable base representation; `frequency_encoded` for medium cardinality; `target_encoded` (with out-of-fold logic handled by the FE/training stage) for high cardinality; never raw label / arbitrary integer codes, which inject fake order.
  - **Identity-significant categoricals (e.g. occupation, industry, product-type) should almost always be `deferred`.** Frequency-encoding them here collapses category identity into prevalence — two categories with the same row count become indistinguishable, which destroys per-category signal a tree model could otherwise split on. If you think `frequency_encoded` at preprocess is right, ask: would two different categories with similar frequency lose all discriminative power? If yes, defer.
- `multi_value_set` -> `binary_membership` by default unless repeated counts per token are semantically meaningful.

**When `representation_intent: "deferred"` is declared:** preprocessing must clean the column (garbage tokens replaced, imputation applied) but leave it as an object-dtype string column in the output frame. The FE node is responsible for producing numeric columns per view. Do not also set an `encoding` for a deferred column — the FE stage owns that decision.

## Primitive hints

When `allowed_cleaning_primitives` is present, treat it as the approved deterministic cleaning library for downstream preprocessing codegen.

- Keep `semantic_role` on the existing 12-role taxonomy.
- Use `primitive` only when one of the provided primitives is a direct semantic fit.
- Use `primitive_params` for concrete bounds, token lists, delimiter choices, or group columns that make the primitive actionable.
- `semantic_subtype` is optional. Use it only to sharpen meaning without inventing a new top-level role, for example `"currency_amount"`, `"duration_months"`, or `"human_age"`.
- If no primitive applies, leave `primitive` and `primitive_params` null and describe the cleaning normally.

Recommended mappings when the data supports them:

- Dirty numeric strings with symbols / separators -> `parse_dirty_numeric`
- Human age stored as mixed text -> `parse_age_series`
- `"X Years and Y Months"` style durations -> `parse_duration_months`
- Placeholder-string normalization -> `missing_string_mask`
- Delimited multi-value membership sets -> `multi_hot_membership`
- Entity-stable numeric imputation -> `fill_numeric_by_group_then_global`
- Credit history months bounded by applicant age -> `cap_credit_history_by_adulthood`

## Considerations for each column

- **Relevance**: identifier? group key? target? Feature?
- **Data quality**: garbage values, type mismatches, placeholder patterns?
- **Outliers**: large gap between mean/p99/max suggests corrupted tail — pick a tighter bound.
- **Structure**: single-value or delimited multi-value?
- **Representation compactness**: ask whether this encoding keeps one concept as one feature when possible, or explodes it prematurely before feature engineering has a chance to reason about it.
- **Missingness**: MNAR flag from EDA means missingness carries signal.
- **Discriminative power**: prioritize features with high **model-eligible** MI or ANOVA F. If `eda_insights.raw_top_discriminative_features` contains identifier-like or leakage-alert columns, treat those as audit warnings, not evidence that the raw field should survive preprocessing.

## Inputs

- `columns`, `sample_rows`, `column_profiles`, `dataset_policy_spec`, `allowed_cleaning_primitives`, optional `eda_insights`.
  - When present, `eda_insights.top_discriminative_features` is the backward-compatible alias for **model-eligible** MI rankings.
  - `eda_insights.model_eligible_top_discriminative_features` is the authoritative list for modeling relevance.
  - `eda_insights.raw_top_discriminative_features` and `eda_insights.leakage_alerts` are audit context only. Use them to justify drop/quarantine decisions or call out suspicious raw signal, but do not preserve an identifier/leakage field just because its raw MI is high.
  - `allowed_cleaning_primitives` lists the approved deterministic runtime helpers. Prefer them whenever the column semantics match.

## Output format

Return **only** a raw JSON object (no markdown fences, no explanation text):

```
{
  "transforms": {
    "<column_name>": {
      "action": "keep" | "drop" | "quarantine",
      "semantic_role": "<one of the 12 roles>",
      "representation_intent": "<intent string or null>",
      "primitive": "<allowed primitive name or null>",
      "primitive_params": {"<param>": "<value>", ...} | null,
      "semantic_subtype": "<optional subtype string or null>",
      "cleaning": "<specific instruction with concrete bounds, or null>",
      "imputation": "median" | "mode" | "group_median" | "group_mode" | "fallback_formula" | null,
      "group_impute_by": "<column name>" | ["<col1>", "<col2>", ...] | null,
      "bucket_spec": {"source": "<col>", "edges": [<numeric>, ...], "labels": [<int>, ...]} | null,
      "fallback_formula": "<pandas-evaluable expression over other columns>" | null,
      "ordinal_mapping": {"<category>": <int>, ...} | null,
      "garbage_tokens": ["<token>", ...] | null,
      "encoding": "one_hot" | "ordinal" | "label" | "multi_hot" | null
    },
    ...
  },
  "reasoning": {
    "<column_name>": "<explanation grounded in profile and EDA numbers, including why this role+intent>",
    ...
  }
}
```

Include an entry for **every** column in both `transforms` and `reasoning`.

## Example

Given columns `["Patient_ID", "Age", "Blood_Pressure", "Smoker", "Severity", "Symptoms", "Diagnosis"]` where `Patient_ID` is an identifier, `Symptoms` is comma-delimited, and `Diagnosis` is the target:

```json
{
  "transforms": {
    "Patient_ID": {"action": "drop", "semantic_role": "group_identifier", "representation_intent": null, "primitive": null, "primitive_params": null, "semantic_subtype": null, "cleaning": null, "imputation": null, "group_impute_by": null, "bucket_spec": null, "fallback_formula": null, "ordinal_mapping": null, "encoding": null},
    "Age": {"action": "keep", "semantic_role": "numeric_continuous", "representation_intent": "raw", "primitive": "parse_age_series", "primitive_params": {"bounds": [0, 120]}, "semantic_subtype": "human_age", "cleaning": "extract age token, then mark values outside [0, 120] as NaN", "imputation": "group_median", "group_impute_by": "Patient_ID", "bucket_spec": null, "fallback_formula": null, "ordinal_mapping": null, "encoding": null},
    "Blood_Pressure": {"action": "keep", "semantic_role": "numeric_continuous", "representation_intent": "raw", "primitive": "parse_dirty_numeric", "primitive_params": {"bounds": [40, 250]}, "semantic_subtype": null, "cleaning": "strip non-numeric artifacts, convert to float, clip to [40, 250]", "imputation": "group_median", "group_impute_by": "Patient_ID", "bucket_spec": null, "fallback_formula": null, "ordinal_mapping": null, "encoding": null},
    "Diastolic_BP": {"action": "keep", "semantic_role": "numeric_continuous", "representation_intent": "raw", "primitive": "parse_dirty_numeric", "primitive_params": {"bounds": [30, 150]}, "semantic_subtype": null, "cleaning": "strip non-numeric artifacts, convert to float, clip to [30, 150]", "imputation": "fallback_formula", "group_impute_by": null, "bucket_spec": null, "fallback_formula": "Blood_Pressure * 0.65", "ordinal_mapping": null, "encoding": null},
    "Med_Dose_mg": {"action": "keep", "semantic_role": "numeric_continuous", "representation_intent": "raw", "primitive": "fill_numeric_by_group_then_global", "primitive_params": {"group_by": ["Age_Bucket", "Smoker"], "bounds": [0, 500]}, "semantic_subtype": "currency_amount", "cleaning": "convert to float, clip to [0, 500]", "imputation": "group_median", "group_impute_by": ["Age_Bucket", "Smoker"], "bucket_spec": null, "fallback_formula": null, "ordinal_mapping": null, "encoding": null},
    "Visit_Count": {"action": "keep", "semantic_role": "numeric_count", "representation_intent": "raw", "primitive": "parse_dirty_numeric", "primitive_params": {"lower_bound": 0, "upper_bound": 100}, "semantic_subtype": null, "cleaning": "mark values < 0 as NaN, clip upper 100", "imputation": "group_median", "group_impute_by": "Severity_Bucket", "bucket_spec": {"source": "Severity_Score", "edges": [-1, 3, 6, 10], "labels": [0, 1, 2]}, "fallback_formula": null, "ordinal_mapping": null, "encoding": null},
    "Smoker": {"action": "keep", "semantic_role": "binary_flag", "representation_intent": null, "primitive": "missing_string_mask", "primitive_params": {"garbage_tokens": ["Unknown"]}, "semantic_subtype": null, "cleaning": "map {'Yes':1,'No':0}", "imputation": "mode", "group_impute_by": null, "bucket_spec": null, "fallback_formula": null, "ordinal_mapping": null, "garbage_tokens": null, "encoding": null},
    "Severity": {"action": "keep", "semantic_role": "ordered_categorical", "representation_intent": null, "primitive": "missing_string_mask", "primitive_params": {"garbage_tokens": ["_", "Unknown"]}, "semantic_subtype": null, "cleaning": "replace placeholder tokens with NaN, then map", "imputation": "group_mode", "group_impute_by": "Patient_ID", "bucket_spec": null, "fallback_formula": null, "ordinal_mapping": {"Mild": 0, "Moderate": 1, "Severe": 2}, "garbage_tokens": ["_", "Unknown"], "encoding": "ordinal"},
    "Symptoms": {"action": "keep", "semantic_role": "multi_value_set", "representation_intent": "binary_membership", "primitive": "multi_hot_membership", "primitive_params": {"delimiter": ",", "strip_whitespace": true}, "semantic_subtype": null, "cleaning": "split on ', ', strip whitespace, remove leading 'and '", "imputation": null, "group_impute_by": null, "bucket_spec": null, "fallback_formula": null, "ordinal_mapping": null, "encoding": "multi_hot"},
    "Diagnosis": {"action": "drop", "semantic_role": "target", "representation_intent": null, "primitive": null, "primitive_params": null, "semantic_subtype": null, "cleaning": null, "imputation": null, "group_impute_by": null, "bucket_spec": null, "fallback_formula": null, "ordinal_mapping": null, "encoding": null}
  },
  "reasoning": {
    "Patient_ID": "group_identifier — same patient recurs across visits; used by splitter and as imputation group",
    "Age": "numeric_continuous. p99=85 but max=999 and some negatives. Mark out-of-range as NaN rather than clipping, then fill from the same patient's other visits (age shouldn't vary within a patient); global median only as last-resort fallback.",
    "Blood_Pressure": "numeric_continuous. Within-patient BP variation is real signal, so hard-clip to plausible [40,250] preserves it. Remaining NaN filled by patient group_median to keep per-patient level.",
    "Diastolic_BP": "numeric_continuous. Systolic and diastolic track mechanically (~0.65×). When diastolic is missing but systolic is present, the mechanical relation is a higher-fidelity fill than any median.",
    "Med_Dose_mg": "numeric_continuous. Dose depends jointly on age-stratum and smoking status, so group_median over the pair is more specific than a single-column group.",
    "Visit_Count": "numeric_count. Visit count correlates with severity bucket, not patient identity directly. Declare a derived severity bucket and impute within it.",
    "Smoker": "binary_flag — two values (Yes/No), map to {0,1}",
    "Severity": "ordered_categorical — 3 values (Mild<Moderate<Severe), ordinal_mapping declared explicitly so generator does not rely on alphabetical order",
    "Symptoms": "multi_value_set with binary_membership — one cell may list several symptoms. str.get_dummies(sep=', ') with whitespace strip and 'and ' prefix removal. Indicators must be {0,1} presence, never counts",
    "Diagnosis": "Target column — dropped per policy"
  }
}
```
