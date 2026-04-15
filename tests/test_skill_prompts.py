import pytest

from bt5151_credit_risk.skill_prompts import load_skill_prompt


def test_load_skill_prompt_reads_skill_file():
    prompt = load_skill_prompt("train-models")
    assert "train-models" in prompt


def test_load_skill_prompt_raises_for_missing_file():
    missing_name = "does-not-exist"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_skill_prompt(missing_name)

    assert str(exc_info.value) == f"Skill prompt not found: {missing_name}"


@pytest.mark.parametrize("skill_name", ["../README", "/tmp/skill", "nested/../../README"])
def test_load_skill_prompt_rejects_path_traversal(skill_name):
    with pytest.raises(ValueError) as exc_info:
        load_skill_prompt(skill_name)

    assert str(exc_info.value) == f"Invalid skill name: {skill_name}"


def test_generate_feature_engineering_prompt_makes_ordering_unambiguous():
    prompt = load_skill_prompt("generate-feature-engineering-code")

    assert "EDA hypotheses are prioritized ideas, not directives" in prompt
    assert "Build interaction features from raw values first" in prompt
    assert "linear_view" in prompt
    assert "tree_view" in prompt
    assert "view_metadata.json" in prompt
    assert "single-view fallback" in prompt
    assert "prefer dual-view output" in prompt
    assert "Apply log or other monotonic transforms to standalone parent numeric columns" in prompt
    assert "upstream directives, not suggestions" not in prompt
    assert "highest-priority feature requests" not in prompt
    assert "implement every feasible one" not in prompt
    assert "transform the most skewed columns first" not in prompt


def test_repair_feature_engineering_prompt_preserves_raw_parent_semantics():
    prompt = load_skill_prompt("repair-feature-engineering-code")

    assert "Preserve semantic feature meaning." in prompt
    assert "interactions or ratios, make sure they use raw parent columns before any log" in prompt
    assert "Interactions come before log transforms." in prompt
    assert "Dual-view outputs must stay internally aligned." in prompt


def test_preprocessing_prompts_cover_percentile_clipping_and_duration_connectors():
    generate_prompt = load_skill_prompt("generate-preprocessing-code")
    repair_prompt = load_skill_prompt("repair-preprocessing-code")
    spec_prompt = load_skill_prompt("column-transform-spec")

    assert "percentile-based two-sided clipping" in generate_prompt
    assert "Duration strings often contain connector words between units." in generate_prompt
    assert "compact canonical base table" in generate_prompt
    assert "Keep compact encodings compact." in generate_prompt
    assert "Duration strings often include connector words." in repair_prompt
    assert "add percentile-based clipping" in repair_prompt
    assert "Preserve the compact base-table contract" in repair_prompt
    assert "Do not \"repair\" a compact role into a wider encoding." in repair_prompt
    assert "Preprocessing should preserve a compact canonical base table" in spec_prompt
    assert "Structured strings should preserve information, not collapse it." in spec_prompt
    assert "`unordered_categorical` | `deferred`, `one_hot`, `ordinal_proxy`" in spec_prompt


def test_column_transform_spec_declares_new_generalized_fields():
    """The spec schema must expose group_impute_by, bucket_spec, fallback_formula,
    ordinal_mapping, and garbage_tokens so the reasoning model can use them."""
    spec_prompt = load_skill_prompt("column-transform-spec")

    assert "group_impute_by" in spec_prompt
    assert "bucket_spec" in spec_prompt
    assert "fallback_formula" in spec_prompt
    assert "ordinal_mapping" in spec_prompt
    assert "garbage_tokens" in spec_prompt
    # Principles, not just schema fields
    assert "Exploit group structure when it exists." in spec_prompt
    assert "Derived fallbacks are valid imputation." in spec_prompt
    assert "Bucket-conditional imputation." in spec_prompt
    assert "Numeric bounds are mandatory, not optional." in spec_prompt
    assert "Ordinal means ordinal." in spec_prompt
    assert "Garbage and sentinel tokens are first-class cleaning targets." in spec_prompt
    # Target-row drop is handled at codegen step 2, so per-column drop_rows must not reappear
    assert '"drop_rows"' not in spec_prompt


def test_generate_preprocessing_prompt_handles_new_spec_fields():
    """Worker must have implementation recipes for every new spec field and must keep
    raw_df row-aligned when target rows are dropped, so downstream grouping/splitting
    stays correct."""
    prompt = load_skill_prompt("generate-preprocessing-code")

    # Row-alignment contract after target-row drop
    assert "raw_df = raw_df.loc[valid].reset_index(drop=True)" in prompt
    assert "Never impute a label" in prompt
    # Garbage tokens replaced BEFORE numeric coercion
    assert "Replace garbage tokens with NaN" in prompt
    assert "Do this FIRST" in prompt
    # Non-numeric stripping as its own step
    assert "Strip non-numeric artifacts" in prompt
    # group_impute_by: per-column source resolution, list-capable, no shared-source assumption
    assert "resolve them **independently**" in prompt
    assert "list of column names" in prompt
    # bucket_spec implementation
    assert "bucket_spec" in prompt
    assert "pd.cut(df[bs['source']]" in prompt
    # fallback_formula implementation
    assert "fallback_formula" in prompt
    assert "df.eval(spec['fallback_formula'])" in prompt
    # ordinal_mapping implementation
    assert "ordinal_mapping" in prompt
    assert "spec['ordinal_mapping']" in prompt


def test_generate_feature_engineering_prompt_enforces_ratio_stability():
    prompt = load_skill_prompt("generate-feature-engineering-code")
    assert "Engineered ratios must be stabilized" in prompt
    assert "train_ratio.quantile(0.99)" in prompt


def test_deferred_encoding_contract_across_three_skills():
    """Deferred intent must be declared in the spec, skipped by preprocessing codegen,
    and handled per-view by FE codegen with cardinality-aware encoding + a
    fully-numeric assertion before writing views."""
    spec_prompt = load_skill_prompt("column-transform-spec")
    preproc_prompt = load_skill_prompt("generate-preprocessing-code")
    fe_prompt = load_skill_prompt("generate-feature-engineering-code")

    # 1. Spec declares deferred as a valid intent and as the preferred default
    #    for unordered_categorical when dual-view FE is available.
    assert "`deferred`" in spec_prompt
    assert "deferred" in spec_prompt.lower()
    assert "preferred default when a dual-view feature engineering stage is available" in spec_prompt
    # Raw label encoding explicitly disallowed as default
    assert "never raw label" in spec_prompt

    # 2. Preprocessing codegen skips deferred columns at encoding step.
    assert 'representation_intent: "deferred"' in preproc_prompt
    assert "Skip columns with" in preproc_prompt

    # 3. FE codegen has per-cardinality rules and a numeric assertion.
    assert "Deferred categoricals must be encoded per view" in fe_prompt
    assert "Very low cardinality" in fe_prompt
    assert "Medium cardinality" in fe_prompt
    assert "High cardinality" in fe_prompt
    assert "frequency encoding" in fe_prompt.lower()
    assert "target encoding" in fe_prompt.lower()
    assert "out-of-fold" in fe_prompt
    assert "Never use raw label encoding" in fe_prompt
    assert "select_dtypes(exclude='number').empty" in fe_prompt
    # Input contract wording relaxed
    assert "may arrive already encoded" in fe_prompt
    assert "deferred as cleaned string columns" in fe_prompt


def test_feature_engineering_prompts_discourage_epsilon_ratio_artifacts():
    generate_prompt = load_skill_prompt("generate-feature-engineering-code")
    repair_prompt = load_skill_prompt("repair-feature-engineering-code")

    assert "Do not hide zero denominators behind `denominator + 1e-6`" in generate_prompt
    assert "Use zero-aware logic instead" in generate_prompt
    assert "Avoid epsilon-denominator artifacts." in generate_prompt
    assert "Do not paper over zero denominators with epsilon hacks." in repair_prompt
    assert "replace `/(denominator + 1e-6)` with zero-aware branching" in repair_prompt
