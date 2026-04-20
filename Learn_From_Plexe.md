# The deep logic that I think I should steal

1. LLM proposes experiments; deterministic code decides what is true

Plexe’s architecture pushes LLMs into roles like task analysis, metric selection, hypothesis generation, planning, feature processing, and model definition, but the workflow, search policy, checkpoints, and evaluation loop are still programmatic. The search policy chooses what to expand; the journal normalizes score direction; the workflow controls phases and resume behavior.

Your repo currently says almost every analytical stage is LLM-driven, from dataset-policy spec through column transforms, preprocessing codegen, FE codegen, training diagnostics, XAI interpretation, and explanation. That is powerful, but it also means drift can happen in multiple semantic-contract nodes.

So the practical lesson for your repo is:

let the LLM suggest what to try
let deterministic validators decide whether the output is acceptable

That means:

LLM may propose Type_of_Loan_missing
validator must fail if a multi-hot Type_of_Loan expansion occurs without the missing flag
LLM may propose one-hot for Occupation
validator must fail if Occupation is frequency-encoded in tree_view
LLM may summarize model comparison
model selection metric and split policy must already be fixed in code

That is also safer under the course AI policy, which says model choices, evaluation interpretation, and business judgment must genuinely be yours, not delegated to AI.

3. Standard metrics should be hardcoded, not negotiated

Plexe has a StandardMetric concept and only falls back to metric-code generation when the metric is not standard. Its code index also makes clear that metric selection and metric implementation are distinct roles.

For your repo, the equivalent contract should be fixed:

task type: multiclass classification
primary selection metric: Macro F1
report all required metrics: per-class precision, recall, F1, macro F1, weighted F1
required visualization: confusion-matrix heatmap, plus optionally per-class bars
same held-out split for all candidate models
same grouped policy for baseline preview, tuning, and final evaluation

That is not just good engineering; it is directly aligned with the rubric’s required metrics and fair-comparison expectations.

4. Smaller agent responsibility is better than fewer agents

Plexe’s agents are specialized: splitter, sampler, baseline builder, planner, feature processor, model definer, evaluator, and so on. The key is not the number. The key is that each role is narrow.

Your repo currently has a strong analytical story, but some nodes are doing too much semantic risk-bearing work:

column-transform-spec
preprocessing code generation
FE code generation
review/audit
XAI interpretation synthesis

So the redesign principle is:

contract nodes should be deterministic and validator-backed
proposal nodes can stay LLM-driven
interpretation nodes should come after model selection

That means you do not reduce node count just to “save turns.” You reduce drift by narrowing what each node is allowed to decide.

5. Observability should come from instrumentation, not another reasoning layer

Plexe’s callback system is exactly the right model for your frontend idea. It treats observability as lifecycle instrumentation: build start, iteration start, iteration end, build end, with datasets, schemas, iteration number, current node/solution, and metrics available in the callback context.

That means your Gradio trace tab should be built from:

node start/end events
state keys read
state keys written
warnings
metrics
artifacts produced

not from another LLM trying to “explain the log.”

That is both more reliable and more demo-friendly.

we can also discuss if we can create a run_registry.jsonl

One line per experiment candidate:

run_id
timestamp
mode (search or full)
feature_changes
model_family
params_summary
split_policy
primary_metric
macro_f1
weighted_f1
status
notes

# The deterministic contracts you should hardcode now

A. Validation-policy contract

This belongs in code, not prompts.

Rules:

baseline preview, tuning, and final evaluation must use the same grouped policy family
no candidate model may be compared on a different split
if a fold lacks classes, skip or degrade gracefully, but log it
group leakage is forbidden

This is exactly the kind of hard fix that already worked in your repo. Your own recent patch log shows baseline CV was moved from ad hoc StratifiedKFold logic to the policy-aware split builder used by tuning, and that change held because it changed deterministic Python, not prompt wording.

B. Metric contract

Hardcode:

task_type = multiclass_classification
primary_metric = macro_f1
required report metrics = per-class precision/recall/F1 + macro F1 + weighted F1
final visual = confusion matrix
model selection compares candidates on the same held-out data

The LLM may explain why XGBoost won. It should not define the scoring logic.

C. Preprocessing contract

Validator should fail if:

target column remains in features
encoded target derivatives remain
required missing flags are missing for specified MNAR columns
forbidden raw placeholders are still present
critical parsed fields fall below expected uniqueness floors

Your recent preprocessing hardening already showed this pattern works better than prompt-only guidance. The patched run caught target leakage and forced repair; that is exactly the right style of contract.

D. Feature-engineering contract

Validator should fail if:

linear_view and tree_view are inconsistent with the model mapping
Occupation in tree_view is frequency-encoded
Type_of_Loan_missing is absent when Type_of_Loan was expanded
numeric-only expectation for training views is violated
feature names drift in a way that breaks grouped PFI / SHAP grouping

This is where your current repo still needs stronger hard guarantees.

E. Candidate-model contract

For each candidate model:

same train/val split
same sample regime
same primary metric
same leakage policy
logged hyperparameter budget
deterministic seed where possible

That makes the comparison fair and easier to justify in the notebook and report. The rubric explicitly rewards fair comparison, reproducibility, and justified model selection.


# What I would patch first

A. train.py and graph.py

This is where the truth contract for evaluation should live.

Hardcode:

task type = multiclass classification
primary metric = macro F1
required metrics = per-class precision/recall/F1 + macro F1 + weighted F1
same grouped split family for preview, tuning, and final evaluation
same candidate set comparison logic

graph.py should orchestrate.
train.py should own the metric/split truth.

The LLM can still suggest model grids or feature ideas, but not the scoring rule.

B. preprocess.py

This should become your first strong validator wall.

Make it fail or repair on:

target leakage
encoded target derivatives
missing flags required for known MNAR columns
placeholder cleanup not done
parsed fields in invalid format
any column contract the downstream stage depends on

This is also where Type_of_Loan_missing should ultimately become deterministic, not just a prompt hint.

C. feature_engineering.py

This should become the second validator wall.

Make it fail or repair on:

Occupation being frequency-encoded in tree_view
missing required engineered flags
illegal view-specific encodings
inconsistent view/model expectations
broken numeric-only assumptions for model input

This is where I would move the Occupation protection into real enforcement.

D. prompt / skill files
Keep them, but downgrade their authority.

They should answer:

what to try
how to summarize results

# The exact order I would do this in
Freeze metric contract in Python
train.py
selection rule, split rule, metric outputs
Freeze preprocessing contract
preprocess.py
especially Type_of_Loan_missing and leakage/missingness checks
Freeze FE contract
feature_engineering.py
especially Occupation tree-view behavior

observability should come from instrumentation, not another reasoning layer

I would keep that in mind while patching, even if you do not build the frontend yet. When touching graph.py, have each node emit structured trace info:

node name
state keys read
state keys written
warnings
metrics
artifacts produced

That will make your later Gradio trace tab much easier.