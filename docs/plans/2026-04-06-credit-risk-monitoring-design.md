# BT5151 Credit Risk Monitoring System Design

Date: 2026-04-06

## 1. Problem Statement

This project will build a multi-agent machine learning system for monthly credit risk monitoring. The system takes a customer-month record, predicts the customer's credit standing as `Good`, `Standard`, or `Poor`, and then translates that prediction into a business-facing risk explanation and recommended next action.

The design is anchored to the BT5151 group project brief in [bt5151_group_project_2026.pdf](../../bt5151_group_project_2026.pdf). The solution must satisfy the required ML comparison, SKILL.md stage decomposition, LangGraph orchestration, Gradio output, and rubric expectations for rigorous preprocessing, evaluation, and business interpretation.

## 2. Why This Dataset Fits

The selected dataset in [train.csv](../../train.csv) is a strong fit for the assignment because it already contains:

- a labelled target column: `Credit_Score`
- three business-meaningful classes: `Good`, `Standard`, `Poor`
- repeated monthly records for each customer
- realistic financial behavior signals such as debt, utilization, payment delays, inquiries, and loan mix
- real cleaning challenges such as placeholders, malformed numeric values, invalid ages, and missing values

These properties make it suitable for a multi-class credit risk monitoring problem and provide enough complexity to justify a dedicated `preprocess-data` stage and a fair model comparison.

## 3. Business Case

### Users

- credit risk analyst
- credit operations officer
- lending portfolio manager

### Decision Context

The system supports monthly portfolio review rather than one-time loan approval. Each monthly record is treated as a current snapshot of a customer's repayment and credit behavior. The output helps a business user decide whether a customer appears healthy, should be monitored, or should be escalated for intervention.

### Business Value

- earlier detection of deteriorating customer quality
- more consistent triage of moderate-risk accounts
- faster manual review for risky customers
- clearer communication of risk to non-technical stakeholders

## 4. ML Task Definition

The primary task is **multi-class classification**:

- input: one customer-month record
- output: predicted credit class `Good`, `Standard`, or `Poor`

This design intentionally keeps the original task label rather than collapsing it into a binary risk label. That preserves the information in the dataset and aligns with the brief's multi-class metric requirements.

## 5. Design Principles

This system follows five first-principles rules:

1. The target should not be changed unless the data forces a change.
2. Cleaning, feature engineering, and leakage control must be treated as separate concerns.
3. Evaluation must reflect realistic generalization, not memorization of repeated customers.
4. Raw model outputs are not enough; downstream stages must convert them into business action.
5. Every pipeline stage must have an explicit state contract documented in its own `SKILL.md`.

## 6. Pipeline Architecture

The minimum required LangGraph chain is:

`preprocess-data -> train-models -> evaluate-models -> select-model -> run-inference`

This project will extend that minimum with two downstream business-facing stages:

`preprocess-data -> train-models -> evaluate-models -> select-model -> run-inference -> explain-risk -> recommend-action`

### Stage Responsibilities

#### `preprocess-data`

- load raw CSV input
- profile schema and target balance
- replace invalid placeholders
- parse numeric and date-like fields
- remove unusable identifiers from modeling inputs
- create split-safe processed feature tables
- write preprocessing report and split metadata to agent state

#### `train-models`

- train at least two candidate models on the same processed training data
- record model configuration and fit summary
- persist fitted model artifacts in state

#### `evaluate-models`

- evaluate all candidate models on the same held-out data
- compute all required multi-class metrics
- generate evaluation plots
- write structured metric tables to agent state

#### `select-model`

- compare candidates using macro F1, weighted F1, per-class performance, and business cost of mistakes
- record a written justification for the final selection

#### `run-inference`

- accept one new customer-month input
- apply saved preprocessing
- return class probabilities and predicted label

#### `explain-risk`

- translate prediction and confidence into business-language risk reasoning
- state the likely risk level and the main drivers behind the prediction

#### `recommend-action`

- convert the business explanation into a next-step recommendation
- distinguish low-risk, monitor, and escalation paths

## 7. State Design

Agent state is the contract that connects the system. Each stage reads only the fields it needs and writes only the fields the next stage needs.

Planned state keys:

- `raw_dataset_path`
- `dataset_profile`
- `preprocessing_rules`
- `feature_columns`
- `split_metadata`
- `train_frame`
- `validation_frame`
- `test_frame`
- `candidate_model_specs`
- `trained_models`
- `evaluation_results`
- `evaluation_visual_paths`
- `selected_model_name`
- `selection_justification`
- `inference_input`
- `prediction_output`
- `risk_explanation`
- `recommended_action`

This structure is designed to make the state chain precise and auditable, which directly supports the rubric for SKILL.md completeness and pipeline state specification.

## 8. Data Cleaning and Leakage Policy

### Cleaning Scope

The preprocessing stage will handle:

- placeholder tokens such as `_` and malformed text markers
- corrupted numeric values
- invalid ages and impossible count fields
- missing values
- parsing of `Credit_History_Age`
- normalization of selected categoricals

### Leakage Controls

The preprocessing stage must also guard against invalid evaluation. The key rule is:

**No split-blind preprocessing that uses future or test information to fill training inputs.**

Specific controls:

- do not include `Credit_Score` as a modeling feature
- do not include direct personal identifiers in modeling inputs
- split by `Customer_ID` so the same customer does not appear in both train and test
- fit imputers and encoders on training data only, then apply them to validation and test data

This distinction matters because leakage control is not merely cleaning. It is pipeline integrity.

## 9. Split Strategy

The recommended evaluation strategy is grouped splitting by `Customer_ID`.

Reasoning:

- each customer appears for 8 months
- a random row split would let the model see the same customer in both train and test
- this would inflate performance and weaken the credibility of the model comparison

The grouped split is the safest default for the BT5151 brief because it supports a fair held-out evaluation and a stronger report justification.

## 10. Candidate Models

To satisfy the assignment while staying interpretable and robust, the system will compare:

- **Model A:** multinomial logistic regression
- **Model B:** random forest classifier

Why this pair:

- logistic regression provides a simple and explainable baseline
- random forest captures nonlinear relationships and feature interactions
- the comparison is easy to explain to both technical and non-technical audiences

If time permits, a third candidate such as gradient boosting may be added as an extension, but the base design assumes exactly two candidates to preserve focus.

## 11. Required Evaluation

Because the task is multi-class classification, the system must report:

- per-class precision
- per-class recall
- per-class F1-score
- macro F1
- weighted F1

Planned visualizations:

- confusion matrix heatmap
- per-class metric bar chart

Interpretation rules:

- metrics must be explained in business terms
- misclassifying `Poor` accounts as safer classes is treated as especially costly
- the report must discuss limitations honestly

## 12. Downstream Business Translation

The Gradio interface must expose the downstream business output, not only raw model scores.

### Planned downstream output

- predicted credit class
- confidence band
- business-readable risk summary
- recommended action
- caution note if confidence is low

Example action mapping:

- `Good` with high confidence -> continue standard handling
- `Standard` with moderate confidence -> monitor and review recent behavior
- `Poor` or low-confidence high-risk case -> escalate for manual review

## 13. Gradio Experience

The Gradio app will be designed for a non-technical business user.

The main output pane will prioritize:

- risk summary
- recommended action
- confidence level

Technical details such as class probabilities may be shown in a secondary panel for transparency, but they should not dominate the interface.

## 14. Testing and Failure Analysis

The design intentionally includes end-to-end evaluation beyond model metrics.

Planned full-system tests:

- one clear low-risk case
- one borderline medium-risk case
- one challenging or messy case

For each test:

- record the input
- state the expected business-facing output
- capture the actual output
- record confidence

The report will also document at least two actual failure modes observed from runs, such as:

- `Poor` misclassified as `Standard`
- vague recommendation under low confidence
- unstable output when many fields are missing

## 15. Deliverable Mapping

### Deliverable 1: Notebook

The notebook will show:

- data loading
- preprocessing
- candidate model training
- evaluation tables and visualizations
- model selection justification
- pipeline definition
- downstream output generation
- Gradio launch with `share=True`

### Deliverable 2: `skills/`

The project will include:

- `skills/preprocess-data.md`
- `skills/train-models.md`
- `skills/evaluate-models.md`
- `skills/select-model.md`
- `skills/run-inference.md`
- `skills/explain-risk.md`
- `skills/recommend-action.md`

### Deliverable 3: Report

The report will mirror the brief's required sections:

- system architecture
- business case
- data and ML methodology
- model evaluation
- SKILL.md design rationale
- system evaluation
- AI usage declaration

### Deliverable 4: Slides and presentation

The demo must show:

- the Gradio workflow
- the chosen model's outputs
- the comparison evidence
- how the LangGraph stages connect

## 16. AI Usage Guardrail

AI tools may help with code drafting, debugging, and boilerplate, but final ownership of:

- dataset understanding
- model choice
- evaluation interpretation
- SKILL.md design decisions
- report writing
- presentation explanation

must remain with the group.

## 17. Final Recommendation

Proceed with a multi-class credit risk monitoring system using grouped customer splitting, logistic regression versus random forest, and a downstream business translation layer that turns model predictions into risk explanations and recommended actions.

This is the most rubric-safe, business-coherent, and implementation-manageable design for the current dataset and assignment brief.
