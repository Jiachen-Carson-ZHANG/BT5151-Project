# BT5151 Credit Risk Monitoring System

This repository is for the BT5151 group project: a multi-agent machine learning system that predicts monthly customer credit standing and translates the prediction into a business-facing risk recommendation.

The project is designed to follow the assignment brief in `bt5151_group_project_2026.pdf`, with one working assumption agreed by the team: we are building and testing in a local or Codespaces-style environment instead of treating Google Colab as the main development environment.

## 1. Project Objective

From first principles, the assignment asks us to build one complete chain:

1. start from a labelled dataset
2. preprocess it properly
3. train at least two candidate models
4. evaluate them fairly on held-out data
5. justify the final model choice
6. run inference on a new input
7. pass the prediction into downstream agent skills
8. show a business-facing result in Gradio

Our chosen business case is:

`Monthly credit risk monitoring for lending or portfolio management`

Our chosen ML task is:

`Multi-class classification of Credit_Score as Good / Standard / Poor`

Why this is the right framing:

- the dataset already contains a labelled target, `Credit_Score`
- the target classes are business-meaningful
- the same customer appears across months, so the dataset supports monitoring rather than one-off prediction
- the assignment requires downstream business-facing output, not raw model scores only

## 2. Assignment Alignment

This repo is being built to satisfy the BT5151 requirements below.

### Required pipeline stages

The brief requires at minimum:

`preprocess-data -> train-models -> evaluate-models -> select-model -> run-inference`

Our planned pipeline extends that with downstream business skills:

`preprocess-data -> train-models -> evaluate-models -> select-model -> run-inference -> explain-risk -> recommend-action`

### Required model comparison

The brief requires at least two candidate models trained and compared on the same held-out data.

Our current candidate models are:

- `logistic_regression`
- `random_forest`

This choice is intentional:

- logistic regression gives us an interpretable baseline
- random forest gives us a stronger nonlinear comparison
- together they support fair and explainable model selection

### Required downstream business output

The brief says Gradio must show the downstream skill output, not just probabilities.

Our downstream outputs are:

- business-facing risk explanation
- recommended action

### Required multi-class metrics

Because this is a multi-class classification task, the assignment requires:

- per-class precision
- per-class recall
- per-class F1-score
- macro F1
- weighted F1

And at least one relevant visualisation, such as:

- confusion matrix heatmap
- per-class metric bar chart

### Required SKILL.md design

The brief requires one `SKILL.md` per pipeline stage with clear state inputs, outputs, execution instructions, and business-readable behavior.

Current skill files:

- `skills/preprocess-data.md`
- `skills/train-models.md`
- `skills/evaluate-models.md`
- `skills/select-model.md`
- `skills/run-inference.md`
- `skills/explain-risk.md`
- `skills/recommend-action.md`

## 3. Business Case

The system is meant for a non-technical business user such as:

- credit risk analyst
- credit operations officer
- lending portfolio manager

The intended business question is:

`Given a customer's latest monthly profile, should this account be treated as healthy, monitored, or escalated?`

Expected business value:

- earlier identification of deteriorating credit quality
- more consistent treatment of medium-risk accounts
- faster triage for clearly risky accounts
- clearer communication from model output to business action

## 4. Dataset And Core Analytical Decisions

Expected dataset file:

- `train.csv` at the repository root

Important characteristics of the dataset:

- target column: `Credit_Score`
- repeated monthly records per `Customer_ID`
- messy raw data with placeholders, malformed values, and missing values

First-principles decisions we have already locked in:

### Keep the original target

We are not collapsing the task into binary classification unless a later analytical reason forces us to.

Reason:

- the dataset already provides three classes
- the assignment supports multi-class evaluation
- collapsing to binary would throw away useful business signal

### Split by customer, not by row

We must avoid letting the same customer appear in both training and test sets.

Reason:

- each customer appears across multiple months
- random row splitting would create leakage
- leakage would weaken both the evaluation and the report justification

### Separate cleaning from leakage control

We treat these as different concerns:

- cleaning: placeholders, malformed values, impossible values, missingness
- feature engineering: transforming fields into usable model inputs
- leakage control: making sure training and testing remain fair

This distinction matters because the rubric rewards justified preprocessing, not superficial preprocessing.

## 5. Current Repository Structure

### Source code

- `src/bt5151_credit_risk/config.py`
- `src/bt5151_credit_risk/state.py`
- `src/bt5151_credit_risk/profile.py`
- `src/bt5151_credit_risk/preprocess.py`
- `src/bt5151_credit_risk/train.py`
- `src/bt5151_credit_risk/evaluate.py`
- `src/bt5151_credit_risk/business.py`
- `src/bt5151_credit_risk/llm.py`
- `src/bt5151_credit_risk/graph.py`

### Notebook

- `bt5151_credit_risk_pipeline.ipynb`

### Skills

- `skills/`

### Tests

- `tests/`

### Planning docs

- `docs/plans/`
- `docs/notes/`

## 6. What Is Already Implemented

The current branch already includes a real working backbone.

### Implemented now

- Python package scaffold for the project
- shared config and pipeline state model
- dataset profiling helper
- initial split-safe preprocessing scaffold
- two candidate model definitions
- multi-class metric computation and model selection helper
- downstream business explanation and recommendation stages
- real LangGraph pipeline with distinct nodes
- OpenAI-backed JSON response layer for downstream skills
- notebook wired to the compiled graph
- environment setup using `.env` and `python-dotenv`
- automated tests for package structure, state, preprocessing, training, evaluation, graph, and notebook wiring

### Verified now

Current local verification command:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest -v
```

Current status:

- `14` tests passing

## 7. What Is Not Done Yet

This section is important. The current repo is a strong backbone, but it is not yet the finished BT5151 submission.

### Still needed for the assignment

- deeper dataset-specific preprocessing beyond the current minimal scaffold
- explicit handling and justification of missing values, noisy values, and class imbalance
- final train or validation or test methodology write-up and evidence
- notebook evaluation visualisations required by the brief
- stronger written model selection reasoning tied to business consequences
- at least three documented end-to-end pipeline runs
- at least one challenging or failure-prone test case
- discussion of at least two actual failure modes from real runs
- Gradio interface polish for non-technical business use
- final report
- final slides and recorded presentation
- AI Usage Declaration in the report

### Important warning

The notebook now runs the real graph path, but the current business stages depend on `OPENAI_API_KEY` being present in `.env`. No hardcoded keys should ever be committed.

## 8. Roadmap To Reach Submission Quality

This roadmap is based directly on the assignment requirements and rubric priorities.

### Phase 1: Finish the ML foundation

Goal:

Make the preprocessing and evaluation fully defensible.

Tasks:

- audit raw `train.csv` carefully
- document missing values, placeholders, malformed values, and suspicious outliers
- improve preprocessing logic for real dataset columns
- preserve grouped splitting by `Customer_ID`
- confirm no target leakage or split leakage
- consider whether feature engineering is needed for fields like `Credit_History_Age` or loan-related text columns

Why this matters:

- Component 1 places the highest weight on dataset preparation and fair training
- weak preprocessing or weak split logic will undermine the whole project

### Phase 2: Strengthen model comparison and evaluation

Goal:

Make the ML comparison rigorous and report-ready.

Tasks:

- train both candidate models on the finalized processed data
- report all required multi-class metrics on held-out data
- build confusion matrix heatmap
- build per-class metric visualisation
- compare the two models clearly and fairly
- write a justified final selection based on evidence, not preference

Why this matters:

- the brief explicitly says one-model-only work cannot score above 70 overall
- the rubric rewards reproducibility, fairness, and business interpretation

### Phase 3: Finalize the SKILL.md contracts

Goal:

Make each stage explainable, coherent, and ready for report inclusion.

Tasks:

- audit every `skills/*.md` file against Section 4.2 of the brief
- ensure each file clearly states:
  - what it reads from state
  - what it writes back to state
  - how it executes
  - what output format it returns
- make downstream skills business-readable

Why this matters:

- Component 2 is worth 20 points
- unclear or inconsistent state contracts will cost marks even if the code runs

### Phase 4: Improve full pipeline and business output

Goal:

Make the full system persuasive to a non-technical user.

Tasks:

- confirm graph works end-to-end on real dataset cases
- make sure downstream skills use prediction, confidence, and evaluation context coherently
- improve Gradio so the output is business-facing first
- ensure confidence is visible without forcing the user to understand model internals

Why this matters:

- the brief requires downstream output, not raw scores
- Component 3 rewards meaningful ML-to-agent integration

### Phase 5: Do the required end-to-end testing

Goal:

Collect evidence for Component 5 and the report.

Tasks:

- document at least three complete pipeline runs
- include at least one challenging or messy case
- for each run, record:
  - input
  - expected business output
  - actual output
  - confidence
- identify at least two actual failure modes from observed runs
- explain why they happen and what business consequence they would have

### Phase 6: Produce the final submission materials

Goal:

Complete the remaining deliverables cleanly.

Tasks:

- write the technical report
- prepare slides
- record the presentation
- verify the ZIP contains all required items except the team's agreed Colab workflow variation

## 9. Suggested Team Work Split

This is the cleanest way to split work while keeping ownership clear.

### Workstream A: Data and preprocessing

Owner responsibilities:

- raw data audit
- cleaning rules
- missing value strategy
- feature engineering
- grouped split justification

Deliverables:

- improved preprocessing code
- preprocessing notes for the report
- evidence that leakage has been controlled

### Workstream B: Model training and evaluation

Owner responsibilities:

- train both candidate models
- compute required metrics
- create evaluation tables and visualisations
- write model selection justification draft

Deliverables:

- final metrics
- confusion matrix and per-class plots
- evidence-backed winner selection

### Workstream C: Agent pipeline and skills

Owner responsibilities:

- audit LangGraph state flow
- keep stage boundaries clean
- align code and `SKILL.md`
- improve downstream prompt behavior if needed

Deliverables:

- final graph behavior
- final `skills/*.md`
- clear state chain for report diagram

### Workstream D: Gradio, system testing, and demo evidence

Owner responsibilities:

- improve Gradio UX
- document three full test runs
- collect screenshots and demo evidence
- record failure cases and improvements

Deliverables:

- polished demo flow
- testing evidence for report
- presentation-ready screenshots

### Workstream E: Report and slides

Owner responsibilities:

- compile evidence from all workstreams
- draft the report structure required by the brief
- prepare slides and presentation narrative
- write the AI Usage Declaration honestly

Deliverables:

- final PDF report
- final `.pptx`
- final presentation storyline

## 10. Setup And Run

### Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

Create `.env` in the project root:

```env
OPENAI_API_KEY=your_real_key_here
OPENAI_MODEL=gpt-4o-mini
```

### Run tests

```bash
source .venv/bin/activate
PYTHONPATH=src pytest -v
```

### Open the notebook

```bash
source .venv/bin/activate
jupyter notebook bt5151_credit_risk_pipeline.ipynb
```

### Execute the notebook from terminal

```bash
source .venv/bin/activate
jupyter nbconvert --execute --to notebook --inplace bt5151_credit_risk_pipeline.ipynb
```

## 11. Team Rules For Staying Aligned With The Brief

To avoid drifting away from the assignment, we should keep these rules visible:

- do not switch to a different business problem unless the team agrees explicitly
- do not use random row splitting
- do not train only one model
- do not show raw probabilities as the main user-facing output
- do not hardcode API keys
- do not let AI make the final analytical decisions for model choice, evaluation interpretation, or report writing
- do not invent failure analysis without real runs

## 12. AI Usage Guardrail

The assignment allows AI to help with coding, debugging, refactoring, and boilerplate, but the group must still own:

- dataset understanding
- preprocessing rationale
- model choice
- evaluation interpretation
- SKILL.md design decisions
- report writing
- presentation explanation

That means:

- AI can help us build faster
- AI cannot replace our analytical judgement
- every teammate should be able to explain the code and decisions in the final presentation

## 13. Current Bottom Line

This repo already has the backbone of a BT5151-compliant system:

- real LangGraph pipeline
- two candidate ML models
- downstream business stages
- notebook wired to the real graph
- tests passing

The work now is not to restart the project. The work now is to harden the data logic, produce evaluation evidence, document actual failures, and turn the backbone into a submission-ready system.
