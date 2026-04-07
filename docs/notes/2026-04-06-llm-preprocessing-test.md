# LLM Preprocessing Test Log

Date: 2026-04-06

## Purpose

This note records a small experiment: how far a general-purpose LLM-generated preprocessing notebook can go on the BT5151 credit-score dataset when given broad coding flexibility.

The notebook under test was [test.ipynb](/home/tough/BT5151%20GroupProject/test.ipynb), applied to [train.csv](/home/tough/BT5151%20GroupProject/train.csv).

## Dataset Context

- Dataset: customer-month credit score records
- Size: 100,000 rows
- Customers: 12,500
- Time structure: 8 months per customer
- Target: `Credit_Score` with classes `Good`, `Standard`, `Poor`

This is a multi-class credit risk monitoring problem with repeated customer records over time.

## What The Notebook Did Well

The generated notebook produced a credible first-pass preprocessing pipeline:

- standardized raw string fields
- replaced several obvious placeholder values with nulls
- extracted numeric values from corrupted numeric-like fields
- capped impossible or highly suspicious values
- parsed `Credit_History_Age` into numeric months
- normalized several categorical text fields
- performed broad imputation
- produced two outputs:
  - `credit_score_cleaned.csv`
  - `credit_score_model_ready.csv`
- converted categorical variables into model-friendly columns
- expanded `Type_of_Loan` into multi-hot encoded features
- created a numeric target label

For a one-shot script, this is a strong draft. It handled much of the mechanical cleaning work correctly and ran end-to-end.

## What The Test Showed

The notebook is useful as a preprocessing draft, but it is not sufficient as a final preprocessing stage for a valid ML pipeline.

### Strengths

- Good at mechanical cleanup
- Good at quickly converting messy columns into usable numeric/categorical forms
- Good at producing a model-oriented table with minimal manual effort

### Weaknesses

- It left the original target column `Credit_Score` inside the so-called model-ready dataset, so the export was not fully safe for training.
- It used customer-level imputation before any train/test split logic. Because the same customer appears across multiple months, this creates leakage risk across evaluation boundaries.
- It did not define train/validation/test split artifacts.
- It did not produce an explicit preprocessing audit report for downstream stages.
- It did not save preprocessing rules as a reusable fitted transformation object.
- It did not define any agent-state contract for an ML pipeline.
- It did not include assertions or tests proving the exported table was safe for modeling.

## Classification Of The Notebook Work

### True data cleaning

- placeholder removal
- malformed numeric cleanup
- invalid value capping
- text normalization
- missing value imputation

### Feature engineering

- `Credit_History_Age` to numeric months
- multi-hot encoding of `Type_of_Loan`
- one-hot encoding of categorical fields
- month numeric encoding

### Leakage and experimental design concerns

- target column retained in model-ready export
- preprocessing performed without split-aware safeguards
- customer-wise imputation before fair evaluation design

These last items are not just "cleaning" problems. They are pipeline integrity problems spanning preprocessing, feature engineering, splitting, and evaluation.

## Holistic Judgment

This notebook is:

- good as a one-shot LLM-generated preprocessing draft
- promising as the base of a `preprocess-data` stage
- not sufficient as a final BT5151 preprocessing stage without human revision

Practical judgment:

- mechanical quality: good
- methodological quality: incomplete
- agentic-pipeline readiness: partial

## Main Lesson

A general LLM can generate a large share of the syntax and transformation logic for preprocessing, but it is less reliable on the boundaries that make an ML pipeline valid:

- leakage control
- split design
- train-only versus test-safe preprocessing
- downstream artifact design
- evaluation integrity

The experiment suggests that a one-shot LLM prompt can get a team roughly most of the way to a usable preprocessing draft, but not all the way to a trustworthy ML pipeline stage.

## Carry-Forward Into BT5151 Design

We are keeping the previously approved BT5151 direction:

- business case: monthly credit risk monitoring
- task: multi-class classification of `Good`, `Standard`, `Poor`
- split strategy: grouped by `Customer_ID`
- candidate models: interpretable baseline plus stronger nonlinear model
- required downstream stages: translate prediction into business-facing risk explanation and recommended action

This test reinforces the need for a human-designed `preprocess-data` stage that is:

- split-aware
- leakage-aware
- documented through `SKILL.md`
- connected cleanly to downstream evaluation and inference stages
