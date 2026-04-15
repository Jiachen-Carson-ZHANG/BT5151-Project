# BT5151 Agentic Credit Risk Pipeline

An end-to-end agentic ML pipeline for multiclass credit risk classification (`Good` / `Standard` / `Poor`), built on LangGraph. Every analytical stage — from data cleaning to business explanation — is driven by a chain of LLM calls with programmatic validation, self-repair loops, and a three-tier hypothesis system (tested → supported → exploratory) that chains observations across EDA, training, and XAI.

Dataset: `train.csv` (100k rows, 28 columns, Kaggle credit score dataset).
Design intent: the pipeline core is dataset-agnostic; swapping the dataset requires only a new CSV and updated `.env` config.

---

## Architecture

```mermaid
flowchart TD
    subgraph SPEC["Spec Layer"]
        A([Raw CSV]) --> DPS[dataset-policy-spec\nLLM: target · split · leakage policy]
        DPS --> EDA[exploratory-data-analysis\nMI · ANOVA · correlations · skew]
        EDA --> EH[generate-eda-hypotheses\nLLM o4-mini: 3-tier directional predictions]
        EH --> CTS[column-transform-spec\nLLM o3: per-column roles · encoding · imputation]
    end

    subgraph PREP["Preprocessing Loop  ↺ up to 5 attempts"]
        CTS --> GPC[generate-preprocessing-code\nLLM: codegen from spec]
        GPC --> IPC[inspect-preprocessing-code\nstatic analysis · blocklist]
        IPC --> EXP[execute-generated-preprocessing\nsubprocess · 180s timeout]
        EXP --> VAL[validate-preprocessing-output\ndeterministic schema checks]
        VAL --> AUD[review-preprocessing-quality\nLLM: audit feature frame]
        AUD -->|needs_repair| RPC[repair-preprocessing-code\nLLM: fix from audit feedback]
        RPC --> IPC
        AUD -->|pass| FE_START
    end

    subgraph FE["Feature Engineering Loop  ↺ up to 3 attempts"]
        FE_START([FE start]) --> GFE[generate-feature-engineering-code\nLLM o4-mini: dual-view codegen]
        GFE --> IFE[inspect-feature-engineering-code\nstatic analysis]
        IFE --> EFE[execute-feature-engineering\nsubprocess]
        EFE --> VFE[validate-feature-engineering\nnumeric contract · view alignment]
        VFE -->|fail| RFE[repair-feature-engineering-code\nLLM: fix from validation error]
        RFE --> IFE
        VFE -->|pass| TM
    end

    subgraph TRAIN["Training Layer"]
        TM[train-models\nLR · RF · XGB · Optuna · grouped CV]
        TM --> EM[evaluate-models\nper-class metrics · confusion · confidence]
        EM --> TD[training-diagnostics\nLLM o4-mini: capacity · confusion flow · hypothesis validation]
        TD --> SM[select-model\nLLM: justification]
    end

    subgraph XAI["XAI Layer"]
        SM --> GX[global-xai\nSHAP · grouped PFI · PDP + ALE]
        GX --> LX[local-xai\ncasebook: representative · borderline · worst misclass per class]
        LX --> IGX[interpret-global-xai\nLLM o4-mini: cross-method consensus · feature effects]
        IGX --> ILX[interpret-local-xai\nLLM o4-mini: per-class stories · boundary analysis]
        ILX --> PAB[package-analysis-bundle\nfull semantic bundle → lab/logs/]
    end

    subgraph INFER["Inference + Explanation"]
        PAB --> RI[run-inference\npredict · per-prediction SHAP]
        RI --> ER[explain-risk\nLLM: evidence-traced explanation + recommended action]
        ER --> OUT([Output])
    end
```

**Dual-view FE:** The FE stage produces two model-specific feature frames — `linear_view` (one-hot encoded, log-transformed, standardized) and `tree_view` (frequency/ordinal encoded, raw scale) — so each model family gets the representation it performs best with.

**Hypothesis chain:** EDA hypotheses (three-tier) → validated by training diagnostics → cross-checked by global XAI interpretation → grounded in per-case local XAI stories → synthesized by explain-risk into customer-facing output. Every claim carries a `tier` (tested / supported / exploratory) and `layer` (eda / training / global_xai / local_xai) tag.

---

## Models

| Model | Tuning | View |
|---|---|---|
| Logistic Regression | Optuna C search (log scale) | `linear_view` |
| Random Forest | Optuna depth + min_samples_split | `tree_view` |
| XGBoost | Early stopping on 20% holdout, retrain full train on best round | `tree_view` |

Validation: `GroupShuffleSplit` by `Customer_ID` (entity-level, no leakage across splits).

---

## XAI Methods

| Method | When | Purpose |
|---|---|---|
| SHAP (global + per-case) | Always | Feature attribution, beeswarm, dependence plots |
| Grouped PFI | Always | Correct PFI for one-hot features — permutes entire original feature as a group |
| PDP (per-class curves) | Top continuous features | Average marginal effect view across the feature range |
| ALE (per-class curves) | Same top continuous features | Correlation-robust complement to PDP on the same features |

Local casebook per class: **representative** (most confident correct), **borderline** (least confident correct), **worst misclassification** (most confident wrong + which class it confused with). Up to 9 cases total.

---

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

```bash
# Full pipeline (row 42 for inference demo)
PYTHONPATH=src python run_stage.py full 42

# Stop early at specific stages
PYTHONPATH=src python run_stage.py specs        # EDA + column-transform-spec
PYTHONPATH=src python run_stage.py preprocess   # + preprocessing loop + FE loop
PYTHONPATH=src python run_stage.py evaluate     # + training + model selection
```

```bash
# Tests
PYTHONPATH=src pytest tests/ -q   # 93 passing
```

---

## Per-node Model Config

Each LLM node can be overridden independently in `.env`:

```env
OPENAI_MODEL=gpt-4o-mini                         # global default
OPENAI_MODEL_COLUMN_TRANSFORM_SPEC=o3            # spec node — most consequential, uses o3
OPENAI_REASONING_EFFORT_COLUMN_TRANSFORM_SPEC=high
OPENAI_MODEL_GENERATE_FEATURE_ENGINEERING_CODE=o4-mini
OPENAI_MODEL_GENERATE_EDA_HYPOTHESES=o4-mini
OPENAI_MODEL_GENERATE_TRAINING_DIAGNOSTICS=o4-mini
OPENAI_MODEL_INTERPRET_GLOBAL_XAI=o4-mini
OPENAI_MODEL_INTERPRET_LOCAL_XAI=o4-mini
OPENAI_MODEL_EXPLAIN_RISK=o4-mini
```

---

## Repository Layout

```
src/bt5151_credit_risk/   # pipeline modules
  graph.py                # LangGraph graph definition + all node functions
  preprocess.py           # preprocessing codegen, validation, repair
  feature_engineering.py  # FE codegen, validation, repair
  train.py                # model definitions, Optuna tuning, grouped CV
  evaluate.py             # per-class metrics, confusion matrix, confidence stats
  xai.py                  # SHAP, grouped PFI, PDP, ALE, casebook selection
  hypotheses.py           # LLM calls: EDA hypotheses, training diagnostics, XAI interpretation
  business.py             # LLM calls: explain-risk (merged explain + recommend)
  state.py                # CreditRiskState (Pydantic)
  llm.py                  # OpenAI client, per-caller model override, JSON retry
  config.py               # dataset config (target column, group column)

skills/                   # one skill prompt per LLM node
tests/                    # 93 tests covering all modules and graph wiring

lab/
  experiments/            # one record per pipeline run (goals, results, findings)
  logs/                   # run logs (stage_full_YYYYMMDD_HHMMSS.log) + analysis bundles
  analysis/               # architectural decision notes and design trade-offs
  backlog.md              # deferred ideas with rationale

run_stage.py              # CLI entry point — run pipeline to any stage
```

---

## Gradio Demo App

After a full pipeline run, an interactive prediction UI is available:

```bash
# Step 1 — run pipeline and save trained state (one-off, ~1–2 h)
PYTHONPATH=src python run_stage.py full 42 --save-cache

# Step 2 — launch the app (instant reload from cache)
PYTHONPATH=src python app.py
# → http://localhost:7860
```

Three tabs: **Customer Prediction** (row picker → prediction + SHAP waterfall + LLM explanation + recommended action), **Model Overview** (eval metrics + global SHAP importance), **EDA Hypotheses** (three-tier hypothesis output from EDA layer).

---

## XAI Analytics Flow

The pipeline doesn't just produce numbers — it chains observations, hypotheses, and interpretations across five analytical layers. Each layer receives the prior layer's findings as context, so the final customer explanation is grounded in a complete evidence trail.

```mermaid
flowchart TD

    subgraph L1["① EDA — Discover"]
        EDA_PROG["Programmatic analysis\nMutual Information · ANOVA F-stats\nPearson correlations · skewness\nmissingness patterns · high cardinality"]
        EDA_LLM(["LLM  generate-eda-hypotheses"])
        EDA_T["🔬 Tested predictions\n'LR macro_f1 ≥ 0.55 from top-2 MI features'\n'Annual_Income is the single strongest separator'"]
        EDA_S["💡 Supported conjectures\n'Standard class will have lowest recall —\nclass-conditional means overlap both neighbours'"]
        EDA_E["🔭 Exploratory leads\n'Monthly_Balance × Num_Credit_Inquiries\nmay separate borderline Standard from Good'"]
        EDA_PROG --> EDA_LLM
        EDA_LLM --> EDA_T & EDA_S & EDA_E
    end

    subgraph L2["② Training — Validate"]
        TRAIN_DATA["Per-class F1 · Precision · Recall\nConfusion matrix · Optuna best params\nXGB learning curve (1500 rounds max)\nConfidence on correct vs wrong preds\nWhich class confuses which"]
        TRAIN_LLM(["LLM  training-diagnostics"])
        HYPS_VAL["Hypothesis validation  ✓ / ✗\nEDA tested predictions confirmed or refuted\nwith actual metric values from training"]
        CAPACITY["Capacity diagnosis\nLR: C value → over-regularised?\nRF: max_depth → interaction-heavy signal?\nXGB: best_n_trees vs cap → still learning?"]
        CONFUSION["Confusion flow analysis\nAsymmetric? Good→Standard >> Standard→Good?\nWhich class is hardest and why?\nConfidence gap on correct vs wrong preds"]
        NEW_T2["New training hypotheses\nfor XAI layer to cross-check"]
        TRAIN_DATA --> TRAIN_LLM
        TRAIN_LLM --> HYPS_VAL & CAPACITY & CONFUSION & NEW_T2
    end

    subgraph L3["③ Global XAI — Cross-method Consensus"]
        SHAP_G["SHAP\nMean ❙SHAP❙ importance ranking\nBeeswarm: value × direction per class\nDependence plots: interaction effects"]
        PFI_G["Grouped PFI\nPermutes entire original feature\n(all one-hot columns together)\nCross-validates SHAP ranking"]
        PDP_G["PDP  per-class curves\nAverage marginal effect\nfor uncorrelated top features\n(avoids corr. bias)"]
        ALE_G["ALE  per-class curves\nCorrelation-robust complement\ncomputed where ❙r❙ > 0.5\nDivergence from PDP = corr. signal"]
        IGX_LLM(["LLM  interpret-global-xai"])
        IGX_OUT["Cross-method consensus\nSHAP ↔ PFI: agree or diverge?\nPDP ↔ ALE divergence = corr. bias detected\nFeature effect shapes: monotone · non-linear · threshold\nWhich features shift probability mass across classes"]
        SHAP_G & PFI_G & PDP_G & ALE_G --> IGX_LLM --> IGX_OUT
    end

    subgraph L4["④ Local XAI — Case Stories"]
        CASEBOOK["9 strategic cases  (3 per class)\n🟢 Representative — most confident correct\n🟡 Borderline — least confident correct\n🔴 Worst misclass — most confident wrong"]
        SHAP_LOCAL["Per-case SHAP waterfall\nTop-10 features · base value · contributions\nWhich features drove the wrong class prediction"]
        ILX_LLM(["LLM  interpret-local-xai"])
        ILX_OUT["Per-class failure stories\nBorderline case thinness — how close is the boundary?\nSHAP signature of confident errors\nConfusion asymmetry pattern across class pairs\nDecision boundary feature analysis"]
        CASEBOOK & SHAP_LOCAL --> ILX_LLM --> ILX_OUT
    end

    subgraph L5["⑤ Inference — Individual Evidence"]
        PRED["run-inference  row N\npredict_proba → predicted label\nPer-prediction SHAP waterfall  top-10\nPDP position: where on the risk curve?\nConfidence diagnosis vs typical correct-pred conf\nNearest casebook case  cosine similarity on SHAP"]
        ER_LLM(["LLM  explain-risk"])
        ER_OUT["Risk level + confidence band\nKey drivers with raw feature values + global rank\nHypothesis validation notes  confirmed / refuted from bundle\nCounterfactual when boundary evidence supports it\nRecommended action + monitoring conditions"]
        PRED --> ER_LLM --> ER_OUT
    end

    %% Cross-layer hypothesis chain
    EDA_T & EDA_S & EDA_E -->|"tested predictions to validate"| TRAIN_LLM
    CAPACITY & CONFUSION & NEW_T2 -->|"capacity + confusion hypotheses"| IGX_LLM
    IGX_OUT -->|"global context for local grounding"| ILX_LLM
    HYPS_VAL & IGX_OUT & ILX_OUT -->|"full analysis bundle summary"| ER_LLM
```

**Reading the diagram:** Rounded nodes `( )` are LLM calls. Square nodes `[ ]` are data, observations, or findings. Arrows between layers carry the hypothesis chain — each LLM receives not just the raw numbers from its own layer but the interpreted findings from every prior layer, so the final explanation is evidence-traced back to EDA.

---

## Current Best Result

Run 009 (before XAI overhaul): **XGBoost macro_f1 = 0.8017** on grouped entity split.

Active work is on the XAI overhaul (runs 010–012): split interpret nodes, verbatim analysis bundle, dual-view encoding contracts, deferred categorical handling. Next run expected to recover Run 009 performance with full XAI chain operational.
