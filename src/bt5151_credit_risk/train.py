import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from bt5151_credit_risk.config import RANDOM_SEED
from bt5151_credit_risk.llm import call_json_response
from bt5151_credit_risk.skill_prompts import load_skill_prompt

# Subsample threshold for Optuna tuning — datasets larger than this are
# downsampled during the hyperparameter search.  Hyperparameter rankings are
# stable with 15-20k rows so this does not hurt quality, but it cuts per-fit
# time dramatically (RF on 66k rows ≈ 26s vs ~4s on 15k).
_TUNE_SUBSAMPLE_MAX = 15_000

logger = logging.getLogger(__name__)

# Suppress Optuna's per-trial logging — we log results ourselves.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _normalize_validation_policy(validation_policy=None, group_values=None, time_values=None):
    policy = dict(validation_policy or {})
    policy_type = policy.get("type")
    if not policy_type:
        if time_values is not None:
            policy_type = "temporal"
        elif group_values is not None:
            policy_type = "grouped_entity"
        else:
            policy_type = "iid_stratified"
    return {
        "type": policy_type,
        "group_column": policy.get("group_column"),
        "time_column": policy.get("time_column"),
        "stratify_target": bool(policy.get("stratify_target", policy_type == "iid_stratified")),
    }


def _slice_sidecar(values, indices):
    if values is None:
        return None
    return np.asarray(values)[indices]


def _prepare_policy_aligned_data(
    frame,
    target,
    sample_weights=None,
    group_values=None,
    time_values=None,
    validation_policy=None,
):
    policy = _normalize_validation_policy(validation_policy, group_values, time_values)
    if policy["type"] != "temporal" or time_values is None:
        return frame, target, sample_weights, group_values, time_values, policy

    raw_times = pd.Series(time_values)
    parsed_times = pd.to_datetime(raw_times, errors="coerce")
    if parsed_times.notna().any():
        fill_value = parsed_times.max()
        sort_key = parsed_times.fillna(fill_value)
    else:
        sort_key = raw_times.astype(str)

    order = np.argsort(sort_key.to_numpy(), kind="mergesort")
    return (
        frame.iloc[order].reset_index(drop=True),
        target.iloc[order].reset_index(drop=True),
        _slice_sidecar(sample_weights, order),
        _slice_sidecar(group_values, order),
        _slice_sidecar(time_values, order),
        policy,
    )


def _select_rows(
    frame,
    target,
    indices,
    sample_weights=None,
    group_values=None,
    time_values=None,
):
    return (
        frame.iloc[indices].reset_index(drop=True),
        target.iloc[indices].reset_index(drop=True),
        _slice_sidecar(sample_weights, indices),
        _slice_sidecar(group_values, indices),
        _slice_sidecar(time_values, indices),
    )


def _build_holdout_indices(target, policy, group_values=None, test_size=0.15):
    total_rows = len(target)
    if total_rows < 2:
        raise ValueError("Need at least 2 rows to build a validation holdout.")

    all_indices = np.arange(total_rows)

    if policy["type"] == "grouped_entity" and group_values is not None:
        groups = np.asarray(group_values)
        unique_groups = pd.Series(groups).dropna().nunique()
        if unique_groups >= 2:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=RANDOM_SEED,
            )
            train_idx, val_idx = next(splitter.split(all_indices, target, groups=groups))
            return np.sort(train_idx), np.sort(val_idx)
        logger.warning("    grouped validation requested but fewer than 2 unique groups found; falling back to IID split")

    if policy["type"] == "temporal":
        val_rows = max(1, int(np.ceil(total_rows * test_size)))
        val_rows = min(val_rows, total_rows - 1)
        split_at = total_rows - val_rows
        return np.arange(split_at), np.arange(split_at, total_rows)

    target_array = np.asarray(target)
    stratify_target = None
    if policy.get("stratify_target", True) and len(np.unique(target_array)) > 1:
        stratify_target = target_array
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=test_size,
        stratify=stratify_target,
        random_state=RANDOM_SEED,
    )
    return np.sort(train_idx), np.sort(val_idx)


def _build_cv_splits(target, policy, group_values=None, n_splits=5):
    total_rows = len(target)
    if total_rows < 2:
        raise ValueError("Need at least 2 rows to build validation folds.")

    if policy["type"] == "grouped_entity" and group_values is not None:
        groups = np.asarray(group_values)
        unique_groups = pd.Series(groups).dropna().nunique()
        fold_count = min(n_splits, unique_groups)
        if fold_count >= 2:
            try:
                from sklearn.model_selection import StratifiedGroupKFold

                splitter = StratifiedGroupKFold(
                    n_splits=fold_count,
                    shuffle=True,
                    random_state=RANDOM_SEED,
                )
                return list(splitter.split(np.zeros((total_rows, 1)), target, groups=groups))
            except Exception:
                splitter = GroupKFold(n_splits=fold_count)
                return list(splitter.split(np.zeros((total_rows, 1)), target, groups=groups))
        logger.warning("    grouped validation requested but fewer than 2 unique groups found; falling back to non-group CV")

    if policy["type"] == "temporal":
        fold_count = min(n_splits, total_rows - 1)
        if fold_count >= 2:
            splitter = TimeSeriesSplit(n_splits=fold_count)
            return list(splitter.split(np.zeros((total_rows, 1))))
        splitter = KFold(n_splits=2, shuffle=False)
        return list(splitter.split(np.zeros((total_rows, 1))))

    min_class_size = int(pd.Series(target).value_counts().min())
    if min_class_size >= 2:
        fold_count = min(n_splits, min_class_size)
        splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=RANDOM_SEED)
        return list(splitter.split(np.zeros((total_rows, 1)), target))

    splitter = KFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
    return list(splitter.split(np.zeros((total_rows, 1))))


def _subsample_for_tuning(
    frame,
    target,
    sample_weights=None,
    group_values=None,
    time_values=None,
    validation_policy=None,
):
    policy = _normalize_validation_policy(validation_policy, group_values, time_values)
    if len(frame) <= _TUNE_SUBSAMPLE_MAX:
        return frame, target, sample_weights, group_values, time_values

    if policy["type"] == "grouped_entity" and group_values is not None:
        groups = np.asarray(group_values)
        unique_groups = pd.Series(groups).dropna().nunique()
        if unique_groups >= 2:
            train_fraction = min(0.95, _TUNE_SUBSAMPLE_MAX / len(frame))
            splitter = GroupShuffleSplit(
                n_splits=1,
                train_size=train_fraction,
                random_state=RANDOM_SEED,
            )
            subset_idx, _ = next(splitter.split(frame, target, groups=groups))
            subset_idx = np.sort(subset_idx)
            return _select_rows(
                frame,
                target,
                subset_idx,
                sample_weights=sample_weights,
                group_values=group_values,
                time_values=time_values,
            )

    if policy["type"] == "temporal" and time_values is not None:
        evenly_spaced = np.linspace(0, len(frame) - 1, _TUNE_SUBSAMPLE_MAX, dtype=int)
        evenly_spaced = np.unique(evenly_spaced)
        return _select_rows(
            frame,
            target,
            evenly_spaced,
            sample_weights=sample_weights,
            group_values=group_values,
            time_values=time_values,
        )

    subset_idx, _ = train_test_split(
        np.arange(len(frame)),
        train_size=_TUNE_SUBSAMPLE_MAX,
        stratify=target,
        random_state=RANDOM_SEED,
    )
    subset_idx = np.sort(subset_idx)
    return _select_rows(
        frame,
        target,
        subset_idx,
        sample_weights=sample_weights,
        group_values=group_values,
        time_values=time_values,
    )


# Create the candidate models we compare in the project.
def build_candidate_models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
        ),
    }


# Ask the LLM to reason appropriate hyperparameter search grids.
def reason_hyperparameter_grids(model_names, train_rows, feature_count, class_distribution, current_metrics):
    system_prompt = load_skill_prompt("reason-hyperparameter-grid")
    payload = {
        "model_names": model_names,
        "train_rows": train_rows,
        "feature_count": feature_count,
        "class_distribution": class_distribution,
        "current_metrics": current_metrics,
    }
    return call_json_response(system_prompt, payload, caller="reason-hyperparameter-grid")


# Translate an LLM-reasoned grid spec into Optuna trial suggestions.
def _suggest_params(trial, grid_spec, model_name):
    params = {}
    for param_name, spec in grid_spec.items():
        # Strip LR params that require saga solver — lbfgs only supports l2.
        if model_name == "logistic_regression" and param_name in (
            "model__l1_ratio", "model__penalty", "model__solver"
        ):
            continue

        param_type = spec.get("type", "float")
        if param_type == "int":
            step = spec.get("step", 1)
            params[param_name] = trial.suggest_int(param_name, spec["low"], spec["high"], step=step)
        elif param_type == "float":
            log = spec.get("log", False)
            params[param_name] = trial.suggest_float(param_name, spec["low"], spec["high"], log=log)
        elif param_type == "categorical":
            choices = spec.get("choices", [])
            params[param_name] = trial.suggest_categorical(param_name, choices)
    return params


# Run Optuna Bayesian optimization for each model using LLM-reasoned grids.
def tune_models(
    models,
    grids,
    train_frame,
    train_target,
    sample_weights=None,
    validation_policy=None,
    train_group_values=None,
    train_time_values=None,
):
    tuned_models = {}
    tuning_results = {}
    trial_histories = {}

    train_frame, train_target, sample_weights, train_group_values, train_time_values, policy = (
        _prepare_policy_aligned_data(
            train_frame,
            train_target,
            sample_weights=sample_weights,
            group_values=train_group_values,
            time_values=train_time_values,
            validation_policy=validation_policy,
        )
    )

    # Subsample for tuning if dataset is large — hyperparameter rankings are
    # stable with 15k rows, but fit time scales linearly with row count.
    if len(train_frame) > _TUNE_SUBSAMPLE_MAX:
        tune_frame, tune_target, tune_weights, tune_group_values, tune_time_values = _subsample_for_tuning(
            train_frame,
            train_target,
            sample_weights=sample_weights,
            group_values=train_group_values,
            time_values=train_time_values,
            validation_policy=policy,
        )
        logger.info("    Subsampled %d → %d rows for tuning CV",
                    len(train_frame), len(tune_frame))
    else:
        tune_frame = train_frame
        tune_target = train_target
        tune_weights = sample_weights
        tune_group_values = train_group_values
        tune_time_values = train_time_values

    # Log class balance of tuning subset — flag if any class drops below 5%.
    _log_class_balance("tuning subset", tune_target)

    cv_splits = _build_cv_splits(
        tune_target,
        policy,
        group_values=tune_group_values,
        n_splits=5,
    )
    logger.info("    tuning validation policy: %s (%d fold(s))", policy["type"], len(cv_splits))

    for model_name, model in models.items():
        grid_spec = grids.get(model_name)
        if not grid_spec:
            logger.info("    %s: no grid provided, using defaults", model_name)
            tuned_models[model_name] = model
            continue

        # Safety net: cap max_depth if the LLM returns an unreasonably high range.
        # Unlimited or very deep trees on 50k+ rows are the #1 training time killer.
        if "max_depth" in grid_spec and grid_spec["max_depth"].get("high", 0) > 20:
            logger.warning("    %s: capping max_depth upper bound from %d to 20",
                           model_name, grid_spec["max_depth"]["high"])
            grid_spec["max_depth"]["high"] = 20

        # Remove n_estimators from tuning grids — RF uses a fixed count and
        # XGBoost relies on early stopping, so tuning this wastes budget.
        if model_name in ("random_forest", "xgboost") and "n_estimators" in grid_spec:
            logger.info("    %s: removing n_estimators from grid (handled by defaults/early stopping)",
                        model_name)
            del grid_spec["n_estimators"]

        # For RF tuning, use fewer trees — 100 is enough to rank hyperparameter
        # configs.  The final retrain uses the model's default (200).
        tune_model = clone(model)
        if model_name == "random_forest":
            tune_model.set_params(n_estimators=100)

        # Build objective closure for this model.
        def make_objective(m_name, m_model, g_spec):
            def objective(trial):
                params = _suggest_params(trial, g_spec, m_name)
                scores = []
                for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
                    X_train = tune_frame.iloc[train_idx]
                    X_val = tune_frame.iloc[val_idx]
                    y_train = tune_target.iloc[train_idx]
                    y_val = tune_target.iloc[val_idx]
                    # Log fold balance on trial 0 only to avoid log spam.
                    if trial.number == 0:
                        _log_class_balance(f"{m_name} fold {fold_i} train", y_train)
                        _log_class_balance(f"{m_name} fold {fold_i} val", y_val)

                    cloned = clone(m_model)
                    cloned.set_params(**params)

                    fit_kwargs = {}
                    if m_name == "xgboost":
                        # Early stopping: 500-round ceiling with patience 30.
                        # On subsampled 12k rows, convergence is fast — 30
                        # rounds is enough patience to avoid premature stops.
                        cloned.set_params(n_estimators=500, early_stopping_rounds=30)
                        fit_kwargs["eval_set"] = [(X_val, y_val)]
                        fit_kwargs["verbose"] = False
                        if tune_weights is not None:
                            fit_kwargs["sample_weight"] = tune_weights[train_idx]
                    elif m_name != "logistic_regression" and tune_weights is not None:
                        # RF doesn't support sample_weight in fit via this path;
                        # it uses class_weight='balanced' instead.
                        pass

                    cloned.fit(X_train, y_train, **fit_kwargs)
                    preds = cloned.predict(X_val)
                    scores.append(f1_score(y_val, preds, average="macro", zero_division=0))

                return float(np.mean(scores))
            return objective

        objective_fn = make_objective(model_name, tune_model, grid_spec)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        n_trials = 10
        logger.info("    %s: tuning with Optuna (%d trials, 5-fold CV)...", model_name, n_trials)
        study.optimize(objective_fn, n_trials=n_trials, n_jobs=1)

        best_params = study.best_trial.params
        best_cv_score = round(study.best_value, 4)
        logger.info("    %s: best_cv_score=%.4f best_params=%s", model_name, best_cv_score, best_params)

        # Log trial history
        trials_log = []
        for t in study.trials:
            trials_log.append({
                "number": t.number,
                "value": round(t.value, 4) if t.value is not None else None,
                "params": t.params,
            })
        trial_histories[model_name] = trials_log

        # Retrain best model on full training data.
        final_model = clone(model)
        final_model.set_params(**best_params)

        fit_kwargs = {}
        if model_name == "xgboost":
            # Two-step early stopping (following XAI case study pattern):
            # Step 1: Fit with early stopping on 15% held-out split to find
            #         optimal tree count. This model is kept only for learning
            #         curves — it trains on 85% of data.
            train_idx, es_idx = _build_holdout_indices(
                train_target,
                policy,
                group_values=train_group_values,
                test_size=0.15,
            )
            X_tr = train_frame.iloc[train_idx]
            X_es = train_frame.iloc[es_idx]
            y_tr = train_target.iloc[train_idx]
            y_es = train_target.iloc[es_idx]
            logger.info(
                "    xgboost early-stopping policy: %s (train=%d, eval=%d)",
                policy["type"],
                len(train_idx),
                len(es_idx),
            )
            es_model = clone(final_model)
            es_model.set_params(n_estimators=1000, early_stopping_rounds=50)
            es_fit_kwargs = {
                "eval_set": [(X_tr, y_tr), (X_es, y_es)],
                "verbose": False,
            }
            if sample_weights is not None:
                from sklearn.utils.class_weight import compute_sample_weight
                es_fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_tr)
            es_model.fit(X_tr, y_tr, **es_fit_kwargs)
            best_n = es_model.best_iteration + 1
            logger.info("    xgboost: early stopping found best_n_trees=%d", best_n)

            # Step 2: Retrain on FULL training data with fixed tree count
            #         (no early stopping). All models see the same data volume.
            final_model.set_params(n_estimators=best_n)
            if sample_weights is not None:
                fit_kwargs["sample_weight"] = sample_weights
            final_model.fit(train_frame, train_target, **fit_kwargs)

            # Attach early-stopped model's evals_result for learning curves
            final_model._es_evals_result = es_model.evals_result()
        else:
            final_model.fit(train_frame, train_target)

        tuned_models[model_name] = final_model
        tuning_results[model_name] = {
            "best_params": best_params,
            "best_cv_score": best_cv_score,
        }

    return tuned_models, tuning_results, trial_histories


# Extract XGBoost learning curves from evals_result after fitting with eval_set.
def extract_learning_curves(model, model_name):
    if model_name != "xgboost":
        return None
    try:
        # Use the early-stopped model's evals_result (attached during two-step
        # training) — the final model was retrained without eval_set.
        evals = getattr(model, "_es_evals_result", None) or model.evals_result()
        if not evals:
            return None
        curves = {}
        for ds_name, metrics in evals.items():
            for metric_name, values in metrics.items():
                key = f"{ds_name}_{metric_name}"
                curves[key] = [round(v, 6) for v in values]
        # Log convergence summary
        val_key = None
        for k in curves:
            if "validation_1" in k:
                val_key = k
                break
        if val_key and curves[val_key]:
            n_rounds = len(curves[val_key])
            final_val = curves[val_key][-1]
            best_val = min(curves[val_key])
            best_round = curves[val_key].index(best_val) + 1
            logger.info("    %s learning curve: %d rounds, best val_loss=%.6f at round %d, final=%.6f",
                        model_name, n_rounds, best_val, best_round, final_val)
        return curves
    except Exception:
        return None


def _log_class_balance(label: str, target) -> None:
    """Log per-class counts and warn if any class falls below 5% of total."""
    counts = pd.Series(target).value_counts().sort_index()
    total = counts.sum()
    parts = ", ".join(f"cls{k}={v} ({100*v/total:.1f}%)" for k, v in counts.items())
    logger.info("    class balance [%s]: %s", label, parts)
    for cls, cnt in counts.items():
        if cnt / total < 0.05:
            logger.warning(
                "    class balance [%s]: class %s has only %.1f%% of samples — "
                "imbalance may cause unstable CV scores",
                label, cls, 100 * cnt / total,
            )


def _grid_size(grid):
    size = 1
    for values in grid.values():
        if isinstance(values, list):
            size *= len(values)
        else:
            size *= 5  # Approximate for range-based specs
    return size
