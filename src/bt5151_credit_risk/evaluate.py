from sklearn.metrics import classification_report, f1_score


# Compute the core multi-class metrics used for model comparison.
def compute_multiclass_metrics(y_true, y_pred, class_names):
    label_indices = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "per_class": {name: report[name] for name in class_names},
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# Pick the winning model from the evaluation summary.
def choose_best_model(results):
    model_name = max(
        results,
        key=lambda name: (results[name]["macro_f1"], results[name]["weighted_f1"]),
    )
    return {
        "model_name": model_name,
        "justification": (
            f"Selected {model_name} based on stronger macro_f1 and weighted_f1."
        ),
    }
