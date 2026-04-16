import numpy as np
import pandas as pd
import pytest

from bt5151_credit_risk.shortcut_audit import (
    ablate_suspects,
    detect_shortcut_suspects,
    run_shortcut_audit,
)


def test_detect_shap_pfi_divergence():
    """SHAP rank <= 5 but PFI rank > 10 → shap_pfi_divergence signal."""
    shap = [
        {"feature": "Ghost", "mean_abs_shap": 0.10},
        {"feature": "Real_A", "mean_abs_shap": 0.08},
        {"feature": "Real_B", "mean_abs_shap": 0.07},
        {"feature": "Real_C", "mean_abs_shap": 0.06},
        {"feature": "Real_D", "mean_abs_shap": 0.05},
        {"feature": "F6", "mean_abs_shap": 0.04},
        {"feature": "F7", "mean_abs_shap": 0.03},
        {"feature": "F8", "mean_abs_shap": 0.02},
        {"feature": "F9", "mean_abs_shap": 0.01},
        {"feature": "F10", "mean_abs_shap": 0.005},
    ]
    # PFI ranks Ghost at position 15 — divergence
    pfi = [
        {"original_feature": f"Real_{c}", "importance_mean": 0.2 - i * 0.01}
        for i, c in enumerate("ABCD")
    ]
    pfi += [{"original_feature": f"F{i}", "importance_mean": 0.01 * (20 - i)} for i in range(5, 15)]
    pfi += [{"original_feature": "Ghost", "importance_mean": 0.001}]

    suspects = detect_shortcut_suspects(shap, pfi)
    ghost = next(s for s in suspects if s["feature"] == "Ghost")
    assert "shap_pfi_divergence" in ghost["signals"]
    assert ghost["shap_rank"] == 1
    assert ghost["pfi_rank"] == 15


def test_detect_dominance():
    """A single feature holding >20% of top-10 SHAP mass → dominance signal."""
    shap = [
        {"feature": "Dominator", "mean_abs_shap": 1.0},
        {"feature": "B", "mean_abs_shap": 0.5},
        {"feature": "C", "mean_abs_shap": 0.4},
        {"feature": "D", "mean_abs_shap": 0.3},
        {"feature": "E", "mean_abs_shap": 0.2},
        {"feature": "F", "mean_abs_shap": 0.15},
        {"feature": "G", "mean_abs_shap": 0.12},
        {"feature": "H", "mean_abs_shap": 0.1},
        {"feature": "I", "mean_abs_shap": 0.08},
        {"feature": "J", "mean_abs_shap": 0.05},
    ]
    pfi = [{"original_feature": e["feature"], "importance_mean": e["mean_abs_shap"]} for e in shap]
    suspects = detect_shortcut_suspects(shap, pfi)
    dominator = next(s for s in suspects if s["feature"] == "Dominator")
    assert "dominance" in dominator["signals"]
    assert dominator["shap_share"] > 0.20


def test_detect_calendar_shortcut():
    """`Month` in SHAP top-10 → calendar_shortcut signal."""
    shap = [{"feature": f"F{i}", "mean_abs_shap": 0.1 - i * 0.005} for i in range(9)]
    shap.insert(2, {"feature": "Month", "mean_abs_shap": 0.08})
    pfi = [{"original_feature": e["feature"], "importance_mean": e["mean_abs_shap"]} for e in shap]
    suspects = detect_shortcut_suspects(shap, pfi)
    month = next(s for s in suspects if s["feature"] == "Month")
    assert "calendar_shortcut" in month["signals"]


def test_ablation_cap_respected():
    """5 flagged suspects → only max_ablations=2 are actually ablated."""
    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    X = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0],
        "B": [0.1, 0.2, 0.3, 0.4],
        "C": [5, 6, 7, 8],
        "D": [9, 10, 11, 12],
        "E": [0, 1, 0, 1],
    })
    y = np.array([0, 0, 0, 0])
    suspects = [{"feature": c, "signals": ["dominance"], "shap_rank": i+1, "pfi_rank": i+1, "shap_share": 0.3}
                for i, c in enumerate(X.columns)]

    result = ablate_suspects(DummyModel(), X, y, suspects, max_ablations=2)
    assert len(result) == 2


def test_verdict_thresholds_weak_signal():
    """|Δ macro_f1| < 0.005 → weak_signal."""
    class Model:
        def predict(self, X):
            # Return same prediction regardless of feature zeroing.
            return np.tile([0, 1, 2], len(X) // 3 + 1)[: len(X)]

    X = pd.DataFrame({
        "suspect": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "safe": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })
    y = np.array([0, 1, 2, 0, 1, 2])
    suspects = [{"feature": "suspect", "signals": ["dominance"], "shap_rank": 1, "pfi_rank": 1, "shap_share": 0.5}]
    result = ablate_suspects(Model(), X, y, suspects, max_ablations=1)
    assert result[0]["verdict"] == "weak_signal"


def test_verdict_thresholds_real_signal():
    """Δ macro_f1 < -0.02 → real_signal."""
    class Model:
        def predict(self, X):
            # If suspect == median value, predict all wrong; otherwise correct.
            median_val = X["suspect"].median()
            y = []
            for i, row in X.iterrows():
                if row["suspect"] == median_val:
                    y.append((i + 1) % 3)  # wrong prediction
                else:
                    y.append(i % 3)  # correct
            return np.array(y)

    X = pd.DataFrame({
        "suspect": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    y = np.array([0, 1, 2, 0, 1, 2])
    suspects = [{"feature": "suspect", "signals": ["dominance"], "shap_rank": 1, "pfi_rank": 1, "shap_share": 0.5}]
    result = ablate_suspects(Model(), X, y, suspects, max_ablations=1)
    assert result[0]["verdict"] == "real_signal"
    assert result[0]["delta_macro_f1"] < -0.02


def test_run_shortcut_audit_end_to_end():
    """run_shortcut_audit wires detect + ablate and returns the expected dict shape."""
    class Model:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    shap = [{"feature": "Month", "mean_abs_shap": 0.5}]
    shap += [{"feature": f"F{i}", "mean_abs_shap": 0.1 - 0.01 * i} for i in range(1, 10)]
    pfi = [{"original_feature": e["feature"], "importance_mean": 0.01} for e in shap]

    X = pd.DataFrame({e["feature"]: [1.0, 2.0, 3.0, 4.0] for e in shap})
    y = np.array([0, 0, 0, 0])

    audit = run_shortcut_audit(Model(), X, y, shap, pfi, class_names=["A", "B", "C"], max_ablations=2)
    assert "suspects" in audit
    assert "ablations" in audit
    assert len(audit["ablations"]) <= 2
    month_ablation = next(a for a in audit["ablations"] if a["feature"] == "Month")
    assert "verdict" in month_ablation
