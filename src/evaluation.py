"""Metrics, plots, timing and significance testing for the four models."""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import learning_curve

POS = "p"  # poisonous is the positive class: missing it is the dangerous error


def _proba_pos(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(POS)]


def metrics_table(models, X_train, y_train, X_test, y_test):
    y_bin = (y_test == POS).astype(int)
    rows = []
    for name, m in models.items():
        pred = m.predict(X_test)
        proba = _proba_pos(m, X_test)
        rows.append({
            "Model": name,
            "Train Acc": m.score(X_train, y_train),
            "Test Acc": accuracy_score(y_test, pred),
            "Precision (p)": precision_score(y_test, pred, pos_label=POS),
            "Recall (p)": recall_score(y_test, pred, pos_label=POS),
            "F1 (p)": f1_score(y_test, pred, pos_label=POS),
            "ROC-AUC": roc_auc_score(y_bin, proba),
            "Log-loss": log_loss(y_bin, proba),
            "Poisonous called edible": int(((y_test == POS) & (pred == "e")).sum()),
        })
    return pd.DataFrame(rows)


def timing_table(models, X_train, y_train, X_test, n_repeats=5):
    rows = []
    for name, m in models.items():
        fits, preds = [], []
        for _ in range(n_repeats):
            mm = clone(m)
            t0 = time.perf_counter(); mm.fit(X_train, y_train); fits.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); mm.predict(X_test); preds.append(time.perf_counter() - t0)
        rows.append({"Model": name,
                     "Median Fit Time (ms)": round(np.median(fits) * 1000, 1),
                     "Median Predict Time (ms)": round(np.median(preds) * 1000, 2)})
    return pd.DataFrame(rows)


def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact McNemar test. Returns (A right & B wrong, A wrong & B right, p-value)."""
    y_true = np.asarray(y_true)
    a_ok, b_ok = np.asarray(pred_a) == y_true, np.asarray(pred_b) == y_true
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, p


def mcnemar_table(ensemble, base_models, X_test, y_test):
    ens_pred = ensemble.predict(X_test)
    rows = []
    for name, m in base_models.items():
        b, c, p = mcnemar_exact(y_test, ens_pred, m.predict(X_test))
        rows.append({"Ensemble vs": name, "Ensemble right / model wrong": b,
                     "Model right / ensemble wrong": c, "p-value": p,
                     "Significant at 0.05?": p < 0.05})
    return pd.DataFrame(rows)


def save_evaluation_figures(models, X_train, y_train, X_test, y_test, cv,
                            out_dir="figures", prefix=""):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    y_bin = (y_test == POS).astype(int)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, m in models.items():
        proba = _proba_pos(m, X_test)
        fpr, tpr, _ = roc_curve(y_bin, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y_bin, proba):.4f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves {prefix}".strip()); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out / f"{prefix}roc_curves.png", dpi=120); plt.close()

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))
    for ax, (name, m) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_estimator(m, X_test, y_test, labels=["e", "p"],
                                              display_labels=["Edible", "Poisonous"],
                                              cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(name)
    plt.tight_layout(); plt.savefig(out / f"{prefix}confusion_matrices.png", dpi=120); plt.close()

    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.5), sharey=True)
    for ax, (name, m) in zip(axes, models.items()):
        sizes, tr, va = learning_curve(m, X_train, y_train, cv=cv, scoring="accuracy",
                                       train_sizes=np.linspace(0.05, 1.0, 8), n_jobs=-1)
        ax.plot(sizes, tr.mean(axis=1), "o-", label="Training")
        ax.plot(sizes, va.mean(axis=1), "o-", label="Validation (CV)")
        ax.set_title(name); ax.set_xlabel("Training examples"); ax.grid(True)
    axes[0].set_ylabel("Accuracy"); axes[0].legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out / f"{prefix}learning_curves.png", dpi=120); plt.close()
