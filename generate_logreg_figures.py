"""
generate_logreg_figures.py
--------------------------
Run AFTER python logistic_regression.py.
Produces 5 figures for the final report.

Outputs to Figures/ folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from logistic_regression import LogisticRegression
from preprocess import load_preprocessed

os.makedirs("Figures", exist_ok=True)
sns.set_style("whitegrid")

# Load preprocessed data and saved model outputs
X_train, X_test, y_train, y_test, feature_names, _ = load_preprocessed()
y_pred = np.load("models/logreg_y_pred.npy")
y_proba = np.load("models/logreg_y_proba.npy")
importances = np.load("models/logreg_importances.npy")


# ── Figure 1: Loss Curves ───────────────────────────────────────────
print("[1/5] Loss curves...")
rng = np.random.default_rng(42)
val_idx = rng.choice(len(X_train), size=int(0.2 * len(X_train)), replace=False)
tr_idx = np.setdiff1d(np.arange(len(X_train)), val_idx)
X_tr, X_val = X_train[tr_idx], X_train[val_idx]
y_tr, y_val = y_train[tr_idx], y_train[val_idx]

# retrain best config without early stopping so we see the full curve
model_for_curve = LogisticRegression(
    learning_rate=0.1, n_iterations=2000, lambda_reg=0.01, early_stopping=False
)
model_for_curve.fit(X_tr, y_tr, X_val, y_val)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(model_for_curve.train_loss_history, label="Training loss", color="steelblue")
ax.plot(model_for_curve.val_loss_history, label="Validation loss", color="coral")
ax.set_xlabel("Iteration")
ax.set_ylabel("Binary Cross-Entropy Loss")
ax.set_title("Logistic Regression Training Curves (lr=0.1, L2=0.01)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Figures/logreg_loss_curve.png", dpi=200, bbox_inches="tight")
plt.close()


# ── Figure 2: Permutation Feature Importance ────────────────────────
print("[2/5] Feature importance...")
order = np.argsort(importances)[::-1]
sorted_names = np.array(feature_names)[order]
sorted_imps = importances[order]
colors = ["#1f77b4" if x >= 0 else "#d62728" for x in sorted_imps]

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(range(len(sorted_names)), sorted_imps, color=colors, edgecolor="black")
ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels(sorted_names)
ax.invert_yaxis()
ax.set_xlabel(r"Permutation Importance ($\Delta$F1)")
ax.set_title("Logistic Regression: Permutation Feature Importance")
ax.axvline(0, color="gray", linewidth=0.8)
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("Figures/logreg_perm_importance.png", dpi=200, bbox_inches="tight")
plt.close()


# ── Figure 3: Confusion Matrix ──────────────────────────────────────
print("[3/5] Confusion matrix...")
cm = np.array([
    [int(np.sum((y_pred == 0) & (y_test == 0))),
     int(np.sum((y_pred == 1) & (y_test == 0)))],
    [int(np.sum((y_pred == 0) & (y_test == 1))),
     int(np.sum((y_pred == 1) & (y_test == 1)))],
])

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Pred: No", "Pred: Yes"],
    yticklabels=["Actual: No", "Actual: Yes"],
    cbar=False, annot_kws={"size": 14}, ax=ax,
)
ax.set_title("Logistic Regression — Confusion Matrix")
plt.tight_layout()
plt.savefig("Figures/logreg_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()


# ── Figure 4: ROC Curve ─────────────────────────────────────────────
print("[4/5] ROC curve...")
thresholds = np.sort(np.unique(y_proba))[::-1]
tprs, fprs = [0.0], [0.0]
pos = y_test.sum()
neg = len(y_test) - pos
for t in thresholds:
    pred = (y_proba >= t).astype(float)
    tp = np.sum((pred == 1) & (y_test == 1))
    fp = np.sum((pred == 1) & (y_test == 0))
    tprs.append(tp / pos)
    fprs.append(fp / neg)
auc = np.trapezoid(tprs, fprs)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fprs, tprs, color="steelblue", linewidth=2,
        label=f"Logistic Regression (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Logistic Regression")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Figures/logreg_roc_curve.png", dpi=200, bbox_inches="tight")
plt.close()


# ── Figure 5: Grid Search Heatmap ───────────────────────────────────
print("[5/5] Grid search heatmap...")
gs = pd.read_csv("models/logreg_grid_search.csv")
pivot = gs.pivot(index="lr", columns="n_iter", values="f1")

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu",
            cbar_kws={"label": "Validation F1"}, ax=ax)
ax.set_title("Grid Search F1 across (lr, n_iter)")
plt.tight_layout()
plt.savefig("Figures/logreg_grid_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()


print("\n✓ All 5 figures saved to Figures/")
print("  - logreg_loss_curve.png")
print("  - logreg_perm_importance.png")
print("  - logreg_confusion_matrix.png")
print("  - logreg_roc_curve.png")
print("  - logreg_grid_heatmap.png")
