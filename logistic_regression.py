"""
logistic_regression.py
----------------------
Logistic Regression (NumPy only) for binary diabetes classification.
Usage:
    python logistic_regression.py
Requires preprocessed data in models/ (run preprocess.py first).
Saves best model config and predictions to models/.
"""

import numpy as np
import os
import time
from preprocess import load_preprocessed
from knn_model import compute_metrics, print_metrics, permutation_importance


# ── Logistic Regression Class ────────────────────────────────────────────────

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=0.01,
                 early_stopping=True, patience=20):
        self.model_name     = "Logistic Regression"
        self.lr             = learning_rate
        self.n_iter         = n_iterations
        self.lambda_reg     = lambda_reg
        self.early_stopping = early_stopping
        self.patience       = patience

        # learned parameters (set after fit)
        self.w = None
        self.b = None

        # for diagnostics
        self.train_loss_history = []
        self.val_loss_history   = []

    # ── Helpers ──────────────────────────────────────────────────────────

    def _sigmoid(self, z):
        # clip to avoid overflow when z is large in magnitude
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _bce_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        # L2 penalty on weights only (not bias)
        if self.lambda_reg > 0 and self.w is not None:
            loss += (self.lambda_reg / 2) * np.sum(self.w ** 2)
        return loss

    # ── Fit ──────────────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train via batch gradient descent. Optional early stopping if
        validation set is provided."""
        n_samples, n_features = X_train.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        best_val_loss = float('inf')
        best_w, best_b = self.w.copy(), self.b
        epochs_no_improve = 0

        for i in range(self.n_iter):
            # forward
            z = X_train @ self.w + self.b
            y_pred = self._sigmoid(z)

            # gradients (vectorised, mean over batch)
            error = y_pred - y_train
            dw = (X_train.T @ error) / n_samples + self.lambda_reg * self.w
            db = np.mean(error)

            # update
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # track loss
            self.train_loss_history.append(self._bce_loss(y_train, y_pred))

            if X_val is not None and y_val is not None:
                val_pred = self._sigmoid(X_val @ self.w + self.b)
                val_loss = self._bce_loss(y_val, val_pred)
                self.val_loss_history.append(val_loss)

                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_w, best_b = self.w.copy(), self.b
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.patience:
                            self.w, self.b = best_w, best_b
                            break

        return self

    # ── Predict ──────────────────────────────────────────────────────────

    def predict_proba(self, X_query):
        """Return P(y=1 | X) for each query."""
        assert self.w is not None, "Call fit() before predict_proba()."
        return self._sigmoid(X_query @ self.w + self.b)

    def predict(self, X_query, threshold=0.5):
        """Predict binary labels using the given threshold (default 0.5)."""
        return (self.predict_proba(X_query) >= threshold).astype(np.float64)


# ── Grid Search ──────────────────────────────────────────────────────────────

def grid_search(X_train, y_train, X_val, y_val,
                lr_values=(0.001, 0.01, 0.1),
                iter_values=(500, 1000, 2000),
                lambda_reg=0.01):
    """Grid search over learning rate and iterations. Returns best config and
    full results table."""
    results = []
    best_f1, best_cfg = -1, None

    for lr in lr_values:
        for n_iter in iter_values:
            print(f"  Testing lr={lr}, n_iter={n_iter} ...", end=" ", flush=True)
            t0 = time.time()
            model = LogisticRegression(
                learning_rate=lr, n_iterations=n_iter,
                lambda_reg=lambda_reg, early_stopping=False
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            m = compute_metrics(y_val, y_pred)
            elapsed = time.time() - t0
            print(f"F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}  ({elapsed:.1f}s)")
            results.append({"lr": lr, "n_iter": n_iter, **m, "time_s": elapsed})
            if m["f1"] > best_f1:
                best_f1  = m["f1"]
                best_cfg = {"lr": lr, "n_iter": n_iter}

    return best_cfg, results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 55)
    print("  Logistic Regression — Diabetes Binary Classification")
    print("═" * 55)

    # 1. Load preprocessed data
    X_train, X_test, y_train, y_test, feature_names, _ = load_preprocessed()
    print(f"\nLoaded: X_train={X_train.shape}, X_test={X_test.shape}")

    # 2. Carve out a validation split from train (matches KNN convention)
    rng     = np.random.default_rng(42)
    val_idx = rng.choice(len(X_train), size=int(0.2 * len(X_train)), replace=False)
    tr_idx  = np.setdiff1d(np.arange(len(X_train)), val_idx)
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    # 3. Grid search
    print("\n── Grid Search ─────────────────────────────────────")
    best_cfg, gs_results = grid_search(X_tr, y_tr, X_val, y_val)
    print(f"\n★ Best config: lr={best_cfg['lr']}, n_iter={best_cfg['n_iter']}")

    # 4. Train best model on full train set with early stopping
    print("\n── Training best model on full train set ───────────")
    best_model = LogisticRegression(
        learning_rate=best_cfg["lr"],
        n_iterations=best_cfg["n_iter"],
        lambda_reg=0.01,
        early_stopping=True,
        patience=30,
    )
    # use same val split for early stopping (small leak is acceptable here;
    # final eval is still on the held-out test set)
    best_model.fit(X_tr, y_tr, X_val, y_val)
    print(f"  Stopped after {len(best_model.train_loss_history)} iterations")

    # 5. Evaluate on test set
    print("── Evaluating on test set ──────────────────────────")
    t0      = time.time()
    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    print(f"Inference time: {time.time()-t0:.1f}s")

    test_metrics = compute_metrics(y_test, y_pred, y_proba)
    print_metrics(test_metrics, label=f"Test Results  lr={best_cfg['lr']} n_iter={best_cfg['n_iter']}")

    # 6. Permutation feature importance (same function KNN uses → fair comparison)
    print("\n── Permutation Feature Importance (test set) ───────")
    importances = permutation_importance(
        best_model, X_test, y_test, feature_names, n_repeats=3, metric="f1"
    )

    # Print ranked features
    ranked = np.argsort(importances)[::-1]
    print("\n  Rank  Feature                      Importance (ΔF1)")
    for rank, idx in enumerate(ranked):
        print(f"  {rank+1:>4}  {feature_names[idx]:<28} {importances[idx]:+.4f}")

    # 7. Save results (mirrors KNN naming convention)
    os.makedirs("models", exist_ok=True)
    np.save("models/logreg_y_pred.npy",      y_pred)
    np.save("models/logreg_y_proba.npy",     y_proba)
    np.save("models/logreg_importances.npy", importances)
    np.save("models/logreg_weights.npy",     best_model.w)
    np.save("models/logreg_bias.npy",        np.array([best_model.b]))
    np.save("models/logreg_best_config.npy", np.array([best_cfg["lr"],
                                                        best_cfg["n_iter"]], dtype=object))

    # weight-magnitude importance (model-specific, complements permutation)
    weight_importance = np.abs(best_model.w)
    np.save("models/logreg_weight_importance.npy", weight_importance)

    # Save grid search results as CSV
    import csv
    with open("models/logreg_grid_search.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=gs_results[0].keys())
        writer.writeheader()
        writer.writerows(gs_results)

    print("\n[saved] models/logreg_y_pred.npy")
    print("[saved] models/logreg_y_proba.npy")
    print("[saved] models/logreg_importances.npy")
    print("[saved] models/logreg_weights.npy")
    print("[saved] models/logreg_bias.npy")
    print("[saved] models/logreg_weight_importance.npy")
    print("[saved] models/logreg_grid_search.csv")
    print("\n✓ Logistic Regression training complete.")
