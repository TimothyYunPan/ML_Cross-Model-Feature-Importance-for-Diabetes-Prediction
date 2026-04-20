"""
knn_model.py
------------
K-Nearest Neighbors (NumPy only) for binary diabetes classification.

Usage:
    python knn_model.py

Requires preprocessed data in models/ (run preprocess.py first).
Saves best model config and predictions to models/.
"""

import numpy as np
import os
import time
from preprocess import load_preprocessed

# ── KNN Class ────────────────────────────────────────────────────────────────

class KNN:
    def __init__(self, num_neighbors=5, metric="Euclidean"):
        self.model_name    = "K Nearest Neighbor"
        self.num_neighbors = num_neighbors
        self.metric        = metric
        self.X_train       = None
        self.y_train       = None

    # ── Distance Methods ──────────────────────────────────────────────────

    def euclidean_distance(self, feature_matrix, query):
        """L2 distance from query to every row in feature_matrix. (vectorized)"""
        return np.sqrt(np.sum((feature_matrix - query) ** 2, axis=1))

    def manhattan_distance(self, feature_matrix, query):
        """L1 distance from query to every row in feature_matrix."""
        return np.sum(np.abs(feature_matrix - query), axis=1)

    def _compute_distances(self, feature_matrix, query):
        if self.metric == "Euclidean":
            return self.euclidean_distance(feature_matrix, query)
        elif self.metric == "Manhattan":
            return self.manhattan_distance(feature_matrix, query)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    # ── Batch Distance (fast) ─────────────────────────────────────────────

    def _batch_distances(self, X_train, X_query):
        """
        Compute distances between all queries and all train points at once.
        Uses the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b^T
        Returns (n_query, n_train) distance matrix.
        Works for Euclidean only; falls back to loop for Manhattan.
        """
        if self.metric == "Euclidean":
            # squared norms
            train_sq = np.sum(X_train ** 2, axis=1)          # (n_train,)
            query_sq = np.sum(X_query ** 2, axis=1)           # (n_query,)
            cross    = X_query @ X_train.T                    # (n_query, n_train)
            dist_sq  = query_sq[:, None] + train_sq[None, :] - 2 * cross
            dist_sq  = np.maximum(dist_sq, 0)                 # numerical safety
            return np.sqrt(dist_sq)
        else:
            # Manhattan: loop over queries (still faster than naive double loop)
            dists = np.empty((len(X_query), len(X_train)), dtype=np.float64)
            for i, q in enumerate(X_query):
                dists[i] = self.manhattan_distance(X_train, q)
            return dists

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(self, X_train, y_train):
        """Store training data (KNN is lazy — no actual training)."""
        self.X_train = X_train
        self.y_train = y_train

    # ── Predict ───────────────────────────────────────────────────────────

    def predict(self, X_query, batch_size=512):
        """
        Predict class labels for X_query.
        Uses batched distance computation to avoid memory issues.
        Returns np.ndarray of shape (n_query,).
        """
        assert self.X_train is not None, "Call fit() before predict()."
        n = len(X_query)
        predictions = np.empty(n, dtype=np.float64)

        for start in range(0, n, batch_size):
            end   = min(start + batch_size, n)
            batch = X_query[start:end]                            # (B, d)
            dists = self._batch_distances(self.X_train, batch)    # (B, n_train)

            # k nearest neighbours for each query in batch
            k_idx    = np.argpartition(dists, self.num_neighbors, axis=1)[:, :self.num_neighbors]
            k_labels = self.y_train[k_idx]                        # (B, k)
            # majority vote
            predictions[start:end] = (k_labels.mean(axis=1) >= 0.5).astype(np.float64)

        return predictions

    def predict_proba(self, X_query, batch_size=512):
        """Return fraction of positive neighbors as probability estimate."""
        assert self.X_train is not None, "Call fit() before predict_proba()."
        n = len(X_query)
        proba = np.empty(n, dtype=np.float64)

        for start in range(0, n, batch_size):
            end   = min(start + batch_size, n)
            batch = X_query[start:end]
            dists = self._batch_distances(self.X_train, batch)
            k_idx = np.argpartition(dists, self.num_neighbors, axis=1)[:, :self.num_neighbors]
            k_labels = self.y_train[k_idx]
            proba[start:end] = k_labels.mean(axis=1)

        return proba

# ── Evaluation Metrics (NumPy only) ──────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba=None):
    """Return dict with accuracy, precision, recall, f1, and optionally auc_roc."""
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    accuracy  = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    metrics = dict(accuracy=accuracy, precision=precision,
                   recall=recall, f1=f1, TP=int(TP), TN=int(TN),
                   FP=int(FP), FN=int(FN))

    if y_proba is not None:
        metrics["auc_roc"] = _auc_roc(y_true, y_proba)

    return metrics


def _auc_roc(y_true, y_score):
    """Trapezoidal AUC-ROC (NumPy only)."""
    thresholds = np.sort(np.unique(y_score))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = y_true.sum()
    neg = len(y_true) - pos
    for t in thresholds:
        pred = (y_score >= t).astype(float)
        tprs.append(np.sum((pred == 1) & (y_true == 1)) / pos)
        fprs.append(np.sum((pred == 1) & (y_true == 0)) / neg)
    tprs.append(1.0); fprs.append(1.0)
    tprs, fprs = np.array(tprs), np.array(fprs)
    return float(np.trapezoid(tprs, fprs) if hasattr(np, 'trapezoid') else np.trapz(tprs, fprs))


def print_metrics(metrics, label=""):
    header = f"── {label} " if label else "── "
    print(f"\n{header}{'─' * (50 - len(header))}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    if "auc_roc" in metrics:
        print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"  TP={metrics['TP']}  TN={metrics['TN']}  FP={metrics['FP']}  FN={metrics['FN']}")

# ── Permutation Feature Importance ───────────────────────────────────────────

def permutation_importance(model, X_test, y_test, feature_names,
                           n_repeats=3, metric="f1"):
    """
    Permutation importance: shuffle each feature and measure performance drop.
    Returns array of importance scores (shape: n_features).
    """
    # baseline
    y_pred = model.predict(X_test)
    base   = compute_metrics(y_test, y_pred)[metric]

    importances = np.zeros(X_test.shape[1])
    rng = np.random.default_rng(42)

    for j in range(X_test.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            rng.shuffle(X_perm[:, j])
            y_perm_pred = model.predict(X_perm)
            scores.append(compute_metrics(y_test, y_perm_pred)[metric])
        importances[j] = base - np.mean(scores)
        print(f"  [{j+1:02d}/{X_test.shape[1]}] {feature_names[j]:<28} Δ{metric}={importances[j]:+.4f}")

    return importances

# ── Hyperparameter Search ─────────────────────────────────────────────────────

def grid_search(X_train, y_train, X_val, y_val,
                k_values=(3, 5, 7, 9, 11),
                metrics_list=("Euclidean", "Manhattan")):
    """
    Simple grid search over k and distance metric.
    Returns best config dict and results table.
    """
    results = []
    best_f1, best_cfg = -1, None

    for metric in metrics_list:
        for k in k_values:
            print(f"  Testing k={k}, metric={metric} ...", end=" ", flush=True)
            t0  = time.time()
            knn = KNN(num_neighbors=k, metric=metric)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)
            m = compute_metrics(y_val, y_pred)
            elapsed = time.time() - t0
            print(f"F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}  ({elapsed:.1f}s)")
            results.append({"k": k, "metric": metric, **m, "time_s": elapsed})
            if m["f1"] > best_f1:
                best_f1  = m["f1"]
                best_cfg = {"k": k, "metric": metric}

    return best_cfg, results

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 55)
    print("  KNN — Diabetes Binary Classification")
    print("═" * 55)

    # 1. Load preprocessed data
    X_train, X_test, y_train, y_test, feature_names, _ = load_preprocessed()
    print(f"\nLoaded: X_train={X_train.shape}, X_test={X_test.shape}")

    # 2. Small validation split from train for grid search (20% of train)
    rng     = np.random.default_rng(42)
    val_idx = rng.choice(len(X_train), size=int(0.2 * len(X_train)), replace=False)
    tr_idx  = np.setdiff1d(np.arange(len(X_train)), val_idx)
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    # 3. Grid search
    print("\n── Grid Search ─────────────────────────────────────")
    # best_cfg, gs_results = grid_search(X_tr, y_tr, X_val, y_val)
    best_cfg, gs_results = grid_search(
        X_tr, y_tr, X_val, y_val,
        k_values=(5,),
        metrics_list=("Euclidean",)
    )
    print(f"\n★ Best config: k={best_cfg['k']}, metric={best_cfg['metric']}")

    # 4. Train best model on full train set
    print("\n── Training best model on full train set ───────────")
    best_knn = KNN(num_neighbors=best_cfg["k"], metric=best_cfg["metric"])
    best_knn.fit(X_train, y_train)

    # 5. Evaluate on test set
    print("── Evaluating on test set ──────────────────────────")
    t0      = time.time()
    y_pred  = best_knn.predict(X_test)
    y_proba = best_knn.predict_proba(X_test)
    print(f"Inference time: {time.time()-t0:.1f}s")

    test_metrics = compute_metrics(y_test, y_pred, y_proba)
    print_metrics(test_metrics, label=f"Test Results  k={best_cfg['k']} {best_cfg['metric']}")

    # 6. Permutation feature importance
    print("\n── Permutation Feature Importance (test set) ───────")
    importances = permutation_importance(
        best_knn, X_test, y_test, feature_names, n_repeats=3, metric="f1"
    )

    # Print ranked features
    ranked = np.argsort(importances)[::-1]
    print("\n  Rank  Feature                      Importance (ΔF1)")
    for rank, idx in enumerate(ranked):
        print(f"  {rank+1:>4}  {feature_names[idx]:<28} {importances[idx]:+.4f}")

    # 7. Save results
    os.makedirs("models", exist_ok=True)
    np.save("models/knn_y_pred.npy",        y_pred)
    np.save("models/knn_y_proba.npy",       y_proba)
    np.save("models/knn_importances.npy",   importances)
    np.save("models/knn_best_config.npy",   np.array([best_cfg["k"],
                                                       best_cfg["metric"]], dtype=object))

    # Save grid search results as CSV
    import csv
    with open("models/knn_grid_search.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=gs_results[0].keys())
        writer.writeheader()
        writer.writerows(gs_results)

    print("\n[saved] models/knn_y_pred.npy")
    print("[saved] models/knn_y_proba.npy")
    print("[saved] models/knn_importances.npy")
    print("[saved] models/knn_grid_search.csv")
    print("\n✓ KNN training complete.")