"""
ann_model.py
------------
Artificial Neural Network (PyTorch) for binary diabetes classification.

Architecture (per proposal):
    h = ReLU(W1 x + b1)
    y_hat = sigmoid(W2 h + b2)
Loss: Binary Cross-Entropy.
Optimizer: Adam (gradient-based, backprop).

Usage:
    python ann_model.py

Requires preprocessed data in models/ (run preprocess.py first).
Saves best model weights, predictions, and feature importances to models/.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from preprocess import load_preprocessed
from knn_model import compute_metrics, print_metrics, permutation_importance


# ── Reproducibility ──────────────────────────────────────────────────────────

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ── Network ──────────────────────────────────────────────────────────────────

class ANNNet(nn.Module):
    """Feedforward network with one hidden layer (per proposal Eq. 5)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logits; sigmoid is applied outside (via BCEWithLogitsLoss)
        return self.fc2(self.relu(self.fc1(x))).squeeze(-1)


# ── Wrapper (matches sklearn-ish interface used by KNN / LogReg) ────────────

class ANN:
    """
    Thin wrapper around ANNNet that exposes fit / predict / predict_proba,
    so it plugs directly into compute_metrics + permutation_importance.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        early_stopping: bool = True,
        patience: int = 10,
        device: torch.device = DEVICE,
        verbose: bool = False,
    ):
        self.model_name = "Artificial Neural Network"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.device = device
        self.verbose = verbose

        self.net: ANNNet | None = None
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []

    # ── Fit ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "ANN":
        self.net = ANNNet(self.input_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()  # numerically stable sigmoid + BCE

        X_tr = torch.as_tensor(X_train, dtype=torch.float32)
        y_tr = torch.as_tensor(y_train, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=self.batch_size,
            shuffle=True,
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = torch.as_tensor(X_val, dtype=torch.float32).to(self.device)
            y_v = torch.as_tensor(y_val, dtype=torch.float32).to(self.device)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.net.train()
            epoch_loss = 0.0
            n_seen = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                n_seen += xb.size(0)
            train_loss = epoch_loss / n_seen
            self.train_loss_history.append(train_loss)

            if has_val:
                self.net.eval()
                with torch.no_grad():
                    val_logits = self.net(X_v)
                    val_loss = criterion(val_logits, y_v).item()
                self.val_loss_history.append(val_loss)

                if self.early_stopping:
                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        best_state = {k: v.detach().clone()
                                      for k, v in self.net.state_dict().items()}
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.patience:
                            if self.verbose:
                                print(f"    [early stop] epoch {epoch+1}, "
                                      f"best val_loss={best_val_loss:.4f}")
                            break

            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                tail = f"  val_loss={self.val_loss_history[-1]:.4f}" if has_val else ""
                print(f"    epoch {epoch+1:>3}/{self.epochs}  "
                      f"train_loss={train_loss:.4f}{tail}")

        if self.early_stopping and best_state is not None:
            self.net.load_state_dict(best_state)

        return self

    # ── Predict ──────────────────────────────────────────────────────────

    def predict_proba(self, X_query: np.ndarray) -> np.ndarray:
        assert self.net is not None, "Call fit() before predict_proba()."
        self.net.eval()
        with torch.no_grad():
            X = torch.as_tensor(X_query, dtype=torch.float32).to(self.device)
            logits = self.net(X)
            proba = torch.sigmoid(logits).cpu().numpy()
        return proba.astype(np.float64)

    def predict(self, X_query: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_query) >= threshold).astype(np.float64)


# ── Grid Search ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ANNConfig:
    lr: float
    hidden_dim: int
    epochs: int


def grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr_values: tuple[float, ...] = (0.001, 0.01),
    hidden_values: tuple[int, ...] = (32, 64, 128),
    epoch_values: tuple[int, ...] = (50, 100, 200),
) -> tuple[ANNConfig, list[dict]]:
    """Grid search matching the proposal hyperparameter ranges."""
    results: list[dict] = []
    best_f1, best_cfg = -1.0, None
    input_dim = X_train.shape[1]

    for lr in lr_values:
        for hd in hidden_values:
            for n_epochs in epoch_values:
                print(f"  Testing lr={lr}, hidden={hd}, epochs={n_epochs} ...",
                      end=" ", flush=True)
                t0 = time.time()
                model = ANN(
                    input_dim=input_dim,
                    hidden_dim=hd,
                    learning_rate=lr,
                    epochs=n_epochs,
                    early_stopping=False,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                m = compute_metrics(y_val, y_pred)
                elapsed = time.time() - t0
                print(f"F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}  ({elapsed:.1f}s)")
                results.append({
                    "lr": lr, "hidden_dim": hd, "epochs": n_epochs,
                    **m, "time_s": elapsed,
                })
                if m["f1"] > best_f1:
                    best_f1 = m["f1"]
                    best_cfg = ANNConfig(lr=lr, hidden_dim=hd, epochs=n_epochs)

    assert best_cfg is not None
    return best_cfg, results


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 55)
    print("  ANN (PyTorch) — Diabetes Binary Classification")
    print("═" * 55)
    print(f"  device = {DEVICE}")

    # 1. Load preprocessed data
    X_train, X_test, y_train, y_test, feature_names, _ = load_preprocessed()
    print(f"\nLoaded: X_train={X_train.shape}, X_test={X_test.shape}")

    # 2. Validation split (same convention as KNN / LogReg)
    rng = np.random.default_rng(SEED)
    val_idx = rng.choice(len(X_train), size=int(0.2 * len(X_train)), replace=False)
    tr_idx = np.setdiff1d(np.arange(len(X_train)), val_idx)
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    # 3. Grid search
    #    For the mid checkpoint we keep the search small; the full proposal grid
    #    (lr ∈ {0.001, 0.01}, hidden ∈ {32, 64, 128}, epochs ∈ {50, 100, 200})
    #    can be enabled by removing the overrides below.
    print("\n── Grid Search ─────────────────────────────────────")
    best_cfg, gs_results = grid_search(
        X_tr, y_tr, X_val, y_val,
        lr_values=(0.01,),
        hidden_values=(64,),
        epoch_values=(50,),
    )
    print(f"\n★ Best config: lr={best_cfg.lr}, hidden={best_cfg.hidden_dim}, "
          f"epochs={best_cfg.epochs}")

    # 4. Train best model on full train set with early stopping
    print("\n── Training best model on full train set ───────────")
    best_model = ANN(
        input_dim=X_train.shape[1],
        hidden_dim=best_cfg.hidden_dim,
        learning_rate=best_cfg.lr,
        epochs=best_cfg.epochs,
        early_stopping=True,
        patience=15,
        verbose=True,
    )
    best_model.fit(X_tr, y_tr, X_val, y_val)
    print(f"  Trained for {len(best_model.train_loss_history)} epochs")

    # 5. Evaluate on test set
    print("── Evaluating on test set ──────────────────────────")
    t0 = time.time()
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    print(f"Inference time: {time.time()-t0:.1f}s")

    test_metrics = compute_metrics(y_test, y_pred, y_proba)
    print_metrics(
        test_metrics,
        label=(f"Test Results  lr={best_cfg.lr} "
               f"hd={best_cfg.hidden_dim} ep={best_cfg.epochs}"),
    )

    # 6. Permutation feature importance (fair cross-model comparison)
    print("\n── Permutation Feature Importance (test set) ───────")
    importances = permutation_importance(
        best_model, X_test, y_test, feature_names, n_repeats=3, metric="f1"
    )

    ranked = np.argsort(importances)[::-1]
    print("\n  Rank  Feature                      Importance (ΔF1)")
    for rank, idx in enumerate(ranked):
        print(f"  {rank+1:>4}  {feature_names[idx]:<28} {importances[idx]:+.4f}")

    # 7. Save results (mirrors KNN / LogReg naming convention)
    os.makedirs("models", exist_ok=True)
    np.save("models/ann_y_pred.npy",      y_pred)
    np.save("models/ann_y_proba.npy",     y_proba)
    np.save("models/ann_importances.npy", importances)
    torch.save(best_model.net.state_dict(), "models/ann_weights.pt")
    np.save(
        "models/ann_best_config.npy",
        np.array([best_cfg.lr, best_cfg.hidden_dim, best_cfg.epochs], dtype=object),
    )
    np.save("models/ann_train_loss.npy", np.array(best_model.train_loss_history))
    np.save("models/ann_val_loss.npy",   np.array(best_model.val_loss_history))

    with open("models/ann_grid_search.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=gs_results[0].keys())
        writer.writeheader()
        writer.writerows(gs_results)

    print("\n[saved] models/ann_y_pred.npy")
    print("[saved] models/ann_y_proba.npy")
    print("[saved] models/ann_importances.npy")
    print("[saved] models/ann_weights.pt")
    print("[saved] models/ann_best_config.npy")
    print("[saved] models/ann_grid_search.csv")
    print("\n✓ ANN training complete.")
