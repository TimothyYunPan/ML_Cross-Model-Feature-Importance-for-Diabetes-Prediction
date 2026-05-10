"""
ann_model.py
------------
Artificial Neural Network (NumPy only) for binary diabetes classification.

Architecture (per proposal §IV.A.2 Eq. 5):
    h     = ReLU(W1·x + b1)
    y_hat = sigmoid(W2·h + b2)
Loss:         Binary Cross-Entropy in logit space (numerically stable).
Optimizer:    Adam (Kingma & Ba 2014), implemented from scratch in NumPy.
Regularizers: L2 weight decay + early stopping (per proposal §V.B).

Usage
-----
    python ann_model.py

Requires preprocessed data in ``models/`` (run ``preprocess.py`` first).
Saves best model weights, predictions, and feature importances to ``models/``.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass

import numpy as np

from preprocess import load_preprocessed
from knn_model import compute_metrics, print_metrics, permutation_importance


# ── Reproducibility ──────────────────────────────────────────────────────────

SEED = 42


# ── Numerically Stable Primitives ────────────────────────────────────────────

def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid that avoids ``exp`` overflow for large ``|z|``.

    Branches on the sign of ``z``:
    ``z >= 0`` uses ``1 / (1 + exp(-z))``, ``z < 0`` uses ``exp(z) / (1 + exp(z))``.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def bce_with_logits(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean binary cross-entropy from raw logits.

    Uses ``max(z, 0) - z·y + log1p(exp(-|z|))`` for numerical stability,
    which sidesteps ``log(0)`` and supersedes the proposal §V.B "clip
    probabilities to [ε, 1-ε]" mitigation.

    Parameters
    ----------
    logits : np.ndarray
        Pre-sigmoid output, shape ``(N,)``.
    y : np.ndarray
        Binary labels in ``{0, 1}``, shape ``(N,)``.
    """
    return float(np.mean(
        np.maximum(logits, 0.0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
    ))


# ── Adam Optimizer (NumPy) ───────────────────────────────────────────────────

class AdamOptimizer:
    """Adam optimizer (Kingma & Ba, 2014) implemented in NumPy.

    Updates each parameter with bias-corrected first/second moments::

        m_t = β1·m_{t-1} + (1-β1)·g
        v_t = β2·v_{t-1} + (1-β2)·g²
        θ  -= lr · (m_t / (1-β1^t)) / (sqrt(v_t / (1-β2^t)) + ε)

    Parameters
    ----------
    param_shapes : dict[str, tuple[int, ...]]
        Parameter name → shape. Used to allocate moment buffers.
    lr, beta1, beta2, eps : float
        Standard Adam hyperparameters.
    """

    def __init__(
        self,
        param_shapes: dict[str, tuple[int, ...]],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros(s, dtype=np.float64) for k, s in param_shapes.items()}
        self.v = {k: np.zeros(s, dtype=np.float64) for k, s in param_shapes.items()}

    def step(
        self,
        params: dict[str, np.ndarray],
        grads: dict[str, np.ndarray],
    ) -> None:
        """Apply one in-place Adam update to ``params`` using ``grads``."""
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── Network ──────────────────────────────────────────────────────────────────

class ANNNet:
    """Single-hidden-layer feedforward network (proposal Eq. 5).

    Pure-NumPy forward and backward. He initialization (He et al., 2015)
    keeps activation variance roughly preserved through the ReLU layer.

    Parameters
    ----------
    input_dim : int
        Feature dimensionality ``D``.
    hidden_dim : int, default=64
        Hidden layer width ``H``.
    seed : int, default=SEED
        RNG seed for weight initialization.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, seed: int = SEED):
        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {
            "W1": rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim),
            "b1": np.zeros(hidden_dim, dtype=np.float64),
            "W2": rng.standard_normal((hidden_dim, 1)) * np.sqrt(2.0 / hidden_dim),
            "b2": np.zeros(1, dtype=np.float64),
        }

    @property
    def param_shapes(self) -> dict[str, tuple[int, ...]]:
        """Map of parameter name → shape (used to size optimizer buffers)."""
        return {k: v.shape for k, v in self.params.items()}

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Run a forward pass.

        Parameters
        ----------
        X : np.ndarray
            Input batch, shape ``(N, D)``.

        Returns
        -------
        logits : np.ndarray
            Pre-sigmoid output, shape ``(N,)``.
        cache : dict
            Intermediates (``X``, ``z1``, ``h``) needed for ``backward``.
        """
        z1 = X @ self.params["W1"] + self.params["b1"]   # (N, H)
        h = np.maximum(0.0, z1)                           # ReLU
        z2 = h @ self.params["W2"] + self.params["b2"]    # (N, 1)
        logits = z2.squeeze(-1)                           # (N,)
        return logits, {"X": X, "z1": z1, "h": h}

    def backward(
        self,
        cache: dict,
        y: np.ndarray,
        logits: np.ndarray,
        weight_decay: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Backpropagate to obtain parameter gradients.

        Uses ``d(BCE)/dz_out = σ(z_out) - y``; the stable loss formulation
        only changes how ``L`` is computed, not its derivative.

        Parameters
        ----------
        cache : dict
            Forward-pass intermediates.
        y : np.ndarray
            Binary labels, shape ``(N,)``.
        logits : np.ndarray
            Output of ``forward``, shape ``(N,)``.
        weight_decay : float, default=0.0
            L2 coefficient added to weight (not bias) gradients.

        Returns
        -------
        dict[str, np.ndarray]
            Gradients keyed by parameter name.
        """
        X, z1, h = cache["X"], cache["z1"], cache["h"]
        N = X.shape[0]

        dz2 = (stable_sigmoid(logits) - y)[:, None] / N   # (N, 1)
        dW2 = h.T @ dz2                                    # (H, 1)
        db2 = dz2.sum(axis=0)                              # (1,)

        dh = dz2 @ self.params["W2"].T                     # (N, H)
        dz1 = dh * (z1 > 0)                                # ReLU' (subgradient at 0 = 0)

        dW1 = X.T @ dz1                                    # (D, H)
        db1 = dz1.sum(axis=0)                              # (H,)

        if weight_decay > 0:
            dW1 = dW1 + weight_decay * self.params["W1"]
            dW2 = dW2 + weight_decay * self.params["W2"]

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return a deep-copied snapshot of parameters."""
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Overwrite parameters from a snapshot produced by ``state_dict``."""
        for k in self.params:
            self.params[k] = state[k].copy()


# ── Wrapper (matches sklearn-ish interface used by KNN / LogReg) ────────────

class ANN:
    """Sklearn-style wrapper around ``ANNNet``.

    Exposes ``fit`` / ``predict`` / ``predict_proba`` so the model plugs
    directly into ``compute_metrics`` and ``permutation_importance``.

    Parameters
    ----------
    input_dim : int
        Feature dimensionality.
    hidden_dim : int, default=64
        Width of the hidden layer.
    learning_rate : float, default=0.01
        Adam step size.
    epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=256
        Mini-batch size.
    weight_decay : float, default=1e-4
        L2 coefficient.
    early_stopping : bool, default=True
        Stop when validation loss has not improved for ``patience`` epochs.
    patience : int, default=10
        Early-stopping patience.
    seed : int, default=SEED
        RNG seed for weight init and batch shuffling.
    verbose : bool, default=False
        Print periodic loss updates.
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
        seed: int = SEED,
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
        self.seed = seed
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
        """Train via mini-batch Adam with optional early stopping.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training features ``(N, D)`` and labels ``(N,)`` in ``{0, 1}``.
        X_val, y_val : np.ndarray, optional
            Validation set; required for early stopping.

        Returns
        -------
        ANN
            ``self``, for sklearn-style fluent chaining.
        """
        self.net = ANNNet(self.input_dim, self.hidden_dim, seed=self.seed)
        optim = AdamOptimizer(self.net.param_shapes, lr=self.lr)

        rng = np.random.default_rng(self.seed)
        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=np.float64)
        N = X_tr.shape[0]

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = np.asarray(X_val, dtype=np.float64)
            y_v = np.asarray(y_val, dtype=np.float64)

        best_val_loss = float("inf")
        best_state: dict | None = None
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            perm = rng.permutation(N)
            epoch_loss = 0.0
            n_seen = 0

            for start in range(0, N, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, yb = X_tr[idx], y_tr[idx]

                logits, cache = self.net.forward(xb)
                loss = bce_with_logits(logits, yb)
                grads = self.net.backward(cache, yb, logits, self.weight_decay)
                optim.step(self.net.params, grads)

                epoch_loss += loss * xb.shape[0]
                n_seen += xb.shape[0]

            train_loss = epoch_loss / n_seen
            self.train_loss_history.append(train_loss)

            if has_val:
                val_logits, _ = self.net.forward(X_v)
                val_loss = bce_with_logits(val_logits, y_v)
                self.val_loss_history.append(val_loss)

                if self.early_stopping:
                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        best_state = self.net.state_dict()
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
        """Return ``P(y=1 | x)`` for each row of ``X_query``."""
        assert self.net is not None, "Call fit() before predict_proba()."
        logits, _ = self.net.forward(np.asarray(X_query, dtype=np.float64))
        return stable_sigmoid(logits)

    def predict(self, X_query: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return hard labels ``{0, 1}`` using ``threshold`` on ``predict_proba``."""
        return (self.predict_proba(X_query) >= threshold).astype(np.float64)


# ── Two-Hidden-Layer Network (extension beyond proposal Eq. 5) ───────────────

class ANNNet2:
    """Two-hidden-layer feedforward network.

    Architecture::

        h1    = ReLU(W1·x  + b1)
        h2    = ReLU(W2·h1 + b2)
        y_hat = σ   (W3·h2 + b3)

    Same He init / pure-NumPy backprop discipline as :class:`ANNNet`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (128, 64),
        seed: int = SEED,
    ):
        rng = np.random.default_rng(seed)
        h1, h2 = hidden_dims
        self.params: dict[str, np.ndarray] = {
            "W1": rng.standard_normal((input_dim, h1)) * np.sqrt(2.0 / input_dim),
            "b1": np.zeros(h1, dtype=np.float64),
            "W2": rng.standard_normal((h1, h2)) * np.sqrt(2.0 / h1),
            "b2": np.zeros(h2, dtype=np.float64),
            "W3": rng.standard_normal((h2, 1)) * np.sqrt(2.0 / h2),
            "b3": np.zeros(1, dtype=np.float64),
        }

    @property
    def param_shapes(self) -> dict[str, tuple[int, ...]]:
        return {k: v.shape for k, v in self.params.items()}

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        z1 = X @ self.params["W1"] + self.params["b1"]   # (N, H1)
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self.params["W2"] + self.params["b2"]  # (N, H2)
        h2 = np.maximum(0.0, z2)
        z3 = h2 @ self.params["W3"] + self.params["b3"]  # (N, 1)
        return z3.squeeze(-1), {"X": X, "z1": z1, "h1": h1, "z2": z2, "h2": h2}

    def backward(
        self,
        cache: dict,
        y: np.ndarray,
        logits: np.ndarray,
        weight_decay: float = 0.0,
    ) -> dict[str, np.ndarray]:
        X = cache["X"]
        z1, h1, z2, h2 = cache["z1"], cache["h1"], cache["z2"], cache["h2"]
        N = X.shape[0]

        dz3 = (stable_sigmoid(logits) - y)[:, None] / N   # (N, 1)
        dW3 = h2.T @ dz3                                   # (H2, 1)
        db3 = dz3.sum(axis=0)

        dh2 = dz3 @ self.params["W3"].T                    # (N, H2)
        dz2 = dh2 * (z2 > 0)
        dW2 = h1.T @ dz2                                   # (H1, H2)
        db2 = dz2.sum(axis=0)

        dh1 = dz2 @ self.params["W2"].T                    # (N, H1)
        dz1 = dh1 * (z1 > 0)
        dW1 = X.T @ dz1                                    # (D, H1)
        db1 = dz1.sum(axis=0)

        if weight_decay > 0:
            dW1 = dW1 + weight_decay * self.params["W1"]
            dW2 = dW2 + weight_decay * self.params["W2"]
            dW3 = dW3 + weight_decay * self.params["W3"]

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def state_dict(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for k in self.params:
            self.params[k] = state[k].copy()


class ANN2:
    """Sklearn-style wrapper around :class:`ANNNet2`.

    Mirrors :class:`ANN` exactly except the hidden layer is a 2-tuple.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (128, 64),
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        early_stopping: bool = True,
        patience: int = 10,
        seed: int = SEED,
        verbose: bool = False,
    ):
        self.model_name = "ANN (2 hidden layers)"
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.seed = seed
        self.verbose = verbose

        self.net: ANNNet2 | None = None
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "ANN2":
        self.net = ANNNet2(self.input_dim, self.hidden_dims, seed=self.seed)
        optim = AdamOptimizer(self.net.param_shapes, lr=self.lr)

        rng = np.random.default_rng(self.seed)
        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=np.float64)
        N = X_tr.shape[0]

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = np.asarray(X_val, dtype=np.float64)
            y_v = np.asarray(y_val, dtype=np.float64)

        best_val_loss = float("inf")
        best_state: dict | None = None
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            perm = rng.permutation(N)
            epoch_loss = 0.0
            n_seen = 0

            for start in range(0, N, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, yb = X_tr[idx], y_tr[idx]

                logits, cache = self.net.forward(xb)
                loss = bce_with_logits(logits, yb)
                grads = self.net.backward(cache, yb, logits, self.weight_decay)
                optim.step(self.net.params, grads)

                epoch_loss += loss * xb.shape[0]
                n_seen += xb.shape[0]

            train_loss = epoch_loss / n_seen
            self.train_loss_history.append(train_loss)

            if has_val:
                val_logits, _ = self.net.forward(X_v)
                val_loss = bce_with_logits(val_logits, y_v)
                self.val_loss_history.append(val_loss)

                if self.early_stopping:
                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        best_state = self.net.state_dict()
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

    def predict_proba(self, X_query: np.ndarray) -> np.ndarray:
        assert self.net is not None, "Call fit() before predict_proba()."
        logits, _ = self.net.forward(np.asarray(X_query, dtype=np.float64))
        return stable_sigmoid(logits)

    def predict(self, X_query: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_query) >= threshold).astype(np.float64)


# ── Grid Search ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ANNConfig:
    """Hyperparameter triple selected by grid search."""
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
    """Exhaustive grid search over the proposal hyperparameter ranges.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : np.ndarray
        Train/validation splits.
    lr_values, hidden_values, epoch_values : tuple
        Candidate values for each axis.

    Returns
    -------
    best_cfg : ANNConfig
        Configuration with highest validation F1.
    results : list[dict]
        Per-configuration metrics + wall time, suitable for CSV export.
    """
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
    print("  ANN (NumPy) — Diabetes Binary Classification")
    print("═" * 55)

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

    # 7. Save results (mirrors KNN / LogReg naming convention).
    #    .npz bundles the four weight arrays into one NumPy-native archive.
    os.makedirs("models", exist_ok=True)
    np.save("models/ann_y_pred.npy",      y_pred)
    np.save("models/ann_y_proba.npy",     y_proba)
    np.save("models/ann_importances.npy", importances)
    np.savez(
        "models/ann_weights.npz",
        W1=best_model.net.params["W1"],
        b1=best_model.net.params["b1"],
        W2=best_model.net.params["W2"],
        b2=best_model.net.params["b2"],
    )
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
    print("[saved] models/ann_weights.npz")
    print("[saved] models/ann_best_config.npy")
    print("[saved] models/ann_grid_search.csv")
    print("\n✓ ANN training complete.")
