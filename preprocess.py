"""
preprocess.py
-------------
Data loading, exploration, and preprocessing pipeline for the
CDC BRFSS 2015 Diabetes Health Indicators dataset.

Usage:
    python preprocess.py

Outputs:
    - Prints EDA summary to console
    - Saves preprocessed train/test splits as .npy files in models/
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ── Constants ────────────────────────────────────────────────────────────────

BALANCED_DATA_PATH = "data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
FULL_DATA_PATH     = "data/diabetes_binary_health_indicators_BRFSS2015.csv"

TARGET_COL = "Diabetes_binary"

BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]
ORDINAL_FEATURES = ["GenHlth", "Age", "Education", "Income"]
NUMERICAL_FEATURES = ["BMI", "MentHlth", "PhysHlth"]

ALL_FEATURES = BINARY_FEATURES + ORDINAL_FEATURES + NUMERICAL_FEATURES  # 21 features

CORR_THRESHOLD = 0.7   # drop one of a pair if |r| > this
TEST_SIZE      = 0.2
RANDOM_SEED    = 42

# ── 1. Load Data ─────────────────────────────────────────────────────────────

def _resolve_dataset_path(path: str) -> str:
    """
    Resolve a usable dataset path.
    Priority:
      1) explicit `path`
      2) known fallback path
      3) first CSV found in ./data
    """
    candidates = [path, FULL_DATA_PATH]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    if os.path.isdir("data"):
        data_csvs = sorted(
            [f for f in os.listdir("data") if f.lower().endswith(".csv")]
        )
        if data_csvs:
            auto_path = os.path.join("data", data_csvs[0])
            print(f"[preprocess] Using detected dataset: '{auto_path}'")
            return auto_path

    raise FileNotFoundError(
        "\n[ERROR] Dataset not found.\n"
        f"Tried:\n  - {path}\n  - {FULL_DATA_PATH}\n"
        "and no CSV was found in the data/ folder.\n"
        "Please download from:\n"
        "  https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset\n"
        "then place the CSV inside data/."
    )


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and do basic validation."""
    resolved_path = _resolve_dataset_path(path)
    df = pd.read_csv(resolved_path)
    print(f"[load] Loaded {len(df):,} rows × {df.shape[1]} cols from '{resolved_path}'")
    return df

# ── 2. EDA ───────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame, save_plots: bool = False) -> None:
    """Print EDA summary and optionally save plots."""
    print("\n── EDA ──────────────────────────────────────────────")
    print(df.describe().T[["mean", "std", "min", "max"]])

    # Class distribution
    counts = Counter(df[TARGET_COL].astype(int))
    print(f"\nClass distribution: {dict(counts)}")

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found.")
    else:
        print(f"Missing values:\n{missing[missing > 0]}")

    if save_plots:
        os.makedirs("plots", exist_ok=True)

        # Histogram of numerical features
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, col in zip(axes, NUMERICAL_FEATURES):
            ax.hist(df[col], bins=40, color="steelblue", edgecolor="white")
            ax.set_title(col)
        plt.tight_layout()
        plt.savefig("plots/numerical_histograms.png", dpi=150)
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(df[ALL_FEATURES].corr(), annot=False, cmap="coolwarm", center=0)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("plots/correlation_heatmap.png", dpi=150)
        plt.close()

        print("Plots saved to plots/")

# ── 3. Preprocessing ─────────────────────────────────────────────────────────

def remove_correlated_features(df: pd.DataFrame, threshold: float = CORR_THRESHOLD):
    """
    Drop one feature from each highly correlated pair (|r| > threshold).
    Returns filtered df and list of dropped columns.
    """
    corr_matrix = df[ALL_FEATURES].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        print(f"[preprocess] Dropping highly correlated features: {to_drop}")
    df = df.drop(columns=to_drop)
    return df, to_drop


def remove_bmi_outliers(df: pd.DataFrame):
    """Remove BMI outliers using IQR method."""
    if "BMI" not in df.columns:
        return df
    Q1, Q3 = df["BMI"].quantile(0.25), df["BMI"].quantile(0.75)
    IQR = Q3 - Q1
    before = len(df)
    df = df[(df["BMI"] >= Q1 - 1.5 * IQR) & (df["BMI"] <= Q3 + 1.5 * IQR)]
    print(f"[preprocess] BMI outlier removal: {before:,} → {len(df):,} rows")
    return df


def minmax_normalize(X: np.ndarray, feature_min=None, feature_max=None):
    """
    Min-Max normalize to [0, 1].
    Pass feature_min/feature_max from train set when transforming test set.
    """
    if feature_min is None:
        feature_min = X.min(axis=0)
    if feature_max is None:
        feature_max = X.max(axis=0)
    denom = feature_max - feature_min
    denom[denom == 0] = 1  # avoid division by zero for constant columns
    return (X - feature_min) / denom, feature_min, feature_max


def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_size: float = TEST_SIZE,
                     random_seed: int = RANDOM_SEED):
    """Stratified 80/20 train-test split (NumPy only)."""
    rng = np.random.default_rng(random_seed)
    classes = np.unique(y)
    train_idx, test_idx = [], []

    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = int(len(idx) * test_size)
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ── 4. Full Pipeline ──────────────────────────────────────────────────────────

def preprocess_pipeline(path: str = BALANCED_DATA_PATH,
                        save: bool = True,
                        run_eda_flag: bool = True):
    """
    Full preprocessing pipeline.

    Returns:
        X_train, X_test, y_train, y_test (all np.ndarray)
        feature_names (list of str)
        scaler_params (dict with 'min' and 'max' for inference)
    """
    # 1. Load
    df = load_data(path)

    # 2. EDA
    if run_eda_flag:
        run_eda(df, save_plots=False)

    # 3. Remove BMI outliers
    df = remove_bmi_outliers(df)

    # 4. Drop highly correlated features
    df, dropped_cols = remove_correlated_features(df)

    # 5. Separate features and target
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    X = df[available_features].values.astype(np.float64)
    y = df[TARGET_COL].values.astype(np.float64)
    print(f"[preprocess] Feature matrix: {X.shape}, Target: {y.shape}")

    # 6. Stratified train/test split (before normalization to prevent leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"[preprocess] Train: {X_train.shape}, Test: {X_test.shape}")

    # 7. Min-Max normalization (fit on train, apply to test)
    X_train, feat_min, feat_max = minmax_normalize(X_train)
    X_test, _, _                = minmax_normalize(X_test, feat_min, feat_max)

    scaler_params = {"min": feat_min, "max": feat_max}

    # 8. Save
    if save:
        os.makedirs("models", exist_ok=True)
        np.save("models/X_train.npy", X_train)
        np.save("models/X_test.npy",  X_test)
        np.save("models/y_train.npy", y_train)
        np.save("models/y_test.npy",  y_test)
        np.save("models/feat_min.npy", feat_min)
        np.save("models/feat_max.npy", feat_max)
        np.save("models/feature_names.npy", np.array(available_features))
        print("[preprocess] Saved splits and scaler params to models/")

    print("\n── Preprocessing complete ───────────────────────────")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")
    print(f"  Class balance (train) — 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")

    return X_train, X_test, y_train, y_test, available_features, scaler_params


def load_preprocessed():
    """Load previously saved train/test splits."""
    X_train = np.load("models/X_train.npy")
    X_test  = np.load("models/X_test.npy")
    y_train = np.load("models/y_train.npy")
    y_test  = np.load("models/y_test.npy")
    feat_min = np.load("models/feat_min.npy")
    feat_max = np.load("models/feat_max.npy")
    feature_names = np.load("models/feature_names.npy", allow_pickle=True).tolist()
    scaler_params = {"min": feat_min, "max": feat_max}
    return X_train, X_test, y_train, y_test, feature_names, scaler_params


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preprocess_pipeline(save=True, run_eda_flag=True)
