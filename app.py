from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from ann_model import ANN, ANNNet, DEVICE
from knn_model import KNN
from logistic_regression import LogisticRegression
from preprocess import ALL_FEATURES


st.set_page_config(page_title="Diabetes Risk Prediction", layout="wide")

MODELS_DIR = Path("models")
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.npy"
FEAT_MIN_PATH = MODELS_DIR / "feat_min.npy"
FEAT_MAX_PATH = MODELS_DIR / "feat_max.npy"
ANN_CFG_PATH = MODELS_DIR / "ann_best_config.npy"
ANN_WEIGHTS_PATH = MODELS_DIR / "ann_weights.pt"
ANN_IMPORTANCE_PATH = MODELS_DIR / "ann_importances.npy"
LOGREG_WEIGHTS_PATH = MODELS_DIR / "logreg_weights.npy"
LOGREG_BIAS_PATH = MODELS_DIR / "logreg_bias.npy"
LOGREG_IMPORTANCE_PATH = MODELS_DIR / "logreg_importances.npy"
KNN_CFG_PATH = MODELS_DIR / "knn_best_config.npy"
KNN_IMPORTANCE_PATH = MODELS_DIR / "knn_importances.npy"
X_TRAIN_PATH = MODELS_DIR / "X_train.npy"
Y_TRAIN_PATH = MODELS_DIR / "y_train.npy"


def _normalize_row(row: np.ndarray, feat_min: np.ndarray, feat_max: np.ndarray) -> np.ndarray:
    denom = feat_max - feat_min
    denom[denom == 0] = 1.0
    return (row - feat_min) / denom


@st.cache_resource
def load_shared_artifacts() -> tuple[list[str], np.ndarray, np.ndarray]:
    required_paths = [FEATURE_NAMES_PATH, FEAT_MIN_PATH, FEAT_MAX_PATH]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files in models/: "
            + ", ".join(missing)
            + ". Please run preprocess.py first."
        )

    feature_names = np.load(FEATURE_NAMES_PATH, allow_pickle=True).tolist()
    feat_min = np.load(FEAT_MIN_PATH)
    feat_max = np.load(FEAT_MAX_PATH)
    return feature_names, feat_min, feat_max


def load_ann_model(feature_names: list[str]) -> ANN | None:
    required_paths = [ANN_CFG_PATH, ANN_WEIGHTS_PATH]
    if not all(p.exists() for p in required_paths):
        return None

    best_cfg = np.load(ANN_CFG_PATH, allow_pickle=True)
    lr, hidden_dim, epochs = float(best_cfg[0]), int(best_cfg[1]), int(best_cfg[2])

    model = ANN(
        input_dim=len(feature_names),
        hidden_dim=hidden_dim,
        learning_rate=lr,
        epochs=epochs,
        device=DEVICE,
    )
    model.net = ANNNet(len(feature_names), hidden_dim).to(DEVICE)

    state = torch.load(ANN_WEIGHTS_PATH, map_location=DEVICE)
    model.net.load_state_dict(state)
    model.net.eval()
    return model


def load_logreg_model(feature_names: list[str]) -> LogisticRegression | None:
    required_paths = [LOGREG_WEIGHTS_PATH, LOGREG_BIAS_PATH]
    if not all(p.exists() for p in required_paths):
        return None

    model = LogisticRegression()
    model.w = np.load(LOGREG_WEIGHTS_PATH)
    model.b = float(np.load(LOGREG_BIAS_PATH)[0])
    if len(model.w) != len(feature_names):
        return None
    return model


def load_knn_model() -> KNN | None:
    required_paths = [KNN_CFG_PATH, X_TRAIN_PATH, Y_TRAIN_PATH]
    if not all(p.exists() for p in required_paths):
        return None

    best_cfg = np.load(KNN_CFG_PATH, allow_pickle=True)
    k = int(best_cfg[0])
    metric = str(best_cfg[1])

    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)

    model = KNN(num_neighbors=k, metric=metric)
    model.fit(X_train, y_train)
    return model


def load_importances(model_key: str) -> np.ndarray | None:
    path_map = {
        "ANN": ANN_IMPORTANCE_PATH,
        "Logistic Regression": LOGREG_IMPORTANCE_PATH,
        "KNN": KNN_IMPORTANCE_PATH,
    }
    path = path_map[model_key]
    if path.exists():
        return np.load(path)
    return None


def default_value(feature: str) -> float:
    defaults = {
        "BMI": 27.5,
        "MentHlth": 0.0,
        "PhysHlth": 0.0,
        "GenHlth": 3.0,
        "Age": 8.0,
        "Education": 5.0,
        "Income": 6.0,
    }
    if feature in defaults:
        return defaults[feature]
    if feature in ALL_FEATURES:
        return 0.0
    return 0.0


st.title("Diabetes Risk Prediction")
st.caption("Input patient features and compare ANN / Logistic Regression / KNN.")

try:
    feature_names, feat_min, feat_max = load_shared_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info(
        "Run the following first:\n"
        "1) `python3 preprocess.py`\n"
        "2) `python3 ann_model.py` (optional for ANN)\n"
        "3) `python3 logistic_regression.py` (optional for Logistic Regression)\n"
        "4) `python3 knn_model.py` (optional for KNN)\n"
        "Then run `streamlit run app.py` again."
    )
    st.stop()

ann_model = load_ann_model(feature_names)
logreg_model = load_logreg_model(feature_names)
knn_model = load_knn_model()

available_models: dict[str, object] = {}
if ann_model is not None:
    available_models["ANN"] = ann_model
if logreg_model is not None:
    available_models["Logistic Regression"] = logreg_model
if knn_model is not None:
    available_models["KNN"] = knn_model

if not available_models:
    st.error("No trained model artifacts found in models/.")
    st.info(
        "Run one or more of the following:\n"
        "- `python3 ann_model.py`\n"
        "- `python3 logistic_regression.py`\n"
        "- `python3 knn_model.py`\n"
        "Then run `streamlit run app.py` again."
    )
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Input Variables")
    input_values: dict[str, float] = {}

    binary_features = {
        "HighBP",
        "HighChol",
        "CholCheck",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "DiffWalk",
        "Sex",
    }
    ordinal_ranges = {
        "GenHlth": (1, 5),
        "Age": (1, 13),
        "Education": (1, 6),
        "Income": (1, 8),
    }

    for feature in feature_names:
        if feature in binary_features:
            value = st.toggle(feature, value=bool(default_value(feature)))
            input_values[feature] = float(value)
        elif feature in ordinal_ranges:
            min_v, max_v = ordinal_ranges[feature]
            value = st.slider(
                feature,
                min_value=min_v,
                max_value=max_v,
                value=int(default_value(feature)),
                step=1,
            )
            input_values[feature] = float(value)
        else:
            value = st.number_input(
                feature,
                value=float(default_value(feature)),
                step=0.1,
                format="%.2f",
            )
            input_values[feature] = float(value)

    predict_btn = st.button("Predict Risk", type="primary", use_container_width=True)

with col2:
    st.subheader("Prediction Results")
    selected_model_name = st.selectbox("Model", list(available_models.keys()))
    selected_model = available_models[selected_model_name]

    if predict_btn:
        raw_vector = np.array([input_values[f] for f in feature_names], dtype=np.float64)
        normalized = _normalize_row(raw_vector, feat_min, feat_max).reshape(1, -1)
        risk_prob = float(selected_model.predict_proba(normalized)[0])
        risk_label = "High Risk" if risk_prob >= 0.5 else "Low Risk"

        st.metric("Selected Model", selected_model_name)
        st.metric("Diabetes Probability", f"{risk_prob * 100:.2f}%")
        st.metric("Predicted Class", risk_label)
    else:
        st.info("Adjust inputs and click 'Predict Risk' to see the result.")

    st.divider()
    st.subheader("Feature Importance")
    st.caption("Permutation importance from offline evaluation.")

    importances = load_importances(selected_model_name)
    if importances is None:
        train_cmd = {
            "ANN": "`python3 ann_model.py`",
            "Logistic Regression": "`python3 logistic_regression.py`",
            "KNN": "`python3 knn_model.py`",
        }[selected_model_name]
        st.warning(f"No importance file for {selected_model_name}. Run {train_cmd} to generate it.")
    else:
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances.astype(float)}
        ).sort_values("Importance", ascending=False)

        chart = (
            alt.Chart(importance_df)
            .mark_bar()
            .encode(
                x=alt.X("Importance:Q", title="Delta F1"),
                y=alt.Y("Feature:N", sort="-x", title="Feature"),
                tooltip=["Feature", "Importance"],
            )
            .properties(height=400)
        )
        st.altair_chart(chart, use_container_width=True)
