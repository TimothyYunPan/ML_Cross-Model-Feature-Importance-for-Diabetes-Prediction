from __future__ import annotations

import ast
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ann_model import ANN, ANN2, ANNNet, ANNNet2
from knn_counterfactual import explain_query
from knn_model import KNN
from logistic_regression import LogisticRegression
from preprocess import ALL_FEATURES


st.set_page_config(page_title="Diabetes Risk Prediction", layout="wide")

# ── Human-readable feature metadata ──────────────────────────────────────────
# Sourced from diabetes-health-indicators-dataset-notebook.ipynb (the cleaning
# script that derived these 21 columns from BRFSS 2015). The notebook is the
# authoritative codebook reference here; do not paraphrase its mapping without
# re-checking the cells.

# Toggle widgets: stored values are 0.0/1.0 to match the trained model's input.
BINARY_INPUT_FEATURES: frozenset[str] = frozenset(
    {
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
    }
)

# Each entry: "label" (always), optional "help" tooltip, optional "options" dict
# for ordinal features (raw int code → human-readable string).
FEATURE_META: dict[str, dict] = {
    # Demographics
    "Sex": {
        "label": "Male",
        "help": "On = male, Off = female. (BRFSS encodes Male=1, Female=0.)",
    },
    "Age": {
        "label": "Age group",
        "help": "BRFSS uses 5-year buckets, not raw age.",
        "options": {
            1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39",
            5: "40–44", 6: "45–49", 7: "50–54", 8: "55–59",
            9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79",
            13: "80 or older",
        },
    },
    "Education": {
        "label": "Education",
        "options": {
            1: "Never attended / Kindergarten",
            2: "Grades 1–8 (elementary)",
            3: "Grades 9–11 (some high school)",
            4: "High school graduate / GED",
            5: "Some college / Technical school",
            6: "College graduate (4 years or more)",
        },
    },
    "Income": {
        "label": "Annual household income",
        "options": {
            1: "Less than $10,000",
            2: "Less than $15,000",
            3: "Less than $20,000",
            4: "Less than $25,000",
            5: "Less than $35,000",
            6: "Less than $50,000",
            7: "Less than $75,000",
            8: "$75,000 or more",
        },
    },
    # Lifestyle
    "BMI": {
        "label": "BMI (kg/m²)",
        "help": "Body mass index. Normal ~18.5–25, overweight 25–30, obese ≥30.",
    },
    "Smoker": {
        "label": "Smoker",
        "help": "Have smoked at least 100 cigarettes in your lifetime.",
    },
    "PhysActivity": {
        "label": "Physical activity in past 30 days",
        "help": "Any physical activity outside of your regular job in the last 30 days.",
    },
    "Fruits": {
        "label": "Eat fruit at least once per day",
    },
    "Veggies": {
        "label": "Eat vegetables at least once per day",
    },
    "HvyAlcoholConsump": {
        "label": "Heavy alcohol use",
        "help": "Men >14 drinks/week or women >7 drinks/week.",
    },
    # Health status
    "GenHlth": {
        "label": "General health rating",
        "help": "Self-assessed overall health.",
        "options": {
            1: "1 — Excellent",
            2: "2 — Very good",
            3: "3 — Good",
            4: "4 — Fair",
            5: "5 — Poor",
        },
    },
    "MentHlth": {
        "label": "Poor mental-health days (past 30)",
        "help": "Days in the last 30 with stress, depression, or emotional problems (0–30).",
    },
    "PhysHlth": {
        "label": "Poor physical-health days (past 30)",
        "help": "Days in the last 30 with physical illness or injury (0–30).",
    },
    "DiffWalk": {
        "label": "Serious difficulty walking or climbing stairs",
    },
    # Medical history
    "HighBP": {
        "label": "High blood pressure",
        "help": "Told by a doctor that you have high blood pressure.",
    },
    "HighChol": {
        "label": "High cholesterol",
        "help": "Told by a doctor that you have high cholesterol.",
    },
    "CholCheck": {
        "label": "Cholesterol check in past 5 years",
    },
    "Stroke": {
        "label": "Ever had a stroke",
    },
    "HeartDiseaseorAttack": {
        "label": "History of heart disease or heart attack",
    },
    # Healthcare access
    "AnyHealthcare": {
        "label": "Has any healthcare coverage",
    },
    "NoDocbcCost": {
        "label": "Skipped seeing a doctor due to cost (past year)",
    },
}

# Rendering order: groups patient inputs so a non-clinician can fill them in
# top-down (who you are → how you live → how you feel → what doctors told you).
INPUT_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Demographics",       ("Sex", "Age", "Education", "Income")),
    ("Lifestyle",          ("BMI", "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump")),
    ("Health status",      ("GenHlth", "MentHlth", "PhysHlth", "DiffWalk")),
    ("Medical history",    ("HighBP", "HighChol", "CholCheck", "Stroke", "HeartDiseaseorAttack")),
    ("Healthcare access",  ("AnyHealthcare", "NoDocbcCost")),
)

# Human-readable action labels for the personalized-insights panel. Indexed by
# feature name; the engine's raw verb ("Lower", "Stop", "Watch") is replaced by
# a hand-written phrase that reads naturally to non-clinicians.
FRIENDLY_RECOMMENDATIONS: dict[str, str] = {
    "BMI": "Lower your BMI",
    "Smoker": "Stop smoking",
    "PhysActivity": "Start regular physical activity",
    "Fruits": "Eat fruit every day",
    "Veggies": "Eat vegetables every day",
    "HvyAlcoholConsump": "Cut back on heavy drinking",
    "GenHlth": "Improve your overall health",
    "MentHlth": "Reduce your poor mental-health days",
    "PhysHlth": "Reduce your poor physical-health days",
}
FRIENDLY_WARNINGS: dict[str, str] = {
    "BMI": "Watch your BMI",
    "Smoker": "You currently smoke",
    "PhysActivity": "You aren't physically active",
    "Fruits": "You don't eat fruit daily",
    "Veggies": "You don't eat vegetables daily",
    "HvyAlcoholConsump": "You report heavy drinking",
    "GenHlth": "Your general-health rating is concerning",
    "MentHlth": "Watch your mental-health days",
    "PhysHlth": "Watch your physical-health days",
}


MODELS_DIR = Path("models")
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.npy"
FEAT_MIN_PATH = MODELS_DIR / "feat_min.npy"
FEAT_MAX_PATH = MODELS_DIR / "feat_max.npy"
ANN_CFG_PATH = MODELS_DIR / "ann_best_config.npy"
ANN_WEIGHTS_PATH = MODELS_DIR / "ann_weights.npz"
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


def load_ann_model(feature_names: list[str]) -> tuple[ANN | ANN2, float] | None:
    """Load the saved ANN plus its tuned decision threshold.

    Supports two config formats:

    - Legacy 3-tuple ``[lr, hidden_dim, epochs]`` produced by the proposal-grid
      script: always a 1-hidden-layer net, no tuned threshold → uses 0.5.
    - Extended 7-tuple ``[arch, hidden, lr, wd, batch, epochs, threshold]``
      produced by the refined / 2-layer search: ``arch`` is ``"1L"`` or
      ``"2L"``, ``hidden`` is an int or stringified tuple (e.g. ``"(128, 64)"``),
      ``threshold`` is the validation-tuned decision threshold from §5.8.
    """
    required_paths = [ANN_CFG_PATH, ANN_WEIGHTS_PATH]
    if not all(p.exists() for p in required_paths):
        return None

    best_cfg = np.load(ANN_CFG_PATH, allow_pickle=True)
    first = best_cfg[0]
    is_new_format = isinstance(first, str) and first in {"1L", "2L"}

    in_dim = len(feature_names)
    threshold = 0.5
    if is_new_format:
        arch = str(best_cfg[0])
        hidden_repr = best_cfg[1]
        lr = float(best_cfg[2])
        if len(best_cfg) >= 7:
            threshold = float(best_cfg[6])
        if arch == "2L":
            hidden_dims = (
                tuple(ast.literal_eval(hidden_repr))
                if isinstance(hidden_repr, str)
                else tuple(hidden_repr)
            )
            model: ANN | ANN2 = ANN2(input_dim=in_dim, hidden_dims=hidden_dims, learning_rate=lr)
            model.net = ANNNet2(in_dim, hidden_dims)
            weight_keys = ("W1", "b1", "W2", "b2", "W3", "b3")
        else:
            hidden_dim = int(
                ast.literal_eval(hidden_repr) if isinstance(hidden_repr, str) else hidden_repr
            )
            model = ANN(input_dim=in_dim, hidden_dim=hidden_dim, learning_rate=lr)
            model.net = ANNNet(in_dim, hidden_dim)
            weight_keys = ("W1", "b1", "W2", "b2")
    else:
        lr, hidden_dim, epochs = float(first), int(best_cfg[1]), int(best_cfg[2])
        model = ANN(input_dim=in_dim, hidden_dim=hidden_dim, learning_rate=lr, epochs=epochs)
        model.net = ANNNet(in_dim, hidden_dim)
        weight_keys = ("W1", "b1", "W2", "b2")

    with np.load(ANN_WEIGHTS_PATH) as data:
        state = {k: np.asarray(data[k]) for k in weight_keys}
    model.net.load_state_dict(state)
    return model, threshold


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


def _render_personalized_insights(
    knn: KNN,
    x_norm: np.ndarray,
    feature_names: list[str],
    feat_min: np.ndarray,
    feat_max: np.ndarray,
    is_high_risk: bool,
) -> None:
    """Show counterfactual-style recommendations or warnings based on KNN neighborhood."""
    st.divider()
    if is_high_risk:
        st.subheader("Personalized Recommendations")
        st.caption(
            "Lifestyle features where you sit furthest from the matched-healthy "
            "neighbors in KNN's training set. Educational only — not medical advice."
        )
        insights = explain_query(knn, x_norm, feature_names, feat_min, feat_max, direction="healthier")
        empty_msg = "Your lifestyle features already match the matched-healthy cohort."
    else:
        st.subheader("Warning Features")
        st.caption(
            "Lifestyle features where you most resemble the matched-diabetic neighbors. "
            "Educational only — not medical advice."
        )
        insights = explain_query(knn, x_norm, feature_names, feat_min, feat_max, direction="warning")
        empty_msg = "No lifestyle features cross the warning threshold — your profile clearly aligns with the healthy cohort."

    if not insights:
        st.success(empty_msg)
        return

    friendly = FRIENDLY_RECOMMENDATIONS if is_high_risk else FRIENDLY_WARNINGS
    for ins in insights:
        action_text = friendly.get(ins.feature, ins.action)
        score_label = (
            f"gap {ins.score:.2f}" if is_high_risk
            else f"{min(ins.score, 9.99) * 100:.0f}% of healthy→diabetic span"
        )
        with st.container(border=True):
            st.markdown(f"**{action_text}** — *{score_label}*")
            st.caption(ins.explanation)


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

ann_load = load_ann_model(feature_names)
logreg_model = load_logreg_model(feature_names)
knn_model = load_knn_model()

available_models: dict[str, object] = {}
# Per-model decision threshold for the High/Low Risk label. Defaults to 0.5;
# ANN uses the validation-tuned threshold from §5.8 (currently 0.39 — recall-
# biased, appropriate for screening). KNN and Logistic Regression were not
# threshold-tuned, so they keep 0.5.
model_thresholds: dict[str, float] = {}
ann_model: ANN | ANN2 | None = None
if ann_load is not None:
    ann_model, ann_threshold = ann_load
    available_models["ANN"] = ann_model
    model_thresholds["ANN"] = ann_threshold
if logreg_model is not None:
    available_models["Logistic Regression"] = logreg_model
    model_thresholds["Logistic Regression"] = 0.5
if knn_model is not None:
    available_models["KNN"] = knn_model
    model_thresholds["KNN"] = 0.5

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
    st.caption("Fill in like a health questionnaire. Hover the ⓘ icon next to any field for details.")
    input_values: dict[str, float] = {}
    available = set(feature_names)

    for section_name, section_features in INPUT_SECTIONS:
        section_features_present = [f for f in section_features if f in available]
        if not section_features_present:
            continue
        st.markdown(f"**{section_name}**")
        for feature in section_features_present:
            meta = FEATURE_META[feature]
            label = meta["label"]
            help_text = meta.get("help")
            widget_key = f"input_{feature}"

            if feature in BINARY_INPUT_FEATURES:
                value = st.toggle(
                    label,
                    value=bool(default_value(feature)),
                    help=help_text,
                    key=widget_key,
                )
                input_values[feature] = float(value)
            elif "options" in meta:
                opts: dict[int, str] = meta["options"]
                option_keys = list(opts.keys())
                default_key = int(default_value(feature)) or option_keys[0]
                if default_key not in opts:
                    default_key = option_keys[0]
                value = st.select_slider(
                    label,
                    options=option_keys,
                    value=default_key,
                    format_func=lambda k, _opts=opts: _opts[k],
                    help=help_text,
                    key=widget_key,
                )
                input_values[feature] = float(value)
            elif feature == "BMI":
                value = st.number_input(
                    label,
                    value=float(default_value(feature)),
                    min_value=10.0,
                    max_value=70.0,
                    step=0.5,
                    format="%.1f",
                    help=help_text,
                    key=widget_key,
                )
                input_values[feature] = float(value)
            elif feature in {"MentHlth", "PhysHlth"}:
                value = st.number_input(
                    label,
                    value=int(default_value(feature)),
                    min_value=0,
                    max_value=30,
                    step=1,
                    format="%d",
                    help=help_text,
                    key=widget_key,
                )
                input_values[feature] = float(value)
            else:
                value = st.number_input(
                    label,
                    value=float(default_value(feature)),
                    step=0.1,
                    format="%.2f",
                    help=help_text,
                    key=widget_key,
                )
                input_values[feature] = float(value)

    # Fallback: render any unmapped feature with the bare column name so a
    # future preprocessing change can't silently leave columns missing.
    for feature in feature_names:
        if feature in input_values:
            continue
        st.markdown("**Other**")
        value = st.number_input(
            feature,
            value=float(default_value(feature)),
            step=0.1,
            format="%.2f",
            key=f"input_{feature}",
        )
        input_values[feature] = float(value)

    predict_btn = st.button("Predict Risk", type="primary", width="stretch")

with col2:
    st.subheader("Prediction Results")
    selected_model_name = st.selectbox("Model", list(available_models.keys()))
    selected_model = available_models[selected_model_name]

    if predict_btn:
        raw_vector = np.array([input_values[f] for f in feature_names], dtype=np.float64)
        normalized = _normalize_row(raw_vector, feat_min, feat_max).reshape(1, -1)
        risk_prob = float(selected_model.predict_proba(normalized)[0])
        threshold = model_thresholds.get(selected_model_name, 0.5)
        is_high_risk = risk_prob >= threshold
        risk_label = "High Risk" if is_high_risk else "Low Risk"
        threshold_help = (
            "Validation-tuned threshold from §5.8 of the ANN notebook "
            "(maximizes F1, biases toward recall — appropriate for medical screening)."
            if selected_model_name == "ANN" and not np.isclose(threshold, 0.5)
            else "Default decision threshold."
        )

        st.metric("Selected Model", selected_model_name)
        st.metric("Diabetes Probability", f"{risk_prob * 100:.2f}%")
        st.metric(
            "Predicted Class",
            risk_label,
            help=f"Decision threshold: {threshold:.2f}. {threshold_help}",
        )

        if selected_model_name == "KNN" and knn_model is not None:
            _render_personalized_insights(
                knn_model,
                normalized.ravel(),
                feature_names,
                feat_min,
                feat_max,
                is_high_risk=is_high_risk,
            )
    else:
        st.info("Adjust inputs and click 'Predict Risk' to see the result.")

    st.divider()
    st.subheader("Feature Importance")
    st.caption(
        "Permutation importance: aggregate ΔF1 on the hold-out test set (precomputed). "
        "It updates when you change Model above, not when you edit inputs like Age or BMI; "
        "for input sensitivity, use Diabetes Probability above."
    )

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
        st.altair_chart(chart, width="stretch")
