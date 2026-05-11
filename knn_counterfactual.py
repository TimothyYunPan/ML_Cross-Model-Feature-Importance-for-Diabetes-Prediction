"""
knn_counterfactual.py
---------------------
Local, per-patient explanations for the KNN diabetes classifier.

Two symmetric use cases, both restricted to *modifiable* lifestyle features and
both direction-aware so we never recommend moving toward a less-healthy state:

  direction = "healthier"  (patient is predicted DIABETIC)
      Reference cohort = K' nearest training points with label=0 (healthy).
      For each lifestyle feature we measure how far the patient sits from this
      reference in the *unhealthy direction* (e.g. higher BMI is unhealthy,
      lower PhysActivity is unhealthy). Top-K features with the largest such
      gap become "recommended changes" — places where the patient is most
      unlike the matched-healthy cohort and could improve.

  direction = "warning"   (patient is predicted NON-DIABETIC)
      Reference cohorts = K' nearest label=0 AND K' nearest label=1 neighbors.
      For each lifestyle feature we compute a normalized "risk proximity":
            0.0  → aligned with the healthy mean
            1.0  → aligned with the diabetic mean
           >1.0  → worse than the diabetic mean
      Features whose risk proximity exceeds a threshold are surfaced as "you
      are within X% of the matched-diabetic profile here" warnings.

This is a counterfactual / nearest-opposite-class explanation, restricted to
features the patient can plausibly act on. Global permutation importance
(knn_importances.npy) answers "which features matter on average"; this module
answers "which features matter for THIS patient, right now".

NOT medical advice — intended for educational visualization only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

Direction = Literal["healthier", "warning"]

# Lifestyle features the patient can plausibly modify. Excludes:
#   - demographics: Age, Sex, Education, Income
#   - medical history: HighBP, HighChol, Stroke, HeartDiseaseorAttack, DiffWalk, CholCheck
#   - healthcare access: AnyHealthcare, NoDocbcCost
MODIFIABLE_FEATURES: tuple[str, ...] = (
    "BMI",
    "Smoker",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
)

# Per-feature direction of "healthier": "lower" → lower values are healthier;
# "higher" → higher values are healthier. Encoded so the engine can flag *only*
# improvements toward the healthy direction and never recommend moving the
# wrong way (e.g. lowering Veggies or raising BMI).
HEALTHIER_DIRECTION: dict[str, str] = {
    "BMI": "lower",
    "Smoker": "lower",
    "PhysActivity": "higher",
    "Fruits": "higher",
    "Veggies": "higher",
    "HvyAlcoholConsump": "lower",
    "GenHlth": "lower",  # 1 = excellent ... 5 = poor
    "MentHlth": "lower",  # poor mental-health days in past 30
    "PhysHlth": "lower",  # poor physical-health days in past 30
}

# Features that are 0/1 indicators. Reference values are shown as rates
# ("70% of similar healthy people"); suggested actions are switches.
BINARY_LIFESTYLE_FEATURES: frozenset[str] = frozenset(
    {"Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"}
)

# Warning threshold on risk_proximity (0 = healthy, 1 = diabetic). Features
# below this aren't surfaced as warnings — the patient is closer to the healthy
# reference than to the diabetic reference on that axis.
DEFAULT_WARNING_THRESHOLD = 0.30


@dataclass(frozen=True)
class FeatureInsight:
    """One row in the personalized insight panel."""

    feature: str
    current_value: float  # in original units
    reference_value: (
        float  # in original units (μ_healthy for "healthier", μ_diabetic for "warning")
    )
    delta: float  # current - reference, original units (for display)
    score: float  # ranking score in normalized space
    #   healthier: signed gap toward healthy reference (≥ 0)
    #   warning:   risk proximity to diabetic reference (0..1+)
    action: str  # short imperative for the UI ("Lower BMI", "Watch BMI", ...)
    explanation: str  # one-sentence rationale


def _direction_sign(feature: str) -> int:
    """+1 if higher values are healthier, -1 if lower values are healthier."""
    return +1 if HEALTHIER_DIRECTION[feature] == "higher" else -1


def _nearest_label_indices(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_norm: np.ndarray,
    target_label: float,
    n_reference: int,
    metric: str,
) -> np.ndarray:
    """Return indices into X_train of the n_reference closest rows with y == target_label."""
    mask = y_train == target_label
    candidate_idx = np.where(mask)[0]
    candidates = X_train[mask]
    if metric == "Manhattan":
        dists = np.sum(np.abs(candidates - x_norm), axis=1)
    else:
        dists = np.sqrt(np.sum((candidates - x_norm) ** 2, axis=1))
    n = min(n_reference, len(candidates))
    if n == 0:
        return np.empty(0, dtype=np.int64)
    nearest_local = np.argpartition(dists, n - 1)[:n]
    return candidate_idx[nearest_local]


def _denormalize(value_norm: float, feat_min_i: float, feat_max_i: float) -> float:
    span = feat_max_i - feat_min_i
    if span == 0:
        return float(feat_min_i)
    return float(value_norm * span + feat_min_i)


def _build_recommend_insight(
    feature: str,
    patient_norm: float,
    healthy_ref_norm: float,
    feat_min_i: float,
    feat_max_i: float,
    score: float,
) -> FeatureInsight:
    """direction='healthier' — one row of the recommendation panel."""
    cur_orig = _denormalize(patient_norm, feat_min_i, feat_max_i)
    ref_orig = _denormalize(healthy_ref_norm, feat_min_i, feat_max_i)
    delta_orig = cur_orig - ref_orig

    if feature in BINARY_LIFESTYLE_FEATURES:
        ref_rate = ref_orig  # mean of 0/1 is a rate
        if HEALTHIER_DIRECTION[feature] == "higher":
            # e.g. Veggies: patient=0, ref_rate=0.85 → start eating veggies
            action = f"Start {feature}"
            explanation = (
                f"You report {feature}=0; {ref_rate * 100:.0f}% of similar "
                f"healthy people do."
            )
        else:
            # e.g. Smoker: patient=1, ref_rate=0.30 → stop smoking
            action = f"Stop {feature}"
            explanation = (
                f"You report {feature}=1; only {ref_rate * 100:.0f}% of "
                f"similar healthy people do."
            )
    else:
        if HEALTHIER_DIRECTION[feature] == "lower":
            action = f"Lower {feature}"
        else:
            action = f"Raise {feature}"
        explanation = (
            f"You: {cur_orig:.1f}. Matched-healthy peers average "
            f"{ref_orig:.1f} (gap {delta_orig:+.1f})."
        )

    return FeatureInsight(
        feature=feature,
        current_value=cur_orig,
        reference_value=ref_orig,
        delta=delta_orig,
        score=score,
        action=action,
        explanation=explanation,
    )


def _build_warning_insight(
    feature: str,
    patient_norm: float,
    healthy_ref_norm: float,
    diabetic_ref_norm: float,
    feat_min_i: float,
    feat_max_i: float,
    risk_proximity: float,
) -> FeatureInsight:
    """direction='warning' — one row of the warning panel."""
    cur_orig = _denormalize(patient_norm, feat_min_i, feat_max_i)
    healthy_orig = _denormalize(healthy_ref_norm, feat_min_i, feat_max_i)
    diabetic_orig = _denormalize(diabetic_ref_norm, feat_min_i, feat_max_i)
    delta_to_diabetic = cur_orig - diabetic_orig

    if feature in BINARY_LIFESTYLE_FEATURES:
        diabetic_rate = diabetic_orig
        if HEALTHIER_DIRECTION[feature] == "higher":
            # Patient probably has feature=0 (unhealthy side), aligned with diabetic peers.
            action = f"Start {feature}"
            explanation = (
                f"You report {feature}=0; only "
                f"{diabetic_rate * 100:.0f}% of similar diabetic patients do — "
                f"you match their pattern."
            )
        else:
            # Patient probably has feature=1 (unhealthy side).
            action = f"Watch: {feature}=1"
            explanation = (
                f"You report {feature}=1, matching "
                f"{diabetic_rate * 100:.0f}% of similar diabetic patients."
            )
    else:
        action = f"Watch {feature}"
        explanation = (
            f"You: {cur_orig:.1f}. Matched-healthy avg {healthy_orig:.1f}, "
            f"matched-diabetic avg {diabetic_orig:.1f} — you are "
            f"{risk_proximity * 100:.0f}% of the way toward the diabetic profile."
        )

    return FeatureInsight(
        feature=feature,
        current_value=cur_orig,
        reference_value=diabetic_orig,
        delta=delta_to_diabetic,
        score=risk_proximity,
        action=action,
        explanation=explanation,
    )


def _validate_inputs(knn, x_norm: np.ndarray, feature_names: list[str]) -> np.ndarray:
    if knn.X_train is None or knn.y_train is None:
        raise ValueError("KNN model is not fitted.")
    x_norm = np.asarray(x_norm, dtype=np.float64).ravel()
    if x_norm.shape != (len(feature_names),):
        raise ValueError(
            f"x_norm shape {x_norm.shape} does not match {len(feature_names)} features."
        )
    return x_norm


def _recommend(
    knn,
    x_norm: np.ndarray,
    feature_names: list[str],
    feat_min: np.ndarray,
    feat_max: np.ndarray,
    top_k: int,
    n_reference: int,
    modifiable: set[str],
) -> list[FeatureInsight]:
    healthy_idx = _nearest_label_indices(
        knn.X_train, knn.y_train, x_norm, 0.0, n_reference, knn.metric
    )
    if len(healthy_idx) == 0:
        return []
    mu_healthy = knn.X_train[healthy_idx].mean(axis=0)

    scored: list[tuple[int, float]] = []
    for i, name in enumerate(feature_names):
        if name not in modifiable:
            continue
        sign = _direction_sign(name)  # +1 higher healthier, -1 lower healthier
        # Gap toward unhealthy direction relative to healthy reference, in [0, 1].
        # Positive only when patient is on the unhealthy side of μ_healthy.
        gap = sign * (mu_healthy[i] - x_norm[i])
        if gap > 0:
            scored.append((i, float(gap)))

    scored.sort(key=lambda t: t[1], reverse=True)
    top = scored[:top_k]

    return [
        _build_recommend_insight(
            feature=feature_names[i],
            patient_norm=float(x_norm[i]),
            healthy_ref_norm=float(mu_healthy[i]),
            feat_min_i=float(feat_min[i]),
            feat_max_i=float(feat_max[i]),
            score=score,
        )
        for i, score in top
    ]


def _warn(
    knn,
    x_norm: np.ndarray,
    feature_names: list[str],
    feat_min: np.ndarray,
    feat_max: np.ndarray,
    top_k: int,
    n_reference: int,
    modifiable: set[str],
    threshold: float,
) -> list[FeatureInsight]:
    healthy_idx = _nearest_label_indices(
        knn.X_train, knn.y_train, x_norm, 0.0, n_reference, knn.metric
    )
    diabetic_idx = _nearest_label_indices(
        knn.X_train, knn.y_train, x_norm, 1.0, n_reference, knn.metric
    )
    if len(healthy_idx) == 0 or len(diabetic_idx) == 0:
        return []
    mu_healthy = knn.X_train[healthy_idx].mean(axis=0)
    mu_diabetic = knn.X_train[diabetic_idx].mean(axis=0)

    scored: list[tuple[int, float]] = []
    for i, name in enumerate(feature_names):
        if name not in modifiable:
            continue
        sign = _direction_sign(name)
        # In a healthy-axis frame, larger value = healthier.
        # We define a unhealthy span (h → d), then project patient onto it:
        #   risk_proximity = (μ_healthy_health_axis - patient_health_axis) /
        #                    (μ_healthy_health_axis - μ_diabetic_health_axis)
        # Multiplying both numerator and denominator by `sign` to keep signs aligned:
        unhealthy_span = sign * (
            mu_healthy[i] - mu_diabetic[i]
        )  # always ≥ 0 in expectation
        patient_unhealthy = sign * (mu_healthy[i] - x_norm[i])
        if unhealthy_span <= 1e-9:
            continue
        risk_proximity = patient_unhealthy / unhealthy_span
        if risk_proximity >= threshold:
            scored.append((i, float(risk_proximity)))

    scored.sort(key=lambda t: t[1], reverse=True)
    top = scored[:top_k]

    return [
        _build_warning_insight(
            feature=feature_names[i],
            patient_norm=float(x_norm[i]),
            healthy_ref_norm=float(mu_healthy[i]),
            diabetic_ref_norm=float(mu_diabetic[i]),
            feat_min_i=float(feat_min[i]),
            feat_max_i=float(feat_max[i]),
            risk_proximity=score,
        )
        for i, score in top
    ]


def explain_query(
    knn,
    x_norm: np.ndarray,
    feature_names: list[str],
    feat_min: np.ndarray,
    feat_max: np.ndarray,
    direction: Direction,
    top_k: int = 3,
    n_reference_neighbors: int = 25,
    modifiable_features: Iterable[str] = MODIFIABLE_FEATURES,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
) -> list[FeatureInsight]:
    """
    Local counterfactual explanation for one query patient.

    Args:
        knn: fitted KNN model exposing X_train, y_train, metric.
        x_norm: query feature vector in min-max-normalized [0, 1] space, shape (n_features,).
        feature_names: list of feature names matching column order.
        feat_min, feat_max: per-feature min/max used during normalization, for
            denormalizing display values back into original units.
        direction: "healthier" (recommend changes) or "warning" (flag risks).
        top_k: maximum number of features to return.
        n_reference_neighbors: cohort size used to compute reference means.
            Larger → smoother reference; smaller → more locally specific.
        modifiable_features: feature names eligible to be surfaced.
        warning_threshold: minimum risk_proximity to qualify as a warning
            (only applies when direction="warning"). 0.5 ≈ halfway from
            healthy to diabetic; default 0.3 is intentionally permissive.

    Returns:
        List of FeatureInsight, length ≤ top_k, sorted by score descending.
        May be shorter than top_k (or empty) if not enough features clear the
        direction filter or warning threshold.
    """
    x_norm = _validate_inputs(knn, x_norm, feature_names)
    modifiable = set(modifiable_features)

    if direction == "healthier":
        return _recommend(
            knn,
            x_norm,
            feature_names,
            feat_min,
            feat_max,
            top_k,
            n_reference_neighbors,
            modifiable,
        )
    if direction == "warning":
        return _warn(
            knn,
            x_norm,
            feature_names,
            feat_min,
            feat_max,
            top_k,
            n_reference_neighbors,
            modifiable,
            warning_threshold,
        )
    raise ValueError(f"Unknown direction: {direction!r}")
