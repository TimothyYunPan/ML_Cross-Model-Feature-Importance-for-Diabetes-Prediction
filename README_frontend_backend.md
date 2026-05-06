# Frontend / Backend Integration

## Current setup

This project currently uses a single Streamlit app (`app.py`) for both UI and prediction logic.

- Frontend: Streamlit inputs and layout
- Backend: model loading, normalization, and inference
- Artifacts: all trained files are read from `models/`

There is no separate API server right now.

## Request flow

1. User enters health indicators in Streamlit.
2. App builds a feature vector using `feature_names.npy`.
3. Input is normalized with `feat_min.npy` and `feat_max.npy`.
4. Selected model (ANN / Logistic Regression / KNN) runs `predict_proba`.
5. App shows probability, risk class, and feature-importance chart.

## Files involved

- `app.py`: UI + inference flow
- `preprocess.py`: preprocessing and saved normalization stats
- `ann_model.py`, `logistic_regression.py`, `knn_model.py`: model training and prediction interfaces

## Input / output

Input: 21 health features (binary, ordinal, continuous).  
Output:
- `selected_model`
- `diabetes_probability` (0 to 1)
- `predicted_class` (`High Risk` if >= 0.5, else `Low Risk`)

## Error handling

- If preprocessing artifacts are missing, app stops with setup instructions.
- If some model files are missing, only available models are shown.
- If no model is available, app stops with an error message.

## Run locally

```bash
python preprocess.py
python ann_model.py          # or logistic_regression.py / knn_model.py
streamlit run app.py
```

## Optional: split into real frontend + backend

If needed later, move prediction logic into an API service (e.g., `POST /predict`) and let Streamlit call it through HTTP.
