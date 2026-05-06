# ML_Cross-Model-Feature-Importance-for-Diabetes-Prediction
## Dataset & Preprocessing

**Must download the dataset manually from Kaggle before running any code.**
 
1. Go to: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
2. Click **Download** (requires a free Kaggle account)
3. Unzip the downloaded file
4. Place the following file inside the `data/` folder in this repo:
```
data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv
```
 
> We use the **balanced binary 50/50 split version** (70,692 records, 50% diabetic / 50% non-diabetic) for training. The full unbalanced dataset (`diabetes_binary_health_indicators_BRFSS2015.csv`) is used for real-world generalization evaluation.

### 1. Preprocessing & EDA
```bash
python preprocess.py
```
 **Steps:**
 
1. **Load** — Load the balanced binary dataset (50/50 class split, 35,346 per class), confirm no missing values.
2. **EDA** — Inspect mean/std/min/max for all 21 features and verify class distribution.
3. **BMI Outlier Removal** — Remove rows where BMI falls outside Q1 − 1.5×IQR or Q3 + 1.5×IQR. Reduced dataset from 70,692 → 68,511 rows.
4. **Correlation Filtering** — Compute pairwise Pearson correlation; drop one feature from any pair with |r| > 0.7 to reduce multicollinearity. No features were dropped for this dataset (no pair exceeded the threshold).
5. **Stratified Train/Test Split** — 80/20 split stratified by class label to preserve class proportions in both sets. Implemented with NumPy only.
6. **Min-Max Normalization** — Scale all features to [0, 1]. Fit on train set only, then apply to test set to prevent data leakage.
7. **Save** — Write `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `feat_min.npy`, `feat_max.npy`, and `feature_names.npy` to `models/`.
**Result:** X_train: (54,810 × 21), X_test: (13,701 × 21)
 
> **Note:** After BMI outlier removal, the train set class balance is 27,888 (negative) vs. 26,922 (positive) — a slight imbalance. This is expected: BMI outliers were not evenly distributed across classes, so removing them introduced a small skew. This is not a bug and does not significantly affect model training given the near-equal split.

## Frontend-Backend Integration (Summary)

This project currently uses a single Streamlit app (`app.py`) for both frontend and backend logic.

- Frontend: user inputs and UI rendering in Streamlit
- Backend: feature normalization + model inference (ANN / Logistic Regression / KNN)
- Data flow: input -> normalize -> `predict_proba` -> probability + risk class + feature-importance chart

For detailed integration notes, see:
- `README_frontend_backend.md`