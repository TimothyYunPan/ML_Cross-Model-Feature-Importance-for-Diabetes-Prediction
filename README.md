# ML_Cross-Model-Feature-Importance-for-Diabetes-Prediction
## Dataset & Preprocessing

** Must download the dataset manually from Kaggle before running any code.**
 
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
 