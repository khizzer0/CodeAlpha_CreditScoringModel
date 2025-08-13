import os, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib

RAW_PATH = "data/raw/cs-training.csv"
MODEL_DIR = "models"
PIPELINE_PATH = os.path.join(MODEL_DIR, "best_pipeline.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")
TARGET = "SeriousDlqin2yrs"
DROP_COLS = ["Unnamed: 0", "Unnamed: 0.1"]  # any stray index cols

os.makedirs(MODEL_DIR, exist_ok=True)

# 1) Load raw
df = pd.read_csv(RAW_PATH)

# 2) Define features (remove target + junk cols)
all_cols = [c for c in df.columns if c != TARGET]
feature_names = [c for c in all_cols if c not in DROP_COLS]

X = df[feature_names]
y = df[TARGET]

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Build pipeline: impute -> scale -> model
pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=True)),
    ("model", XGBClassifier(
        eval_metric="logloss",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=5
    ))
])

# 5) Fit & evaluate
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
preds = (proba >= 0.5).astype(int)

print(classification_report(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, proba))

# 6) Save pipeline + feature names
joblib.dump(pipe, PIPELINE_PATH)
with open(FEATURES_PATH, "w") as f:
    json.dump(feature_names, f)

print(f"✅ Saved pipeline -> {PIPELINE_PATH}")
print(f"✅ Saved feature set -> {FEATURES_PATH}")
