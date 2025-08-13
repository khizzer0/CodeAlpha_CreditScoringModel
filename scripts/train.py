import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def load_processed_data(processed_dir):
    """Load processed train/test data."""
    X_train = pd.read_csv(f"{processed_dir}/X_train.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    X_test = pd.read_csv(f"{processed_dir}/X_test.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    y_train = pd.read_csv(f"{processed_dir}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{processed_dir}/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model and print metrics."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, preds))
    print("ROC-AUC Score:", roc_auc_score(y_test, proba))
    return roc_auc_score(y_test, proba)

def train_and_select_best(X_train, X_test, y_train, y_test):
    """Train multiple models and select the best."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced"),
        "XGBoost": XGBClassifier(eval_metric="logloss", scale_pos_weight=5)
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)
        score = evaluate_model(model, X_test, y_test)
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    print(f"\n✅ Best Model: {best_name} with ROC-AUC: {best_score:.4f}")
    return best_model, best_name

if __name__ == "__main__":
    processed_dir = "data/processed"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)

    # Train & select best model
    best_model, best_name = train_and_select_best(X_train, X_test, y_train, y_test)

    # Save best model
    model_path = f"{model_dir}/{best_name}_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"💾 Model saved at {model_path}")
