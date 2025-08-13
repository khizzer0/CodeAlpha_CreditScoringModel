import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

# -------- Paths --------
X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"
MODEL_PATH  = "models/XGBoost_model.pkl"  # Change if your file name is different
FIG_DIR     = "reports/figures"

# -------- Load Test Data --------
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

# Drop accidental index column if present
drop_cols = [c for c in X_test.columns if c.lower().startswith("unnamed")]
if drop_cols:
    X_test = X_test.drop(columns=drop_cols, errors="ignore")

# -------- Load Model --------
model = joblib.load(MODEL_PATH)

# -------- Predictions --------
proba = model.predict_proba(X_test)[:, 1]
preds = (proba >= 0.5).astype(int)

# -------- Metrics --------
print("\n=== Classification Report ===")
print(classification_report(y_test, preds))
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

# -------- Save Plots --------
os.makedirs(FIG_DIR, exist_ok=True)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure()
plt.plot(fpr, tpr, label="ROC curve")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"), bbox_inches="tight")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
plt.figure()
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, val, ha="center", va="center", color="red")
plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), bbox_inches="tight")
plt.close()

print("\n✅ Evaluation complete. Plots saved in reports/figures/")
