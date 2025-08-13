
import sys, json
import pandas as pd
import joblib
import numpy as np

PIPELINE_PATH = "models/best_pipeline.pkl"
FEATURES_PATH = "models/feature_names.json"

def load_pipeline_and_features():
    pipe = joblib.load(PIPELINE_PATH)
    with open(FEATURES_PATH, "r") as f:
        feature_names = json.load(f)
    return pipe, feature_names

def align_columns(df, feature_names):
    # Ensure exact same columns/order; fill missing with NaN
    return df.reindex(columns=feature_names, fill_value=np.nan)

def predict_from_csv(csv_path):
    pipe, feature_names = load_pipeline_and_features()
    df = pd.read_csv(csv_path)
    df = align_columns(df, feature_names)
    proba = pipe.predict_proba(df)[:, 1]
    preds = (proba >= 0.5).astype(int)
    out = pd.DataFrame({"Prediction": preds, "Probability_of_Default": proba})
    print(out.head(10))
    return out

def predict_single(example_dict):
    pipe, feature_names = load_pipeline_and_features()
    df = pd.DataFrame([example_dict])
    df = align_columns(df, feature_names)
    proba = pipe.predict_proba(df)[0, 1]
    pred = int(proba >= 0.5)
    print(f"Prediction: {pred} (0=No Default, 1=Default)")
    print(f"Probability of Default: {proba:.2f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_from_csv(sys.argv[1])
    else:
        # Example input with raw features (NO target, NO Unnamed columns)
        example_input = {
            "RevolvingUtilizationOfUnsecuredLines": 0.5,
            "age": 45,
            "NumberOfTime30-59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.30,
            "MonthlyIncome": 5000,
            "NumberOfOpenCreditLinesAndLoans": 5,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 2
        }
        predict_single(example_input)
