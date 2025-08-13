import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
import os

def load_data(path):
    """Load dataset from CSV file."""
    return pd.read_csv(path)

def handle_missing_values(df):
    """Fill missing values in numeric columns with median."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df

def remove_outliers(df, cols, z_thresh=3):
    """Remove rows where z-score > threshold for selected columns."""
    for col in cols:
        df = df[(np.abs(zscore(df[col])) < z_thresh)]
    return df

def scale_features(X):
    """Scale numeric features to mean=0, std=1."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split into train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def preprocess_pipeline(input_path, output_dir, target_col):
    """Full preprocessing pipeline."""
    # 1. Load data
    df = load_data(input_path)
    
    # 2. Handle missing values
    df = handle_missing_values(df)
    
    # 3. Remove outliers from key numeric columns
    df = remove_outliers(df, cols=['age', 'DebtRatio'])
    
    # 4. Split into train and test sets
    X_train, X_test, y_train, y_test = split_data(df, target_col)
    
    # 5. Scale features
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Save processed data
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    print("✅ Preprocessing complete. Files saved to:", output_dir)

if __name__ == "__main__":
    preprocess_pipeline(
        input_path="data/raw/cs-training.csv",
        output_dir="data/processed",
        target_col="SeriousDlqin2yrs"
    )
