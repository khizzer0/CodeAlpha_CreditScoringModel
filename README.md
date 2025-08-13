# CodeAlpha Credit Scoring Model

## 📌 Overview
This project is part of the **CodeAlpha Machine Learning Internship**.  
The goal is to build a **Credit Scoring Machine Learning Model** that predicts the likelihood of a customer defaulting on a loan within the next two years.  
The model uses historical credit and demographic data to make predictions.

---

## 🛠 Features
- **Data Loading & EDA**: Understand dataset structure, missing values, and feature relationships.
- **Data Preprocessing**: Handle missing values, outliers, and feature scaling.
- **Model Training**: Train an XGBoost classifier with optimized parameters.
- **Evaluation**: Generate classification reports, ROC curves, and confusion matrices.
- **Prediction Script**: Predict for single input or batch CSV file.
- **Modular Architecture**: Clean code with `src/` for data, features, models, and utils.

---

## 📂 Project Structure
CodeAlpha_CreditScoringModel/
│ README.md
│ requirements.txt
│ .gitignore
│
├───configs/ # Configuration files
├───data/
│ ├───raw/ # Original dataset (train/test CSVs)
│ ├───interim/ # Intermediate data after EDA
│ └───processed/ # Cleaned data for modeling
├───models/ # Trained model files (.pkl)
├───notebooks/ # Jupyter notebooks for EDA & experiments
├───reports/
│ └───figures/ # Plots (ROC curve, confusion matrix)
├───scripts/ # Training, prediction, and evaluation scripts
└───src/ # Python modules for data, features, and models


---

## 📊 Dataset
- **Source**: Kaggle – Give Me Some Credit dataset
- **Files**:
  - `cs-training.csv` (training data)
  - `cs-test.csv` (testing data)
  - `Data Dictionary.xls` (feature descriptions)
- **Target Column**: `SeriousDlqin2yrs`  
  - `1` = Default within 2 years  
  - `0` = No default  

---

## 🚀 Installation & Setup
### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/CodeAlpha_CreditScoringModel.git
cd CodeAlpha_CreditScoringModel

2. Create Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

📊 Results
Model Used: XGBoost Classifier

Test ROC-AUC: 0.85

Accuracy: 90%

ROC Curve

Confusion Matrix

📌 Tech Stack
Python 3.11+

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

XGBoost

Joblib

📜 License
This project is for educational purposes under the CodeAlpha Internship Program.

👨‍💻 Author
Khizzer ul Islam
BS Artificial Intelligence | CodeAlpha ML Intern
GitHub: khizzer0
