# CreditWise - Loan Approval Prediction System

A machine learning web application that predicts whether a loan application will be approved or rejected, based on applicant details such as income, credit score, employment status, and property area. The project compares three classification algorithms and deploys the best-performing pipeline using Streamlit.

Live app: [add your Streamlit Cloud link here]

## Problem Statement

Banks and NBFCs need a quick, consistent way to screen loan applications before manual underwriting. This project builds a binary classifier that takes applicant and loan details as input and predicts approval status (Yes/No), along with a confidence score.

## Dataset

The dataset (`loan_approval_data.csv`) contains 1000 applicant records with the following information:

- Demographic details: Age, Gender, Marital Status, Dependents, Education Level
- Financial details: Applicant Income, Coapplicant Income, Savings, Existing Loans, DTI Ratio, Credit Score
- Loan details: Loan Amount, Loan Term, Loan Purpose, Collateral Value
- Employment details: Employment Status, Employer Category
- Property details: Property Area
- Target variable: Loan_Approved (Yes/No)

Missing values were handled using mean/median/mode imputation depending on the column. Categorical variables were encoded using Label Encoding (for binary/ordinal fields) and One-Hot Encoding (for nominal fields with more than two categories).

## Feature Engineering

A few derived features were created to improve model performance:

- Applicant_Income_log: log-transformed applicant income (np.log1p) to reduce skew
- Credit_Score_sq: squared credit score to capture non-linear effects
- DTI_Ratio_sq: squared debt-to-income ratio for the same reason

The original Credit_Score and DTI_Ratio columns were dropped in favor of their transformed versions.

## Models Used

Three classification algorithms were trained and compared:

1. Logistic Regression
2. Gaussian Naive Bayes
3. K-Nearest Neighbors (KNN)

All models were trained on the same scaled feature set (StandardScaler) and evaluated on a held-out test set using precision, recall, F1-score, and accuracy.

### Model Comparison

| Model | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Gaussian Naive Bayes | TBD | TBD | TBD | TBD |
| K-Nearest Neighbors | TBD | TBD | TBD | TBD |

[Add one or two lines here on which model performed best and why you'd pick it for this use case - e.g., if recall on the approved class matters more than raw accuracy]

## Tech Stack

- Python
- Pandas, NumPy for data handling
- Scikit-learn for preprocessing, modeling, and evaluation
- Streamlit for the web interface
- Joblib for model persistence

## Project Structure

```
credit-wise-loan-approval/
├── data/
│   └── loan_approval_data.csv
├── notebooks/
│   └── credit_wise.ipynb
├── models/
│   ├── le_education.pkl
│   ├── le_target.pkl
│   ├── ohe.pkl
│   ├── scaler.pkl
│   ├── logistic_model.pkl
│   ├── gaussian_nb_model.pkl
│   ├── knn_model.pkl
│   ├── ohe_input_cols.json
│   └── feature_columns.json
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run Locally

```bash
git clone https://github.com/yourusername/credit-wise-loan-approval.git
cd credit-wise-loan-approval
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## How It Works

1. The user fills in applicant details through the Streamlit form.
2. Categorical inputs are transformed using the same Label Encoder and One-Hot Encoder fitted during training.
3. Engineered features (log income, squared credit score, squared DTI ratio) are recomputed from raw inputs.
4. The final feature vector is reordered to match the training column order and scaled using the saved StandardScaler.
5. The selected model predicts the approval status and returns a probability score.

## Notes on Deployment

The notebook (`credit_wise.ipynb`) contains the full pipeline: data cleaning, EDA, encoding, feature engineering, training, and evaluation. At the end of the notebook, the trained models, encoders, scaler, and column order are serialized using joblib and saved to the `models/` folder. The Streamlit app loads these saved artifacts directly rather than retraining on every run.

## Future Improvements

- Address class imbalance (current dataset is roughly 70/30 rejected/approved) using techniques like SMOTE or class-weighted training
- Add more models such as Random Forest or XGBoost for comparison
- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Add SHAP or feature importance plots for model explainability
- Containerize with Docker for more portable deployment

## Author

[Ankur Gangwar]
[LinkedIn URL]
[GitHub URL]