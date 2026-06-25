# CreditWise - Loan Approval Prediction System

A machine learning web app that predicts whether a loan application will be approved or rejected, based on applicant details such as income, credit score, employment status, and property area. Four classification algorithms were trained and compared, with the pipeline deployed using Streamlit.

Live app: https://loan-classifier.streamlit.app/

## Problem Statement

Banks and NBFCs need a quick, consistent way to screen loan applications before manual underwriting. This project builds a binary classifier that takes applicant and loan details as input and predicts approval status (Yes/No) with a confidence score.

## Dataset

`loan_approval_data.csv` contains 1000 applicant records covering demographic details (age, gender, marital status, dependents, education), financial details (income, savings, existing loans, DTI ratio, credit score), loan details (amount, term, purpose, collateral value), employment details, and property area. Target: Loan_Approved (Yes/No).

Missing values were imputed using mean/median/mode. Categorical variables were encoded using Label Encoding (binary/ordinal fields) and One-Hot Encoding (nominal fields with multiple categories).

## Feature Engineering

Three derived features were added: log-transformed applicant income (`np.log1p`), squared credit score, and squared DTI ratio, to capture non-linear effects and reduce skew. The original Credit_Score and DTI_Ratio columns were dropped in favor of these.

## Models Used

Logistic Regression, Gaussian Naive Bayes, K-Nearest Neighbors, and Decision Tree were trained on the same scaled feature set (StandardScaler) and evaluated on a held-out test set.

### Model Comparison

| Model | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|
| Decision Tree | 0.92 | 0.92 | 0.92 | 0.92 |
| Logistic Regression | 0.87 | 0.87 | 0.87 | 0.87 |
| Gaussian Naive Bayes | 0.86 | 0.86 | 0.86 | 0.86 |
| K-Nearest Neighbors | 0.74 | 0.75 | 0.74 | 0.75 |

(weighted averages across both classes)

Decision Tree performed best overall and is the default model in the app, with 0.92 accuracy and 0.92 recall on the approved class — well ahead of the other three. This matters here since missing a genuinely approvable applicant is a more costly mistake than a false alarm. Logistic Regression and Naive Bayes were close to each other in the high-80s, while KNN lagged behind all three, likely due to the imbalanced class distribution affecting distance-based voting.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Streamlit, Joblib

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
│   ├── tree_model.pkl
│   ├── ohe_input_cols.json
│   └── feature_columns.json
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run Locally

```bash
git clone https://github.com/ankurgangwar4575-tech/Mini-ML-Project.git
cd Mini-ML-Project
pip install -r requirements.txt
streamlit run app.py
```

App opens at `http://localhost:8501`.

## How It Works

The user fills in applicant details through the form. Categorical inputs are transformed using the same Label Encoder and One-Hot Encoder fitted during training, and engineered features are recomputed from raw inputs. The final feature vector is reordered to match the training column order, scaled using the saved StandardScaler, and passed to the selected model, which returns a prediction and confidence score.

The notebook (`credit_wise.ipynb`) contains the full pipeline: cleaning, EDA, encoding, feature engineering, training, and evaluation. The trained models, encoders, scaler, and column order are saved with joblib at the end of the notebook, and the app loads these directly instead of retraining on every run.

## Future Improvements

- Address class imbalance (current dataset is roughly 70/30 rejected/approved) using SMOTE or class-weighted training
- Add Random Forest or XGBoost for comparison
- Tune Decision Tree depth and post-pruning (ccp_alpha) to check for overfitting on unseen data
- Add SHAP plots for model explainability

## Author

Ankur Gangwar
- LinkedIn: https://www.linkedin.com/in/ankur-gangwar/
- GitHub: https://github.com/ankurgangwar4575-tech
- Project repo: https://github.com/ankurgangwar4575-tech/Mini-ML-Project
