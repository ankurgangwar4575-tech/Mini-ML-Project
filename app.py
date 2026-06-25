import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="CreditWise Loan Approval", page_icon="💰", layout="centered")

@st.cache_resource
def load_artifacts():
    tree_model = joblib.load('models/tree_model.pkl') 
    le_education = joblib.load('models/le_education.pkl')
    le_target = joblib.load('models/le_target.pkl')
    ohe = joblib.load('models/ohe.pkl')
    scaler = joblib.load('models/scaler.pkl')
    log_model = joblib.load('models/logistic_model.pkl')
    gnb_model = joblib.load('models/gaussian_nb_model.pkl')
    knn_model = joblib.load('models/knn_model.pkl')
    with open('models/ohe_input_cols.json', 'r') as f:
        ohe_input_cols = json.load(f)
    with open('models/feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    return le_education, le_target, ohe, scaler,tree_model, log_model, gnb_model, knn_model, ohe_input_cols, feature_columns

le_education, le_target, ohe, scaler,tree_model, log_model, gnb_model, knn_model, ohe_input_cols, feature_columns = load_artifacts()

st.title("CreditWise — Loan Approval Predictor")
st.markdown("Fill in the applicant details below to predict loan approval status.")
st.divider()

model_choice = st.selectbox(
    "Choose a model for prediction:",
    ["Decision Tree","Logistic Regression", "Gaussian Naive Bayes", "K-Nearest Neighbors"]
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income (₹/month)", min_value=2000, max_value=20000, value=10852, step=500)
    coapplicant_income = st.number_input("Coapplicant Income (₹/month)", min_value=0, max_value=10000, value=5082, step=500)
    age = st.number_input("Age", min_value=21, max_value=59, value=40)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=3, value=1)
    credit_score = st.slider("Credit Score", 550, 799, 676)
    dti_ratio = st.slider("DTI Ratio (Debt-to-Income, as decimal)", 0.10, 0.60, 0.35, step=0.01)
    existing_loans = st.number_input("Existing Loans (count)", min_value=0, max_value=4, value=2)
    savings = st.number_input("Savings (₹)", min_value=0, max_value=20000, value=9940, step=500)

with col2:
    collateral_value = st.number_input("Collateral Value (₹)", min_value=0, max_value=50000, value=24802, step=1000)
    loan_amount = st.number_input("Loan Amount Requested (₹)", min_value=1000, max_value=40000, value=20522, step=1000)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84], index=3)
    education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
    loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Car", "Business", "Home", "Education"])

property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
employer_category = st.selectbox("Employer Category", ["Private", "Government", "Unemployed", "MNC", "Business"])

st.divider()

if st.button("Predict Loan Approval", use_container_width=True):

    education_encoded = le_education.transform([education_level])[0]

    cat_input_df = pd.DataFrame([{
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Gender": gender,
        "Employer_Category": employer_category
    }])
    ohe_encoded = ohe.transform(cat_input_df[ohe_input_cols])
    ohe_encoded_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(ohe_input_cols))

    dti_ratio_sq = dti_ratio ** 2
    credit_score_sq = credit_score ** 2
    applicant_income_log = np.log1p(applicant_income)

    numeric_df = pd.DataFrame([{
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Existing_Loans": existing_loans,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": education_encoded,
        "DTI_Ratio_sq": dti_ratio_sq,
        "Credit_Score_sq": credit_score_sq,
        "Applicant_Income_log": applicant_income_log
    }])

    final_input = pd.concat([numeric_df, ohe_encoded_df], axis=1)
    final_input = final_input.reindex(columns=feature_columns, fill_value=0)
    final_input_scaled = scaler.transform(final_input)

    model_choice = st.selectbox(
    "Choose a model for prediction:",
    ["Decision Tree", "Logistic Regression", "Gaussian Naive Bayes", "K-Nearest Neighbors"]
)
    model_map = {
        "Decision Tree": tree_model,
        "Logistic Regression": log_model,
        "Gaussian Naive Bayes": gnb_model,
        "K-Nearest Neighbors": knn_model
    }
    selected_model = model_map[model_choice]

    prediction = selected_model.predict(final_input_scaled)[0]
    probability = selected_model.predict_proba(final_input_scaled)[0]
    predicted_label = le_target.inverse_transform([prediction])[0]

    st.divider()
    if str(predicted_label).strip().lower() == "yes":
        st.success(f"✅ Loan Approved! (Confidence: {max(probability)*100:.1f}%)")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {max(probability)*100:.1f}%)")

    with st.expander("See prediction probabilities"):
        st.write(f"No (Rejected) probability: {probability[0]*100:.2f}%")
        st.write(f"Yes (Approved) probability: {probability[1]*100:.2f}%")

st.divider()
st.caption("Built with Streamlit | Models: Logistic Regression, GaussianNB, KNN")