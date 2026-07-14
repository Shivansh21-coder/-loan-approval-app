import streamlit as st
import joblib
import numpy as np

model = joblib.load("loan_approval_model.pkl")
sc = joblib.load("scaler.pkl")

st.title("Loan Approval Prediction")
st.write("Enter applicant details")

person_age = st.number_input("Age")
person_gender = st.number_input("Gender")
person_education = st.number_input("Education")
person_income = st.number_input("Income")
person_emp_exp = st.number_input("Employment Experience")
person_home_ownership = st.number_input("Home Ownership")
loan_amnt = st.number_input("Loan Amount")
loan_intent = st.number_input("Loan Intent")
loan_int_rate = st.number_input("Interest Rate")
loan_percent_income = st.number_input("Loan Percent Income")
cb_person_cred_hist_length = st.number_input("Credit History Length")
credit_score = st.number_input("Credit Score")
previous_loan_defaults_on_file = st.number_input("Previous Loan Default")

income_to_loan_ratio = st.number_input("Income to Loan Ratio")
monthly_income = st.number_input("Monthly Income")
estimated_emi = st.number_input("Estimated EMI")
emi_burden = st.number_input(
    "EMI Burden",
    min_value=0.0,
    step=0.001,
    format="%.3f"
)
experience_ratio = st.number_input("Experience Ratio")
credit_experience_ratio = st.number_input("Credit Experience Ratio")
credit_strength = st.number_input("Credit Strength")
income_credit_score = st.number_input("Income × Credit Score")
loan_risk_index = st.number_input("Loan Risk Index")
interest_burden = st.number_input("Interest Burden")
income_after_loan = st.number_input("Income After Loan")
income_per_credit_score = st.number_input("Income per Credit Score")
loan_interest_burden_ratio = st.number_input("Loan Interest Burden Ratio")
debt_capacity_score = st.number_input("Debt Capacity Score")
credit_reliability_score = st.number_input("Credit Reliability Score")
repayment_ability = st.number_input("Repayment Ability")

if st.button("Predict"):

    input_data = [[
        person_age,
        person_gender,
        person_education,
        person_income,
        person_emp_exp,
        person_home_ownership,
        loan_amnt,
        loan_intent,
        loan_int_rate,
        loan_percent_income,
        cb_person_cred_hist_length,
        credit_score,
        previous_loan_defaults_on_file,
        income_to_loan_ratio,
        monthly_income,
        estimated_emi,
        emi_burden,
        experience_ratio,
        credit_experience_ratio,
        credit_strength,
        income_credit_score,
        loan_risk_index,
        interest_burden,
        income_after_loan,
        income_per_credit_score,
        loan_interest_burden_ratio,
        debt_capacity_score,
        credit_reliability_score,
        repayment_ability
    ]]

    input_data = sc.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved") 
    else:
        st.error("❌ Loan Rejected")