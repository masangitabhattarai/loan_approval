import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model
model_path = os.path.join("notebook", "logistic_regression_final_model.pkl")
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()
model = joblib.load(model_path)

st.title('Loan Approval Prediction')
st.write('Enter applicant details to predict loan approval')

# Inputs
Gender = st.selectbox('Gender', ['Male', 'Female'])
Married = st.selectbox('Married', ['No', 'Yes'])
Dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
Self_Employed = st.selectbox('Self Employed', ['No', 'Yes'])
ApplicantIncome = st.number_input('Applicant Income', min_value=0)
CoapplicantIncome = st.number_input('Coapplicant Income', min_value=0)
LoanAmount = st.number_input('Loan Amount', min_value=0)
Loan_Amount_Term = st.number_input('Loan Amount Term (days)', min_value=0)
Credit_History = st.selectbox('Credit History', [0.0, 1.0])
Property_Area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])

if st.button('Predict'):
    # Encoding categorical inputs (must match training encoding)
    gender_map = {'Male':1, 'Female':0}
    married_map = {'Yes':1, 'No':0}
    education_map = {'Graduate':1, 'Not Graduate':0}
    self_employed_map = {'Yes':1, 'No':0}
    property_area_map = {'Urban':2, 'Semiurban':1, 'Rural':0}
    dependents_map = {'0':0, '1':1, '2':2, '3+':3}

    data = {
        'Gender': [gender_map[Gender]],
        'Married': [married_map[Married]],
        'Dependents': [dependents_map[Dependents]],
        'Education': [education_map[Education]],
        'Self_Employed': [self_employed_map[Self_Employed]],
        'Credit_History': [Credit_History],
        'Property_Area': [property_area_map[Property_Area]],
        'LoanAmount_log': [np.log1p(LoanAmount)],
        'TotalIncome_log': [np.log1p(ApplicantIncome + CoapplicantIncome)],
        'ApplicantIncome_log': [np.log1p(ApplicantIncome)],
        'CoapplicantIncome_log': [np.log1p(CoapplicantIncome)],
        'Loan_Amount_Term_log': [np.log1p(Loan_Amount_Term)]
    }

    feature_order = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'Credit_History', 'Property_Area',
        'LoanAmount_log', 'TotalIncome_log', 'ApplicantIncome_log',
        'CoapplicantIncome_log', 'Loan_Amount_Term_log'
    ]

    input_df = pd.DataFrame(data)
    input_df = input_df[feature_order]

    pred = model.predict(input_df)
    prob = model.predict_proba(input_df) if hasattr(model, 'predict_proba') else None

    if pred[0] == 1:
        st.success('Loan is likely to be APPROVED')
    else:
        st.error(' Loan is likely to be NOT APPROVED')

    if prob is not None:
        st.write(f"Approval probability: {prob[0][1]:.2%}")



