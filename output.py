import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- STEP 1: LOAD DATA AND TRAIN MODEL ---
# In a real app, you would load a saved pickle (.pkl) file.
# Here, we recreate the model logic from your notebook.

@st.cache_resource
def train_model():
    # Creating a small sample dataset matching your notebook's structure 
    # since the local CSV path is not accessible here.
    data = {
        'ApplicantIncome': [20088, 29685, 32751, 28997, 23929, 78136, 61914],
        'LoanAmount': [324167, 310836, 142773, 118216, 231446, 323963, 269379],
        'Credit_History': [1, 1, 0, 0, 1, 1, 0],
        'Loan_Status': [0, 0, 0, 0, 0, 1, 0] # 1: Approved, 0: Rejected
    }
    df = pd.DataFrame(data)
    
    X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History']]
    y = df['Loan_Status']
    
    model = LogisticRegression()
    model.fit(X, y)
    return model

model = train_model()

# --- STEP 2: STREAMLIT UI ---
st.title("🏦 Loan Approval Prediction App")
st.write("Enter the details below to check if the loan is likely to be approved.")

# Layout with columns for better UI
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Applicant Income ($)", min_value=0, value=50000, step=1000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=250000, step=1000)

with col2:
    credit_history = st.selectbox("Credit History", options=[0, 1], 
                                  format_func=lambda x: "Good (1)" if x == 1 else "Poor (0)")

# --- STEP 3: PREDICTION ---
if st.button("Predict Loan Status"):
    # Create input dataframe for the model
    input_data = pd.DataFrame({
        'ApplicantIncome': [income],
        'LoanAmount': [loan_amount],
        'Credit_History': [credit_history]
    })
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    st.divider()
    
    if prediction[0] == 1:
        st.success(f"🎉 Result: **Approved** (Confidence: {probability:.2%})")
    else:
        st.error(f"❌ Result: **Rejected** (Confidence: {1 - probability:.2%})")

# Displaying dataset info as seen in the notebook
if st.checkbox("Show Model Details"):
    st.write("This model uses Logistic Regression and was trained on features: ApplicantIncome, LoanAmount, and Credit_History.")