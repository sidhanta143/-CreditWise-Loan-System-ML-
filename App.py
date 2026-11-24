
import streamlit as st
import numpy as np
import joblib
import os



cols=['Applicant_Income', 'Coapplicant_Income', 'Age', 'Marital_Status',
       'Dependents', 'Credit_Score', 'Existing_Loans', 'DTI_Ratio', 'Savings',
       'Collateral_Value', 'Loan_Amount', 'Loan_Term', 'Gender',
        'Employment_Status_Salaried',
       'Employment_Status_Self-employed', 'Employment_Status_Unemployed',
       'Loan_Purpose_Car', 'Loan_Purpose_Education', 'Loan_Purpose_Home',
       'Loan_Purpose_Personal', 'Property_Area_Semiurban',
       'Property_Area_Urban', 'Education_Level_Not Graduate',
       'Employer_Category_Government', 'Employer_Category_MNC',
       'Employer_Category_Private', 'Employer_Category_Unemployed']


# Page configuration
st.set_page_config(
    page_title="CreditWise Loan System",
    page_icon="🏦",
    layout="centered"
)

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# App title
st.title("🏦 CreditWise Loan System")
st.write("Welcome! Enter the required details to check your loan prediction.")
st.divider()

# Input section
st.subheader("📋 Enter Loan Details")

features = []

col1, col2 = st.columns(2)

for i in range(len(cols)):
    with col1 if i % 2 == 0 else col2:
        value = st.number_input(
            f"{cols[i].replace('_', ' ')}",
            value=0.0,
            key=f"feature_{i}"
        )
        features.append(value)

st.divider()

# Prediction
if st.button(" Predict Loan Status", type="primary", use_container_width=True):

    try:
        input_data = np.array(features).reshape(1, -1)

        # Scale input data
        scaled_data = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(scaled_data)[0]

        st.subheader(" Prediction Result")

        if prediction == 1:
            st.success("Loan Status: Approved")
        else:
            st.error("Loan Status: Not Approved")

        # Prediction probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_data)[0]

            st.write("### Prediction Confidence")

            for cls, probability in zip(model.classes_, probabilities):
                st.write(f"Class {cls}: **{probability:.2%}**")

                st.progress(float(probability))

    except Exception as e:
        st.error(f"Prediction error: {e}")
