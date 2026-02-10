import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("final_random_forest_model.pkl")

# App title and description
st.title("Income Prediction Application")
st.write(
    "This application predicts whether an individual earns more than $50,000 per year "
    "based on demographic and work-related information."
)

st.divider()

# Input options
marital_statuses = [
    'Married-civ-spouse',
    'Never-married',
    'Divorced',
    'Separated',
    'Widowed'
]

# User inputs
age = st.slider("Age", min_value=18, max_value=90, value=30)
education_num = st.slider("Education Level (Numeric)", min_value=1, max_value=16, value=10)
hours_per_week = st.slider("Hours Worked Per Week", min_value=1, max_value=100, value=40)

capital_gain = st.number_input(
    "Capital Gain",
    min_value=0,
    max_value=100000,
    value=0
)

capital_loss = st.number_input(
    "Capital Loss",
    min_value=0,
    max_value=100000,
    value=0
)

marital_status_selected = st.selectbox(
    "Marital Status",
    marital_statuses
)

# Prediction
if st.button("Predict Income"):

    if age <= 0 or hours_per_week <= 0:
        st.error("Please enter valid values for age and working hours.")
    else:
        # Prepare input data
        df_input = pd.DataFrame({
            "age": [age],
            "educational-num": [education_num],
            "hours-per-week": [hours_per_week],
            "capital-gain": [np.log1p(capital_gain)],
            "capital-loss": [np.log1p(capital_loss)],
            "marital-status_Married-civ-spouse": [
                1 if marital_status_selected == "Married-civ-spouse" else 0
            ]
        })

        # Align features with training data
        df_input = df_input.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        # Generate prediction
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Predicted Income: More than $50K")
        else:
            st.info("Predicted Income: $50K or less")

        st.write(f"Probability of earning more than $50K: {probability:.2f}")

# Page styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    </style>
    """,
    unsafe_allow_html=True
)
