import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('mental_health_model.pkl')

# Title
st.title("🧠 Mental Health Treatment Predictor")

st.markdown("Predict whether a person is likely to seek treatment for mental health based on a few simple inputs.")

# User Inputs
age = st.slider("Age", 18, 100, 25)

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
gender_encoded = {"Male": 1, "Female": 0, "Other": 2}[gender]

self_employed = st.selectbox("Are you self-employed?", ["No", "Yes"])
self_employed_encoded = {"No": 0, "Yes": 1}[self_employed]

family_history = st.selectbox("Family history of mental illness?", ["No", "Yes"])
family_history_encoded = {"No": 0, "Yes": 1}[family_history]

work_interfere = st.selectbox("Mental health interferes with work?", 
                              ["Never", "Rarely", "Sometimes", "Often"])
work_interfere_encoded = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}[work_interfere]

benefits = st.selectbox("Company offers mental health benefits?", ["No", "Yes", "Don't know"])
benefits_encoded = {"No": 0, "Yes": 1, "Don't know": 2}[benefits]

# Prepare the input for prediction
features = np.array([[age, gender_encoded, self_employed_encoded, 
                      family_history_encoded, work_interfere_encoded, benefits_encoded]])

# Predict
if st.button("Predict"):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.success("✅ The person is likely to seek mental health treatment.")
    else:
        st.warning("❌ The person is unlikely to seek mental health treatment.")
