# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🧠 Diabetes Prediction App")
st.markdown("Predict whether a patient is diabetic based on medical measurements.")

# Load sample data if available
try:
    sample_data = pd.read_csv("data_sample.csv")
    sample_available = True
except FileNotFoundError:
    sample_data = None
    sample_available = False

# Sidebar inputs
st.sidebar.header("Input Features")

def user_input_features():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)
    glucose = st.sidebar.slider("Glucose", 0, 200, 110)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 140, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 900, 79)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 32.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.sidebar.slider("Age", 10, 100, 33)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    return pd.DataFrame([data])

# Optional: Choose from sample data
if sample_available:
    st.sidebar.subheader("OR select from sample data")
    selected_index = st.sidebar.selectbox("Sample Row", sample_data.index)
    if st.sidebar.button("Use Sample Row"):
        input_df = sample_data.drop(columns=["Outcome"]).iloc[[selected_index]]
        st.success("Sample row loaded!")
    else:
        input_df = user_input_features()
else:
    input_df = user_input_features()

# Show input
st.subheader("🔢 Input Features")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0]

st.subheader("📊 Prediction Result")
st.success(f"Prediction: **{'Diabetic' if prediction == 1 else 'Not Diabetic'}**")

st.subheader("📈 Prediction Probability")
st.bar_chart(pd.DataFrame({"Probability": proba}, index=["Not Diabetic", "Diabetic"]))

# Feature importance
st.subheader("🧠 Feature Importance")
feature_imp = model.feature_importances_
features = input_df.columns

fig, ax = plt.subplots()
ax.barh(features, feature_imp)
ax.set_xlabel("Importance Score")
ax.set_title("Top Influential Features")
st.pyplot(fig)
