import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("models/landslide_model.pkl")

st.title("🌍 Suraksha.AI – Flood & Landslide Risk Predictor")

st.write("Enter environmental parameters to assess risk:")

slope = st.number_input("Slope (degrees)", min_value=0.0, max_value=90.0, step=0.1)
ndvi = st.number_input("NDVI (vegetation index)", min_value=-1.0, max_value=1.0, step=0.01)
rain_3d = st.number_input("Rainfall in last 3 days (mm)", min_value=0.0, step=1.0)

if st.button("Predict Risk"):
    user_data = np.array([[slope, ndvi, rain_3d]])
    prob = model.predict_proba(user_data)[0][1]
    risk = model.predict(user_data)[0]

    risk_label = "⚠️ High Risk" if risk == 1 else "✅ Low Risk"
    st.subheader(f"Prediction: {risk_label}")
    st.write(f"**Probability of Landslide:** {prob*100:.2f}%")
