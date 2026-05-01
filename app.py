import streamlit as st
import numpy as np
import joblib
import os

# -------------------------------
# Load trained model safely
# -------------------------------
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Bike Demand Predictor", layout="centered")

st.title("🚴 Bike Sharing Demand Prediction")
st.write("Predict daily bike rentals based on weather and calendar conditions")

# -------------------------------
# User Inputs
# -------------------------------
temp = st.slider("Temperature (Normalized)", 0.0, 1.0, 0.5)
atemp = st.slider("Feels Like Temperature", 0.0, 1.0, 0.5)
humidity = st.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.slider("Windspeed", 0.0, 1.0, 0.2)

season = st.selectbox("Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)", [1, 2, 3, 4])
weathersit = st.selectbox("Weather Situation (1=Clear, 2=Mist, 3=Light Snow/Rain, 4=Heavy Rain)", [1, 2, 3, 4])

year = st.slider("Year", 2011, 2012, 2012)
month = st.slider("Month", 1, 12, 6)
workingday = st.selectbox("Working Day (0=No, 1=Yes)", [0, 1])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Demand"):

    features = np.array([[temp, atemp, humidity, windspeed,
                          season, weathersit,
                          year, month, workingday]])

    prediction_log = model.predict(features)[0]
    prediction = np.expm1(prediction_log)

    st.success(f"🚴 Predicted Bike Rentals: {int(prediction)}")