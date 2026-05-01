import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("🚴 Bike Demand Prediction")

temp = st.slider("Temperature", 0.0, 1.0, 0.5)
atemp = st.slider("Feels Like Temp", 0.0, 1.0, 0.5)
humidity = st.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.slider("Windspeed", 0.0, 1.0, 0.2)

season = st.selectbox("Season", [1,2,3,4])
weathersit = st.selectbox("Weather Situation", [1,2,3,4])

year = st.slider("Year", 2011, 2012, 2012)
month = st.slider("Month", 1, 12, 6)
workingday = st.selectbox("Working Day", [0,1])

# prediction
if st.button("Predict"):
    
    features = np.array([[temp, atemp, humidity, windspeed,
                          season, weathersit,
                          year, month, workingday]])

    pred_log = model.predict(features)[0]
    pred = np.expm1(pred_log)

    st.success(f"🚴 Predicted Rentals: {int(pred)}")