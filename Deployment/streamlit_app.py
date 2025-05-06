import streamlit as st
import numpy as np
import pickle
from datetime import datetime

# Load trained model
with open("price_prediction.pkl", "rb") as file:
    model = pickle.load(file)

st.title("✈️ Flight Price Prediction App")
st.write("Enter flight details to predict the price.")

# Input fields
airline = st.selectbox("Airline", [
    'Air India', 'GoAir', 'IndiGo', 'Jet Airways',
    'Multiple carriers', 'SpiceJet', 'Vistara'
])

source = st.selectbox("Source", ['Chennai', 'Delhi', 'Kolkata', 'Mumbai'])
destination = st.selectbox("Destination", [
    'Cochin', 'Delhi', 'Hyderabad', 'Kolkata',
    'New Delhi', 'Banglore', 'Chennai', 'Mumbai'
])

total_stops = st.selectbox("Total Stops", ['non-stop', '1 stop', '2 stops', '3 stops'])

journey_date = st.date_input("Date of Journey", min_value=datetime.today())
dep_time = st.time_input("Departure Time")
arr_time = st.time_input("Arrival Time")

def preprocess():
    # Date/time breakdown
    journey_day = journey_date.day
    journey_month = journey_date.month
    dep_hour = dep_time.hour
    dep_min = dep_time.minute
    arr_hour = arr_time.hour
    arr_min = arr_time.minute

    # Duration calculation
    dep_dt = datetime.combine(datetime.today(), dep_time)
    arr_dt = datetime.combine(datetime.today(), arr_time)
    duration = abs((arr_dt - dep_dt).total_seconds()) / 60
    duration_hour = int(duration // 60)
    duration_min = int(duration % 60)

    # Total stops mapping
    stop_map = {
        'non-stop': 0,
        '1 stop': 1,
        '2 stops': 2,
        '3 stops': 3
    }
    stops = stop_map[total_stops]

    # One-hot encoding (matching model training columns)
    airline_ohe = [1 if airline == x else 0 for x in
                   ['Air India', 'GoAir', 'IndiGo', 'Jet Airways',
                    'Multiple carriers', 'SpiceJet', 'Vistara']]

    source_ohe = [1 if source == x else 0 for x in
                  ['Chennai', 'Delhi', 'Kolkata', 'Mumbai']]

    destination_ohe = [1 if destination == x else 0 for x in
                       ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata',
                        'New Delhi', 'Banglore', 'Chennai', 'Mumbai']]

    # Final input vector (length = 28)
    final_input = [
        stops, journey_day, journey_month,
        dep_hour, dep_min, arr_hour, arr_min,
        duration_hour, duration_min
    ] + airline_ohe + source_ohe + destination_ohe

    return np.array([final_input])

# Prediction
if st.button("Predict Price"):
    input_data = preprocess()

    if len(input_data[0]) != 28:
        st.error(f"⚠️ Feature mismatch: Got {len(input_data[0])}, expected 28.")
    else:
        prediction = model.predict(input_data)
        st.success(f"Predicted Flight Price: ₹{round(prediction[0], 2)}")
