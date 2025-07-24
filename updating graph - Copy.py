
import streamlit as st
import pandas as pd
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from streamlit_autorefresh import st_autorefresh

# 🔁 Auto-refresh every 3 minutes (180,000 ms)
st_autorefresh(interval=10000, key="refresh")

# 🧠 Cache the dataset
@st.cache_data
def load_data():
    return pd.read_csv("Crop Data Set.csv")

# 🔁 Load inputs from JSON

def load_inputs():
    with open("inputs.json", "r") as f:
        return json.load(f)

# 🎯 Train model and encode labels (cached)
@st.cache_resource
def train_model(df):
    le = LabelEncoder()
    df['Crop_encoded'] = le.fit_transform(df['Crop'])
    X = df.drop(columns=['Crop', 'Yield'])
    y = df['Yield']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, le

# Load data and model
df = load_data()
model, le = train_model(df)
crop_names = le.classes_

# Bounds of features
feature_bounds = {
    col: (df[col].min(), df[col].max()) 
    for col in df.columns if col not in ['Crop', 'Yield']
}

# 🌾 Predict yields based on inputs
@st.cache_data(ttl=60)  # Recompute every 60 seconds
def predict_yields(inputs):
    if any(not (low <= inputs[col] <= high) 
           for col, (low, high) in feature_bounds.items()):
        return [(crop, 0.0) for crop in crop_names], True

    predictions = []
    for crop in crop_names:
        input_data = pd.DataFrame([{
            **{col: inputs[col] for col in feature_bounds},
            'Crop_encoded': le.transform([crop])[0]
        }])
        predictions.append((crop, round(model.predict(input_data)[0], 2)))
    
    return sorted(predictions, key=lambda x: -x[1]), False

# 🖼️ UI setup
st.set_page_config(layout="centered")
st.title("🌾 Crop Yield Predictor")

# 📥 Load inputs
inputs = load_inputs()

# 📦 Show input JSON
with st.expander("📥 Current Input Values"):
    st.json(inputs)

# 🔮 Make predictions
predictions, out_of_scope = predict_yields(inputs)

# ⚠️ Warning for out-of-range input
if out_of_scope:
    st.warning("Some inputs are outside the training range. Yield predictions are set to 0.")

# 📊 Show predictions
st.bar_chart(pd.DataFrame(predictions, columns=["Crop", "Predicted Yield"]))

# 🔄 Manual refresh option
if st.button("🔄 Refresh Now"):
    st.rerun()
