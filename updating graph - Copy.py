import streamlit as st
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Cache data and model to avoid reloading on every refresh
@st.cache_data
def load_data():
    return pd.read_csv("Crop Data Set.csv")

@st.cache_data
def load_inputs():
    with open("inputs.json", "r") as f:
        return json.load(f)

@st.cache_resource
def train_model(df):
    le = LabelEncoder()
    df['Crop_encoded'] = le.fit_transform(df['Crop'])
    X = df.drop(columns=['Crop', 'Yield'])
    y = df['Yield']
    model = RandomForestRegressor(n_estimators=50, random_state=42)  # Reduced for speed
    model.fit(X, y)
    return model, le

# Load data and train model (cached)
df = load_data()
model, le = train_model(df)
crop_names = le.classes_

# Get feature bounds
feature_bounds = {
    col: (df[col].min(), df[col].max()) 
    for col in df.columns if col not in ['Crop', 'Yield']
}

# Prediction function (cached)
@st.cache_data(ttl=60)  # Refresh every 60 seconds
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

# Streamlit UI
st.set_page_config(layout="centered")
st.title("Crop Yield Predictor 🌱")

# Auto-refresh every 3 minutes
st.auto_refresh(interval=180000)  # 180,000 ms = 3 minutes

# Display current inputs
inputs = load_inputs()
with st.expander("📥 Current Input Values"):
    st.json(inputs)

# Get predictions
predictions, out_of_scope = predict_yields(inputs)

# Show warning if out of scope
if out_of_scope:
    st.warning("Inputs outside training range! Yields set to 0.")

# Display results
st.bar_chart(
    pd.DataFrame(predictions, columns=["Crop", "Predicted Yield"])
)

# Optional: Add manual refresh button
if st.button("🔄 Refresh Now"):
    st.rerun()