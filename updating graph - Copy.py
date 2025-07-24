import streamlit as st
import pandas as pd
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from streamlit_autorefresh import st_autorefresh

# 🖼️ Page config must be first Streamlit command
st.set_page_config(
    page_title="Crop Yield Predictor",
    layout="centered",
    initial_sidebar_state="auto"
)

# 🔁 Auto-refresh every 3 minutes (180,000 ms)
st_autorefresh(interval=180000, key="refresh")

# 🧠 Cache the dataset (with error handling)
@st.cache_data(show_spinner=False)
def load_data(path="Crop Data Set.csv"):
    if not os.path.exists(path):
        st.error(f"Data file not found: {path}")
        st.stop()
    return pd.read_csv(path)

# 🔁 Load inputs from JSON (with error handling)
@st.cache_data(show_spinner=False)
def load_inputs(path="inputs.json"):
    if not os.path.exists(path):
        st.error(f"Inputs file not found: {path}")
        st.stop()
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in {path}")
        st.stop()

# 🎯 Train model and encode labels (cached)
@st.cache_resource
def train_model(df: pd.DataFrame):
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
@st.cache_data(ttl=60, show_spinner=False)
def predict_yields(inputs: dict):
    # Out‑of‑range detection
    if any(inputs.get(col, None) is None or not (low <= inputs[col] <= high)
           for col, (low, high) in feature_bounds.items()):
        return [(crop, 0.0) for crop in crop_names], True

    preds = []
    for crop in crop_names:
        row = {col: inputs[col] for col in feature_bounds}
        row['Crop_encoded'] = int(le.transform([crop])[0])
        pred = model.predict(pd.DataFrame([row]))[0]
        preds.append((crop, round(pred, 2)))

    return sorted(preds, key=lambda x: -x[1]), False

# 🖼️ UI
st.title("🌾 Crop Yield Predictor")

# 📥 Load inputs
inputs = load_inputs()

with st.expander("📥 Current Input Values"):
    st.json(inputs)

predictions, out_of_scope = predict_yields(inputs)

if out_of_scope:
    st.warning("Some inputs are outside the training range. Yields set to 0.")

# Display bar chart
chart_data = pd.DataFrame(predictions, columns=["Crop", "Predicted Yield"])
st.bar_chart(chart_data.set_index("Crop"))

# 🔄 Manual refresh
if st.button("🔄 Refresh Now"):
    st.rerun()
