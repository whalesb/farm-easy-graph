import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from streamlit_autorefresh import st_autorefresh
import altair as alt

# 🔁 Auto-refresh every 4 hours (in milliseconds)
st_autorefresh(interval=14400000, key="refresh")

# 📦 Load CSV
@st.cache_data
def load_data():
    return pd.read_csv("Crop Data Set.csv")

# 📥 Load input JSON
def load_inputs():
    with open("inputs.json", "r") as f:
        return json.load(f)

# 🎓 Train model
@st.cache_resource
def train_model(df):
    le = LabelEncoder()
    df['Crop_encoded'] = le.fit_transform(df['Crop'])
    X = df.drop(columns=['Crop', 'Yield'])
    y = df['Yield']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, le

# 🚀 Load model + data
df = load_data()
model, le = train_model(df)
feature_bounds = {col: (df[col].min(), df[col].max()) for col in df.columns if col not in ['Crop', 'Yield']}

# 🎯 Predict yield for selected crop
def predict_yield_for_crop(inputs):
    crop = inputs["Crop"]
    messages = []

    # Soil moisture warnings
    if inputs['Soil_Moisture'] > 90:
        messages.append("🌧️ Soil moisture is too high (above 90%)")
    elif inputs['Soil_Moisture'] < 10:
        messages.append("🏜️ Soil moisture is too low (below 10%)")
    elif inputs['Soil_Moisture'] < 15.72:
        messages.append("🌫️ Soil moisture is a bit too low (below 15.72%)")

    # Temperature warnings
    if inputs['Temperature'] > 43:
        messages.append("🔥 Temperature is too high (above 43°C)")
    elif inputs['Temperature'] < 8.82:
        messages.append("❄️ Temperature is too low (below 8.82°C)")

    # Predict yield
    input_data = pd.DataFrame([{
        **{col: inputs[col] for col in feature_bounds},
        'Crop_encoded': le.transform([crop])[0]
    }])
    predicted_yield = round(model.predict(input_data)[0], 2)

    return crop, predicted_yield, messages

# 🗂️ Log yield to file
def log_yield(crop, predicted_yield):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df = pd.DataFrame([{
        "Timestamp": now,
        "Crop": crop,
        "Predicted_Yield": predicted_yield
    }])
    log_df.to_csv("yield_log.csv", mode='a', header=not os.path.exists("yield_log.csv"), index=False)

# 📈 Plot yield vs. month for selected crop
def plot_yield_chart(crop):
    try:
        df = pd.read_csv("yield_log.csv")
        df = df[df["Crop"] == crop]

        if df.empty:
            st.info(f"No yield records found yet for {crop}.")
            return

        # Auto-handle inconsistent date formats
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format='mixed', errors='coerce')

        # Drop invalid dates
        df = df.dropna(subset=["Timestamp"])

        # Keep daily resolution but label by month
        chart = alt.Chart(df).mark_line(interpolate='linear').encode(
            x=alt.X(
                'Timestamp:T',
                axis=alt.Axis(format='%b %Y', labelAngle=-45, title="Month")
            ),
            y=alt.Y('Predicted_Yield:Q', title='Yield (kg/ha)'),
            tooltip=[alt.Tooltip('Timestamp:T', title='Date'), 'Predicted_Yield']
        )

        points = alt.Chart(df).mark_point(size=60, filled=True).encode(
            x='Timestamp:T',
            y='Predicted_Yield:Q',
            tooltip=[alt.Tooltip('Timestamp:T', title='Date'), 'Predicted_Yield']
        )

        st.altair_chart((chart + points).properties(width=700, height=400), use_container_width=True)

    except FileNotFoundError:
        st.warning("⚠️ Yield log file not found. Run a few predictions to generate it.")

# 🖼️ UI Section
st.title("Predictive Yield Analysis")

# 📥 Inputs
inputs = load_inputs()

# 🔮 Predict
crop, predicted_yield, messages = predict_yield_for_crop(inputs)

# 💬 Show messages
# Uncomment the lines below to display environmental warning messages
# for msg in messages:
#     st.warning(msg)

# ✅ Show result
# Uncomment the line below to display the predicted yield
# st.success(f"✅ Predicted Yield for {crop}: {predicted_yield} kg/ha")

# 📁 Log yield
log_yield(crop, predicted_yield)

# 📊 Line chart: Yield vs Time
plot_yield_chart(crop)

# 🔄 Manual refresh
if st.button("🔄 Refresh Now"):
    st.rerun()
