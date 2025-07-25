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

        # Set full monthly range from Jan of earliest year to current month
        df = df.sort_values("Timestamp")
        start = df["Timestamp"].min().replace(day=1)
        end = pd.Timestamp.now().replace(day=1)

        # Create full monthly range
        full_months = pd.date_range(start=start, end=end, freq='MS')  # MS = Month Start
        full_month_df = pd.DataFrame({"Timestamp": full_months})
        full_month_df["Month"] = full_month_df["Timestamp"].dt.strftime("%b %Y")

        # Format data for plotting
        df["Month"] = df["Timestamp"].dt.strftime("%b %Y")
        df = df.groupby("Month")["Predicted_Yield"].mean().reset_index()

        # Merge with full range to fill missing months
        chart_data = pd.merge(full_month_df, df, on="Month", how="left")

        chart = alt.Chart(chart_data).mark_line(point=True).encode(
            x=alt.X('Month:N', sort=list(chart_data["Month"]), title="Month"),
            y=alt.Y('Predicted_Yield:Q', title='Yield (kg/ha)'),
            tooltip=["Month", "Predicted_Yield"]
        ).properties(
            #title=f"📈 Yield Trend for {crop}",
            width=700,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)

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
