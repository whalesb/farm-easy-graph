import streamlit as st
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from streamlit_autorefresh import st_autorefresh

# 🔁 Auto-refresh every 10s
st_autorefresh(interval=10000, key="refresh")

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
crop_names = le.classes_

# ⚙️ Get bounds
feature_bounds = {
    col: (df[col].min(), df[col].max())
    for col in df.columns if col not in ['Crop', 'Yield']
}

# 🎯 Yield prediction
@st.cache_data(ttl=60)
def predict_yields(inputs):
    # Initialize messages list
    messages = []
    out_of_bounds = False
    
    # Check soil moisture conditions
    if inputs['Soil_Moisture'] > 90:
        messages.append("🌧️ Soil moisture is too high (above 90%)")
    elif inputs['Soil_Moisture'] < 10:
        messages.append("🏜️ Soil moisture is too low (below 10%)")
    
    # Check temperature conditions
    if inputs['Temperature'] > 43:
        messages.append("🔥 Temperature is too high (above 43°C)")
    elif inputs['Temperature'] < 8.82:
        messages.append("❄️ Temperature is too low (below 8.82°C)")
    

    # Return 0 yields if out of bounds
    if out_of_bounds:
        return [(crop, 0.0) for crop in crop_names], messages

    # Otherwise make predictions
    predictions = []
    for crop in crop_names:
        input_data = pd.DataFrame([{
            **{col: inputs[col] for col in feature_bounds},
            'Crop_encoded': le.transform([crop])[0]
        }])
        pred = model.predict(input_data)[0]
        predictions.append((crop, round(pred, 2)))
    
    return sorted(predictions, key=lambda x: -x[1]), messages

# 🖼️ UI
#st.set_page_config(layout="centered")
#st.title("🌾 Crop Yield Predictor")

# 📥 Inputs
inputs = load_inputs()

#with st.expander("📥 Current Input Values"):
#    st.json(inputs)

# 🔮 Predict
predictions, messages = predict_yields(inputs)

# Display warning messages
if messages:
    for msg in messages:
        st.warning(msg)

# 📊 Chart
import altair as alt

# Create DataFrame for chart
chart_df = pd.DataFrame(predictions, columns=["Crop", "Predicted Yield"])

# Create an Altair bar chart
bar_chart = alt.Chart(chart_df).mark_bar().encode(
    x=alt.X("Crop", sort='-y', title="Crop Type"),
    y=alt.Y("Predicted Yield", title="Yield (kg/ha)"),
    tooltip=["Crop", "Predicted Yield"],
    color=alt.Color("Predicted Yield", scale=alt.Scale(scheme='greens'))
).properties(
    title="📊 Predicted Crop Yields",
    width=600,
    height=400
)

# Show chart in Streamlit
st.altair_chart(bar_chart, use_container_width=True)


# 🔄 Manual refresh
if st.button("🔄 Refresh Now"):
    st.rerun()