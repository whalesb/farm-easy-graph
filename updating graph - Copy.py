import streamlit as st
import pandas as pd
import json
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Crop Data Set.csv")

# Encode Crop
le = LabelEncoder()
df['Crop_encoded'] = le.fit_transform(df['Crop'])

# Features and target
X = df.drop(columns=['Crop', 'Yield'])
y = df['Yield']

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Crop names
crop_names = le.classes_

# Get min and max bounds from training data
feature_bounds = {
    "Nitrogen": (df["Nitrogen"].min(), df["Nitrogen"].max()),
    "Phosphorus": (df["Phosphorus"].min(), df["Phosphorus"].max()),
    "Potassium": (df["Potassium"].min(), df["Potassium"].max()),
    "Temperature": (df["Temperature"].min(), df["Temperature"].max()),
    "Humidity": (df["Humidity"].min(), df["Humidity"].max()),
    "Soil_pH": (df["Soil_pH"].min(), df["Soil_pH"].max()),
    "Soil_Moisture": (df["Soil_Moisture"].min(), df["Soil_Moisture"].max())
}

# Function to check if inputs are out of scope
def is_out_of_scope(inputs):
    for feature, (low, high) in feature_bounds.items():
        if not (low <= inputs[feature] <= high):
            return True
    return False

# Function to predict yield for all crops
def predict_yield_for_all_crops(inputs):
    if is_out_of_scope(inputs):
        return [(crop, 0.0) for crop in crop_names], True  # All 0 yields, flag as out-of-scope

    predictions = []
    for crop in crop_names:
        crop_encoded = le.transform([crop])[0]
        input_data = pd.DataFrame([{
            "Nitrogen":        inputs["Nitrogen"],
            "Phosphorus":      inputs["Phosphorus"],
            "Potassium":       inputs["Potassium"],
            "Temperature":     inputs["Temperature"],
            "Humidity":        inputs["Humidity"],
            "Soil_pH":         inputs["Soil_pH"],
            "Soil_Moisture":   inputs["Soil_Moisture"],
            "Crop_encoded":    crop_encoded
        }])
        yield_pred = model.predict(input_data)[0]
        predictions.append((crop, round(yield_pred, 2)))

    return predictions, False

# Load input values from JSON file
def load_inputs_from_file():
    with open("inputs.json", "r") as f:
        return json.load(f)

# Streamlit UI
st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.title("")

# Load inputs and show them
inputs = load_inputs_from_file()
with st.expander("📥 Current Input Values"):
    st.json(inputs)

# Predict yields
yield_predictions, out_of_scope_flag = predict_yield_for_all_crops(inputs)
yield_predictions.sort(key=lambda x: x[1], reverse=True)

# Show warning if inputs are out of scope
if out_of_scope_flag:
    st.warning(
        "🚫 Input values are outside the scope of the training dataset. "
        "Yield is estimated as 0 for all crops."
    )

# Display bar chart of all predictions
st.bar_chart(
    pd.DataFrame(yield_predictions, columns=["Crop", "Predicted Yield (tons/ha)"])
      .set_index("Crop")
)

# Wait and auto-refresh
time.sleep(180)
st.rerun()
