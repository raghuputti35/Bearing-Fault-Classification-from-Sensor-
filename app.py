"""Bearing Fault Prediction - Streamlit UI only (no training here).
Run  python train_model.py  first to create the saved models."""
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = "models/bearing_models.joblib"

FAULT_DICT = {
  "Ball_007_1": "Ball bearing fault with a 0.007-inch defect",
  "Ball_014_1": "Ball bearing fault with a 0.014-inch defect",
  "Ball_021_1": "Ball bearing fault with a 0.021-inch defect",

  "IR_007_1": "Inner part of the bearing has a 0.007-inch defect",
  "IR_014_1": "Inner part of the bearing has a 0.014-inch defect",
  "IR_021_1": "Inner part of the bearing has a 0.021-inch defect",

  "Normal_1": "Bearing is working normally with no detected fault",

  "OR_007_6_1": "Outer part of the bearing has a 0.007-inch defect",
  "OR_014_6_1": "Outer part of the bearing has a 0.014-inch defect",
  "OR_021_6_1": "Outer part of the bearing has a 0.021-inch defect"
}

st.set_page_config(page_title="Bearing Fault Prediction", page_icon="⚙️")
st.title("⚙️ Bearing Fault Prediction")

# ---------------------------------------------------------------- load saved models
if not os.path.exists(MODEL_FILE):
    st.error("No trained model found. Run `python train_model.py` first, then refresh this page.")
    st.stop()


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_FILE)


bundle = load_bundle()
models = bundle["models"]
features = bundle["features"]
defaults = bundle["defaults"]

model_name = st.sidebar.selectbox("Model", list(models.keys()))
st.sidebar.caption(f"Test accuracy: {bundle['accuracies'][model_name]:.2%}")
model = models[model_name]

mode = st.radio("How do you want to give input?", ["Enter values manually", "Upload a CSV file"],
                horizontal=True)

# ---------------------------------------------------------------- manual input
if mode == "Enter values manually":
    st.write("Enter the sensor values:")
    values = {}
    cols = st.columns(3)
    for i, name in enumerate(features):
        values[name] = cols[i % 3].number_input(name, value=float(defaults[name]), format="%.5f")

    if st.button("Predict", type="primary"):
        prediction = model.predict(pd.DataFrame([values]))[0]
        st.success(f"Predicted fault type: **{prediction}**")
        st.success(f"**{FAULT_DICT[prediction]}**")

# ---------------------------------------------------------------- CSV upload
else:
    st.write("Upload a CSV with these columns: " + ", ".join(features))
    file = st.file_uploader("CSV file", type="csv")

    if file is not None:
        data = pd.read_csv(file)
        missing = [c for c in features if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            data = data.dropna(subset=features)
            result = data.copy()
            predictions = model.predict(data[features])
            result["Predicted_Fault"] = predictions
            result["fault_nl"] = pd.Series(predictions).map(FAULT_DICT).values
            st.write(f"Predictions for {len(result)} rows:")
            st.dataframe(result)
            st.download_button("Download predictions", result.to_csv(index=False),
                               "predictions.csv", "text/csv")
