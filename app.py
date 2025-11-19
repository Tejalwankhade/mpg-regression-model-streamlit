# streamlit_autompg_app.py
# Streamlit app to predict MPG using a pre-trained model saved as 'Autompg_model.pkl'
# Save this file and run: streamlit run streamlit_autompg_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path

st.set_page_config(page_title="Auto MPG Predictor", layout="centered")

MODEL_PATH = "Autompg_model.pkl"  # update if your pickle file is in another path

st.title("🚗 Auto MPG Predictor")
st.write("Enter car attributes to predict miles per gallon (MPG) using the pre-trained model.")

# Sidebar: model load status and tips
st.sidebar.header("Model / App Info")
st.sidebar.write("Model file expected: **Autompg_model.pkl**")
st.sidebar.write("Make sure the pickle contains a scikit-learn Pipeline or model that accepts a dataframe with columns:\n`['cylinders','displacement','horsepower','weight','acceleration','model year','origin']`")

# Attempt to load model
@st.cache_resource
def load_model(path: str):
    p = Path(path)
    if not p.exists():
        return None, f"Model file not found at: {path}"
    try:
        # try joblib first
        model = joblib.load(path)
        return model, None
    except Exception:
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            return model, None
        except Exception as e:
            return None, f"Failed to load model: {e}"

model, err = load_model(MODEL_PATH)
if err:
    st.sidebar.error(err)
else:
    st.sidebar.success("Model loaded successfully")

# Default input values (rough medians from Auto MPG dataset)
defaults = {
    'cylinders': 4,
    'displacement': 120.0,
    'horsepower': 95.0,
    'weight': 2800,
    'acceleration': 15.0,
    'model year': 78,
    'origin': 1
}

st.header("Input car features")
col1, col2 = st.columns(2)
with col1:
    cylinders = st.number_input("Cylinders", min_value=2, max_value=16, value=defaults['cylinders'], step=1)
    displacement = st.number_input("Displacement", min_value=50.0, max_value=500.0, value=defaults['displacement'], format="%.1f")
    horsepower = st.number_input("Horsepower", min_value=10.0, max_value=500.0, value=defaults['horsepower'], format="%.1f")
with col2:
    weight = st.number_input("Weight", min_value=1000, max_value=6000, value=defaults['weight'], step=10)
    acceleration = st.number_input("Acceleration", min_value=5.0, max_value=40.0, value=defaults['acceleration'], format="%.1f")
    model_year = st.number_input("Model Year", min_value=70, max_value=99, value=defaults['model year'], step=1)

origin = st.selectbox("Origin (1=USA, 2=Europe, 3=Asia)", options=[1,2,3], index=0)

# Build input DataFrame matching training columns
input_df = pd.DataFrame([{ 
    'cylinders': cylinders,
    'displacement': displacement,
    'horsepower': horsepower,
    'weight': weight,
    'acceleration': acceleration,
    'model year': model_year,
    'origin': origin
}])

st.subheader("Input preview")
st.dataframe(input_df)

# Prediction
if st.button("Predict MPG"):
    if model is None:
        st.error("Model not loaded. Please place 'Autompg_model.pkl' in the app directory.")
    else:
        try:
            # If model is a pipeline that expects a dataframe, pass as-is
            pred = model.predict(input_df)
            # If prediction returns array-like
            mpg_pred = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
            st.success(f"Predicted MPG: {mpg_pred:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Tip: Ensure the saved model accepts a pandas DataFrame with the specified column names. If your model expects scaled/numpy features, consider saving a Pipeline during training that includes preprocessing.")

st.markdown("---")

# Additional utilities
st.header("Utilities")

with st.expander("Upload a CSV to preview predictions (optional)"):
    uploaded = st.file_uploader("Upload CSV with feature columns", type=["csv"])
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            st.write("Preview of uploaded file:")
            st.dataframe(df_upload.head())

            # check required columns
            required_cols = ['cylinders','displacement','horsepower','weight','acceleration','model year','origin']
            missing = [c for c in required_cols if c not in df_upload.columns]
            if missing:
                st.warning(f"Uploaded file is missing columns: {missing}")
            else:
                if st.button("Predict for uploaded CSV"):
                    try:
                        preds = model.predict(df_upload[required_cols])
                        df_upload['predicted_mpg'] = preds
                        st.write(df_upload.head())
                        st.success("Predictions appended as 'predicted_mpg' column")
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

st.write("\n---\nMade with ❤️ — modify the code to match your saved model's preprocessing and column names.")
