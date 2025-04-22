import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt

# -------------------- Load Trained Model -------------------- #
@st.cache_resource
def load_model():
    url = "https://github.com/Levice085/Year5project/raw/refs/heads/main/UHI_model.sav"
    response = requests.get(url)
    with open("UHI_model.sav", "wb") as f:
        f.write(response.content)
    return joblib.load("UHI_model.sav")

model = load_model()

# -------------------- Predict Function -------------------- #
def predict_uhi(features):
    input_array = np.array([features])
    return model.predict(input_array)[0]

# -------------------- App Title -------------------- #
st.title("Urban Heat Island (UHI) Prediction")
st.markdown("Predict UHI from manual input or upload a dataset for batch prediction.")

# -------------------- Session State for History -------------------- #
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- Manual Prediction Form -------------------- #
st.header("Manual Input")

with st.form("manual_form"):
    emm = st.slider("EMM (Emissivity)", 0.0, 1.0, 0.97, 0.01)
    fv = st.slider("FV (Fractional Vegetation)", 0.0, 1.0, 0.3, 0.01)
    lst = st.slider("LST (Land Surface Temperature °C)", 10.0, 60.0, 30.0, 0.1)
    ndvi = st.slider("NDVI", -1.0, 1.0, 0.4, 0.01)
    land_class = st.slider("Class (Land Cover Class)", 0, 10, 1, 1)

    submitted = st.form_submit_button("Predict UHI")

    if submitted:
        features = [emm, fv, lst, ndvi, land_class]
        prediction = predict_uhi(features)
        st.session_state.history.append({
            "EMM": emm, "FV": fv, "LST": lst,
            "NDVI": ndvi, "class": land_class,
            "UHI_Prediction": round(prediction, 2)
        })
        st.success(f"Predicted UHI: **{prediction:.2f}**")

# -------------------- History Table and Chart -------------------- #
if st.session_state.history:
    st.subheader("Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

    st.line_chart(history_df["UHI_Prediction"])


