import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import joblib
import numpy as np
import requests
import os

# ========== Load Trained Model ==========
@st.cache_resource
def load_model():
    """Downloads and loads the trained model from GitHub."""
    model_url = "https://github.com/Levice085/Year5project/raw/main/UHI_model.sav"
    model_path = "UHI_model.sav"

    # Download model only if not already downloaded
    if not os.path.exists(model_path):
        response = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(response.content)

    # Load model
    return joblib.load(model_path)

# Load model
model = load_model()

# ========== Function to Predict UHI ==========
def predict_uhi(features):
    """Predicts UHI values using the trained model."""
    return model.predict(features)

# ========== Streamlit UI ==========
st.title("Urban Heat Island (UHI) Prediction")
st.markdown("This application predicts UHI values based on environmental parameters and visualizes them on an interactive map.")

# File uploader
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Define feature columns required for prediction
    feature_columns = ['EMM', 'FV', 'LST', 'NDVI', 'class', 'suhi']

    # Check for missing columns
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in dataset: {missing_cols}")
    else:
        # Convert feature columns to float for consistency
        df[feature_columns] = df[feature_columns].astype(float)

        # Predict UHI
        df["UHI_Prediction"] = predict_uhi(df[feature_columns].values)

        # Display sample predictions
        st.subheader("Sample Predictions")
        st.dataframe(df[["latitude", "longitude", "UHI_Prediction"]].head())

        # ========== Create Interactive Map ==========
        st.subheader("UHI Prediction Map")

        # Initialize map centered at the mean location
        m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=10)

        # Add data points to the map
        for _, row in df.iterrows():
            color = "red" if row["UHI_Prediction"] > np.percentile(df["UHI_Prediction"], 75) else "blue"
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=f"UHI Prediction: {row['UHI_Prediction']:.2f}",
            ).add_to(m)

        # Display map
        folium_static(m)

        # ========== Option to Download Predictions ==========
        st.download_button(
            label="Download Predictions",
            data=df.to_csv(index=False),
            file_name="uhi_predictions.csv",
            mime="text/csv"
        )
