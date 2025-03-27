import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import joblib
import numpy as np
import requests
import json  # For handling GeoJSON data

# -------------------- Load Trained Model from GitHub -------------------- #
@st.cache_resource
def load_model():
    url = "https://github.com/Levice085/Year5project/raw/refs/heads/main/UHI_model.sav"
    response = requests.get(url)

    # Save the model locally
    with open("UHI_model.sav", "wb") as f:
        f.write(response.content)

    # Load and return the model
    return joblib.load("UHI_model.sav")

model = load_model()

# -------------------- Function to Predict UHI -------------------- #
def predict_uhi(features):
    return model.predict(features)

# -------------------- Function to Extract Coordinates -------------------- #
def extract_coordinates(geo_json):
    try:
        geo_dict = json.loads(geo_json) if isinstance(geo_json, str) else geo_json
        if isinstance(geo_dict, dict) and "coordinates" in geo_dict and geo_dict["coordinates"]:
            lon, lat = geo_dict["coordinates"][0]  # Extract first coordinate pair
            return lat, lon
    except (ValueError, json.JSONDecodeError, IndexError):
        pass
    return None, None
   # -------------------- Check if 'geometry' Column Exists -------------------- #
    if ".geo" not in df.columns:
        st.error("Missing '.geo' column in dataset! Ensure your file contains this column in GeoJSON format.")
    else:
        # Extract latitude and longitude
        df["Latitude"], df["Longitude"] = zip(*df[".geo"].apply(extract_coordinates))
        df = df.dropna(subset=["Latitude", "Longitude"])  # Remove rows with missingcoordinates
        feature_columns = ['EMM', 'FV', 'LST', 'NDVI', 'class']  # Adjust based on your model

        # Ensure feature columns exist and contain valid numbers
        df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")