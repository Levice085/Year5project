import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import joblib
import numpy as np
import ast
import requests
import pickle

# Function to extract coordinates from GeoJSON
def extract_coordinates(geo_json):
    try:
        geo_dict = ast.literal_eval(geo_json)  # Convert string to dictionary
        coordinates = geo_dict.get("coordinates", [])
        if coordinates and isinstance(coordinates[0], list):
            return pd.Series({"Longitude": coordinates[0][0], "Latitude": coordinates[0][1]})
        elif coordinates and len(coordinates) == 2:  # If direct point
            return pd.Series({"Longitude": coordinates[0], "Latitude": coordinates[1]})
        else:
            return pd.Series({"Longitude": None, "Latitude": None})
    except:
        return pd.Series({"Longitude": None, "Latitude": None})

# Load trained model from GitHub
@st.cache_resource
def load_model():
    url = "https://github.com/Levice085/Year5project/raw/main/UHI_model.sav"
    response = requests.get(url)
    
    # Save to temporary file
    with open("UHI_model.sav", "wb") as f:
        f.write(response.content)
    
    # Load model
    with open("UHI_model.sav", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Function to predict UHI
def predict_uhi(features):
    return model.predict(features)

# Streamlit UI
st.title("Urban Heat Island (UHI) Prediction")
st.markdown("This application predicts UHI values based on environmental parameters.")

# File uploader
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Ensure '.geo' column exists
    if ".geo" in df.columns:
        df[["Longitude", "Latitude"]] = df[".geo"].apply(extract_coordinates)
    else:
        st.error("Missing '.geo' column in dataset! Ensure your file contains this column.")
    
    # Display extracted coordinates
    st.write("Extracted Coordinates:", df[["Latitude", "Longitude"]].head())

    # Check if necessary feature columns exist
    feature_columns = ['EMM', 'FV', 'LST', 'NDVI', 'class']
    missing_cols = [col for col in feature_columns if col not in df.columns]

    if missing_cols:
        st.error(f"Missing columns in dataset: {missing_cols}")
    else:
        # Convert feature columns to float
        df[feature_columns] = df[feature_columns].astype(float)

        # Predict UHI
        df["UHI_Prediction"] = model.predict(df[feature_columns].values)

        # Display results before filtering
        st.subheader("Sample Predictions (Before Filtering for Mombasa)")
        st.dataframe(df[["Latitude", "Longitude", "UHI_Prediction"]].head())

        # Save dataset for debugging
        df.to_csv("full_dataset_before_filtering.csv", index=False)
        st.write("Full dataset saved for verification.")

