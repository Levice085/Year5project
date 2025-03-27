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

# -------------------- Streamlit UI -------------------- #
st.title("Urban Heat Island (UHI) Prediction")
st.markdown("Upload a **GeoJSON-compatible** dataset to predict UHI values and visualize them on an interactive map.")

# File uploader
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

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

# Drop rows with missing values in features
        df = df.dropna(subset=feature_columns)

        # -------------------- Check for Required Feature Columns -------------------- #
        
        missing_cols = [col for col in feature_columns if col not in df.columns]

        if missing_cols:
            st.error(f"Missing columns in dataset: {missing_cols}")
        else:
            # Convert feature columns to float, handling errors
            df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")

             # Drop rows with missing values in feature columns
        df = df.dropna(subset=feature_columns)

            # Ensure the DataFrame is not empty before making predictions
        if df.empty:
         st.error("No valid data available for prediction after cleaning.")
        else:
        # Predict UHI values
            df["UHI_Prediction"] = predict_uhi(df[feature_columns].values)

         
            # -------------------- Display Predictions -------------------- #
        st.subheader("Sample Predictions")
        st.dataframe(df[["Latitude", "Longitude", "UHI_Prediction"].head()])

            # -------------------- Create Interactive Folium Map -------------------- #
        st.subheader("UHI Prediction Map")
        m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=10)

            # Add data points to the map
        for _, row in df.iterrows():
         folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=6,
            color="red" if row["UHI_Prediction"] > np.percentile(df["UHI_Prediction"], 75) else "blue",
            fill=True,
            fill_color="red" if row["UHI_Prediction"] > np.percentile(df["UHI_Prediction"], 75) else "blue",
            fill_opacity=0.6,
            popup=f"UHI Prediction: {row['UHI_Prediction']:.2f}",
            ).add_to(m)

            # Display the map
        folium_static(m)

            # -------------------- Download Predictions -------------------- #
        st.download_button(
            label="Download Predictions",
            data=df.to_csv(index=False),
            file_name="uhi_predictions.csv",
            mime="text/csv"
            )
