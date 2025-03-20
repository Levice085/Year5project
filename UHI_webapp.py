import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import joblib
import requests
import ast
import numpy as np

# Load trained model from GitHub
@st.cache_resource
def load_model():
    url = "https://github.com/Levice085/Year5project/raw/main/UHI_model.sav"
    response = requests.get(url)
    with open("UHI_model.sav", "wb") as f:
        f.write(response.content)
    return joblib.load("UHI_model.sav")

model = load_model()

# Function to extract latitude & longitude from GeoJSON format
def extract_coordinates(geo_json):
    try:
        geo_dict = ast.literal_eval(geo_json)  # Convert string to dictionary
        coordinates = geo_dict.get("coordinates", [None, None])  # Extract coordinates
        return pd.Series({"Longitude": coordinates[0], "Latitude": coordinates[1]})
    except:
        return pd.Series({"Longitude": None, "Latitude": None})

# Streamlit UI
st.title("Urban Heat Island (UHI) Prediction for Mombasa")
st.markdown("Upload a dataset to predict and visualize UHI levels in Mombasa.")

# File uploader
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Ensure dataset contains the `.geo` column
    if ".geo" not in df.columns:
        st.error("The dataset must contain a `.geo` column with coordinates.")
    else:
        # Extract latitude and longitude
        df[["Longitude", "Latitude"]] = df[".geo"].apply(extract_coordinates)

        # Drop rows with missing coordinates
        df.dropna(subset=["Longitude", "Latitude"], inplace=True)

        # Filter data for Mombasa (adjust coordinates range if needed)
        mombasa_bounds = {
            "min_lat": -4.1,
            "max_lat": -3.8,
            "min_lon": 39.5,
            "max_lon": 39.8
        }
        df = df[
            (df["Latitude"] >= mombasa_bounds["min_lat"]) & 
            (df["Latitude"] <= mombasa_bounds["max_lat"]) & 
            (df["Longitude"] >= mombasa_bounds["min_lon"]) & 
            (df["Longitude"] <= mombasa_bounds["max_lon"])
        ]

        if df.empty:
            st.error("No data points found in Mombasa. Check dataset coordinates.")
        else:
            # Feature columns for prediction
            feature_columns = ['EMM', 'FV', 'LST', 'NDVI', 'class', 'suhi']
            missing_cols = [col for col in feature_columns if col not in df.columns]

            if missing_cols:
                st.error(f"Dataset is missing required columns: {missing_cols}")
            else:
                df["UHI_Prediction"] = model.predict(df[feature_columns])

                # Display sample predictions
                st.subheader("Sample Predictions")
                st.dataframe(df[["Latitude", "Longitude", "UHI_Prediction"]].head())

                # Create an interactive Folium map centered on Mombasa
                st.subheader("UHI Prediction Map")
                m = folium.Map(location=[-4.05, 39.67], zoom_start=12)

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

                # Option to download predictions
                st.download_button(
                    label="Download Predictions",
                    data=df.to_csv(index=False),
                    file_name="uhi_predictions_mombasa.csv",
                    mime="text/csv"
                )
