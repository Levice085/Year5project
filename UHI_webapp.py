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

# -------------------- Streamlit UI -------------------- #
st.title("Urban Heat Island (UHI) Prediction 🌍🔥")
st.markdown("Upload a **GeoJSON-compatible** dataset to predict UHI values and visualize them on an interactive map.")

# File uploader
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------------------- Check if 'geometry' Column Exists -------------------- #
    if ".geo" not in df.columns:
        st.error("❌ Missing 'geometry' column in dataset! Ensure your file contains this column in GeoJSON format.")
    else:
        # -------------------- Validate GeoJSON Format -------------------- #
        def validate_geojson(geo_str):
            """Checks if the geometry column contains valid GeoJSON format."""
            try:
                geo_dict = json.loads(geo_str) if isinstance(geo_str, str) else geo_str
                if isinstance(geo_dict, dict) and "coordinates" in geo_dict:
                    return geo_dict  # Return valid GeoJSON
            except (ValueError, json.JSONDecodeError):
                pass
            return None  # Return None for invalid data

        df[".geo"] = df[".geo"].apply(validate_geojson)

        # Remove rows where GeoJSON is invalid
        df = df.dropna(subset=[".geo"])

        # -------------------- Check for Required Feature Columns -------------------- #
        feature_columns = ['EMM', 'FV', 'LST', 'NDVI', 'class']  # Adjust based on your model
        missing_cols = [col for col in feature_columns if col not in df.columns]

        if missing_cols:
            st.error(f" Missing columns in dataset: {missing_cols}")
        else:
            # Convert feature columns to float
            df[feature_columns] = df[feature_columns].astype(float)

            # Predict UHI values
            df["UHI_Prediction"] = predict_uhi(df[feature_columns].values)

            # -------------------- Display Predictions -------------------- #
            st.subheader("Sample Predictions")
            st.dataframe(df[[".geo", "UHI_Prediction"]].head())

            # -------------------- Create Interactive Folium Map -------------------- #
            st.subheader("UHI Prediction Map")
            m = folium.Map(location=[0, 0], zoom_start=2)  # Default center

            # Add data points to the map
            for _, row in df.iterrows():
                coords = row[".geo"]["coordinates"]
                if isinstance(coords, list) and len(coords) > 0:
                    lon, lat = coords[0]  # Extract first coordinate pair

                    folium.CircleMarker(
                        location=[lat, lon],
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
                label="📥 Download Predictions",
                data=df.to_csv(index=False),
                file_name="uhi_predictions.csv",
                mime="text/csv"
            )
