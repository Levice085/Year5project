import streamlit as st
import ee
import geemap
import pandas as pd
import folium
from streamlit_folium import folium_static
import joblib
import numpy as np
import requests
import json  # For handling GeoJSON data
import pkg_resources

# -------------------- Initialize Google Earth Engine -------------------- #
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# -------------------- Load Trained Model from GitHub -------------------- #
@st.cache_resource
def load_model():
    url = "https://github.com/Levice085/Year5project/raw/refs/heads/main/UHI_model.sav"
    response = requests.get(url)

    with open("UHI_model.sav", "wb") as f:
        f.write(response.content)

    return joblib.load("UHI_model.sav")

model = load_model()

# -------------------- Function to Predict UHI -------------------- #
def predict_uhi(features):
    return model.predict(features)

# -------------------- Streamlit UI -------------------- #
st.title("Urban Heat Island (UHI) Mapping & Prediction")
st.markdown("This app visualizes UHI-related parameters using Google Earth Engine and predicts UHI values based on uploaded data.")

# -------------------- Define GEE Layers -------------------- #
admin = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
mvita = admin.filter(ee.Filter.eq('ADM2_NAME', 'Mvita'))
geometry = mvita.geometry()

# Define Cloud Masking Function
def cloud_mask(image):
    scored = ee.Algorithms.Landsat.simpleCloudScore(image)
    mask = scored.select(['cloud']).lte(10)
    return image.updateMask(mask)

# Filter Landsat 8 Data
landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
    .filter(ee.Filter.date('2014-01-01', '2024-10-01')) \
    .filter(ee.Filter.bounds(geometry)) \
    .map(cloud_mask)

# Compute Median Composite
median = landsat.median().clip(geometry)

# Compute NDVI
ndvi = median.normalizedDifference(['B5', 'B4']).rename('NDVI')

# Compute LST
thermal_band = median.select('B10')
lst = thermal_band.expression(
    '(Tb / (1 + (0.00115 * (Tb / 1.438)) * log(0.98))) - 273.15',
    {'Tb': thermal_band}
).rename('LST')

# Land Cover Data
land_cover = ee.ImageCollection("ESA/WorldCover/v200").first().select('Map')

# -------------------- Display GEE Map -------------------- #
Map = geemap.Map(center=[-4.05, 39.67], zoom=12)
Map.addLayer(ndvi, {'min': 0, 'max': 1, 'palette': ['white', 'green']}, "NDVI")
Map.addLayer(lst, {'min': 25, 'max': 35, 'palette': ['blue', 'red']}, "LST")
Map.addLayer(land_cover, {'palette': ['yellow', 'red', 'green']}, "Land Cover")

st.subheader("GEE-Based Map Visualization")
Map.to_streamlit(height=500)

# -------------------- Upload Data for UHI Prediction -------------------- #
st.subheader("Upload Dataset for UHI Prediction")
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if ".geo" not in df.columns:
        st.error("Missing 'geometry' column in dataset! Ensure your file contains this column in GeoJSON format.")
    else:
        # Validate GeoJSON Format
        def validate_geojson(geo_str):
            try:
                geo_dict = json.loads(geo_str) if isinstance(geo_str, str) else geo_str
                if isinstance(geo_dict, dict) and "coordinates" in geo_dict:
                    return geo_dict
            except (ValueError, json.JSONDecodeError):
                pass
            return None

        df[".geo"] = df[".geo"].apply(validate_geojson)
        df = df.dropna(subset=[".geo"])

        # Required Feature Columns
        feature_columns = ['EMM', 'FV', 'LST', 'NDVI', 'class']
        missing_cols = [col for col in feature_columns if col not in df.columns]

        if missing_cols:
            st.error(f" Missing columns in dataset: {missing_cols}")
        else:
            df[feature_columns] = df[feature_columns].astype(float)

            # Predict UHI
            df["UHI_Prediction"] = predict_uhi(df[feature_columns].values)

            st.subheader("Sample Predictions")
            st.dataframe(df[[".geo", "UHI_Prediction"]].head())

            # -------------------- Display UHI Predictions on Map -------------------- #
            st.subheader("UHI Prediction Map")
            prediction_map = folium.Map(location=[-4.05, 39.67], zoom_start=12)

            # Add Data Points to Map
            for _, row in df.iterrows():
                coords = row[".geo"]["coordinates"]
                if isinstance(coords, list) and len(coords) > 0:
                    lon, lat = coords[0]

                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=6,
                        color="red" if row["UHI_Prediction"] > np.percentile(df["UHI_Prediction"], 75) else "blue",
                        fill=True,
                        fill_color="red" if row["UHI_Prediction"] > np.percentile(df["UHI_Prediction"], 75) else "blue",
                        fill_opacity=0.6,
                        popup=f"UHI Prediction: {row['UHI_Prediction']:.2f}",
                    ).add_to(prediction_map)

            folium_static(prediction_map)

            # -------------------- Download Predictions -------------------- #
            st.download_button(
                label="Download Predictions",
                data=df.to_csv(index=False),
                file_name="uhi_predictions.csv",
                mime="text/csv"
            )