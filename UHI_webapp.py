import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import geopandas as gpd  # For reading Shapefiles
import joblib
import numpy as np
import requests
import json

# -------------------- Load Trained Model from GitHub -------------------- #
@st.cache_resource
def load_model():
    url = "https://github.com/Levice085/Year5project/raw/refs/heads/main/UHI_model.sav"
    response = requests.get(url)
    
    if response.status_code != 200:
        st.error("Failed to download the model. Check the URL or internet connection.")
        st.stop()
    
    with open("UHI_model.sav", "wb") as f:
        f.write(response.content)
    
    return joblib.load("UHI_model.sav")

model = load_model()

# -------------------- Load County Boundaries (Shapefile) -------------------- #
@st.cache_resource
def load_shapefile():
    """ Load county boundaries from a local Shapefile and convert to GeoJSON. """
    shapefile_path = "C:/Users/levie/OneDrive/Desktop/Year 5/5.2/Carto map design/ken_adm_iebc_20191031_shp/ken_admbnda_adm2_iebc_20191031.shp"
    gdf = gpd.read_file(shapefile_path)  # Read the Shapefile
      # Convert all Timestamp columns to strings to avoid serialization errors
    for col in gdf.columns:
        if gdf[col].dtype == "datetime64[ms]":
            gdf[col] = gdf[col].astype(str)

    return json.loads(gdf.to_json())  # Convert to GeoJSON

# Load county boundaries
county_geojson = load_shapefile()

# -------------------- Function to Predict UHI -------------------- #
def predict_uhi(features):
    return model.predict(features)

# -------------------- Streamlit UI -------------------- #
st.title("Urban Heat Island (UHI) Prediction")
st.markdown("Upload a dataset containing latitude and longitude to predict UHI values and visualize them on an interactive map.")

# File uploader
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------------------- Check if Latitude and Longitude Columns Exist -------------------- #
    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.error("Missing 'latitude' or 'longitude' column in dataset! Ensure your file contains these columns.")
    else:
        feature_columns = ['EMM', 'FV', 'LST', 'NDVI', 'class']

        # Ensure feature columns exist and contain valid numbers
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in dataset: {missing_cols}")
        else:
            df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
            df = df.dropna(subset=feature_columns)  # Drop rows with missing values

            if df.empty:
                st.error("No valid data available for prediction after cleaning.")
            else:
                # Predict UHI values
                df["UHI_Prediction"] = predict_uhi(df[feature_columns].values)
                
                # -------------------- Display Predictions -------------------- #
                st.subheader("Sample Predictions")
                st.dataframe(df[["latitude", "longitude", "UHI_Prediction"]].head())

                # -------------------- Create Interactive Folium Map -------------------- #
                st.subheader("UHI Prediction Map")
                m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=10)

                # Add county boundaries to the map
                folium.GeoJson(
                    county_geojson,
                    name="Kenya Counties",
                    style_function=lambda feature: {
                        "fillColor": "green",
                        "color": "black",
                        "weight": 1,
                        "fillOpacity": 0.1,
                    },
                    tooltip=folium.GeoJsonTooltip(fields=["ADM2_EN"], aliases=["Sub County: "])
                ).add_to(m)

                # Add UHI prediction points to the map
                for _, row in df.iterrows():
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=6,
                        color="red" if row["UHI_Prediction"] > np.percentile(df["UHI_Prediction"], 75) else "blue",
                        fill=True,
                        fill_color="red" if row["UHI_Prediction"] > np.percentile(df["UHI_Prediction"], 75) else "blue",
                        fill_opacity=0.6,
                        popup=f"UHI Prediction: {row['UHI_Prediction']:.2f}",
                    ).add_to(m)

                folium.LayerControl().add_to(m)  # Add layer control for toggling
                folium_static(m)  # Display the map

                # -------------------- Download Predictions -------------------- #
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predictions",
                    data=csv_data,
                    file_name="uhi_predictions.csv",
                    mime="text/csv"
                )
