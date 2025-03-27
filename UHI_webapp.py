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
st.title("Urban Heat Island (UHI) Prediction & Mapping")
st.markdown("Upload a **GeoJSON or CSV dataset** to predict UHI values and visualize them on an interactive map.")

# File uploader
uploaded_file = st.file_uploader(" Upload dataset (CSV or GeoJSON)", type=["csv", "geojson"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "csv":
        uhi = pd.read_csv(uploaded_file)
    elif file_type == "geojson":
        gdf = pd.read_json(uploaded_file)  # Load as JSON
        uhi = pd.DataFrame(gdf["features"].apply(lambda x: x["properties"]))  # Extract properties

        # Extract coordinates
        uhi["Longitude"] = gdf["features"].apply(lambda x: x[".geo"]["coordinates"][0])
        uhi["Latitude"] = gdf["features"].apply(lambda x: x[".geo"]["coordinates"][1])

    # -------------------- Check for Required Feature Columns -------------------- #
    feature_columns = ["EMM", "FV", "LST", "NDVI", "class"]  # Adjust based on your model
    missing_cols = [col for col in feature_columns if col not in uhi.columns]

    if missing_cols:
        st.error(f"Missing columns in dataset: {missing_cols}")
    else:
        # Convert feature columns to float
        uhi[feature_columns] = uhi[feature_columns].astype(float)

        # Predict UHI values
        uhi["UHI_Prediction"] = predict_uhi(uhi[feature_columns].values)

        # -------------------- Display Predictions -------------------- #
        st.subheader("Sample Predictions")
        st.dataframe(uhi[["Latitude", "Longitude", "UHI_Prediction"]].head())

        # -------------------- Create Interactive Folium Map -------------------- #
        st.subheader(" UHI Prediction Map")

        # Center map on dataset
        map_center = [uhi["Latitude"].mean(), uhi["Longitude"].mean()]
        m = folium.Map(location=map_center, zoom_start=12)

        # Define color mapping based on percentiles
        q25, q50, q75 = np.percentile(uhi["UHI_Prediction"], [25, 50, 75])
        
        def get_color(val):
            if val <= q25:
                return "green"  # Low UHI
            elif q25 < val <= q50:
                return "blue"  # Moderate UHI
            elif q50 < val <= q75:
                return "orange"  # High UHI
            else:
                return "red"  # Very High UHI

        # Add points to the map
        for _, row in uhi.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=6,
                color=get_color(row["UHI_Prediction"]),
                fill=True,
                fill_color=get_color(row["UHI_Prediction"]),
                fill_opacity=0.7,
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
