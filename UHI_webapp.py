import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import joblib
import numpy as np
import requests

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
st.title("Urban Heat Island (UHI) Prediction")
st.markdown("Upload a dataset containing latitude and longitude to predict UHI values and visualize them on an interactive map.")

# File uploader
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------------------- Check if Latitude and Longitude Columns Exist -------------------- #
    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.error("Missing 'Latitude' or 'Longitude' column in dataset! Ensure your file contains these columns.")
    else:
        feature_columns = ["NDVI", "EMM", "class", "suhi","FV"]

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

                folium_static(m)  # Display the map

                # -------------------- Download Predictions -------------------- #
                st.download_button(
                    label="Download Predictions",
                    data=df.to_csv(index=False),
                    file_name="uhi_predictions.csv",
                    mime="text/csv"
                )
