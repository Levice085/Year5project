import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import joblib  # For loading the trained model
import numpy as np

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("uhi_model.sav")

model = load_model()

# Function to predict UHI
def predict_uhi(features):
    return model.predict(features)

# Streamlit UI
st.title("Urban Heat Island (UHI) Prediction")
st.markdown("This application predicts UHI values based on environmental parameters and visualizes them on an interactive map.")

# File uploader
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Assuming the dataset contains columns like NDVI, LST, etc.
    feature_columns = ['EMM', 'FV', 'LST', 'NDVI', 'class', 'suhi']  # Adjust based on your model
    if not set(feature_columns).issubset(df.columns):
        st.error(f"Dataset must contain these columns: {feature_columns}")
    else:
        df["UHI_Prediction"] = predict_uhi(df[feature_columns])
        
        # Display results
        st.subheader("Sample Predictions")
        st.dataframe(df[["latitude", "longitude", "UHI_Prediction"]].head())

        # Create an interactive Folium map
        st.subheader("UHI Prediction Map")
        m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=10)

        # Add data points to the map
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

        # Display map
        folium_static(m)

        # Option to download predictions
        st.download_button(
            label="Download Predictions",
            data=df.to_csv(index=False),
            file_name="uhi_predictions.csv",
            mime="text/csv"
        )
