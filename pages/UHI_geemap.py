import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import joblib
import numpy as np
import requests
from datetime import datetime

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
st.markdown("Upload a dataset containing latitude and longitude to predict UHI values and visualize them over time.")

# File uploader
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------------------- Extract date from 'system:index' -------------------- #
    if 'system:index' in df.columns:
        df['date'] = pd.to_datetime(
            df['system:index'].str.extract(r'(\d{8})')[0],
            format='%Y%m%d',
            errors='coerce'
        )

    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.error("Missing 'latitude' or 'longitude' column in dataset!")
    else:
        feature_columns = ['EMM', 'FV', 'LST', 'NDVI', 'class']
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in dataset: {missing_cols}")
        else:
            df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
            df = df.dropna(subset=feature_columns)

            if df.empty:
                st.error("No valid data available for prediction after cleaning.")
            else:
                df["UHI_Prediction"] = predict_uhi(df[feature_columns].values)

                # -------------------- Display Sample Predictions -------------------- #
                st.subheader("Sample Predictions")
                st.dataframe(df[["latitude", "longitude", "UHI_Prediction", "date"]].head())

                # -------------------- Date Filter (Slider) -------------------- #
                st.subheader("Time Series Viewer")
                available_dates = df['date'].dropna().sort_values().unique()

                if len(available_dates) > 0:
                    selected_date = st.slider(
                        "Select Date", 
                        min_value=available_dates[0],
                        max_value=available_dates[-1],
                        value=available_dates[0],
                        format="YYYY-MM-DD"
                    )

                    filtered_df = df[df['date'] == selected_date]

                    if filtered_df.empty:
                        st.warning(f"No data available for {selected_date}")
                    else:
                        st.write(f"Displaying data for {selected_date.strftime('%Y-%m-%d')}")

                        m = folium.Map(location=[filtered_df["latitude"].mean(), filtered_df["longitude"].mean()], zoom_start=10)

                        for _, row in filtered_df.iterrows():
                            folium.CircleMarker(
                                location=[row["latitude"], row["longitude"]],
                                radius=6,
                                color="red" if row["UHI_Prediction"] > np.percentile(filtered_df["UHI_Prediction"], 75) else "blue",
                                fill=True,
                                fill_color="red" if row["UHI_Prediction"] > np.percentile(filtered_df["UHI_Prediction"], 75) else "blue",
                                fill_opacity=0.6,
                                popup=f"UHI: {row['UHI_Prediction']:.2f}",
                            ).add_to(m)

                        folium_static(m)
                else:
                    st.warning("No valid dates found in dataset.")

                # -------------------- Download Predictions -------------------- #
                st.download_button(
                    label="Download All Predictions",
                    data=df.to_csv(index=False),
                    file_name="uhi_predictions.csv",
                    mime="text/csv"
                )
