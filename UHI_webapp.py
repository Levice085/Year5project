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

                # -------------------- Handle Date Column -------------------- #
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                elif 'system:index' in df.columns:
                    unique_indices = df['system:index'].unique()
                    date_range = pd.date_range(start='2014-01-01', periods=len(unique_indices), freq='M')
                    index_date_map = dict(zip(unique_indices, date_range))
                    df['date'] = df['system:index'].map(index_date_map)
                else:
                    df['date'] = pd.NaT

                df.dropna(subset=['date'], inplace=True)
                df['year'] = df['date'].dt.year

                # -------------------- Display Sample Predictions -------------------- #
                st.subheader("Sample Predictions")
                st.dataframe(df[["latitude", "longitude", "UHI_Prediction", "date"]].head())

                # -------------------- Average UHI Per Year (2014-2024) -------------------- #
                st.subheader("Average UHI Per Year")
                yearly_avg = df[(df['year'] >= 2014) & (df['year'] <= 2024)].groupby('year')['UHI_Prediction'].mean().reset_index()
                st.line_chart(yearly_avg.rename(columns={"year": "index"}).set_index("index"))

                # -------------------- Year Range Selector -------------------- #
                st.subheader("Map Viewer: Select Year")
                selected_year = st.selectbox("Select Year (2014–2024)", yearly_avg['year'])

                filtered_df = df[df['year'] == selected_year]

                if filtered_df.empty:
                    st.warning(f"No data available for {selected_year}")
                else:
                    st.write(f"Displaying data for {selected_year}")

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

                # -------------------- Download Predictions -------------------- #
                st.download_button(
                    label="Download All Predictions",
                    data=df.to_csv(index=False),
                    file_name="uhi_predictions.csv",
                    mime="text/csv"
                )
