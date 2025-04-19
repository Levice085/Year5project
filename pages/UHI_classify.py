import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import requests
import joblib
from streamlit_folium import st_folium,folium_static

# Set page config
st.set_page_config(page_title="UHI Risk Classification", layout="wide")
# -------------------- Load Trained Model from GitHub -------------------- #
@st.cache_resource
def load_model():
    url = "https://github.com/Levice085/Year5project/raw/refs/heads/main/UHI_classify.sav"
    response = requests.get(url)
    
    with open("UHI_classify.sav", "wb") as f:
        f.write(response.content)
    
    return joblib.load("UHI_classify.sav")

model = load_model()

st.title("UHI Risk Classification")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    uhi = pd.read_csv(uploaded_file)

    # Display dataset preview
    st.subheader("Uploaded Data Preview")
    st.dataframe(uhi.head())

 # -------------------- Ensure Required Columns -------------------- #
    required_columns = ["latitude", "longitude", "LST", "NDVI", "EMM", "SUHI"]
    missing_cols = [col for col in required_columns if col not in uhi.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()
       
# -------------------- Function to Predict Risk -------------------- #
    def classify_risk(lst):
        if lst > 35:
            return "High Risk"
        elif 30 <= lst <= 35:
            return "Moderate Risk"
        else:
            return "Low Risk"
    # -------------------- Predict Risk Levels -------------------- #
    uhi["Risk_Level"] = uhi["LST"].apply(classify_risk)

    # Encode labels
    risk_mapping = {"Low Risk": 0, "Moderate Risk": 1, "High Risk": 2}
    uhi["Risk_Label"] = uhi["Risk_Level"].map(risk_mapping)

    # Features for prediction
    feature_columns = ["NDVI", "EMM", "class", "SUHI","FV"]
    uhi[feature_columns] = uhi[feature_columns].apply(pd.to_numeric, errors="coerce")
    uhi = uhi.dropna(subset=feature_columns)  # Remove missing values

    # Predict risk levels using the trained model
    uhi["Predicted_Risk"] = model.predict(uhi[feature_columns])
    uhi["Predicted_Risk_Label"] = uhi["Predicted_Risk"].map({v: k for k, v in risk_mapping.items()})

    # -------------------- Display Predictions -------------------- #
    st.subheader("Risk Classification Results")
    st.dataframe(uhi[["latitude", "longitude", "LST", "Predicted_Risk_Label"]])

      # -------------------- Generate Prediction Map -------------------- #
    st.subheader("UHI Risk Prediction Map")
    
    m = folium.Map(location=[uhi["latitude"].mean(), uhi["longitude"].mean()], zoom_start=12)

    # Color coding for risk levels
    color_map = {"High Risk": "red", "Moderate Risk": "orange", "Low Risk": "green"}

    for _, row in uhi.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=color_map[row["Predicted_Risk_Label"]],
            fill=True,
            fill_color=color_map[row["Predicted_Risk_Label"]],
            fill_opacity=0.7,
            popup=f"LST: {row['LST']:.2f}°C\nRisk: {row['Predicted_Risk_Label']}"
        ).add_to(m)

    # Display the Folium map
    folium_static(m)

    # -------------------- Generate Heatmap -------------------- #
    st.subheader("Heatmap of UHI Risk Areas")
    
    heatmap_data = uhi[["latitude", "longitude", "LST"]].values.tolist()
    heatmap_map = folium.Map(location=[uhi["latitude"].mean(), uhi["longitude"].mean()], zoom_start=12)
    HeatMap(heatmap_data, radius=10).add_to(heatmap_map)

    # Display heatmap
    folium_static(heatmap_map)

    # -------------------- Download Classified Data -------------------- #
    st.subheader("Download Classified Data")
    csv = uhi.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download CSV", data=csv, file_name="uhi_classification.csv", mime="text/csv")



   
        




    

