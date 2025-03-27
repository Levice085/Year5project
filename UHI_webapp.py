import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import joblib

# Set page config
st.set_page_config(page_title="UHI Risk Classification", layout="wide")

st.title("UHI Risk Classification")

# File uploader
uploaded_file = st.file_uploader("Upload CSV or GeoJSON file", type=["csv", "geojson"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_type == "geojson":
        gdf = gpd.read_file(uploaded_file)  # Load GeoJSON as GeoDataFrame
        
        # Ensure it contains geometry
        if "geometry" not in gdf.columns:
            st.error("GeoJSON must have a 'geometry' column!")
            st.stop()

        # Convert GeoJSON geometry to latitude & longitude
        gdf["Latitude"] = gdf.geometry.y
        gdf["Longitude"] = gdf.geometry.x
        df = pd.DataFrame(gdf.drop(columns="geometry"))  # Convert to Pandas DataFrame

    # Display dataset
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Ensure required columns exist
    required_columns = ["LST", "NDVI", "EMM", "suhi",".geo"]
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing required columns: {set(required_columns) - set(df.columns)}")
        st.stop()

    # Define risk classification function
    def classify_risk(lst):
        if lst > 35:
            return "High Risk"
        elif 30 <= lst <= 35:
            return "Moderate Risk"
        else:
            return "Low Risk"

    df["Risk_Level"] = df["LST"].apply(classify_risk)

    # Encode labels
    risk_mapping = {"Low Risk": 0, "Moderate Risk": 1, "High Risk": 2}
    df["Risk_Label"] = df["Risk_Level"].map(risk_mapping)

    # Features for model training
    features = ["NDVI", "Emissivity", "SUHI"]
    X = df[features]
    y = df["Risk_Label"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Random Forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Display model accuracy
    st.subheader("📊 Model Performance")
    st.write(f"**Model Accuracy:** {accuracy:.2f}")
    st.text(classification_report(y_test, y_pred, target_names=risk_mapping.keys()))

    # Predict risk levels for full dataset
    df["Predicted_Risk"] = clf.predict(X)
    df["Predicted_Risk_Label"] = df["Predicted_Risk"].map({v: k for k, v in risk_mapping.items()})

    # Save the trained model
    joblib.dump(clf, "uhi_risk_model.pkl")

    # Display classified data
    st.subheader("🗂️ Classified Data")
    st.dataframe(df[["Latitude", "Longitude", "LST", "Predicted_Risk_Label"]])

    # Generate map with hotspots
    st.subheader("🗺️ UHI Hotspots Map")
    m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=12)

    # Color coding for risk levels
    color_map = {"High Risk": "red", "Moderate Risk": "orange", "Low Risk": "green"}

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            color=color_map[row["Predicted_Risk_Label"]],
            fill=True,
            fill_color=color_map[row["Predicted_Risk_Label"]],
            fill_opacity=0.7,
            popup=f"LST: {row['LST']:.2f}°C\nRisk: {row['Predicted_Risk_Label']}"
        ).add_to(m)

    # Display map
    st.components.v1.html(m._repr_html_(), height=600)

    # Heatmap
    st.subheader("🔥 Heatmap of UHI Risk Areas")
    heatmap_data = df[["Latitude", "Longitude", "LST"]].values.tolist()
    heatmap_map = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=12)
    HeatMap(heatmap_data, radius=10).add_to(heatmap_map)
    st.components.v1.html(heatmap_map._repr_html_(), height=600)

    # Allow downloading of classified data
    st.subheader("📥 Download Classified Data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download CSV", data=csv, file_name="uhi_classification.csv", mime="text/csv")
