import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from streamlit_folium import st_folium

# Set page config
st.set_page_config(page_title="UHI Risk Classification", layout="wide")

st.title("UHI Risk Classification")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Display dataset preview
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Ensure required columns exist
    required_columns = ["latitude", "longitude", "LST", "NDVI", "EMM", "suhi"]
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
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
    features = ["NDVI", "EMM", "suhi"]
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
    st.subheader("Model Performance")
    st.write(f"**Model Accuracy:** {accuracy:.2f}")
    st.text(classification_report(y_test, y_pred))

    # Predict risk levels for full dataset
    df["Predicted_Risk"] = clf.predict(X)
    df["Predicted_Risk_Label"] = df["Predicted_Risk"].map({v: k for k, v in risk_mapping.items()})

    # Save the trained model
    joblib.dump(clf, "uhi_risk_model.pkl")

    # Display classified data
    st.subheader("Classified Data")
    st.dataframe(df[["latitude", "longitude", "LST", "Predicted_Risk_Label"]])

    # Generate map with hotspots
    st.subheader("UHI Hotspots Map")
    m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=12)

    # Color coding for risk levels
    color_map = {"High Risk": "red", "Moderate Risk": "orange", "Low Risk": "green"}

    st.subheader("UHI Hotspots Map")

# Ensure correct column names
df.rename(columns={"Latitude": "latitude", "Longitude": "longitude"}, inplace=True)

# Drop rows with missing lat/lon
df = df.dropna(subset=["latitude", "longitude"])

# Generate the map
m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=12)

# Define color mapping
color_map = {"High Risk": "red", "Moderate Risk": "orange", "Low Risk": "green"}

# Add markers
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=6,
        color=color_map.get(row["Predicted_Risk_Label"], "blue"),
        fill=True,
        fill_color=color_map.get(row["Predicted_Risk_Label"], "blue"),
        fill_opacity=0.6,
        popup=f"Risk: {row['Predicted_Risk_Label']}",
    ).add_to(m)

# Display the map
st_folium(m)

