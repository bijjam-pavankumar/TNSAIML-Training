
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load trained model and feature names
model = joblib.load("model.joblib")
feature_names = joblib.load("feature_names.joblib")

# Application title
st.title("Welcome to House Price Prediction Application")

# Sidebar inputs
st.sidebar.header("Property Information")

SQFt = st.sidebar.number_input("SqFt", min_value=0, max_value=100000, )
BedRooms = st.sidebar.number_input("Bedrooms", min_value=0, max_value=10, )
BathRooms = st.sidebar.number_input("Bathrooms", min_value=0, max_value=10, )
Offers = st.sidebar.number_input("Offers", min_value=0, max_value=10, )
Brick=st.sidebar.selectbox("brick(optional)",[1,0])
Neighborhood=st.sidebar.selectbox("Neighborhood(optional)",[1,0])

# Create input DataFrame
input_data = {
    
    "SqFt": SQFt,
    "Bedrooms": BedRooms,
    "Bathrooms": BathRooms,
    "Offers": Offers,
    
}

input_df = pd.DataFrame([input_data])

# Match column names with the model's expected features
# Strip accidental whitespace from feature names
feature_names = [str(col).strip() for col in feature_names]

# Ensure all expected features exist
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

# Keep the same feature order used during training
input_df = input_df[feature_names]


# Display property details
st.subheader("Details of the Property")
st.dataframe(input_df, use_container_width=True,hide_index=True)

# Prediction
if st.button("Predict Price", type="primary"):
    try:
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")
        st.success(f"Estimated House Price: ₹ {prediction:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
if st.button("clear"):
    st.session_state.prediction=None
    st.rerun()
