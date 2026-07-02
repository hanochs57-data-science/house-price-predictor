import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Enter the house details below to predict the price.")

# -----------------------------------
# Load Saved Objects
# -----------------------------------
@st.cache_resource
def load_files():

    with open("model.pkl", "rb") as f:
        model = joblib.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoders = joblib.load(f)

    with open("scaler_X.pkl", "rb") as f:
        scaler_X = joblib.load(f)

    with open("scaler_Y.pkl", "rb") as f:
        scaler_Y = joblib.load(f)

    return model, label_encoders, scaler_X, scaler_Y


model, label_encoders, scaler_X, scaler_Y = load_files()

# -----------------------------------
# Read Training Data
# -----------------------------------
df = pd.read_csv("data/train.csv")

# Target column (last column)
target = df.columns[-1]

# Feature columns
feature_columns = df.columns[:-1]

# -----------------------------------
# User Inputs
# -----------------------------------
st.subheader("House Details")

user_input = {}

for feature in feature_columns:

    # Categorical Feature
    if feature in label_encoders:

        encoder = label_encoders[feature]

        option = st.selectbox(
            feature,
            encoder.classes_
        )

        user_input[feature] = encoder.transform([option])[0]

    # Numerical Feature
    else:

        minimum = float(df[feature].min())
        maximum = float(df[feature].max())
        mean = float(df[feature].mean())

        value = st.number_input(
            feature,
            min_value=minimum,
            max_value=maximum,
            value=mean
        )

        user_input[feature] = value

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict House Price"):

    input_df = pd.DataFrame([user_input])

    # Keep same column order
    input_df = input_df[feature_columns]

    # Scale
    input_scaled = scaler_X.transform(input_df)

    # Predict
    prediction_scaled = model.predict(input_scaled)

    # Reverse Scaling
    prediction = scaler_Y.inverse_transform(
        np.array(prediction_scaled).reshape(-1, 1)
    )[0][0]

    st.success(f"Predicted House Price: ₹ {prediction:,.2f}")

  

# -----------------------------------
# Debug Section
# -----------------------------------
with st.expander("Debug Information"):

    st.write("Feature Columns:")
    st.write(feature_columns.tolist())

    st.write("Encoded Columns:")
    st.write(list(label_encoders.keys()))
