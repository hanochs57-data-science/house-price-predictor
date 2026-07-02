import joblib
from pathlib import Path
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent


def load_pickle(file_name: str):
    path = BASE_DIR / file_name
    with open(path, "rb") as f:
        return joblib.load(f)


def load_assets():
    model = load_pickle("model.pkl")
    scaler_X = load_pickle("scaler_X.pkl")
    scaler_Y = load_pickle("scaler_Y.pkl")
    le = load_pickle("label_encoder.pkl")
    return model, scaler_X, scaler_Y, le


def get_feature_names(model, scaler_X):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(scaler_X, "feature_names_in_"):
        return list(scaler_X.feature_names_in_)
    if hasattr(scaler_X, "n_features_in_"):
        return [f"feature_{i+1}" for i in range(scaler_X.n_features_in_)]
    return [f"feature_{i+1}" for i in range(5)]

st.title("Model Prediction Interface")
st.write("Load the model and scalers, enter input values, and generate a prediction.")

try:
    model, scaler_X, scaler_Y, le = load_assets()
except FileNotFoundError as exc:
    st.error(f"Missing file: {exc.filename}")
    st.stop()
except Exception as exc:
    st.error(f"Error loading assets: {exc}")
    st.stop()

feature_names = get_feature_names(model, scaler_X)
inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(feature, value=0.0, format="%f")

if st.button("Predict"):
    try:
        X = np.array([list(inputs.values())], dtype=float)
        X_scaled = scaler_X.transform(X)
        raw_pred = model.predict(X_scaled)

        try:
            pred_scaled = np.array(raw_pred).reshape(-1, 1)
            inv_pred = scaler_Y.inverse_transform(pred_scaled).flatten()
        except Exception:
            inv_pred = raw_pred

        st.subheader("Prediction results")
        st.write("Raw model output:", raw_pred.tolist())
        st.write("Inverse-scaled output:", inv_pred.tolist())

        try:
            label = le.inverse_transform(np.array(raw_pred, dtype=int))
            st.write("Decoded label:", label.tolist())
        except Exception:
            pass
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
