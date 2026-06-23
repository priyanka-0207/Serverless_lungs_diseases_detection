"""
Reference Streamlit dashboard for the lung disease classifier.

Uploads a chest X-ray, calls the API Gateway /predict endpoint, and shows
the predicted class, a confidence bar chart across all three classes, and
the saliency heatmap overlaid on the original X-ray.

Set the API endpoint via the API_URL environment variable.
"""

import os
import base64

import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/predict")

st.set_page_config(page_title="Lung Disease Classifier", layout="centered")
st.title("Lung Disease Classification")
st.caption("Chest X-ray → Normal / Pneumonia / ILD. Research demo, not for clinical use.")

uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns(2)
    col1.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Run inference"):
        with st.spinner("Calling model..."):
            b64 = base64.b64encode(uploaded.getvalue()).decode()
            resp = requests.post(API_URL, json={"image": b64, "filename": uploaded.name}, timeout=60)

        if resp.status_code != 200:
            st.error(f"Request failed: {resp.status_code}")
        else:
            result = resp.json()
            st.success(f"Prediction: {result['predicted_class']}  ({result['confidence']}%)")
            st.bar_chart(result["probabilities"])
            st.caption(f"Heatmap stored at: {result['heatmap_url']}")
