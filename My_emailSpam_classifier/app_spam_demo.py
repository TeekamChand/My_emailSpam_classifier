"""
Streamlit demo app for the spam classifier saved in model.joblib


Usage:
streamlit run app_spam_demo.py


"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import base64

MODEL_FILE = "model.joblib"


st.set_page_config(page_title="Email Spam Classifier", layout="centered")

# Header
st.title("📧 Attractive Email Spam Classifier")
st.markdown("A simple, demonstrable spam classifier — perfect for presentations. Trained with TF-IDF + Logistic Regression.")

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"Could not load model file '{MODEL_FILE}'. Run training script first: `python train_spam_classifier.py`.\nError: {e}")
        return None


model = load_model()

# Sidebar -- quick controls
st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode:", ["Single email", "Batch upload", "Model info & metrics"])

if mode == "Single email":
    st.subheader("Predict a single email")
    email_text = st.text_area("Paste the email text here", height=200)
    if st.button("Predict"):
        if not model:
            st.error("Model not loaded.")
        elif not email_text.strip():
            st.warning("Please paste some email text.")
        else:
            prob = model.predict_proba([email_text])[0,1]
            pred = model.predict([email_text])[0]
            label = "🔴 SPAM" if pred==1 else "🟢 HAM"
            st.markdown(f"### Prediction: **{label}**")
            st.markdown(f"**Spam probability:** {prob:.3f}")

elif mode == "Batch upload":
    st.subheader("Upload CSV with a `text` column")
    uploaded = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            with st.spinner("Predicting..."):
                probs = model.predict_proba(df['text'].astype(str).tolist())[:,1]
                preds = model.predict(df['text'].astype(str).tolist())
                df['spam_prob'] = probs
                df['predicted_label'] = np.where(preds==1, 'spam', 'ham')
            st.success("Predictions complete")
            st.dataframe(df.head(50))
            # Provide download link
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(f"[Download predictions](data:file/csv;base64,{b64})")

else:
    st.subheader("Model info & metrics")
    st.markdown("This page shows evaluation results produced during training (if available). If you trained here, files `confusion_matrix.png` and `roc_curve.png` should exist.")
    col1, col2 = st.columns(2)
    try:
        cm_img = 'confusion_matrix.png'
        roc_img = 'roc_curve.png'
        col1.image(cm_img, caption='Confusion matrix', use_column_width=True)
        col2.image(roc_img, caption='ROC curve', use_column_width=True)
    except Exception:
        st.info("No metric images found. Run `python train_spam_classifier.py` first to generate them.")

# Footer
st.markdown("---")
st.markdown("Made for presentations. Model is meant for demonstration and fine-tuning is recommended before production use.")