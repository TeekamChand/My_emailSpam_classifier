
"""
Streamlit demo app for the spam classifier saved in model.joblib
Usage:
 streamlit run app_spam_demo.py
"""
"""
📧 Email Spam Classifier (Auto-training version)
Deploy-ready for Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request, zipfile, io

st.set_page_config(page_title="Email Spam Classifier", layout="centered")

MODEL_FILE = "model.joblib"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_FILE = "data/SMSSpamCollection"

# -------------------------------
# STEP 1 — TRAIN MODEL IF MISSING
# -------------------------------
@st.cache_resource(show_spinner=True)
def train_model_if_missing():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)

    st.info("🔄 Model not found. Training a new spam classifier...")

    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_FILE):
        st.write("📥 Downloading dataset...")
        response = urllib.request.urlopen(DATA_URL)
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            z.extractall("data")

    df = pd.read_csv(DATA_FILE, sep='\t', header=None, names=['label', 'text'])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=15000)),
        ("clf", LogisticRegression(max_iter=1000, solver='liblinear'))
    ])
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = report['accuracy']
    st.success(f"✅ Model trained successfully! Accuracy: {acc:.3f}")

    joblib.dump(pipeline, MODEL_FILE)
    st.info("Model saved to model.joblib for reuse.")
    return pipeline

model = train_model_if_missing()

# -------------------------------
# STEP 2 — APP INTERFACE
# -------------------------------
st.title("📧 Email Spam Classifier")
st.markdown("This Streamlit app automatically trains a spam classifier using TF-IDF + Logistic Regression if needed. Try your own emails below!")

mode = st.sidebar.radio("Choose mode:", ["Single Email", "Batch Upload", "About"])

# -------------------------------
# SINGLE EMAIL MODE
# -------------------------------
if mode == "Single Email":
    st.subheader("🔍 Check if a single email is spam or not")
    email_text = st.text_area("Paste your email text here:", height=200)
    if st.button("Predict"):
        if not email_text.strip():
            st.warning("Please paste an email message first.")
        else:
            prob = model.predict_proba([email_text])[0, 1]
            pred = model.predict([email_text])[0]
            label = "🔴 SPAM" if pred == 1 else "🟢 HAM"
            st.markdown(f"### Prediction: **{label}**")
            st.write(f"Spam probability: `{prob:.3f}`")

# -------------------------------
# BATCH MODE
# -------------------------------
elif mode == "Batch Upload":
    st.subheader("📂 Upload CSV for batch prediction")
    uploaded = st.file_uploader("Upload CSV file with a `text` column", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            st.info("Running predictions...")
            df['spam_prob'] = model.predict_proba(df['text'].astype(str))[:, 1]
            df['predicted_label'] = np.where(df['spam_prob'] >= 0.5, 'spam', 'ham')
            st.dataframe(df.head(20))
            st.download_button(
                label="📥 Download predictions as CSV",
                data=df.to_csv(index=False),
                file_name="spam_predictions.csv",
                mime="text/csv"
            )

# -------------------------------
# ABOUT SECTION
# -------------------------------
else:
    st.subheader("ℹ️ About this project")
    st.markdown("""
    **Model:** TF-IDF + Logistic Regression  
    **Dataset:** [SMS Spam Collection (UCI)](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
    **Created for:** Demo & learning purpose  
    **Author:** You 😎  
    ---
    This app automatically trains itself if no saved model is found, 
    so you can deploy it easily on Streamlit Cloud.
    """)

    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    st.caption("Made with ❤️ using Streamlit")


