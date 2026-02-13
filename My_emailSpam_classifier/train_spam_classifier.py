"""
Train a spam classifier and save a pipeline to model.joblib
This script downloads the popular SMS Spam Collection dataset (if not present),
trains a TF-IDF + LogisticRegression classifier, prints evaluation metrics,
and saves the sklearn pipeline using joblib.
Usage:
 python train_spam_classifier.py
"""

import os
import urllib.request
import zipfile
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
MODEL_FILE = "model.joblib"

os.makedirs(DATA_DIR, exist_ok=True)

# Download dataset if missing
if not os.path.exists(DATA_FILE):
    print("Downloading dataset...")
    response = urllib.request.urlopen(DATA_URL)
    with zipfile.ZipFile(io.BytesIO(response.read())) as z:
        z.extractall(DATA_DIR)
    print("Downloaded and extracted dataset.")

# Load dataset
print("Loading data...")
df = pd.read_csv(DATA_FILE, sep='\t', header=None, names=['label', 'text'])

# Map labels to binary
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Quick preprocessing: drop NA and short texts
df = df.dropna().copy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['text'],df['label_num'],test_size=0.2, random_state=42, stratify=df['label_num'])

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=20000)),
    ("clf", LogisticRegression(max_iter=1000, solver='liblinear'))
])
print("Training model...")
pipeline.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred, target_names=['ham','spam']))
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC: {roc_auc:.4f}")

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham','spam'],yticklabels=['ham','spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC curve (AUC = {roc_auc:.3f})')
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.close()

# Save pipeline
joblib.dump(pipeline, MODEL_FILE)
print(f"Saved trained pipeline to {MODEL_FILE}")
print("Done. You can now run the Streamlit app: streamlit run app_spam_demo.py")

