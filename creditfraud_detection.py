# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:06:03 2025

@author: donyf
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)

from imblearn.over_sampling import SMOTE
import joblib

# ---------- CREATE FOLDERS ----------
os.makedirs("images", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ---------- STEP 1: LOAD DATA ----------
print("Loading dataset...")

# Use your correct CSV path here
df = pd.read_csv(r"C:\Users\donyf\Downloads\archive (27)\creditcard.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nClass distribution (0 = non-fraud, 1 = fraud):")
print(df["Class"].value_counts())

# ---------- STEP 2: BASIC EDA ----------
print("\nSummary statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

# Plot class imbalance
plt.figure()
sns.countplot(x="Class", data=df)
plt.title("Fraud (1) vs Non-Fraud (0) Transactions")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/class_distribution.png")
plt.show()

# ---------- STEP 3: PREPROCESSING ----------
# Copy to avoid modifying original
data = df.copy()

# Scale 'Time' and 'Amount'
scaler = StandardScaler()
data[["Time", "Amount"]] = scaler.fit_transform(data[["Time", "Amount"]])

# Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

print("\nBefore SMOTE class distribution:")
print(y.value_counts())

# Handle imbalance with SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

print("\nAfter SMOTE class distribution:")
print(y_resampled.value_counts())

# ---------- STEP 4: TRAIN-TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# ---------- STEP 5: TRAIN MODEL ----------
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model training completed.")

# ---------- STEP 6: EVALUATE MODEL ----------
print("\nEvaluating model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
plt.show()

# ROC-AUC Score
auc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC Score:", auc)

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig("images/roc_curve.png")
plt.show()

# ---------- STEP 7: SAVE MODEL & SCALER ----------
joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel and scaler saved in 'model/' folder.")
print("Images saved in 'images/' folder.")
print("\nPROJECT RUN COMPLETE.")
