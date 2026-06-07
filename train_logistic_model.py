import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("creditcard.csv")

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Save model + scaler
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "logistic_scaler.pkl")

# Get probabilities
y_probs = model.predict_proba(X_test_scaled)[:, 1]

thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]

for threshold in thresholds:
    print("\n==============================")
    print(f"Threshold: {threshold}")

    y_pred = (y_probs > threshold).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
