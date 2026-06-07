import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("creditcard.csv")

# Load trained model
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Prepare features and labels
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale data
X_scaled = scaler.transform(X)

# Predict anomalies
y_pred = model.predict(X_scaled)

# Convert IsolationForest output
y_pred = [1 if x == -1 else 0 for x in y_pred]

# Print evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred))
