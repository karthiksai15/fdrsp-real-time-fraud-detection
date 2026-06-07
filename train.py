import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# ----------------------------
# 1️⃣ Load Dataset
# ----------------------------
df = pd.read_csv("creditcard.csv")  # Make sure this file exists in the same folder

df = df.sort_values("Time")

# ----------------------------
# 2️⃣ Feature Engineering
# ----------------------------
df["amount"] = df["Amount"]

df["round_off"] = np.ceil(df["amount"]) - df["amount"]

df["hour_of_day"] = (df["Time"] // 3600) % 24

# Rolling windows (simple simulation)
df["transactions_last_1hr"] = df["Time"].rolling(window=50).count()
df["transactions_last_24hr"] = df["Time"].rolling(window=200).count()

df["avg_amount_7d"] = df["amount"].rolling(window=200).mean()

df = df.bfill()

df["amount_deviation"] = df["amount"] / (df["avg_amount_7d"] + 1e-5)

df["is_new_user"] = 0
df.loc[:100, "is_new_user"] = 1

# ----------------------------
# 3️⃣ Select Final Features
# ----------------------------
features = [
    "amount",
    "round_off",
    "transactions_last_1hr",
    "transactions_last_24hr",
    "avg_amount_7d",
    "amount_deviation",
    "hour_of_day",
    "is_new_user"
]

X = df[features]

# ----------------------------
# 4️⃣ Normalize
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 5️⃣ Train Isolation Forest
# ----------------------------
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

model.fit(X_scaled)

# ----------------------------
# 6️⃣ Save Model
# ----------------------------
joblib.dump(model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model training complete.")
print("📦 Saved fraud_model.pkl and scaler.pkl")
