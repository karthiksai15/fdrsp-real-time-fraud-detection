from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import time

# -----------------------------------
# Load Trained Behavioral Model
# -----------------------------------
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------------
# IsolationForest calibration values
# ⚠️ IMPORTANT:
# Replace these with actual values from training if available.
# Temporary safe defaults added.
# -----------------------------------
MIN_SCORE = -0.25
MAX_SCORE = 0.20

# -----------------------------------
# Business Risk Threshold
# -----------------------------------
RISK_THRESHOLD = 0.25

app = FastAPI(title="ML Fraud Detection Service")

# -----------------------------------
# Request Schema
# -----------------------------------
class TransactionFeatures(BaseModel):
    amount: float
    round_off: float
    transactions_last_1hr: int
    transactions_last_24hr: int
    avg_amount_7d: float
    amount_deviation: float
    hour_of_day: int
    is_new_user: int


# -----------------------------------
# Health Endpoint
# -----------------------------------
@app.get("/")
def health():
    return {"status": "ML Fraud Service Running"}


# -----------------------------------
# Prediction Endpoint
# -----------------------------------
@app.post("/predict")
def predict(features: TransactionFeatures):

    start_time = time.time()

    # -----------------------------------
    # Convert request to model input
    # -----------------------------------
    input_data = np.array([[
        features.amount,
        features.round_off,
        features.transactions_last_1hr,
        features.transactions_last_24hr,
        features.avg_amount_7d,
        features.amount_deviation,
        features.hour_of_day,
        features.is_new_user
    ]])

    # -----------------------------------
    # Scale features
    # -----------------------------------
    scaled_data = scaler.transform(input_data)

    # -----------------------------------
    # Isolation Forest raw anomaly score
    # -----------------------------------
    raw_score = model.decision_function(scaled_data)[0]

    # -----------------------------------
    # Normalize score based on training distribution
    # lower score = more anomalous
    # -----------------------------------
    normalized_score = (raw_score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)

    # invert → higher = more risky
    ml_risk_score = 1 - normalized_score

    # clamp 0–1
    ml_risk_score = max(0.0, min(1.0, float(ml_risk_score)))

    # -----------------------------------
    # Hybrid Risk Layer (REAL FINTECH APPROACH)
    # Combine ML + financial magnitude intelligence
    # -----------------------------------
    amount_risk = min(features.amount / 5000000, 1.0)

    final_risk_score = (0.7 * ml_risk_score) + (0.3 * amount_risk)

    # -----------------------------------
    # Business anomaly decision
    # -----------------------------------
    is_anomaly = final_risk_score > RISK_THRESHOLD

    end_time = time.time()
    latency_ms = round((end_time - start_time) * 1000, 3)

    # -----------------------------------
    # Response
    # -----------------------------------
    return {
        "is_anomaly": is_anomaly,
        "ml_risk_score": round(ml_risk_score, 4),
        "amount_risk": round(amount_risk, 4),
        "final_risk_score": round(final_risk_score, 4),
        "latency_ms": latency_ms,
        "threshold": RISK_THRESHOLD
    }