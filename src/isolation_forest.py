import pandas as pd
import numpy as np
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD DATASET
# ==========================================

input_path = "data/processed/upi_iqr_results.csv"

df = pd.read_csv(input_path)

print("Dataset loaded successfully!")

print("Dataset Shape:", df.shape)

# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "amount",
    "hour",
    "day_of_week_num",
    "is_weekend",
    "is_night",
    "sender_avg_amount",
    "sender_transaction_count",
    "amount_vs_sender_avg",
    "sender_unique_receivers",
    "is_high_amount"
]

# Check missing features
missing_features = [
    col for col in features
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing Features:{missing_features}"
    )

# ==========================================
# 3. PREPARE DATA
# ==========================================

X = df[features].copy()

# Replace infinite values
X = X.replace(
    [np.inf,-np.inf],np.nan
)

# Fill missing values
X = X.fillna(0)


# ==========================================
# 4. SCALE FEATURES
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 5. TRAIN ISOLATION FOREST
# ==========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled)

# ==========================================
# 6. PREDICT ANOMALIES
# ==========================================

predictions = model.predict(X_scaled)

# Isolation Forest:
# 1 = Normal
# -1 = Anomaly
df["isolation_forest_prediction"] = predictions

df["isolation_forest_anomaly"] = (
    predictions == -1
).astype(int)

# ==========================================
# 7. ANOMALY SCORE
# ==========================================

df["isolation_forest_score"] = (
    model.decision_function(X_scaled)
)

# Lower score = More anomalous
# Higher score = More normal


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

anomaly_count = (
    df["isolation_forest_anomaly"] == 1
).sum()

normal_count = (
    df["isolation_forest_anomaly"] == 0
).sum()


print("\n===================================")
print("ISOLATION FOREST RESULTS")
print("===================================")

print("Total Transactions:", len(df))

print("Normal Transactions:", normal_count)

print("Anomalies Detected:", anomaly_count)


# ==========================================
# 9. DISPLAY TOP ANOMALIES
# ==========================================

print("\nTop 10 Anomalous Transactions:")

top_anomalies = df[
    df["isolation_forest_anomaly"] == 1
].sort_values(
    "isolation_forest_score",
    ascending=True
)

print(
    top_anomalies[
        [
            "transaction_id",
            "amount",
            "sender_id",
            "isolation_forest_score",
            "isolation_forest_anomaly"
        ]
    ].head(10)
)


# ==========================================
# 10. SAVE RESULTS
# ==========================================

output_path = (
    "data/processed/"
    "upi_isolation_forest_results.csv"
)

os.makedirs(
    "data/processed",
    exist_ok=True
)

df.to_csv(
    output_path,
    index=False
)


# ==========================================
# 11. COMPLETION MESSAGE
# ==========================================

print("\n===================================")
print("ISOLATION FOREST COMPLETED")
print("===================================")

print("Saved File:", output_path)