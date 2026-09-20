import pandas as pd
import numpy as np
import os


# ==========================================
# 1. LOAD DATASETS
# ==========================================

isolation_path = (
    "data/processed/"
    "upi_isolation_forest_results.csv"
)

iqr_path = (
    "data/processed/"
    "upi_iqr_results.csv"
)

time_series_path = (
    "data/processed/"
    "upi_time_series_results.csv"
)


df = pd.read_csv(isolation_path)

iqr_df = pd.read_csv(iqr_path)

hourly_df = pd.read_csv(time_series_path)


print("Datasets loaded successfully!")


# ==========================================
# 2. CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "transaction_id",
    "timestamp",
    "amount",
    "isolation_forest_anomaly"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ==========================================
# 3. MERGE IQR RESULTS
# ==========================================

if "iqr_outlier" in iqr_df.columns:

    iqr_flags = iqr_df[
        [
            "transaction_id",
            "iqr_outlier"
        ]
    ].drop_duplicates(
        subset=["transaction_id"]
    )

    df = df.drop(
        columns=["iqr_outlier"],
        errors="ignore"
    )

    df = df.merge(
        iqr_flags,
        on="transaction_id",
        how="left"
    )

else:

    raise ValueError(
        "iqr_outlier column missing!"
    )


# Fill missing IQR flags
df["iqr_outlier"] = (
    df["iqr_outlier"]
    .fillna(0)
    .astype(int)
)


# ==========================================
# 4. CONVERT TIMESTAMP
# ==========================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df = df.dropna(
    subset=["timestamp"]
).copy()


# ==========================================
# 5. ADD TIME-BASED FLAGS
# ==========================================

# Create hour for matching hourly anomalies
df["hour_timestamp"] = (
    df["timestamp"].dt.floor("h")
)


# ==========================================
# 6. PREPARE HOURLY ANOMALY DATA
# ==========================================

hourly_df["timestamp"] = pd.to_datetime(
    hourly_df["timestamp"],
    errors="coerce"
)

hourly_df = hourly_df.dropna(
    subset=["timestamp"]
).copy()

hourly_df["hour_timestamp"] = (
    hourly_df["timestamp"].dt.floor("h")
)


# Keep only required columns
time_flags = hourly_df[
    [
        "hour_timestamp",
        "time_series_anomaly"
    ]
].drop_duplicates(
    subset=["hour_timestamp"]
)


# ==========================================
# 7. MERGE TIME SERIES ANOMALIES
# ==========================================

df = df.merge(
    time_flags,
    on="hour_timestamp",
    how="left"
)


df["time_series_anomaly"] = (
    df["time_series_anomaly"]
    .fillna(0)
    .astype(int)
)


# ==========================================
# 8. CREATE MISSING FEATURES IF REQUIRED
# ==========================================

# If feature engineering columns are missing,
# calculate essential flags.

df["hour"] = df["timestamp"].dt.hour

if "is_night" not in df.columns:

    df["is_night"] = (
        (df["hour"] >= 0) &
        (df["hour"] < 6)
    ).astype(int)


if "is_high_amount" not in df.columns:

    df["is_high_amount"] = (
        df["amount"] > 15000
    ).astype(int)


# ==========================================
# 9. CALCULATE FRAUD RISK SCORE
# ==========================================

df["fraud_risk_score"] = 0


# IQR outlier contribution
df["fraud_risk_score"] += (
    df["iqr_outlier"] * 25
)


# Isolation Forest contribution
df["fraud_risk_score"] += (
    df["isolation_forest_anomaly"] * 35
)


# Night transaction contribution
df["fraud_risk_score"] += (
    df["is_night"] * 10
)


# High amount contribution
df["fraud_risk_score"] += (
    df["is_high_amount"] * 15
)


# Time series contribution
df["fraud_risk_score"] += (
    df["time_series_anomaly"] * 15
)


# Limit score to 100
df["fraud_risk_score"] = (
    df["fraud_risk_score"].clip(0, 100)
)


# ==========================================
# 10. CREATE RISK CATEGORIES
# ==========================================

def assign_risk_category(score):

    if score >= 70:

        return "High Risk"

    elif score >= 40:

        return "Medium Risk"

    else:

        return "Low Risk"


df["risk_category"] = (
    df["fraud_risk_score"]
    .apply(assign_risk_category)
)


# ==========================================
# 11. CREATE ALERT LOGIC
# ==========================================

df["fraud_alert"] = np.where(
    df["fraud_risk_score"] >= 70,
    "Review Required",
    "No Alert"
)


# ==========================================
# 12. DISPLAY RESULTS
# ==========================================

print("\n===================================")
print("FRAUD SCORING RESULTS")
print("===================================")


print(
    "Total Transactions:",
    len(df)
)


print("\nRisk Category Distribution:")

print(
    df["risk_category"]
    .value_counts()
)


print("\nFraud Alert Distribution:")

print(
    df["fraud_alert"]
    .value_counts()
)


# ==========================================
# 13. TOP RISK TRANSACTIONS
# ==========================================

print("\nTop 10 High-Risk Transactions:")

top_risk = df.sort_values(
    "fraud_risk_score",
    ascending=False
)


print(
    top_risk[
        [
            "transaction_id",
            "amount",
            "fraud_risk_score",
            "risk_category",
            "fraud_alert"
        ]
    ].head(10)
)


# ==========================================
# 14. SAVE FINAL DATASET
# ==========================================

output_path = (
    "data/processed/"
    "upi_fraud_scoring_results.csv"
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
# 15. COMPLETION MESSAGE
# ==========================================

print("\n===================================")
print("FRAUD SCORING COMPLETED")
print("===================================")


print(
    "Saved File:",
    output_path
)