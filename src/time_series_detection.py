import pandas as pd
import numpy as np
import os


# ==========================================
# 1. LOAD DATASET
# ==========================================

input_path = (
    "data/processed/"
    "upi_isolation_forest_results.csv"
)

df = pd.read_csv(input_path)

print("Dataset loaded successfully!")

print("Dataset Shape:", df.shape)


# ==========================================
# 2. CONVERT TIMESTAMP
# ==========================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

# Remove invalid timestamps
df = df.dropna(
    subset=["timestamp"]
).copy()


# ==========================================
# 3. SORT BY TIME
# ==========================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ==========================================
# 4. CREATE HOURLY AGGREGATION
# ==========================================

# Set timestamp as index
hourly_data = df.set_index(
    "timestamp"
).resample("h").agg(
    transaction_count=(
        "transaction_id",
        "count"
    ),

    total_amount=(
        "amount",
        "sum"
    ),

    average_amount=(
        "amount",
        "mean"
    ),

    anomaly_count=(
        "isolation_forest_anomaly",
        "sum"
    )
).fillna(0)


# ==========================================
# 5. ROLLING FEATURES
# ==========================================

# Rolling average of transaction count
hourly_data["rolling_avg_count"] = (
    hourly_data["transaction_count"]
    .rolling(window=24, min_periods=1)
    .mean()
)

# Rolling standard deviation
hourly_data["rolling_std_count"] = (
    hourly_data["transaction_count"]
    .rolling(window=24, min_periods=2)
    .std()
    .fillna(0)
)


# ==========================================
# 6. CALCULATE ANOMALY THRESHOLD
# ==========================================

hourly_data["upper_bound"] = (
    hourly_data["rolling_avg_count"] +
    3 * hourly_data["rolling_std_count"]
)


# ==========================================
# 7. DETECT TIME SERIES ANOMALIES
# ==========================================

hourly_data["time_series_anomaly"] = (
    hourly_data["transaction_count"] >
    hourly_data["upper_bound"]
).astype(int)


# ==========================================
# 8. CALCULATE Z-SCORE
# ==========================================

hourly_data["count_z_score"] = np.where(
    hourly_data["rolling_std_count"] > 0,

    (
        hourly_data["transaction_count"] -
        hourly_data["rolling_avg_count"]
    ) /
    hourly_data["rolling_std_count"],

    0
)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

anomaly_count = (
    hourly_data["time_series_anomaly"]
    .sum()
)

normal_count = (
    hourly_data["time_series_anomaly"] == 0
).sum()


print("\n===================================")
print("TIME SERIES ANOMALY RESULTS")
print("===================================")

print(
    "Total Hourly Records:",
    len(hourly_data)
)

print(
    "Normal Hours:",
    normal_count
)

print(
    "Anomalous Hours:",
    anomaly_count
)


# ==========================================
# 10. DISPLAY TOP ANOMALOUS HOURS
# ==========================================

print("\nTop Anomalous Hours:")

top_anomalies = hourly_data[
    hourly_data["time_series_anomaly"] == 1
].sort_values(
    "count_z_score",
    ascending=False
)

print(
    top_anomalies[
        [
            "transaction_count",
            "total_amount",
            "rolling_avg_count",
            "upper_bound",
            "count_z_score",
            "time_series_anomaly"
        ]
    ].head(10)
)


# ==========================================
# 11. SAVE RESULTS
# ==========================================

output_path = (
    "data/processed/"
    "upi_time_series_results.csv"
)

os.makedirs(
    "data/processed",
    exist_ok=True
)

hourly_data.to_csv(
    output_path
)


# ==========================================
# 12. COMPLETION MESSAGE
# ==========================================

print("\n===================================")
print("TIME SERIES DETECTION COMPLETED")
print("===================================")

print("Saved File:", output_path)