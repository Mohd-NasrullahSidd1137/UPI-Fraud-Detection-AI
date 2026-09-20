import pandas as pd
import numpy as np
import os

# ==========================================
# 1. LOAD DATASET
# ==========================================

input_path = "data/raw/upi_transactions.csv"

df = pd.read_csv(input_path)

print("Dataset loaded successfully!")
print("Original Shape:", df.shape)

# ==========================================
# 2. TIMESTAMP CONVERSION
# ==========================================

# Your CSV column is timestamp
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

# Check invalid timestamps
invalid_dates = df["timestamp"].isna().sum()

print("Invalid timestamps:", invalid_dates)

# Remove invalid timestamps
df = df.dropna(
    subset=["timestamp"]
)

# Sort by timestamp
df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

# ==========================================
# 3. TIME-BASED FEATURES
# ==========================================

df["hour"] = df["timestamp"].dt.hour

df["day_of_week_num"] = (
    df["timestamp"].dt.dayofweek
)

df["day"] = df["timestamp"].dt.day

df["month"] = df["timestamp"].dt.month

# Weekend indicator
df["is_weekend"] = (
    df["day_of_week_num"] >= 5
).astype(int)

# Night transaction indicator
df["is_night"] = (
    (df["hour"] >= 0) &
    (df["hour"] < 6)
).astype(int)

# ==========================================
# 4. BEHAVIORAL FEATURES
# ==========================================

# Average amount per sender
df["sender_avg_amount"] = (
    df.groupby("sender_id")["amount"]
    .transform("mean")
)

# Total transactions per sender
df["sender_transaction_count"] = (
    df.groupby("sender_id")["transaction_id"]
    .transform("count")
)

# Amount compared to average
df["amount_vs_sender_avg"] = np.where(
    df["sender_avg_amount"] > 0,
    df["amount"] / df["sender_avg_amount"],
    0
)

# Unique receivers per sender
df["sender_unique_receivers"] = (
    df.groupby("sender_id")["receiver_id"]
    .transform("nunique")
)

# ==========================================
# 5. AMOUNT-BASED FEATURES
# ==========================================

df["is_high_amount"] = (
    df["amount"] > 15000
).astype(int)

# ==========================================
# 6. CLEAN DATA
# ==========================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

numeric_columns = df.select_dtypes(
    include="number"
).columns

df[numeric_columns] = (
    df[numeric_columns].fillna(0)
)

# ==========================================
# 7. SAVE DATASET
# ==========================================

output_path = "data/processed/upi_features.csv"

os.makedirs(
    "data/processed",
    exist_ok=True
)

df.to_csv(
    output_path,
    index=False
)

# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n===================================")
print("FEATURE ENGINEERING COMPLETED")
print("===================================")

print("Total Records:", len(df))

print("Total Columns:", len(df.columns))

print("Final Shape:", df.shape)

print("\nFeature Columns:")

print([
    "hour",
    "day_of_week_num",
    "day",
    "month",
    "is_weekend",
    "is_night",
    "sender_avg_amount",
    "sender_transaction_count",
    "amount_vs_sender_avg",
    "sender_unique_receivers",
    "is_high_amount"
])

print("\nFirst 5 Rows:")

print(df.head())

print("\nSaved File:")

print(output_path)