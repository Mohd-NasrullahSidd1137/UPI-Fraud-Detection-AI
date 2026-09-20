import pandas as pd
import numpy as np
import os

# ==========================================
# 1. LOAD FEATURE DATASET
# ==========================================

input_path = "data/processed/upi_features.csv"

df = pd.read_csv(input_path)

print("Dataset loaded successfully!")

print("Dataset Shape:", df.shape)

# ==========================================
# 2. CHECK AMOUNT COLUMN
# ==========================================

df['amount'] = pd.to_numeric(
    df['amount'],
    errors = "coerce"
)

# Remove invalid amounts
df = df.dropna(
    subset=['amount']
)

# Keep positive transactions
df = df[df["amount"] > 0].copy()


# ==========================================
# 3. CALCULATE IQR
# ==========================================

Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1

# ==========================================
# 4. CALCULATE BOUNDS
# ==========================================

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

# ==========================================
# 5. DETECT OUTLIERS
# ==========================================

df["iqr_outlier"] = (
    (df["amount"] < lower_bound) |
    (df["amount"] > upper_bound)
).astype(int)

# ==========================================
# 6. DISPLAY IQR RESULTS
# ==========================================

outlier_count = df['iqr_outlier'].sum()

normal_count = (
    df['iqr_outlier'] == 0
).sum()

print("\n===================================")
print("IQR OUTLIER DETECTION RESULTS")
print("===================================")

print(f"Q1: {Q1:.2f}")

print(f"Q3: {Q3:.2f}")

print(f"IQR: {IQR:.2f}")

print(f"Lower Bound: {lower_bound:.2f}")

print(f"Upper Bound: {upper_bound:.2f}")

print("\nTransaction Results:")

print("Total Transactions:", len(df))

print("Normal Transactions:", normal_count)

print("Outlier Transactions:", outlier_count)


# ==========================================
# 7. DISPLAY OUTLIER TRANSACTIONS
# ==========================================

print("\nTop 10 Outlier Transactions:")

outliers = df[
    df["iqr_outlier"] == 1
].sort_values(
    "amount",
    ascending=False
)

print(
    outliers[
        [
            "transaction_id",
            "amount",
            "sender_id",
            "receiver_id",
            "iqr_outlier"
        ]
    ].head(10)
)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

output_path = "data/processed/upi_iqr_results.csv"

os.makedirs(
    "data/processed",
    exist_ok=True
)

df.to_csv(
    output_path,
    index=False
)


# ==========================================
# 9. FINAL MESSAGE
# ==========================================

print("\n===================================")
print("IQR DETECTION COMPLETED")
print("===================================")

print("Saved File:", output_path)