# I'am creating or generating some data like 10K but this data is fake not real.

import pandas as pd
import numpy as np
import random
import os

# Reproducibility
np.random.seed(42)
random.seed(42)

# Number of Transactions
N = 1000

# create output directory
os.makedirs("data/raw",exist_ok=True)

# 1) Basic transactions Details
transaction_ids = [
    f"TXN{str(i).zfill(6)}"
    for i in range(1,N+1)
]

sender_ids = [
    f"USER{random.randint(1,1000):04d}"
    for _ in range(N)
]

receiver_ids = [
    f"MERCHANT{random.randint(1,500):04d}"
    for _ in range(N)
]

# Generate timestamps
timestamps = pd.date_range(
    start="2026-01-01",
    end="2026-06-30",
    periods=N
)

# 2) Transaction features
amounts = np.random.lognormal(
    mean=5.2,
    sigma=1.0,
    size = N
)

amounts = np.round(amounts,2)

merchants_categories = [
    "Grocery",
    "Food",
    "Shopping",
    "Travel",
    "Utilities",
    "Entertainment",
    "Healthcare"
]

transaction_types = [
    "P2P",
    "P2M",
    "Bill Payment",
    "Recharge"
]

locations = [
    "Mumbai",
    "Pune",
    "Delhi",
    "Bangalore",
    "Hyderanbad",
    "Chennai"
]

devices_types = [
    "Android",
    "ios",
    "Web"
]

upi_channels = [
    "PhonePe",
    "Google Pay",
    "Paytm",
    "BHIM"
]

data = pd.DataFrame({

    "transaction_id":transaction_ids,
    "timestamp":timestamps,
    "sender_id":sender_ids,
    "receiver_id":receiver_ids,
    "amount":amounts,

    "merchant_category":[
        random.choice(merchants_categories)
        for _ in range(N)
    ],

    "transaction_type":[
        random.choice(transaction_types)
        for _ in range(N)
    ],

    "location":[
        random.choice(locations)
        for _ in range(N)
    ],

    "device_type":[
        random.choice(devices_types)
        for _ in range(N)
    ],

    "upi_channel":[
        random.choice(upi_channels)
        for _ in range(N)
    ]
})

# 3) Time-Based features
data['hour_of_day'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.day_name()

# 4) Fraud Pattern Simulation
data['risk_label'] = 0

# high-value transactions
high_amount = data['amount'] > 10000

# Unusual transaction hours
unusual_hour = (
    (data['hour_of_day']>=0)&
    (data['hour_of_day']<=4)
)

# Random suspicious transactions
random_fraud = np.random.random(N) < 0.03

# combine suspicious patterns
fraud_condition = (
    high_amount |
    unusual_hour |
    random_fraud
)

data.loc[fraud_condition,"risk_label"]=1

# 5) Historical spending pattern
user_average = data.groupby(
    "sender_id"
)['amount'].transform("mean")

data['historical_avg_amount'] = np.round(
    user_average,2
)

data['amount_deviation'] = np.round(
    data['amount'] - data['historical_avg_amount'],2
)

# 6) Save datasets
output_path = "data/raw/upi_transactions.csv"

data.to_csv(
    output_path,index=False
)

# 7) Display Results
print("Dataset generated successfully!")

print(f"Total transactions: {len(data)}")

print("\nDataset shape:")
print(data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nRisk label distribution:")
print(data["risk_label"].value_counts())

print(f"\nSaved to: {output_path}")