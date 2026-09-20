import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ==========================================
# 1. Project Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "upi_fraud_scoring_results.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Load Dataset
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 3. Data Preparation
# ==========================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df["hour"] = df["timestamp"].dt.hour

df["fraud_risk_score"] = pd.to_numeric(
    df["fraud_risk_score"],
    errors="coerce"
)

df["amount"] = pd.to_numeric(
    df["amount"],
    errors="coerce"
)


# ==========================================
# 4. Transaction Amount Distribution
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="amount",
    bins=50,
    kde=True
)

plt.title("UPI Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "transaction_amount_distribution.png"
)

plt.show()
plt.close()


# ==========================================
# 5. Risk Category Distribution
# ==========================================

plt.figure(figsize=(8, 6))

risk_counts = df["risk_category"].value_counts()

sns.barplot(
    x=risk_counts.index,
    y=risk_counts.values
)

plt.title("Fraud Risk Category Distribution")
plt.xlabel("Risk Category")
plt.ylabel("Number of Transactions")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "risk_category_distribution.png"
)

plt.show()
plt.close()


# ==========================================
# 6. Fraud Risk Score Distribution
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="fraud_risk_score",
    bins=30,
    kde=True
)

plt.title("Fraud Risk Score Distribution")
plt.xlabel("Fraud Risk Score")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fraud_score_distribution.png"
)

plt.show()
plt.close()


# ==========================================
# 7. Anomaly Detection Comparison
# ==========================================

anomaly_columns = [
    "iqr_outlier",
    "isolation_forest_anomaly",
    "time_series_anomaly"
]

available_columns = [
    column
    for column in anomaly_columns
    if column in df.columns
]

anomaly_counts = {}

for column in available_columns:

    anomaly_counts[column] = int(
        df[column].astype(bool).sum()
    )

anomaly_series = pd.Series(anomaly_counts)


plt.figure(figsize=(10, 6))

sns.barplot(
    x=anomaly_series.index,
    y=anomaly_series.values
)

plt.title("Anomaly Detection Comparison")
plt.xlabel("Detection Method")
plt.ylabel("Number of Anomalies")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "anomaly_detection_comparison.png"
)

plt.show()
plt.close()


# ==========================================
# 8. Hourly Transaction Trend
# ==========================================

hourly_transactions = (
    df.groupby("hour")
    .size()
    .reset_index(name="transaction_count")
)


plt.figure(figsize=(12, 6))

sns.lineplot(
    data=hourly_transactions,
    x="hour",
    y="transaction_count",
    marker="o"
)

plt.title("Hourly UPI Transaction Trend")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Transactions")

plt.xticks(range(0, 24))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "hourly_transaction_trend.png"
)

plt.show()
plt.close()


# ==========================================
# 9. Average Fraud Risk Score by Hour
# ==========================================

hourly_score = (
    df.groupby("hour")["fraud_risk_score"]
    .mean()
    .reset_index()
)


plt.figure(figsize=(12, 6))

sns.lineplot(
    data=hourly_score,
    x="hour",
    y="fraud_risk_score",
    marker="o"
)

plt.title("Average Fraud Risk Score by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Fraud Risk Score")

plt.xticks(range(0, 24))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "average_fraud_score_by_hour.png"
)

plt.show()
plt.close()


# ==========================================
# 10. Top 20 Risky Transactions
# ==========================================

top_risky_transactions = (
    df.sort_values(
        by="fraud_risk_score",
        ascending=False
    )
    .head(20)
)


print("\nTop 20 Risky Transactions:")

display_columns = [
    "transaction_id",
    "amount",
    "fraud_risk_score",
    "risk_category",
    "fraud_alert"
]

available_display_columns = [
    column
    for column in display_columns
    if column in df.columns
]

print(
    top_risky_transactions[
        available_display_columns
    ].to_string(index=False)
)


top_risky_transactions.to_csv(
    OUTPUT_DIR / "top_20_risky_transactions.csv",
    index=False
)


# ==========================================
# 11. Summary Report
# ==========================================

print("\n========== VISUALIZATION SUMMARY ==========")

print(
    "Total Transactions:",
    len(df)
)

print(
    "High Risk Transactions:",
    (
        df["risk_category"] == "High Risk"
    ).sum()
)

print(
    "Medium Risk Transactions:",
    (
        df["risk_category"] == "Medium Risk"
    ).sum()
)

print(
    "Low Risk Transactions:",
    (
        df["risk_category"] == "Low Risk"
    ).sum()
)

print(
    "Average Fraud Risk Score:",
    round(
        df["fraud_risk_score"].mean(),
        2
    )
)

print(
    "Maximum Fraud Risk Score:",
    round(
        df["fraud_risk_score"].max(),
        2
    )
)

print("\nVisualizations saved at:")
print(OUTPUT_DIR)

print("\nStep 8 Completed Successfully!")