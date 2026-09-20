from fastapi import FastAPI, HTTPException, Query
from pathlib import Path
import pandas as pd


# ==========================================
# 1. FastAPI Application
# ==========================================

app = FastAPI(
    title="UPI Fraud Detection API",
    description="API for UPI fraud risk analysis and anomaly detection",
    version="1.0.0"
)


# ==========================================
# 2. Load Dataset
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "upi_fraud_scoring_results.csv"
)


try:
    df = pd.read_csv(DATA_FILE)

    print("Fraud scoring dataset loaded successfully")
    print("Dataset shape:", df.shape)

except FileNotFoundError:
    df = pd.DataFrame()

    print("Dataset file not found:", DATA_FILE)


# ==========================================
# 3. Convert Data for JSON
# ==========================================

def clean_record(record):
    """
    Convert pandas values into JSON-compatible values.
    """

    cleaned_record = {}

    for key, value in record.items():

        if pd.isna(value):
            cleaned_record[key] = None

        elif hasattr(value, "item"):
            cleaned_record[key] = value.item()

        else:
            cleaned_record[key] = value

    return cleaned_record


# ==========================================
# 4. Root Endpoint
# ==========================================

@app.get("/")
def home():

    return {
        "message": "UPI Fraud Detection API is running",
        "version": "1.0.0",
        "status": "success"
    }


# ==========================================
# 5. Health Check Endpoint
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "dataset_loaded": not df.empty,
        "total_records": len(df)
    }


# ==========================================
# 6. Dataset Summary Endpoint
# ==========================================

@app.get("/summary")
def dataset_summary():

    if df.empty:
        raise HTTPException(
            status_code=500,
            detail="Dataset is not available"
        )

    summary = {
        "total_transactions": int(len(df)),
        "average_transaction_amount": round(
            float(df["amount"].mean()),
            2
        ),
        "average_fraud_risk_score": round(
            float(df["fraud_risk_score"].mean()),
            2
        ),
        "high_risk_transactions": int(
            (
                df["risk_category"] == "High Risk"
            ).sum()
        ),
        "medium_risk_transactions": int(
            (
                df["risk_category"] == "Medium Risk"
            ).sum()
        ),
        "low_risk_transactions": int(
            (
                df["risk_category"] == "Low Risk"
            ).sum()
        )
    }

    return summary


# ==========================================
# 7. Get All Transactions
# ==========================================

@app.get("/transactions")
def get_transactions(
    limit: int = Query(
        default=20,
        ge=1,
        le=500
    )
):

    if df.empty:
        raise HTTPException(
            status_code=500,
            detail="Dataset is not available"
        )

    records = df.head(limit).to_dict(
        orient="records"
    )

    cleaned_records = [
        clean_record(record)
        for record in records
    ]

    return {
        "total_returned": len(cleaned_records),
        "transactions": cleaned_records
    }


# ==========================================
# 8. Get Transaction by ID
# ==========================================

@app.get("/transactions/{transaction_id}")
def get_transaction(
    transaction_id: str
):

    if df.empty:
        raise HTTPException(
            status_code=500,
            detail="Dataset is not available"
        )

    result = df[
        df["transaction_id"].astype(str)
        == str(transaction_id)
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail="Transaction not found"
        )

    record = result.iloc[0].to_dict()

    return clean_record(record)


# ==========================================
# 9. Get High-Risk Transactions
# ==========================================

@app.get("/transactions/risk/high")
def get_high_risk_transactions(
    limit: int = Query(
        default=20,
        ge=1,
        le=500
    )
):

    if df.empty:
        raise HTTPException(
            status_code=500,
            detail="Dataset is not available"
        )

    high_risk = df[
        df["risk_category"] == "High Risk"
    ]

    high_risk = high_risk.sort_values(
        by="fraud_risk_score",
        ascending=False
    )

    records = high_risk.head(limit).to_dict(
        orient="records"
    )

    cleaned_records = [
        clean_record(record)
        for record in records
    ]

    return {
        "total_high_risk": int(len(high_risk)),
        "transactions_returned": len(
            cleaned_records
        ),
        "transactions": cleaned_records
    }


# ==========================================
# 10. Filter Transactions by Risk Category
# ==========================================

@app.get("/transactions/risk/{risk_category}")
def get_transactions_by_risk(
    risk_category: str,
    limit: int = Query(
        default=20,
        ge=1,
        le=500
    )
):

    if df.empty:
        raise HTTPException(
            status_code=500,
            detail="Dataset is not available"
        )

    valid_categories = [
        "High Risk",
        "Medium Risk",
        "Low Risk"
    ]

    matching_category = next(
        (
            category
            for category in valid_categories
            if category.lower()
            == risk_category.lower()
        ),
        None
    )

    if matching_category is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid risk category. "
                "Use High Risk, Medium Risk, "
                "or Low Risk."
            )
        )

    filtered_df = df[
        df["risk_category"] == matching_category
    ]

    records = filtered_df.head(limit).to_dict(
        orient="records"
    )

    cleaned_records = [
        clean_record(record)
        for record in records
    ]

    return {
        "risk_category": matching_category,
        "total_transactions": int(
            len(filtered_df)
        ),
        "transactions": cleaned_records
    }