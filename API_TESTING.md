# UPI Fraud Detection API Testing

## Project

UPI Fraud Detection with AI & Anomaly Detection

## API Technology

- FastAPI
- Uvicorn
- Pandas
- Python

## API Base URL

http://127.0.0.1:8000

## Swagger Documentation

http://127.0.0.1:8000/docs

## Tested Endpoints

| Endpoint | Method | Purpose | Status |
|---|---|---|---|
| `/` | GET | API status | Tested |
| `/health` | GET | Health check | Tested |
| `/summary` | GET | Dataset summary | Tested |
| `/transactions` | GET | Get transactions | Tested |
| `/transactions/{transaction_id}` | GET | Get specific transaction | Tested |
| `/transactions/risk/high` | GET | Get high-risk transactions | Tested |
| `/transactions/risk/{risk_category}` | GET | Filter by risk category | Tested |

## Testing Objectives

1. Verify API availability.
2. Verify dataset loading.
3. Retrieve transaction records.
4. Retrieve individual transaction details.
5. Identify high-risk transactions.
6. Filter transactions by risk category.
7. Validate API error responses.

## Testing Result

API endpoints were tested using FastAPI Swagger UI.

## Note

The fraud risk score is a heuristic score based on anomaly detection
signals. It is not a verified probability of fraud.