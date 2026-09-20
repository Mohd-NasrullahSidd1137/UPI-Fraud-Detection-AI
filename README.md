# 🚨 UPI Fraud Detection with AI & Anomaly Detection

An AI-powered UPI fraud detection project that identifies potentially suspicious transactions using statistical outlier detection, machine learning, time-series analysis, and fraud risk scoring.

The project includes a data processing pipeline, anomaly detection models, a FastAPI backend, and an interactive Streamlit dashboard.

## 🔗 Live Demo

🚀 **Streamlit Dashboard:**  
https://upi-fraud-detection-ai-1137.streamlit.app/

---

## 📌 Project Overview

Digital payment platforms such as UPI process a large number of transactions every day. Detecting suspicious transactions is important for reducing financial risks and improving transaction security.

This project analyzes UPI transaction data and assigns a fraud risk score based on multiple anomaly detection signals.

The system combines:

- Statistical outlier detection
- Machine learning-based anomaly detection
- Time-series anomaly analysis
- Rule-based fraud risk scoring
- REST API integration
- Interactive data visualization

---

## 🎯 Business Problem

UPI transactions can exhibit unusual patterns, including:

- Unusually high transaction amounts
- Transactions during unusual hours
- Abnormal sender transaction behavior
- Unusual transaction frequency
- Sudden changes in transaction activity

The goal of this project is to identify potentially suspicious transactions and categorize them according to their calculated risk level.

> **Note:** This project is a prototype for learning and demonstration. It does not guarantee real-world fraud detection accuracy.

---

## 🚀 Key Features

### 1. Synthetic Data Generation

- Generates sample UPI transaction records
- Includes transaction amount, timestamp, sender, receiver, and transaction details
- Supports experimentation with fraud detection techniques

### 2. Data Preprocessing

- Timestamp conversion
- Feature engineering
- Transaction behavior analysis
- Sender-level aggregation

### 3. IQR Outlier Detection

- Calculates Q1 and Q3
- Computes the Interquartile Range (IQR)
- Identifies statistical outliers in transaction amounts

### 4. Isolation Forest

- Detects potentially anomalous transactions
- Uses transaction behavior and amount-related features
- Produces anomaly predictions and scores

### 5. Time-Series Anomaly Detection

- Aggregates transactions by hour
- Calculates transaction volume and amount statistics
- Uses rolling averages and standard deviation
- Identifies unusual hourly activity

### 6. Fraud Risk Scoring

The project combines multiple detection signals:

- IQR outlier detection
- Isolation Forest anomaly detection
- Night-time transaction indicator
- High-value transaction indicator
- Time-series anomaly detection

The signals are combined to calculate a fraud risk score.

#### Risk Categories

| Risk Score | Category |
|---|---|
| 0–39 | Low Risk |
| 40–69 | Medium Risk |
| 70–100 | High Risk |

> The score is a heuristic risk indicator and should not be interpreted as a verified probability of fraud.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Isolation Forest model |
| FastAPI | Backend API development |
| Uvicorn | API server |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts |
| Git & GitHub | Version control |

---

## 📂 Project Architecture

```text
UPI-Fraud-Detection/
│
├── data/
│   ├── raw/
│   │   └── upi_transactions.csv
│   │
│   └── processed/
│       ├── upi_features.csv
│       ├── upi_iqr_results.csv
│       ├── upi_isolation_forest_results.csv
│       ├── upi_time_series_results.csv
│       └── upi_fraud_scoring_results.csv
│
├── src/
│   ├── data_generation.py
│   ├── feature_engineering.py
│   ├── iqr_detection.py
│   ├── isolation_forest.py
│   ├── time_series_detection.py
│   ├── fraud_scoring.py
│   └── visualization.py
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
├── API_TESTING.md
├── README.md
└── .gitignore
