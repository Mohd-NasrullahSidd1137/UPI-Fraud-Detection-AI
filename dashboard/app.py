import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="UPI Fraud Intelligence",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. PROFESSIONAL DARK THEME
# =====================================================

st.markdown(
    """
    <style>

    /* Main Application Background */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                #172554 0%,
                #0f172a 35%,
                #020617 100%
            );

        color: #f8fafc;
    }


    /* Main Container */

    .main .block-container {
        max-width: 1600px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }


    /* Sidebar */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0f172a 0%,
                #111827 50%,
                #172554 100%
            );

        border-right: 1px solid #334155;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }


    /* Main Heading */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }


    .main-subtitle {
        font-size: 16px;
        color: #94a3b8;
        margin-bottom: 25px;
    }


    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 25px;
        margin-bottom: 15px;
    }


    /* KPI Cards */

    [data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 59, 0.95),
                rgba(15, 23, 42, 0.95)
            );

        border: 1px solid #334155;
        border-radius: 18px;
        padding: 22px;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.25);

        transition: all 0.3s ease;
    }


    [data-testid="stMetric"]:hover {
        border-color: #3b82f6;
        transform: translateY(-3px);

        box-shadow:
            0 10px 30px rgba(37, 99, 235, 0.2);
    }


    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }


    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 27px !important;
        font-weight: 800 !important;
    }


    /* Sidebar Selectbox */

    div[data-baseweb="select"] > div {
        background-color: #1e293b;
        border: 1px solid #475569;
        border-radius: 10px;
    }


    /* Download Button */

    .stDownloadButton button {
        width: 100%;
        background:
            linear-gradient(
                90deg,
                #2563eb,
                #4f46e5
            );

        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 12px;
    }


    .stDownloadButton button:hover {
        background:
            linear-gradient(
                90deg,
                #1d4ed8,
                #4338ca
            );
    }


    /* Dataframe */

    [data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 14px;
        overflow: hidden;
    }


    /* Divider */

    hr {
        border-color: #334155;
    }


    /* Info Box */

    .info-box {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px;
        color: #cbd5e1;
        margin-bottom: 20px;
    }


    /* Footer */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding-top: 30px;
        padding-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 3. PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "upi_fraud_scoring_results.csv"
)


# =====================================================
# 4. LOAD DATASET
# =====================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_FILE)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    data["amount"] = pd.to_numeric(
        data["amount"],
        errors="coerce"
    )

    data["fraud_risk_score"] = pd.to_numeric(
        data["fraud_risk_score"],
        errors="coerce"
    )

    data["hour"] = data["timestamp"].dt.hour

    return data


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "Dataset not found. Run fraud scoring script first."
    )

    st.stop()


# =====================================================
# 5. DASHBOARD HEADER
# =====================================================

st.markdown(
    """
    <div class="main-title">
        🔐 UPI Fraud Intelligence Center
    </div>

    <div class="main-subtitle">
        AI-powered transaction monitoring, anomaly detection
        and financial risk analytics
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="info-box">
        <b>System Status:</b> Dataset loaded successfully.
        Risk scores are generated using anomaly detection
        signals and rule-based scoring.
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 6. SIDEBAR FILTERS
# =====================================================

st.sidebar.markdown(
    "## 🔧 Dashboard Controls"
)

st.sidebar.markdown(
    "Filter transactions by risk category."
)


risk_options = [
    "All",
    "High Risk",
    "Medium Risk",
    "Low Risk"
]


selected_risk = st.sidebar.selectbox(
    "Risk Category",
    risk_options
)


if selected_risk == "All":

    filtered_df = df.copy()

else:

    filtered_df = df[
        df["risk_category"] == selected_risk
    ].copy()


st.sidebar.divider()

st.sidebar.write(
    f"**Selected Records:** {len(filtered_df):,}"
)

st.sidebar.write(
    f"**Total Records:** {len(df):,}"
)


# =====================================================
# 7. KPI CALCULATIONS
# =====================================================

total_transactions = len(filtered_df)

total_amount = filtered_df["amount"].sum()

average_amount = filtered_df["amount"].mean()

average_risk_score = (
    filtered_df["fraud_risk_score"].mean()
)


high_risk_count = (
    filtered_df["risk_category"] == "High Risk"
).sum()

medium_risk_count = (
    filtered_df["risk_category"] == "Medium Risk"
).sum()

low_risk_count = (
    filtered_df["risk_category"] == "Low Risk"
).sum()


# =====================================================
# 8. MAIN KPI CARDS
# =====================================================

st.markdown(
    '<div class="section-title">📌 Key Performance Indicators</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Total Transaction Value",
    f"₹{total_amount:,.2f}"
)

col3.metric(
    "Average Transaction",
    f"₹{average_amount:,.2f}"
)

col4.metric(
    "Average Risk Score",
    f"{average_risk_score:.2f}"
)


st.divider()


# =====================================================
# 9. RISK SUMMARY CARDS
# =====================================================

st.markdown(
    '<div class="section-title">🚨 Risk Overview</div>',
    unsafe_allow_html=True
)


risk_col1, risk_col2, risk_col3 = st.columns(3)


risk_col1.metric(
    "🔴 High Risk",
    f"{int(high_risk_count):,}"
)

risk_col2.metric(
    "🟠 Medium Risk",
    f"{int(medium_risk_count):,}"
)

risk_col3.metric(
    "🟢 Low Risk",
    f"{int(low_risk_count):,}"
)


st.divider()


# =====================================================
# 10. RISK CATEGORY CHART
# =====================================================

st.markdown(
    '<div class="section-title">📊 Risk Category Distribution</div>',
    unsafe_allow_html=True
)


risk_distribution = (
    filtered_df["risk_category"]
    .value_counts()
    .reset_index()
)

risk_distribution.columns = [
    "risk_category",
    "count"
]


fig_risk = px.bar(
    risk_distribution,
    x="risk_category",
    y="count",
    text="count",
    title="Transaction Distribution by Risk Category",
    labels={
        "risk_category": "Risk Category",
        "count": "Number of Transactions"
    },
    template="plotly_dark"
)


fig_risk.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"),
    title_font_size=19,
    margin=dict(l=20, r=20, t=65, b=20)
)


fig_risk.update_traces(
    textposition="outside"
)


st.plotly_chart(
    fig_risk,
    use_container_width=True
)


# =====================================================
# 11. FRAUD RISK SCORE DISTRIBUTION
# =====================================================

st.markdown(
    '<div class="section-title">📈 Fraud Risk Score Analysis</div>',
    unsafe_allow_html=True
)


fig_score = px.histogram(
    filtered_df,
    x="fraud_risk_score",
    nbins=30,
    title="Distribution of Fraud Risk Scores",
    labels={
        "fraud_risk_score": "Fraud Risk Score"
    },
    template="plotly_dark"
)


fig_score.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"),
    title_font_size=19,
    margin=dict(l=20, r=20, t=65, b=20)
)


st.plotly_chart(
    fig_score,
    use_container_width=True
)


# =====================================================
# 12. HOURLY TRANSACTION TREND
# =====================================================

st.markdown(
    '<div class="section-title">🕒 Hourly Transaction Trend</div>',
    unsafe_allow_html=True
)


hourly_data = (
    filtered_df
    .groupby("hour")
    .size()
    .reset_index(name="transaction_count")
)


fig_hourly = px.line(
    hourly_data,
    x="hour",
    y="transaction_count",
    markers=True,
    title="Transaction Activity by Hour",
    labels={
        "hour": "Hour of Day",
        "transaction_count": "Transaction Count"
    },
    template="plotly_dark"
)


fig_hourly.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"),
    title_font_size=19,
    margin=dict(l=20, r=20, t=65, b=20),
    xaxis=dict(
        tickmode="linear",
        dtick=1
    )
)


st.plotly_chart(
    fig_hourly,
    use_container_width=True
)


# =====================================================
# 13. AVERAGE RISK SCORE BY HOUR
# =====================================================

st.markdown(
    '<div class="section-title">🌙 Average Risk Score by Hour</div>',
    unsafe_allow_html=True
)


hourly_score = (
    filtered_df
    .groupby("hour")["fraud_risk_score"]
    .mean()
    .reset_index()
)


fig_hourly_score = px.line(
    hourly_score,
    x="hour",
    y="fraud_risk_score",
    markers=True,
    title="Average Fraud Risk Score by Hour",
    labels={
        "hour": "Hour of Day",
        "fraud_risk_score": "Average Risk Score"
    },
    template="plotly_dark"
)


fig_hourly_score.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"),
    title_font_size=19,
    margin=dict(l=20, r=20, t=65, b=20),
    xaxis=dict(
        tickmode="linear",
        dtick=1
    )
)


st.plotly_chart(
    fig_hourly_score,
    use_container_width=True
)


# =====================================================
# 14. ANOMALY DETECTION COMPARISON
# =====================================================

st.markdown(
    '<div class="section-title">🔍 Anomaly Detection Comparison</div>',
    unsafe_allow_html=True
)


anomaly_columns = [
    "iqr_outlier",
    "isolation_forest_anomaly",
    "time_series_anomaly"
]


available_columns = [
    column
    for column in anomaly_columns
    if column in filtered_df.columns
]


anomaly_results = []


for column in available_columns:

    count = int(
        filtered_df[column]
        .astype(bool)
        .sum()
    )

    anomaly_results.append(
        {
            "method": column,
            "anomaly_count": count
        }
    )


anomaly_df = pd.DataFrame(anomaly_results)


if not anomaly_df.empty:

    fig_anomaly = px.bar(
        anomaly_df,
        x="method",
        y="anomaly_count",
        text="anomaly_count",
        title="Anomalies Detected by Each Method",
        labels={
            "method": "Detection Method",
            "anomaly_count": "Anomaly Count"
        },
        template="plotly_dark"
    )


    fig_anomaly.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        title_font_size=19,
        margin=dict(l=20, r=20, t=65, b=20)
    )


    fig_anomaly.update_traces(
        textposition="outside"
    )


    st.plotly_chart(
        fig_anomaly,
        use_container_width=True
    )

else:

    st.info("No anomaly columns available.")


# =====================================================
# 15. TOP RISKY TRANSACTIONS
# =====================================================

st.markdown(
    '<div class="section-title">🚨 Top 20 Risky Transactions</div>',
    unsafe_allow_html=True
)


top_risky = (
    filtered_df
    .sort_values(
        by="fraud_risk_score",
        ascending=False
    )
    .head(20)
)


display_columns = [
    "transaction_id",
    "timestamp",
    "amount",
    "fraud_risk_score",
    "risk_category",
    "fraud_alert"
]


available_display_columns = [
    column
    for column in display_columns
    if column in top_risky.columns
]


st.dataframe(
    top_risky[available_display_columns],
    use_container_width=True,
    hide_index=True
)


# =====================================================
# 16. DOWNLOAD FILTERED DATA
# =====================================================

st.markdown(
    '<div class="section-title">📥 Export Report</div>',
    unsafe_allow_html=True
)


csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="⬇️ Download Filtered Transactions",
    data=csv_data,
    file_name="filtered_upi_transactions.csv",
    mime="text/csv"
)


# =====================================================
# 17. FOOTER
# =====================================================

st.markdown(
    """
    <div class="footer">
        <hr>
        🔐 UPI Fraud Intelligence Center<br>
        Built with Python, Pandas, Plotly and Streamlit<br>
        <br>
        Risk score is a heuristic indicator and is not
        a verified probability of fraud.
    </div>
    """,
    unsafe_allow_html=True
)