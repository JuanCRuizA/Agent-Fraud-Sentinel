"""
BAFS - Banking Anti-Fraud System
Streamlit Dashboard Prototype

Phase 5: Interactive Model Explainability & Regulatory Dashboard

Run with:
    cd notebooks/dashboard
    streamlit run dashboard_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from PIL import Image
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BAFS - Banking Anti-Fraud System",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────
# Custom CSS - Professional Banking Aesthetic
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1a365d; }
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.85rem;
        padding: 20px 0;
        border-top: 1px solid #dee2e6;
        margin-top: 40px;
    }
    .footer a { color: #1a365d; text-decoration: none; }
    .footer a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Paths (relative to notebooks/dashboard/)
# ─────────────────────────────────────────────────────────────────────
try:
    BASE_PATH = Path(__file__).parent.resolve()
except NameError:
    BASE_PATH = Path.cwd()

MODEL_PATH = BASE_PATH / '..' / '..' / 'models'
DATA_PATH = BASE_PATH / '..' / '..' / 'data' / 'processed'
FIGURES_PATH = BASE_PATH / '..' / '..' / 'figures' / 'shap'


# ─────────────────────────────────────────────────────────────────────
# Data & Model Loading (cached for performance)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    """Load XGBoost model, scaler, and threshold configuration."""
    model = joblib.load(MODEL_PATH / 'xgboost_final.pkl')
    scaler = joblib.load(MODEL_PATH / 'scaler.pkl')
    threshold_config = joblib.load(MODEL_PATH / 'threshold_config.pkl')
    return model, scaler, threshold_config


@st.cache_data
def load_test_data():
    """Load the held-out test set (most recent transactions)."""
    return pd.read_csv(DATA_PATH / 'test.csv')


@st.cache_data
def compute_predictions(_model, df, features):
    """Generate fraud scores for all test transactions."""
    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], [10, -10]).fillna(0)
    y_true = df['isFraud'].values
    y_scores = _model.predict_proba(X)[:, 1]
    return X, y_true, y_scores


# Load everything
try:
    model, scaler, threshold_config = load_model_artifacts()
    df_test = load_test_data()

    FEATURES = threshold_config['features']
    AUTO_BLOCK = threshold_config['auto_block_threshold']
    MANUAL_REVIEW = threshold_config['manual_review_threshold']
    FN_COST = threshold_config.get('fn_cost', 75.0)
    FP_COST = threshold_config.get('fp_cost', 10.0)

    X_test, y_test, fraud_scores = compute_predictions(model, df_test, FEATURES)
except Exception as e:
    st.error(f"Failed to load model or data: {e}")
    st.info(
        "Ensure model artifacts exist in ../../models/ "
        "and test data in ../../data/processed/"
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────
# Banking-Friendly Feature Labels
# ─────────────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    'txn_count_1hr': 'Transaction Velocity (1 hour)',
    'txn_count_24hr': 'Transaction Velocity (24 hours)',
    'amount_deviation': 'Spending Anomaly Score',
    'is_first_transaction': 'First-Time Transaction',
    'hour_of_day': 'Time of Day',
    'is_weekend': 'Weekend Transaction',
    'TransactionAmt': 'Transaction Amount ($)'
}


# ─────────────────────────────────────────────────────────────────────
# Reusable Footer (appears at the bottom of EVERY tab)
# ─────────────────────────────────────────────────────────────────────
FOOTER_HTML = """
<div class="footer">
    <strong>BAFS - Banking Anti-Fraud System</strong><br>
    <a href="https://github.com/JuanCRuizA/Agent-Fraud-Sentinel.git"
       target="_blank">
        https://github.com/JuanCRuizA/Agent-Fraud-Sentinel.git
    </a><br>
    Developed by Juan Carlos Ruiz Arteaga<br>
    Banking Data Scientist<br>
    MSc in Data Science &amp; AI, University of Liverpool<br>
    Contact: j.ruiz-arteaga@liverpool.ac.uk
</div>
"""


def render_footer():
    """Render the standard project footer."""
    st.markdown("---")
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("BAFS")
    st.caption("Banking Anti-Fraud System")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Executive Summary",
            "Model Performance",
            "Case Study Explorer",
            "Regulatory Compliance",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("Global Filters")

    risk_threshold = st.slider(
        "Risk Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(round(MANUAL_REVIEW, 2)),
        step=0.01,
        help="Transactions scoring above this threshold are flagged for review.",
    )

    sample_size = st.selectbox(
        "Sample Size",
        options=[1000, 5000, 10000, 50000, len(y_test)],
        index=4,
        format_func=lambda x: (
            f"{x:,} transactions"
            if x < len(y_test)
            else f"Full dataset ({len(y_test):,})"
        ),
    )

    st.markdown("---")

    with st.expander("About & Methods"):
        st.markdown(
            "- **Dataset:** IEEE-CIS Fraud Detection (590,540 transactions)\n"
            "- **Model:** XGBoost with cost-sensitive optimization\n"
            "- **Cost structure:** $75 FN / $10 FP (ratio 7.5 : 1)\n"
            "- **Explainability:** SHAP TreeExplainer\n"
            "- **Compliance:** Aligned with Federal Reserve SR 11-7"
        )


# ─────────────────────────────────────────────────────────────────────
# Apply Global Filters
# ─────────────────────────────────────────────────────────────────────
if sample_size < len(y_test):
    np.random.seed(42)
    idx = np.random.choice(len(y_test), size=sample_size, replace=False)
    y_filt = y_test[idx]
    scores_filt = fraud_scores[idx]
    X_filt = X_test.iloc[idx]
else:
    y_filt = y_test
    scores_filt = fraud_scores
    X_filt = X_test

y_pred_filt = (scores_filt >= risk_threshold).astype(int)


# =====================================================================
#  PAGE CONTENT
# =====================================================================

# ── TAB 1: Executive Summary ────────────────────────────────────────
if page == "Executive Summary":
    st.header("Executive Summary")
    st.caption(
        "Key performance indicators for the BAFS fraud detection system"
    )

    # Compute KPIs
    tp = int(((y_filt == 1) & (y_pred_filt == 1)).sum())
    fp = int(((y_filt == 0) & (y_pred_filt == 1)).sum())
    fn = int(((y_filt == 1) & (y_pred_filt == 0)).sum())
    tn = int(((y_filt == 0) & (y_pred_filt == 0)).sum())
    total_fraud = int(y_filt.sum())
    recall = tp / total_fraud if total_fraud > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fraud_prevented = tp * FN_COST
    missed_fraud = fn * FN_COST
    review_cost = fp * FP_COST
    total_cost = missed_fraud + review_cost

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Fraud Detected",
        f"{tp:,} / {total_fraud:,}",
        f"{recall:.1%} recall",
    )
    k2.metric(
        "False Positive Rate",
        f"{fpr:.1%}",
        f"{fp:,} false alarms",
    )
    k3.metric(
        "Fraud Prevented",
        f"${fraud_prevented:,.0f}",
        f"{tp:,} transactions blocked",
    )
    k4.metric(
        "Total Operational Cost",
        f"${total_cost:,.0f}",
        f"${total_cost / len(y_filt):.2f} per txn",
    )

    st.markdown("---")

    # Two-column: performance table + risk distribution
    left, right = st.columns(2)

    with left:
        st.subheader("Performance at Current Threshold")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        perf_df = pd.DataFrame(
            {
                "Metric": [
                    "Recall (Fraud Detection Rate)",
                    "Precision (Confirmation Rate)",
                    "F1-Score",
                    "False Positive Rate",
                    "Threshold Applied",
                ],
                "Value": [
                    f"{recall:.2%}",
                    f"{precision:.2%}",
                    f"{f1:.4f}",
                    f"{fpr:.2%}",
                    f"{risk_threshold:.3f}",
                ],
            }
        )
        st.table(perf_df)

    with right:
        st.subheader("Risk Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            scores_filt[y_filt == 0], bins=50, alpha=0.6,
            color="#2196F3", label="Legitimate", density=True,
        )
        ax.hist(
            scores_filt[y_filt == 1], bins=50, alpha=0.6,
            color="#f44336", label="Fraud", density=True,
        )
        ax.axvline(
            risk_threshold, color="#333", linestyle="--",
            linewidth=2, label=f"Threshold ({risk_threshold:.2f})",
        )
        ax.set_xlabel("Fraud Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(
            "Distribution of Fraud Scores",
            fontsize=14, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Cost breakdown
    st.subheader("Cost Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Missed Fraud Cost",
        f"${missed_fraud:,.0f}",
        f"{fn:,} missed x ${FN_COST:.0f}",
    )
    c2.metric(
        "False Alarm Cost",
        f"${review_cost:,.0f}",
        f"{fp:,} reviews x ${FP_COST:.0f}",
    )
    no_model = total_fraud * FN_COST
    savings = no_model - missed_fraud
    c3.metric(
        "Fraud Savings vs No Model",
        f"${savings:,.0f}",
        f"${no_model:,.0f} baseline",
    )

    render_footer()


# ── TAB 2: Model Performance ────────────────────────────────────────
elif page == "Model Performance":
    st.header("Model Performance Analysis")
    st.caption(
        "Detailed evaluation of the XGBoost fraud detection model"
    )

    # Compute confusion matrix values for this page
    cm = confusion_matrix(y_filt, y_pred_filt)
    tn_v, fp_v, fn_v, tp_v = cm.ravel()

    # Row 1: Confusion Matrix + ROC
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.subheader("Confusion Matrix with Cost Overlay")
        labels = np.array(
            [
                [
                    f"TN\n{tn_v:,}\n$0",
                    f"FP\n{fp_v:,}\n${fp_v * FP_COST:,.0f}",
                ],
                [
                    f"FN\n{fn_v:,}\n${fn_v * FN_COST:,.0f}",
                    f"TP\n{tp_v:,}\nPrevented",
                ],
            ]
        )
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm, annot=labels, fmt="", cmap="Blues", ax=ax,
            xticklabels=["Predicted Legit", "Predicted Fraud"],
            yticklabels=["Actual Legit", "Actual Fraud"],
            cbar_kws={"label": "Count"},
        )
        ax.set_title(
            f"Confusion Matrix (threshold = {risk_threshold:.3f})",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with r1c2:
        st.subheader("ROC Curve")
        fpr_c, tpr_c, _ = roc_curve(y_filt, scores_filt)
        roc_auc_val = roc_auc_score(y_filt, scores_filt)

        # Operating point at current threshold
        tpr_op = recall_score(y_filt, y_pred_filt)
        fpr_op = fp_v / (fp_v + tn_v) if (fp_v + tn_v) > 0 else 0

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(
            fpr_c, tpr_c, color="#1565C0", linewidth=2,
            label=f"XGBoost (AUC = {roc_auc_val:.4f})",
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        ax.scatter(
            [fpr_op], [tpr_op], color="red", s=100, zorder=5,
            label=f"Operating Point ({risk_threshold:.2f})",
        )
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
        ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 2: PR Curve + Feature Importance
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.subheader("Precision-Recall Curve")
        prec_c, rec_c, _ = precision_recall_curve(y_filt, scores_filt)
        pr_auc_val = auc(rec_c, prec_c)

        prec_op = (
            precision_score(y_filt, y_pred_filt)
            if y_pred_filt.sum() > 0 else 0
        )
        rec_op = (
            recall_score(y_filt, y_pred_filt)
            if y_filt.sum() > 0 else 0
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(
            rec_c, prec_c, color="#1565C0", linewidth=2,
            label=f"XGBoost (PR-AUC = {pr_auc_val:.4f})",
        )
        ax.axhline(
            y=y_filt.mean(), color="red", linestyle="--",
            alpha=0.5, label=f"Baseline ({y_filt.mean():.4f})",
        )
        ax.scatter(
            [rec_op], [prec_op], color="red", s=100, zorder=5,
            label=f"Operating Point ({risk_threshold:.2f})",
        )
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(
            "Precision-Recall Curve", fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1.05])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with r2c2:
        st.subheader("Feature Importance (Top 7)")
        shap_img = FIGURES_PATH / "shap_feature_importance_bar.png"
        if shap_img.exists():
            st.image(
                Image.open(shap_img), width="stretch",
            )
        else:
            # Fallback: use model's built-in feature importances
            importance = model.feature_importances_
            imp_df = pd.DataFrame(
                {
                    "Feature": [
                        FEATURE_LABELS.get(f, f) for f in FEATURES
                    ],
                    "Importance": importance,
                }
            ).sort_values("Importance", ascending=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(
                imp_df["Feature"], imp_df["Importance"],
                color="steelblue",
            )
            ax.set_xlabel("Importance (Gain)", fontsize=11)
            ax.set_title(
                "Feature Importance", fontsize=13, fontweight="bold",
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Cost-Benefit Analysis Table
    st.subheader("Cost-Benefit Analysis by Threshold")
    thresholds = sorted(
        set(
            [0.20, 0.30, round(MANUAL_REVIEW, 2), 0.50,
             0.60, 0.70, 0.80, round(AUTO_BLOCK, 2)]
        )
    )
    rows = []
    for t in thresholds:
        yp = (scores_filt >= t).astype(int)
        t_tp = int(((y_filt == 1) & (yp == 1)).sum())
        t_fp = int(((y_filt == 0) & (yp == 1)).sum())
        t_fn = int(((y_filt == 1) & (yp == 0)).sum())
        t_rec = t_tp / y_filt.sum() if y_filt.sum() > 0 else 0
        t_pre = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
        t_cost = t_fn * FN_COST + t_fp * FP_COST
        marker = " *" if abs(t - risk_threshold) < 0.005 else ""
        rows.append(
            {
                "Threshold": f"{t:.2f}{marker}",
                "Recall": f"{t_rec:.1%}",
                "Precision": f"{t_pre:.1%}",
                "True Positives": f"{t_tp:,}",
                "False Positives": f"{t_fp:,}",
                "Missed Frauds": f"{t_fn:,}",
                "Total Cost": f"${t_cost:,.0f}",
            }
        )
    st.caption("* = current threshold")
    st.dataframe(
        pd.DataFrame(rows), width="stretch", hide_index=True,
    )

    render_footer()


# ── TAB 3: Case Study Explorer ──────────────────────────────────────
elif page == "Case Study Explorer":
    st.header("Case Study Explorer")
    st.caption(
        "Detailed analysis of individual transaction decisions "
        "with SHAP explanations"
    )

    # Case study definitions (from Phase 4 SHAP analysis)
    cases = {
        "Case 1: True Positive -- Clear Fraud Caught": {
            "score": 0.9094,
            "actual": "FRAUD",
            "decision": "AUTO-BLOCK",
            "features": {
                "Transaction Amount": "$17.52",
                "Time of Day": "6:00 AM",
                "Transactions (1 hour)": "1",
                "Transactions (24 hours)": "9",
                "Weekend": "Yes",
                "Spending Anomaly": "-0.39 std devs",
                "First Transaction": "No",
            },
            "explanation": (
                "This transaction was correctly identified as fraud. "
                "Multiple strong indicators were present: high daily "
                "velocity (9 transactions in 24 hours), weekend timing, "
                "and a small transaction amount typical of card-testing "
                "behaviour. The model automatically blocked this "
                "transaction, preventing potential fraud loss."
            ),
            "drivers": [
                "High 24-hour velocity (9 transactions)",
                "Weekend + early morning timing",
                "Small amount consistent with card testing",
            ],
        },
        "Case 2: True Positive -- Velocity-Driven Detection": {
            "score": 0.7332,
            "actual": "FRAUD",
            "decision": "MANUAL REVIEW",
            "features": {
                "Transaction Amount": "$59.64",
                "Time of Day": "4:00 AM",
                "Transactions (1 hour)": "1",
                "Transactions (24 hours)": "3",
                "Weekend": "No",
                "Spending Anomaly": "+0.54 std devs",
                "First Transaction": "No",
            },
            "explanation": (
                "This fraud was detected primarily through velocity "
                "signals. The combination of elevated 24-hour velocity, "
                "early morning timing, and above-average spending "
                "deviation pushed the score into the manual review zone. "
                "An analyst would confirm this as fraud based on the "
                "pattern."
            ),
            "drivers": [
                "Velocity features elevated",
                "4 AM transaction time",
                "Above-average spending deviation",
            ],
        },
        "Case 3: False Negative -- Missed Fraud": {
            "score": 0.0853,
            "actual": "FRAUD",
            "decision": "AUTO-APPROVE",
            "features": {
                "Transaction Amount": "$57.95",
                "Time of Day": "Business hours",
                "Transactions (1 hour)": "0",
                "Transactions (24 hours)": "0",
                "Weekend": "No",
                "Spending Anomaly": "-1.50 std devs",
                "First Transaction": "Yes",
            },
            "explanation": (
                "The model failed to detect this fraud because all "
                "behavioural features appeared normal. The transaction "
                "had zero velocity (isolated event), a moderate amount, "
                "and occurred during business hours. This represents a "
                "model limitation: sophisticated fraudsters who pace "
                "transactions can evade velocity-based detection."
            ),
            "drivers": [
                "Zero velocity (no pattern to detect)",
                "Normal business hours",
                "First transaction (no history)",
            ],
            "improvement": (
                "Consider adding merchant-category features and device "
                "fingerprinting to catch isolated sophisticated fraud."
            ),
        },
        "Case 4: False Positive -- Legitimate Flagged": {
            "score": 0.9342,
            "actual": "LEGITIMATE",
            "decision": "AUTO-BLOCK",
            "features": {
                "Transaction Amount": "$15.00",
                "Time of Day": "9:00 AM",
                "Transactions (1 hour)": "1",
                "Transactions (24 hours)": "2",
                "Weekend": "Yes",
                "Spending Anomaly": "-0.26 std devs",
                "First Transaction": "No",
            },
            "explanation": (
                "This legitimate transaction was incorrectly blocked. "
                "The model was triggered by the combination of weekend "
                "timing, low amount (similar to card-testing), and "
                "moderate velocity. This false positive demonstrates "
                "why human review and dispute resolution processes are "
                "essential."
            ),
            "drivers": [
                "Weekend + morning timing",
                "Small amount triggered card-testing pattern",
                "Multiple moderate risk factors accumulated",
            ],
        },
        "Case 5: Auto-Block Candidate -- High Confidence": {
            "score": 0.9342,
            "actual": "LEGITIMATE",
            "decision": "AUTO-BLOCK",
            "features": {
                "Transaction Amount": "$15.00",
                "Time of Day": "9:00 AM",
                "Transactions (1 hour)": "1",
                "Transactions (24 hours)": "2",
                "Weekend": "Yes",
                "Spending Anomaly": "-0.26 std devs",
                "First Transaction": "No",
            },
            "explanation": (
                "This case illustrates the risk of auto-blocking: a "
                "high-confidence score (0.93) on a legitimate "
                "transaction. While rare, auto-block false positives "
                "highlight the need for rapid dispute resolution and "
                "continuous model monitoring. The SHAP explanation "
                "provides the audit trail needed for customer "
                "communication."
            ),
            "drivers": [
                "Score exceeded auto-block threshold (0.90)",
                "Pattern matched card-testing signature",
                "Demonstrates need for dispute resolution workflow",
            ],
        },
        "Case 6: Borderline -- Near Review Threshold": {
            "score": 0.3648,
            "actual": "LEGITIMATE",
            "decision": "AUTO-APPROVE",
            "features": {
                "Transaction Amount": "$125.00",
                "Time of Day": "Afternoon",
                "Transactions (1 hour)": "0",
                "Transactions (24 hours)": "0",
                "Weekend": "No",
                "Spending Anomaly": "Normal",
                "First Transaction": "No",
            },
            "explanation": (
                "This borderline case scored just below the manual "
                "review threshold (0.41). The transaction had no "
                "velocity flags and a reasonable amount. The model "
                "correctly approved it, but the score shows it was "
                "not far from the review zone. Small changes in "
                "behaviour could tip similar transactions into review."
            ),
            "drivers": [
                "Score near but below threshold (0.36 vs 0.41)",
                "No velocity anomalies",
                "Normal transaction pattern with slight uncertainty",
            ],
        },
    }

    selected = st.selectbox(
        "Select a case study to analyse:", list(cases.keys()),
    )
    case = cases[selected]

    st.markdown("---")

    # KPI row for selected case
    m1, m2, m3 = st.columns(3)
    m1.metric("Fraud Score", f"{case['score']:.4f}")
    m2.metric("Model Decision", case["decision"])
    m3.metric("Actual Outcome", case["actual"])

    # Feature details
    st.subheader("Transaction Features")
    feat_items = list(case["features"].items())
    mid = (len(feat_items) + 1) // 2
    fc1, fc2 = st.columns(2)
    with fc1:
        for k, v in feat_items[:mid]:
            st.markdown(f"**{k}:** {v}")
    with fc2:
        for k, v in feat_items[mid:]:
            st.markdown(f"**{k}:** {v}")

    # SHAP waterfall plot
    st.subheader("SHAP Explanation")
    waterfall = FIGURES_PATH / "shap_waterfall_cases.png"
    if waterfall.exists():
        st.image(
            Image.open(waterfall),
            caption=(
                "SHAP waterfall plots for all 6 case studies "
                "(from Phase 4 analysis)"
            ),
            width="stretch",
        )
    else:
        st.info(
            "SHAP waterfall plot not available. "
            "Run notebook 04_shap_explainability.ipynb first."
        )

    # Plain-English explanation
    st.subheader("Model Decision Explanation")
    st.write(case["explanation"])

    st.subheader("Key Risk Drivers")
    for d in case["drivers"]:
        st.markdown(f"- {d}")

    if "improvement" in case:
        st.subheader("Recommended Improvement")
        st.write(case["improvement"])

    render_footer()


# ── TAB 4: Regulatory Compliance ────────────────────────────────────
elif page == "Regulatory Compliance":
    st.header("Regulatory Compliance")
    st.caption(
        "Model governance, fair lending review, and audit readiness"
    )

    # SR 11-7 Checklist
    st.subheader("SR 11-7 Model Documentation Checklist")

    done_items = [
        "Model documentation (purpose, inputs, outputs, assumptions)",
        "Performance metrics on held-out test data",
        "Global explainability (feature importance, SHAP summary)",
        "Local explainability (individual transaction SHAP)",
        "Limitations and known risks documented",
        "Fair lending feature review conducted",
        "Right-to-explanation capability demonstrated",
        "Audit trail requirements specified",
    ]
    pending_items = [
        "Disparate impact testing (requires demographic data)",
        "Champion/challenger framework",
        "Ongoing monitoring dashboard (drift detection)",
        "Quarterly model revalidation schedule",
    ]

    chk1, chk2 = st.columns(2)
    with chk1:
        st.markdown("**Completed**")
        for item in done_items:
            st.checkbox(item, value=True, disabled=True, key=f"d_{item}")
    with chk2:
        st.markdown("**Pending**")
        for item in pending_items:
            st.checkbox(item, value=False, disabled=True, key=f"p_{item}")

    st.markdown("---")

    # Fair Lending
    st.subheader("Fair Lending Considerations")

    fl_data = pd.DataFrame(
        [
            {
                "Feature": "Transaction Velocity (1hr, 24hr)",
                "Risk Level": "LOW",
                "Assessment": (
                    "Behavioural pattern. Monitor for disparate "
                    "impact across segments."
                ),
            },
            {
                "Feature": "Spending Anomaly Score",
                "Risk Level": "LOW",
                "Assessment": (
                    "Self-norming (deviation from client's own "
                    "history)."
                ),
            },
            {
                "Feature": "First-Time Transaction",
                "Risk Level": "MEDIUM",
                "Assessment": (
                    "New customers disproportionately flagged. "
                    "Monitor approval rates."
                ),
            },
            {
                "Feature": "Time of Day / Weekend",
                "Risk Level": "MEDIUM",
                "Assessment": (
                    "Shift workers and time zones may be affected. "
                    "Monitor FPR by region."
                ),
            },
            {
                "Feature": "Transaction Amount",
                "Risk Level": "LOW-MEDIUM",
                "Assessment": (
                    "Spending power correlates with income. "
                    "Monitor across segments."
                ),
            },
        ]
    )
    st.dataframe(
        fl_data, width="stretch", hide_index=True,
    )
    st.markdown(
        "**Overall Assessment:** No direct protected attributes used. "
        "Conduct disparate impact analysis when demographic data "
        "becomes available."
    )

    st.markdown("---")

    # Model Governance
    st.subheader("Model Governance Framework")

    gov1, gov2 = st.columns(2)

    with gov1:
        st.markdown("**Model Identification**")
        st.text("Name:     Agent Fraud Sentinel (XGBoost)")
        st.text("Version:  1.0")
        st.text("Type:     Gradient Boosted Decision Tree")
        st.text("Purpose:  Real-time fraud detection")
        st.text("Date:     February 2026")
        st.text("")
        st.markdown("**Monitoring Schedule**")
        sched = pd.DataFrame(
            [
                {
                    "Frequency": "Daily",
                    "Activity": (
                        "Alert volume, auto-block count, queue size"
                    ),
                },
                {
                    "Frequency": "Weekly",
                    "Activity": (
                        "Recall, precision, FPR by risk tier"
                    ),
                },
                {
                    "Frequency": "Monthly",
                    "Activity": (
                        "SHAP drift analysis, feature stability"
                    ),
                },
                {
                    "Frequency": "Quarterly",
                    "Activity": (
                        "Full revalidation, threshold recalibration"
                    ),
                },
                {
                    "Frequency": "Annual",
                    "Activity": "Comprehensive SR 11-7 review",
                },
            ]
        )
        st.dataframe(
            sched, width="stretch", hide_index=True,
        )

    with gov2:
        st.markdown("**Model Risk Classification**")
        st.text("Recommended Tier:  Tier 2")
        st.text("Rationale:         Material financial impact")
        st.text("Review Cycle:      Quarterly")
        st.text("")
        st.markdown("**Key Assumptions**")
        st.markdown(
            "1. Training fraud patterns represent future fraud\n"
            "2. Temporal ordering preserved (no data leakage)\n"
            "3. Client identity: card1 + addr1 + P_emaildomain\n"
            "4. Cost ratio 7.5:1 ($75 FN, $10 FP)\n"
            "5. Minimum 75% recall target"
        )

    st.markdown("---")

    # Right to Explanation
    st.subheader("Right-to-Explanation Capabilities")

    st.markdown(
        "Customers whose transactions are blocked or flagged may "
        "request an explanation. SHAP values provide a complete, "
        "auditable explanation at the individual level.\n\n"
        "**For any transaction, the system can generate:**\n\n"
        "1. **Feature-level attribution** -- which factors "
        "contributed to the decision\n"
        "2. **Quantified contribution** -- how much each factor "
        "affected the score\n"
        "3. **Comparison to baseline** -- score relative to average "
        "fraud probability\n\n"
        "**Dispute Resolution Workflow:**\n\n"
        "1. Customer contacts bank about blocked transaction\n"
        "2. Analyst retrieves SHAP explanation from audit log\n"
        "3. Analyst reviews feature contributions in plain English\n"
        "4. If false positive: approve transaction, note for model "
        "feedback\n"
        "5. If true fraud: confirm block, initiate investigation"
    )

    st.markdown("---")

    # Data Lineage
    st.subheader("Data Lineage and Audit Trail")

    st.markdown(
        "**Data Source:** IEEE-CIS Fraud Detection Dataset "
        "(590,540 transactions)\n\n"
        "**Processing Pipeline:**\n\n"
        "1. Raw data ingestion (434 features)\n"
        "2. Feature engineering: 7 behavioural features derived "
        "from raw data\n"
        "3. Temporal split: 60/20/20 "
        "(train / validation / test)\n"
        "4. Model training: XGBoost with cost-sensitive "
        "optimization\n"
        "5. Threshold calibration: cost-minimising with 75% "
        "recall constraint\n"
        "6. Explainability: SHAP TreeExplainer for all "
        "predictions\n\n"
        "**Audit Requirements:**\n\n"
        "- SHAP values stored at scoring time\n"
        "- Retention: minimum 7 years (regulatory requirement)\n"
        "- Log fields: transaction_id, fraud_score, threshold, "
        "decision, SHAP values, model_version, timestamp"
    )

    render_footer()
