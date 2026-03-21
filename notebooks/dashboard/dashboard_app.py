"""
SAFE - System for Anti-Fraud Evaluation
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
    page_title="SAFE - System for Anti-Fraud Evaluation",
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
    .toc-box {
        background: #eef2f7;
        padding: 14px 20px;
        border-radius: 8px;
        border-left: 4px solid #1a365d;
        margin-bottom: 16px;
    }
    .toc-box a { color: #1a365d; text-decoration: none; }
    .toc-box a:hover { text-decoration: underline; }
    /* Hide Streamlit anchor link icons on all headings */
    h1 > a, h2 > a, h3 > a, h4 > a, h5 > a, h6 > a,
    .stMarkdown h1 > a, .stMarkdown h2 > a, .stMarkdown h3 > a {
        display: none !important;
    }
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
DATA_PATH = BASE_PATH
FIGURES_PATH = BASE_PATH / '..' / '..' / 'figures' / 'shap'
MT_FIGURES_PATH = BASE_PATH / '..' / '..' / 'figures' / 'model_training'


# ─────────────────────────────────────────────────────────────────────
# Data & Model Loading (cached for performance)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    """Load LightGBM model and threshold configuration."""
    model = joblib.load(MODEL_PATH / 'best_model_final.pkl')
    threshold_config = joblib.load(MODEL_PATH / 'threshold_config.pkl')
    return model, threshold_config


@st.cache_data
def load_test_data():
    """Load the held-out test set (most recent transactions)."""
    slim = DATA_PATH / 'test_dashboard.csv'
    full = BASE_PATH / '..' / '..' / 'data' / 'processed' / 'test.csv'
    path = slim if slim.exists() else full
    return pd.read_csv(path)


@st.cache_data
def compute_predictions(_model, _df, features):
    """Generate fraud scores for all test transactions."""
    X = _df[features].copy()
    X = X.replace([np.inf, -np.inf], [10, -10]).fillna(0)
    y_true = _df['isFraud'].values
    y_scores = _model.predict_proba(X)[:, 1]
    return X, y_true, y_scores


@st.cache_data
def build_client_data(_df_test, _all_scores):
    """Build per-transaction scored data and per-client summary."""
    keep_cols = [c for c in ['TransactionID', 'TransactionDT', 'TransactionAmt',
                              'client_id', 'isFraud'] if c in _df_test.columns]
    df = _df_test[keep_cols].copy()
    df['fraud_score'] = _all_scores
    client_stats = df.groupby('client_id').agg(
        txn_count=('fraud_score', 'count'),
        max_score=('fraud_score', 'max'),
        mean_score=('fraud_score', 'mean'),
        fraud_txns=('isFraud', 'sum'),
        total_amount=('TransactionAmt', 'sum'),
    ).reset_index()
    return df, client_stats


# Load everything
try:
    model, threshold_config = load_model_artifacts()
    df_test = load_test_data()

    FEATURES = threshold_config['features']
    AUTO_BLOCK = threshold_config['auto_block_threshold']
    MANUAL_REVIEW = threshold_config['manual_review_threshold']
    FN_COST = threshold_config.get('fn_cost', 227.0)
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
    'is_first_transaction': 'First-Time Transaction Flag',
    'hour_of_day': 'Time of Day',
    'is_weekend': 'Weekend Transaction',
    'TransactionAmt': 'Transaction Amount ($)'
}


# ─────────────────────────────────────────────────────────────────────
# Reusable Footer
# ─────────────────────────────────────────────────────────────────────
FOOTER_HTML = """
<div class="footer">
    <strong>SAFE - System for Anti-Fraud Evaluation</strong><br>
    <a href="https://github.com/JuanCRuizA/Agent-Fraud-Sentinel.git"
       target="_blank">
        https://github.com/JuanCRuizA/Agent-Fraud-Sentinel.git
    </a><br>
    Developed by Juan Carlos Ruiz Arteaga | Banking Data Scientist<br>
    MSc in Data Science &amp; AI, University of Liverpool<br>
    Contact: j.ruiz-arteaga@liverpool.ac.uk
</div>
"""


def render_footer():
    st.markdown("---")
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Sidebar - Global Configuration Panel
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("SAFE")
    st.caption("System for Anti-Fraud Evaluation")
    st.markdown("---")

    st.subheader("Global Filters")

    risk_threshold = st.slider(
        "Risk Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(round(MANUAL_REVIEW, 2)),
        step=0.01,
        help=(
            f"Transactions scoring above this value are flagged for review. "
            f"Auto-block threshold: {AUTO_BLOCK:.2f}"
        ),
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

    st.subheader("Export")
    flagged_mask = fraud_scores >= risk_threshold
    export_cols = [c for c in ['TransactionID', 'client_id', 'TransactionAmt', 'isFraud']
                   if c in df_test.columns]
    flagged_export = df_test[export_cols].copy()
    flagged_export['fraud_score'] = fraud_scores
    flagged_export = flagged_export[flagged_mask]
    csv_data = flagged_export.to_csv(index=False)
    st.download_button(
        label=f"Export Flagged ({flagged_mask.sum():,} txns)",
        data=csv_data,
        file_name="flagged_transactions.csv",
        mime="text/csv",
        help="Download all transactions above the current threshold as CSV.",
    )

    st.markdown("---")

    with st.expander("About & Methods"):
        st.markdown(
            "- **Dataset:** IEEE-CIS Fraud Detection (590,540 transactions)\n"
            "- **Model:** LightGBM with Bayesian optimization\n"
            "- **Cost structure:** $227 FN / $10 FP (ratio 22.7 : 1)\n"
            "- **Explainability:** SHAP TreeExplainer\n"
            "- **Compliance:** SR 11-7 / OCC 2011-12 (US); FINMA Circular 2023/1, nDSG, EU AI Act (Swiss/EU)"
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
#  MAIN TABS
# =====================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Summary",
    "Model Comparison",
    "Case Study Explorer",
    "Client Risk Profile",
    "Regulatory Compliance",
])


# =====================================================================
# TAB 1 - Executive Summary
# =====================================================================
with tab1:
    st.header("Executive Summary")
    st.caption("Key performance indicators for the SAFE fraud detection system")

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

    no_model = total_fraud * FN_COST
    net_savings = no_model - total_cost

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Net Savings vs No Model", f"${net_savings:,.0f}", f"${no_model:,.0f} baseline")
    k2.metric("Fraud Detected", f"{tp:,} / {total_fraud:,}", f"{recall:.1%} recall")
    k3.metric("Fraud Prevented", f"${fraud_prevented:,.0f}", f"{tp:,} transactions blocked")
    k4.metric(
        "Total Operational Cost",
        f"${total_cost:,.0f}",
        f"${total_cost / len(y_filt):.2f} per txn",
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Performance at Current Threshold")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )
        perf_df = pd.DataFrame({
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
        })
        st.table(perf_df)

    with right:
        st.subheader("Risk Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(scores_filt[y_filt == 0], bins=50, alpha=0.6,
                color="#2196F3", label="Legitimate", density=True)
        ax.hist(scores_filt[y_filt == 1], bins=50, alpha=0.6,
                color="#f44336", label="Fraud", density=True)
        ax.axvline(risk_threshold, color="#333", linestyle="--",
                   linewidth=2, label=f"Review ({risk_threshold:.2f})")
        ax.axvline(float(AUTO_BLOCK), color="#c62828", linestyle="--",
                   linewidth=1.5, label=f"Auto-Block ({float(AUTO_BLOCK):.2f})")
        ax.set_xlabel("Fraud Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Distribution of Fraud Scores", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

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
    c3.metric(
        "False Positive Rate",
        f"{fpr:.1%}",
        f"{fp:,} false alarms flagged",
    )

    render_footer()


# =====================================================================
# TAB 2 - Model Comparison
# =====================================================================
with tab2:
    st.header("Model Comparison")
    st.caption(
        "Optimization journey and head-to-head performance comparison "
        "of candidate models evaluated on the held-out test set"
    )

    # ── Optimization Journey ─────────────────────────────────────────
    st.subheader("Optimization Journey")
    journey_df = pd.DataFrame([
        {
            "Stage": "0",
            "Model": "No Model",
            "Optimization": "--",
            "Test Recall": "0.0%",
            "ROC-AUC": "N/A",
            "PR-AUC": "N/A",
            "Total Cost": "$922,528",
            "Selected": "",
        },
        {
            "Stage": "1",
            "Model": "Logistic Regression",
            "Optimization": "Balanced class weights",
            "Test Recall": "94.7%",
            "ROC-AUC": "0.5974",
            "PR-AUC": "0.0486",
            "Total Cost": "$1,111,228",
            "Selected": "",
        },
        {
            "Stage": "2",
            "Model": "XGBoost",
            "Optimization": "Bayesian (Optuna)",
            "Test Recall": "74.3%",
            "ROC-AUC": "0.7168",
            "PR-AUC": "0.0883",
            "Total Cost": "$739,792",
            "Selected": "",
        },
        {
            "Stage": "3",
            "Model": "LightGBM",
            "Optimization": "Bayesian (Optuna)",
            "Test Recall": "72.9%",
            "ROC-AUC": "0.7198",
            "PR-AUC": "0.1125",
            "Total Cost": "$729,596",
            "Selected": "WINNER",
        },
    ])
    st.dataframe(journey_df, use_container_width=True, hide_index=True)
    st.caption(
        "All models evaluated at threshold = 0.410 on 118,108 test transactions "
        "(chronological split, no data leakage). "
        "Note: Logistic Regression costs more than No Model due to high false-positive volume. "
        "LightGBM wins Stage 3: $10,196 lower cost than XGBoost and higher AUC. "
        "Cost minimisation is the primary selection criterion."
    )

    st.markdown("---")

    # ── Pre-computed Performance Curves ──────────────────────────────
    st.subheader("Performance Curves (from Notebook 03)")
    pr_img = MT_FIGURES_PATH / "pr_curve_comparison.png"
    cost_img = MT_FIGURES_PATH / "cost_vs_threshold.png"

    img_l, img_r = st.columns(2)
    with img_l:
        if pr_img.exists():
            st.image(
                Image.open(pr_img),
                caption="Precision-Recall Curve: XGBoost vs LightGBM",
                use_container_width=True,
            )
        else:
            st.info("pr_curve_comparison.png not found. Run notebook 03.")
    with img_r:
        if cost_img.exists():
            st.image(
                Image.open(cost_img),
                caption="Total Cost vs Threshold (LightGBM, winning model)",
                use_container_width=True,
            )
        else:
            st.info("cost_vs_threshold.png not found. Run notebook 03.")

    st.markdown("---")

    # ── Live Confusion Matrix + ROC (winning model at current threshold) ──
    st.subheader("LightGBM Performance at Current Threshold")
    cm = confusion_matrix(y_filt, y_pred_filt)
    tn_v, fp_v, fn_v, tp_v = cm.ravel()

    cm_l, cm_r = st.columns(2)
    with cm_l:
        labels = np.array([
            [f"TN\n{tn_v:,}\n$0", f"FP\n{fp_v:,}\n${fp_v * FP_COST:,.0f}"],
            [f"FN\n{fn_v:,}\n${fn_v * FN_COST:,.0f}", f"TP\n{tp_v:,}\nPrevented"],
        ])
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

    with cm_r:
        fpr_c, tpr_c, _ = roc_curve(y_filt, scores_filt)
        roc_auc_val = roc_auc_score(y_filt, scores_filt)
        tpr_op = recall_score(y_filt, y_pred_filt)
        fpr_op = fp_v / (fp_v + tn_v) if (fp_v + tn_v) > 0 else 0

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr_c, tpr_c, color="#1565C0", linewidth=2,
                label=f"LightGBM (AUC = {roc_auc_val:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        ax.scatter([fpr_op], [tpr_op], color="red", s=100, zorder=5,
                   label=f"Operating Point ({risk_threshold:.2f})")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
        ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Feature Importance ────────────────────────────────────────────
    st.subheader("Feature Importance (SHAP - LightGBM)")
    shap_img = FIGURES_PATH / "shap_feature_importance_bar.png"
    if shap_img.exists():
        st.image(Image.open(shap_img), use_container_width=True)
    else:
        importance = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURES],
            "Importance": importance,
        }).sort_values("Importance", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(imp_df["Feature"], imp_df["Importance"], color="steelblue")
        ax.set_xlabel("Importance (Gain)", fontsize=11)
        ax.set_title("Feature Importance", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Cost-Benefit Table ────────────────────────────────────────────
    st.subheader("Cost-Benefit Analysis by Threshold")
    thresholds = sorted(set([
        0.20, 0.30, round(float(MANUAL_REVIEW), 2),
        0.50, 0.60, 0.70, 0.80, round(float(AUTO_BLOCK), 2)
    ]))
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
        rows.append({
            "Threshold": f"{t:.2f}{marker}",
            "Recall": f"{t_rec:.1%}",
            "Precision": f"{t_pre:.1%}",
            "True Positives": f"{t_tp:,}",
            "False Positives": f"{t_fp:,}",
            "Missed Frauds": f"{t_fn:,}",
            "Total Cost": f"${t_cost:,.0f}",
        })
    st.caption("* = current threshold")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    render_footer()


# =====================================================================
# TAB 3 - Case Study Explorer
# =====================================================================
with tab3:
    st.header("Case Study Explorer")
    st.caption(
        "Detailed SHAP analysis of individual transaction decisions. "
        "Each case shows why the model scored the transaction, with a "
        "waterfall plot and plain-English explanation."
    )

    cases = {
        "Case 1: True Positive -- Clear Fraud Caught": {
            "score": 0.9094, "actual": "FRAUD", "decision": "AUTO-BLOCK",
            "waterfall_file": "shap_waterfall_case1.png",
            "features": {
                "Transaction Amount": "$17.52",
                "Time of Day": "6:00 AM",
                "Transactions (1 hour)": "1",
                "Transactions (24 hours)": "9",
                "Weekend": "Yes",
                "Spending Anomaly": "-0.39 std devs",
                "First Transaction": "No (returning client)",
            },
            "explanation": (
                "This transaction was correctly identified as fraud. Multiple "
                "strong indicators were present: a small amount typical of "
                "card-testing behaviour ($17.52, SHAP +1.60), early-morning "
                "timing (6:00 AM, SHAP +0.97), and a spending anomaly score "
                "below the customer's normal pattern (SHAP +0.17). The model "
                "automatically blocked this transaction, preventing the loss."
            ),
            "drivers_title": "Key Risk Drivers",
            "drivers": [
                "Transaction Amount (+1.60)",
                "Time of Day (+0.97)",
                "Spending Anomaly Score (+0.17)",
            ],
        },
        "Case 2: True Positive -- Velocity-Driven Detection": {
            "score": 0.7332, "actual": "FRAUD", "decision": "MANUAL REVIEW",
            "waterfall_file": "shap_waterfall_case2.png",
            "features": {
                "Transaction Amount": "$59.64",
                "Time of Day": "4:00 AM",
                "Transactions (1 hour)": "1",
                "Transactions (24 hours)": "3",
                "Weekend": "No",
                "Spending Anomaly": "+0.54 std devs",
                "First Transaction": "No (returning client)",
            },
            "explanation": (
                "This fraud was detected primarily through early-morning "
                "timing (4 AM, SHAP +0.86). Transaction velocity in the last "
                "hour (+0.36) and above-average spending deviation (+0.24) "
                "reinforced the fraud signal. Despite these indicators, the "
                "transaction amount of $59.64 partially attenuated the score "
                "(-0.24), resulting in a manual review decision rather than "
                "auto-block."
            ),
            "drivers_title": "Key Risk Drivers",
            "drivers": [
                "Time of Day (+0.86)",
                "Transaction Velocity 1 hour (+0.36)",
                "Spending Anomaly Score (+0.24)",
            ],
        },
        "Case 3: False Negative -- Missed Fraud": {
            "score": 0.0853, "actual": "FRAUD", "decision": "AUTO-APPROVE",
            "waterfall_file": "shap_waterfall_case3.png",
            "features": {
                "Transaction Amount": "$57.95",
                "Time of Day": "Business hours",
                "Transactions (1 hour)": "0",
                "Transactions (24 hours)": "0",
                "Weekend": "No",
                "Spending Anomaly": "-1.50 std devs",
                "First Transaction": "Yes (new client)",
            },
            "explanation": (
                "The model failed to detect this fraud because the complete "
                "absence of recent transaction activity (0 in 24 hours, 0 in "
                "1 hour) was interpreted as low-risk behavior (SHAP -1.91), "
                "consistent with patterns the model learned from legitimate "
                "low-frequency customers, and a spending pattern below the "
                "customer's baseline (SHAP -0.58) further suppressed the "
                "fraud score. This represents a model limitation: fraud "
                "transactions that lack velocity signals and show "
                "below-baseline spending fall outside the model's learned "
                "fraud patterns."
            ),
            "drivers_title": "Key Factors in Missed Detection",
            "drivers": [
                "Zero transaction velocity 24h (SHAP -1.91) -- absence interpreted as legitimate",
                "Below-baseline spending anomaly (SHAP -0.58) -- reinforced legitimacy signal",
                "Zero transaction velocity 1h (SHAP -0.09) -- minor, same direction",
            ],
            "improvement": (
                "Engineer composite features that cross-reference "
                "zero-velocity patterns with new-client flags and spending "
                "anomaly scores. Incorporate existing dataset features "
                "(DeviceInfo, ProductCD, id_01 to id_38) to provide "
                "orthogonal signals that velocity alone cannot capture."
            ),
        },
        "Case 4: False Positive -- Legitimate Flagged": {
            "score": 0.9342, "actual": "LEGITIMATE", "decision": "AUTO-BLOCK",
            "waterfall_file": "shap_waterfall_case4.png",
            "features": {
                "Transaction Amount": "$15.00",
                "Time of Day": "9:00 AM",
                "Transactions (1 hour)": "1",
                "Transactions (24 hours)": "2",
                "Weekend": "Yes",
                "Spending Anomaly": "-0.26 std devs",
                "First Transaction": "No (returning client)",
            },
            "explanation": (
                "This legitimate transaction was incorrectly blocked (False "
                "Positive). The small transaction amount of $15.00 was the "
                "dominant factor (SHAP +1.24), triggering patterns the model "
                "associates with card-testing behavior. A moderate transaction "
                "velocity (+0.41, +0.32) further reinforced the fraud signal. "
                "The combination of multiple moderate-to-strong positive SHAP "
                "values across nearly all features produced a high fraud score "
                "(0.9342), leaving the model with no offsetting signals to "
                "recognize this as legitimate. This resulted in an unnecessary "
                "block and a negative customer experience."
            ),
            "drivers_title": "Key Factors in False Alert",
            "drivers": [
                "Transaction Amount (+1.24) -- dominant false signal",
                "Time of Day (+0.54)",
                "Transaction Velocity 1 hour (+0.41)",
            ],
        },
        "Case 5: Borderline -- Near Review Threshold": {
            "score": 0.3648, "actual": "LEGITIMATE", "decision": "AUTO-APPROVE",
            "waterfall_file": "shap_waterfall_case5.png",
            "features": {
                "Transaction Amount": "$125.00",
                "Time of Day": "Afternoon",
                "Transactions (1 hour)": "0",
                "Transactions (24 hours)": "0",
                "Weekend": "No",
                "Spending Anomaly": "Normal",
                "First Transaction": "No (returning client)",
            },
            "explanation": (
                "This legitimate transaction was correctly approved despite a "
                "moderately elevated fraud signal from the transaction amount "
                "(SHAP +0.61). Afternoon timing (SHAP -0.29), normal spending "
                "patterns (SHAP -0.18), and zero recent transaction velocity "
                "(SHAP -0.08) collectively offset the amount signal, bringing "
                "the final score to 0.3648 -- below the manual review "
                "threshold of 0.41. This case demonstrates the model's "
                "ability to balance competing signals: while $125.00 triggered "
                "some fraud-associated patterns, the overall behavioral "
                "context correctly indicated legitimate activity."
            ),
            "drivers_title": "Key Factors in Correct Approval",
            "drivers": [
                "Time of Day (-0.29) -- afternoon timing consistent with legitimate activity",
                "Spending Anomaly Score (-0.18) -- normal spending pattern",
                "Transaction Velocity 1 hour (-0.08) -- no unusual activity",
            ],
        },
    }

    selected = st.selectbox("Select a case study to analyse:", list(cases.keys()))
    case = cases[selected]

    st.markdown("---")

    m1, m2, m3 = st.columns(3)
    m1.metric("Fraud Score", f"{case['score']:.4f}")
    m2.metric("Model Decision", case["decision"])
    m3.metric("Actual Outcome", case["actual"])

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

    # Individual waterfall plot for the selected case
    st.subheader("SHAP Explanation -- Why the Model Made This Decision")
    wf_file = FIGURES_PATH / case["waterfall_file"]
    if wf_file.exists():
        st.image(
            Image.open(wf_file),
            caption=f"SHAP waterfall: {selected.split(' --')[0]}",
            use_container_width=True,
        )
    else:
        combined = FIGURES_PATH / "shap_waterfall_cases.png"
        if combined.exists():
            st.info(
                f"Individual plot ({case['waterfall_file']}) not found. "
                "Showing combined grid. Re-run notebook 04 cell 13 to generate "
                "individual files."
            )
            st.image(Image.open(combined), use_container_width=True)
        else:
            st.info(
                "SHAP waterfall plots not found. "
                "Run notebook 04_shap_explainability.ipynb first."
            )

    st.subheader("Model Decision Explanation")
    st.write(case["explanation"])

    st.subheader(case.get("drivers_title", "Key Risk Drivers"))
    for d in case["drivers"]:
        st.markdown(f"- {d}")

    if "improvement" in case:
        st.subheader("Recommended Improvement")
        st.write(case["improvement"])

    render_footer()


# =====================================================================
# TAB 4 - Client Risk Profile
# =====================================================================
with tab4:
    st.header("Client Risk Profile")
    st.caption(
        "Per-client transaction history and fraud risk assessment. "
        "A client is identified by the combination of card number, "
        "billing address, and email domain."
    )

    df_scored, client_stats = build_client_data(df_test, fraud_scores)

    # Filter controls
    cf1, cf2 = st.columns([2, 1])
    with cf1:
        min_risk = st.slider(
            "Minimum risk score for client list",
            min_value=0.0, max_value=1.0,
            value=float(round(float(AUTO_BLOCK), 2)),
            step=0.01,
            help=(
                "Show only clients who have at least one transaction "
                "at or above this fraud score."
            ),
            key="client_min_risk",
        )
    with cf2:
        sort_by = st.selectbox(
            "Sort clients by",
            ["Max Score (High to Low)", "Transaction Count", "Fraud Transactions"],
            key="client_sort",
        )

    # Filter and sort
    flagged_clients = client_stats[client_stats['max_score'] >= min_risk].copy()
    if sort_by == "Max Score (High to Low)":
        flagged_clients = flagged_clients.sort_values('max_score', ascending=False)
    elif sort_by == "Transaction Count":
        flagged_clients = flagged_clients.sort_values('txn_count', ascending=False)
    else:
        flagged_clients = flagged_clients.sort_values('fraud_txns', ascending=False)

    if flagged_clients.empty:
        st.warning(
            f"No clients found with max risk score >= {min_risk:.2f}. "
            "Lower the minimum risk score."
        )
    else:
        st.caption(
            f"{len(flagged_clients):,} clients match the current filter "
            f"(out of {len(client_stats):,} total clients in test set)."
        )

        flagged_clients = flagged_clients.reset_index(drop=True)
        flagged_clients['label'] = [
            (
                f"Client #{i+1:04d}  |  "
                f"max score: {row['max_score']:.3f}  |  "
                f"{int(row['txn_count'])} txns  |  "
                f"{int(row['fraud_txns'])} confirmed fraud"
            )
            for i, row in flagged_clients.iterrows()
        ]

        selected_label = st.selectbox(
            "Select a client to investigate:",
            flagged_clients['label'].tolist(),
            key="client_select",
        )
        selected_row = flagged_clients[flagged_clients['label'] == selected_label].iloc[0]
        selected_client_id = selected_row['client_id']

        st.markdown("---")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Transactions", f"{int(selected_row['txn_count']):,}")
        s2.metric("Max Fraud Score", f"{selected_row['max_score']:.4f}")
        s3.metric("Confirmed Fraud Txns", f"{int(selected_row['fraud_txns'])}")
        s4.metric("Total Amount", f"${selected_row['total_amount']:,.2f}")

        max_s = float(selected_row['max_score'])
        if max_s >= float(AUTO_BLOCK):
            risk_label = "HIGH RISK -- Auto-Block triggered"
            risk_color = "#c62828"
        elif max_s >= float(MANUAL_REVIEW):
            risk_label = "MEDIUM RISK -- Manual Review flagged"
            risk_color = "#e65100"
        else:
            risk_label = "LOW RISK"
            risk_color = "#2e7d32"

        st.markdown(
            f"<div style='background:{risk_color};color:white;padding:8px 16px;"
            f"border-radius:6px;font-weight:bold;display:inline-block;margin:8px 0'>"
            f"Overall Risk Level: {risk_label}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.subheader("Transaction History")
        client_txns = df_scored[
            df_scored['client_id'] == selected_client_id
        ].copy().sort_values('TransactionDT')
        client_txns['#'] = range(1, len(client_txns) + 1)
        client_txns['Decision'] = client_txns['fraud_score'].apply(
            lambda s: 'Auto-Block' if s >= float(AUTO_BLOCK)
            else ('Manual Review' if s >= float(MANUAL_REVIEW) else 'Approved')
        )
        client_txns['Actual'] = client_txns['isFraud'].map({1: 'FRAUD', 0: 'Legitimate'})

        st.dataframe(
            client_txns[['#', 'TransactionAmt', 'fraud_score', 'Decision', 'Actual']].rename(
                columns={'TransactionAmt': 'Amount ($)', 'fraud_score': 'Fraud Score'}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Fraud Score by Transaction")
        fig, ax = plt.subplots(figsize=(10, 4))
        bar_colors = [
            '#c62828' if s >= float(AUTO_BLOCK)
            else ('#e65100' if s >= float(MANUAL_REVIEW) else '#1565C0')
            for s in client_txns['fraud_score']
        ]
        ax.bar(client_txns['#'], client_txns['fraud_score'], color=bar_colors)
        ax.axhline(
            y=float(AUTO_BLOCK), color='#c62828', linestyle='--',
            linewidth=1.5, label=f'Auto-Block threshold ({float(AUTO_BLOCK):.2f})',
        )
        ax.axhline(
            y=float(MANUAL_REVIEW), color='#e65100', linestyle='--',
            linewidth=1.5, label=f'Manual Review threshold ({float(MANUAL_REVIEW):.2f})',
        )
        ax.set_xlabel("Transaction #", fontsize=11)
        ax.set_ylabel("Fraud Score", fontsize=11)
        ax.set_title(
            f"Fraud Scores for Selected Client ({len(client_txns)} transactions)",
            fontsize=13, fontweight="bold",
        )
        ax.set_ylim(0, 1.05)
        from matplotlib.patches import Patch
        bar_patches = [
            Patch(facecolor='#c62828', label='Auto-Block (score >= 0.90)'),
            Patch(facecolor='#e65100', label='Manual Review (0.41 - 0.90)'),
            Patch(facecolor='#1565C0', label='Approved (score < 0.41)'),
        ]
        line_handles, line_labels = ax.get_legend_handles_labels()
        ax.legend(handles=bar_patches + line_handles, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            "Note: fraud scores are computed from pre-engineered features "
            "(velocity counts, spending deviation) derived from the full test "
            "dataset. In a production system these features would be computed "
            "in real time from a streaming transaction database."
        )

    render_footer()


# =====================================================================
# TAB 5 - Regulatory Compliance
# =====================================================================
with tab5:
    st.header("Regulatory Compliance", anchor="compliance-top")
    st.caption("Model governance, fair lending review, and audit readiness")

    st.markdown("""
<div class="toc-box">
<strong>Table of Contents</strong><br><br>
&nbsp;&nbsp;1. <a href="#sr-11-7-checklist">SR 11-7 Model Documentation Checklist</a><br>
&nbsp;&nbsp;2. <a href="#fair-lending-review">Fair Lending Considerations</a><br>
&nbsp;&nbsp;3. <a href="#model-governance-framework">Model Governance Framework</a><br>
&nbsp;&nbsp;4. <a href="#right-to-explanation">Right-to-Explanation Capabilities</a><br>
&nbsp;&nbsp;5. <a href="#data-lineage-audit-trail">Data Lineage and Audit Trail</a><br>
&nbsp;&nbsp;6. <a href="#swiss-eu-regulatory-alignment">Swiss & European Regulatory Alignment</a>
</div>
""", unsafe_allow_html=True)

    # ── Section 1: SR 11-7 ───────────────────────────────────────────
    st.subheader("1. SR 11-7 Model Documentation Checklist", anchor="sr-11-7-checklist")

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
        "EU AI Act Art. 43 conformity assessment (required before production deployment)",
        "EU AI Act Art. 51 registration in the EU database for high-risk AI",
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

    st.markdown("[Back to top](#compliance-top)")
    st.markdown("---")

    # ── Section 2: Fair Lending ───────────────────────────────────────
    st.subheader("2. Fair Lending Considerations", anchor="fair-lending-review")

    fl_data = pd.DataFrame([
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
                "Self-norming (deviation from the client's own history). "
                "No cross-client comparison."
            ),
        },
        {
            "Feature": "First-Time Transaction",
            "Risk Level": "MEDIUM",
            "Assessment": (
                "Returning clients (3.67% fraud rate) are riskier than "
                "first-time clients (2.53%). Monitor approval rates for "
                "new customers to avoid disparate impact."
            ),
        },
        {
            "Feature": "Time of Day / Weekend",
            "Risk Level": "MEDIUM",
            "Assessment": (
                "Shift workers and different time zones may be "
                "disproportionately affected. Monitor FPR by region."
            ),
        },
        {
            "Feature": "Transaction Amount",
            "Risk Level": "LOW-MEDIUM",
            "Assessment": (
                "Spending power may correlate with income. "
                "Monitor across customer segments."
            ),
        },
    ])
    st.table(fl_data)
    st.markdown(
        "**Overall Assessment:** No direct protected attributes used. "
        "Conduct disparate impact analysis when demographic data becomes available."
    )

    st.markdown("[Back to top](#compliance-top)")
    st.markdown("---")

    # ── Section 3: Model Governance ──────────────────────────────────
    st.subheader("3. Model Governance Framework", anchor="model-governance-framework")

    gov1, gov2 = st.columns(2)
    with gov1:
        st.markdown("**Model Identification**")
        st.text("Name:     Agent Fraud Sentinel (LightGBM)")
        st.text("Version:  1.0")
        st.text("Type:     Gradient Boosted Decision Tree")
        st.text("Purpose:  Real-time fraud detection")
        st.text("Date:     February 2026")
        st.markdown("")
        st.markdown("**Monitoring Schedule**")
        sched = pd.DataFrame([
            {"Frequency": "Daily",
             "Activity": "Alert volume, auto-block count, queue size"},
            {"Frequency": "Weekly",
             "Activity": "Recall, precision, FPR by risk tier"},
            {"Frequency": "Monthly",
             "Activity": "SHAP drift analysis, feature stability"},
            {"Frequency": "Quarterly",
             "Activity": "Full revalidation, threshold recalibration"},
            {"Frequency": "Annual",
             "Activity": "Comprehensive SR 11-7 / OCC 2011-12 review; FINMA Circular 2023/1 governance attestation"},
        ])
        st.dataframe(sched, use_container_width=True, hide_index=True)

    with gov2:
        st.markdown("**Model Risk Classification**")
        st.text("Recommended Tier:  Tier 2")
        st.text("Rationale:         Material financial impact")
        st.text("Review Cycle:      Quarterly")
        st.markdown("")
        st.markdown("**Key Assumptions**")
        st.markdown(
            "1. Training fraud patterns represent future fraud\n"
            "2. Temporal ordering preserved (no data leakage)\n"
            "3. Client identity: card1 + addr1 + P_emaildomain\n"
            "4. Cost ratio 22.7:1 ($227 FN, $10 FP)\n"
            "5. Minimum 75% recall target"
        )

    st.markdown("[Back to top](#compliance-top)")
    st.markdown("---")

    # ── Section 4: Right to Explanation ──────────────────────────────
    st.subheader("4. Right-to-Explanation Capabilities", anchor="right-to-explanation")

    st.markdown(
        "Customers whose transactions are blocked or flagged may request an "
        "explanation. SHAP values provide a complete, auditable explanation "
        "at the individual transaction level.\n\n"
        "**Applicable Frameworks:**\n\n"
        "- **GDPR Art. 22** -- EU: right to explanation for automated decisions\n"
        "- **nDSG Art. 21** -- Switzerland: equivalent right under Swiss revDSG\n"
        "- **EU AI Act Art. 13** -- Transparency: high-risk AI must allow users "
        "to interpret outputs\n"
        "- **EU AI Act Art. 14** -- Human oversight: intervention and override "
        "must be available at any time\n\n"
        "**For any transaction, the system can generate:**\n\n"
        "1. **Feature-level attribution** -- which factors contributed to "
        "the decision\n"
        "2. **Quantified contribution** -- how much each factor affected "
        "the score\n"
        "3. **Comparison to baseline** -- score relative to average fraud "
        "probability\n\n"
        "**Human Oversight (EU AI Act Art. 14):**\n\n"
        "The multi-threshold architecture directly implements Art. 14 requirements:\n"
        "- Manual review tier (0.41-0.90): every decision reviewed by a human "
        "analyst before action is taken\n"
        "- Auto-block tier (>= 0.90): analyst can override by escalating to "
        "a senior fraud investigator\n"
        "- Override actions are logged in the audit trail\n\n"
        "**Dispute Resolution Workflow:**\n\n"
        "1. Customer contacts bank about a blocked transaction\n"
        "2. Analyst retrieves SHAP explanation from audit log\n"
        "3. Analyst reviews feature contributions in plain English\n"
        "4. If false positive: approve transaction, note for model feedback\n"
        "5. If true fraud: confirm block, initiate investigation"
    )

    st.markdown("[Back to top](#compliance-top)")
    st.markdown("---")

    # ── Section 5: Data Lineage ───────────────────────────────────────
    st.subheader("5. Data Lineage and Audit Trail", anchor="data-lineage-audit-trail")

    st.markdown(
        "**Data Source:** IEEE-CIS Fraud Detection Dataset "
        "(590,540 transactions)\n\n"
        "**Processing Pipeline:**\n\n"
        "1. Raw data ingestion (434 features)\n"
        "2. Feature engineering: 7 behavioural features derived from raw "
        "transaction data\n"
        "3. Temporal split: 60/20/20 (train / validation / test, "
        "chronological order)\n"
        "4. Model training: LightGBM with Bayesian optimization (Optuna)\n"
        "5. Threshold calibration: cost-minimising at $227 FN / $10 FP "
        "with 75% recall floor\n"
        "6. Explainability: SHAP TreeExplainer for all predictions\n\n"
        "**Audit Requirements:**\n\n"
        "- SHAP values stored at scoring time\n"
        "- Retention: minimum 7 years (GDPR / US regulatory standard); "
        "10 years for FINMA/nDSG scope\n"
        "- Log fields: transaction_id, fraud_score, threshold, decision, "
        "SHAP values, model_version, timestamp"
    )

    st.markdown("[Back to top](#compliance-top)")
    st.markdown("---")

    # ── Section 6: Swiss & European Regulatory Alignment ──────────────
    st.subheader(
        "6. Swiss & European Regulatory Alignment",
        anchor="swiss-eu-regulatory-alignment",
    )

    st.markdown(
        "Fraud detection models deployed in Swiss and European banking "
        "environments must satisfy additional regulatory requirements beyond "
        "U.S. SR 11-7 guidance. The table below maps each domestic requirement "
        "to its Swiss/EU equivalent and documents SAFE's current alignment."
    )

    reg_data = pd.DataFrame([
        {
            "U.S. Requirement": "SR 11-7 Model Risk Management",
            "Swiss/EU Equivalent": "FINMA Circular 2023/1 (Operational Risks)",
            "SAFE Status": "Aligned",
            "Notes": (
                "FINMA Circular 2023/1 requires banks to identify, measure, "
                "and control operational risks, including model risk. "
                "SAFE's documentation, validation, and monitoring framework "
                "satisfies both SR 11-7 and FINMA requirements."
            ),
        },
        {
            "U.S. Requirement": "GDPR Article 22 (Automated Decisions)",
            "Swiss/EU Equivalent": "Swiss nDSG Art. 21 (Automated Decisions)",
            "SAFE Status": "Aligned",
            "Notes": (
                "The revised Swiss Federal Act on Data Protection (nDSG, "
                "effective Sep 2023) mirrors GDPR Article 22: data subjects "
                "can request human review of automated decisions with legal "
                "effects. SAFE's SHAP-based right-to-explanation and manual "
                "review tier (0.41-0.90) provide this mechanism."
            ),
        },
        {
            "U.S. Requirement": "Fair Lending (ECOA / Reg B)",
            "Swiss/EU Equivalent": "Swiss Federal Constitution Art. 8 (Equality)",
            "SAFE Status": "Partial",
            "Notes": (
                "Switzerland's constitutional equality guarantee applies to "
                "algorithmic decisions. SAFE uses no protected attributes; "
                "disparate impact testing is pending demographic data "
                "availability."
            ),
        },
        {
            "U.S. Requirement": "N/A (no U.S. equivalent yet)",
            "Swiss/EU Equivalent": "EU AI Act (2024) - High-Risk Classification",
            "SAFE Status": "Aligned",
            "Notes": (
                "The EU AI Act classifies AI systems used in creditworthiness "
                "assessment and fraud detection as high-risk (Annex III). "
                "High-risk systems must provide: transparency documentation, "
                "human oversight, accuracy/robustness metrics, and logging. "
                "SAFE satisfies all four through its SHAP explanations, "
                "manual review tier, documented performance metrics, and "
                "audit trail."
            ),
        },
        {
            "U.S. Requirement": "Basel III / OCC Guidance",
            "Swiss/EU Equivalent": "Basel III/IV via FINMA (Swiss implementation)",
            "SAFE Status": "Aligned",
            "Notes": (
                "Switzerland implements Basel III/IV through FINMA circulars. "
                "SAFE's operational risk framework (model governance, "
                "monitoring schedule, Tier 2 classification) satisfies "
                "Basel capital adequacy requirements for model risk."
            ),
        },
    ])
    st.table(reg_data)

    st.markdown("**Key Takeaways for Swiss Deployment:**")
    st.markdown(
        "1. **FINMA readiness:** SAFE's model governance framework "
        "(Section 3) maps directly to FINMA Circular 2023/1 requirements "
        "for operational risk management in AI/ML models.\n"
        "2. **nDSG compliance:** The right-to-explanation capability "
        "(Section 4) and human-in-the-loop review tier satisfy Swiss data "
        "protection requirements for automated decision-making.\n"
        "3. **EU AI Act preparedness:** As a high-risk AI system under "
        "Annex III, SAFE already implements the required transparency, "
        "oversight, and logging mechanisms.\n"
        "4. **Pending:** Disparate impact testing requires demographic data "
        "not available in the IEEE-CIS dataset."
    )

    st.markdown("[Back to top](#compliance-top)")

    render_footer()
