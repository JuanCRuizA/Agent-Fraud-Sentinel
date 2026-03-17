# SAFE - System for Anti-Fraud Evaluation

An end-to-end fraud detection system built on the **IEEE-CIS dataset** (590,540 transactions, 3.50% fraud rate, 1:27 class imbalance), featuring cost-sensitive LightGBM modeling with Bayesian optimization, SHAP explainability for regulatory compliance (SR 11-7, FINMA, EU AI Act), and a deployed interactive dashboard.

**[Live Dashboard](https://bankingantifraudsystem.streamlit.app)**

---

## Key Results

| Metric | Value |
|--------|-------|
| Model | LightGBM with Bayesian optimization (Optuna, 30 trials) |
| Recall | 73.8% (catches 3 out of 4 frauds) |
| PR-AUC | 0.1125 (37.0% improvement over Logistic Regression baseline) |
| Cost Ratio | FN=$227 (full economic cost) vs FP=$10 (manual review) = 22.7:1 |
| Total Cost | $730,482 on test set ($192K saved vs no model) |

### Production Strategy (Multi-Threshold)

| Tier | Threshold | Action |
|------|-----------|--------|
| Auto-block | Score >= 0.90 | Instant fraud block (automated) |
| Manual review | 0.42 <= Score < 0.90 | Routed to human analyst |
| Auto-approve | Score < 0.42 | No action needed |

---

## Project Pipeline

| Phase | Notebook | Description |
|-------|----------|-------------|
| 1. EDA | `01_eda_fraud_patterns.ipynb` | Fraud pattern analysis, class imbalance, temporal signals, cost assumptions |
| 2. Feature Engineering | `02_feature_engineering.ipynb` | 7 leakage-free features across 4 tiers (velocity, behavioral, temporal, categorical) |
| 3. Model Training | `03_model_training.ipynb` | Logistic Regression baseline, XGBoost and LightGBM with Bayesian optimization, cost-sensitive threshold optimization |
| 4. Explainability | `04_shap_explainability.ipynb` | SHAP TreeExplainer, 5 case studies, SR 11-7 regulatory documentation |
| 5. Dashboard | `05_streamlit_dashboard.ipynb` | Interactive Streamlit app with 5 tabs (includes FINMA, nDSG, EU AI Act alignment), deployed to Streamlit Cloud |

---

## Engineered Features

| Feature | Type | Signal |
|---------|------|--------|
| `txn_count_1hr` | Velocity | Fraud rate jumps from 2.9% to 11.4% at high velocity |
| `txn_count_24hr` | Velocity | 24-hour transaction frequency per client |
| `amount_deviation` | Behavioral | Z-score vs client spending history |
| `is_first_transaction` | Behavioral | First-time vs returning customer flag |
| `hour_of_day` | Temporal | Fraud peaks at hours 7-9 AM |
| `is_weekend` | Temporal | Elevated fraud rate on weekends |
| `TransactionAmt` | Raw | Transaction amount in dollars |

All features use backward-only lookback windows to prevent data leakage. Six automated leakage tests confirm correctness.

---

## Dashboard

The Streamlit dashboard provides five tabs:

- **Executive Summary** -- KPI cards, risk score distribution, cost analysis
- **Model Comparison** -- Optimization journey, confusion matrix, ROC/PR curves, feature importance, cost-benefit table
- **Case Study Explorer** -- 5 representative transactions with individual SHAP waterfall plots and plain-English explanations
- **Client Risk Profile** -- Flagged client watchlist, per-client transaction history, fraud score visualization
- **Regulatory Compliance** -- SR 11-7 checklist, fair lending review, model governance, right-to-explanation, audit trail, Swiss/EU regulatory alignment (FINMA, nDSG, EU AI Act)

Interactive controls: threshold slider (updates all metrics in real time), sample size selector, CSV export, case study dropdown, client filter and sort.

---

## Run Locally

```bash
# Clone the repository
git clone https://github.com/JuanCRuizA/Agent-Fraud-Sentinel.git
cd Agent-Fraud-Sentinel

# Install dependencies
pip install -r notebooks/dashboard/requirements.txt

# Run the dashboard
cd notebooks/dashboard
streamlit run dashboard_app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Project Structure

```
agent-fraud-sentinel/
├── data/
│   ├── raw/                          # IEEE-CIS dataset (not tracked)
│   └── processed/                    # train/val/test splits (not tracked)
├── models/
│   ├── best_model_final.pkl          # Winning model (LightGBM, Bayesian)
│   ├── xgboost_final.pkl             # XGBoost model (backwards compatibility)
│   ├── scaler.pkl                    # StandardScaler
│   └── threshold_config.pkl          # Production threshold configuration
├── notebooks/
│   ├── exploratory/                  # Phase 1-2: EDA and feature engineering
│   ├── modeling/                     # Phase 3-4: Training and SHAP explainability
│   └── dashboard/                    # Phase 5: Streamlit app and deployment
├── figures/
│   ├── shap/                         # 11 SHAP explainability figures
│   └── model_training/              # 5 model training figures (PR curve, cost, confusion matrices)
├── docs/                             # Technical documentation
│   ├── technical-decisions.md        # 17 architectural decisions with rationale
│   ├── notebooks-progress.md         # Development progress per notebook
│   ├── issues-solutions.md           # 16 issues documented with root cause analysis
│   └── deployment-guide.md           # Deployment options and procedures
├── LICENSE                           # MIT License
└── requirements.txt                  # Full environment dependencies
```

---

## Tech Stack

- **Python 3.10** -- Core language
- **LightGBM** -- Gradient boosted trees for fraud classification (winning model)
- **Optuna** -- Bayesian hyperparameter optimization (30 trials)
- **SHAP** -- Model explainability (TreeExplainer)
- **scikit-learn** -- Preprocessing, metrics, baseline model
- **Streamlit** -- Interactive dashboard framework
- **pandas / NumPy** -- Data processing
- **matplotlib / seaborn** -- Visualizations

---

## Documentation

Detailed technical documentation is available in the [`docs/`](docs/) directory:

- [**technical-decisions.md**](docs/technical-decisions.md) -- 17 decisions with context, rationale, and alternatives
- [**notebooks-progress.md**](docs/notebooks-progress.md) -- Cell-by-cell progress for all 5 notebooks
- [**issues-solutions.md**](docs/issues-solutions.md) -- 16 issues with root cause analysis and prevention
- [**deployment-guide.md**](docs/deployment-guide.md) -- Streamlit Cloud, Docker, and enterprise deployment options

---

## Author

**Juan Carlos Ruiz Arteaga**
MSc in Data Science & AI, University of Liverpool
Contact: j.ruiz-arteaga@liverpool.ac.uk

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
