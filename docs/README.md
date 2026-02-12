# Agent Fraud Sentinel - Documentation Index

## Project Status

| Phase | Notebook | Location | Status |
|-------|----------|----------|--------|
| Phase 1 - EDA | `01_eda_fraud_patterns.ipynb` | `notebooks/exploratory/` | Completed (32 cells) |
| Phase 2 - Feature Engineering | `02_feature_engineering.ipynb` | `notebooks/exploratory/` | Completed (37 cells) |
| Phase 3 - Model Training | `03_model_training.ipynb` | `notebooks/modeling/` | Completed (41 cells) |
| Phase 4 - SHAP Explainability | `04_shap_explainability.ipynb` | `notebooks/modeling/` | Completed (38 cells) |
| Phase 5 - Streamlit Dashboard | `05_streamlit_dashboard.ipynb` | `notebooks/dashboard/` | Completed (17 cells) |

**Live Dashboard:** [bankingantifraudsystem.streamlit.app](https://bankingantifraudsystem.streamlit.app)

### Key Results

**Dataset:**
- 590,540 transactions | 3.50% fraud rate | 1:27 class imbalance
- 7 engineered features across 4 tiers (all with confirmed fraud signal)
- Temporal train/val/test split (60/20/20) saved to `data/processed/`

**Model Performance (XGBoost on test set):**
- 76% recall (catches 3 out of 4 frauds) with 75% minimum recall constraint
- PR-AUC: 0.1098 (33.8% improvement over Logistic Regression baseline)
- Cost ratio: 7.5:1 (FN=$75 missed fraud vs FP=$10 manual review)

**Production Strategy (Multi-Threshold):**
- Auto-block: score >= 0.90 (instant fraud block)
- Manual review: score 0.41 - 0.90 (human analyst review)
- Auto-approve: score < 0.41 (no action needed)

---

## Documentation Files

### [notebooks-progress.md](./notebooks-progress.md)
Development progress for each Jupyter notebook:
- Notebook structure, sections, and cell counts
- Key findings and statistical results
- Visualizations produced (17 total across all phases)
- 5 notebooks documented

### [technical-decisions.md](./technical-decisions.md)
Key technical decisions with context and rationale:
- Alternatives considered and trade-offs
- 13 decisions documented (DECISION-001 through DECISION-013)
- Covers: dataset selection, feature engineering, model selection, threshold optimization, explainability, deployment

### [issues-solutions.md](./issues-solutions.md)
Technical problems encountered and solutions:
- 11 issues documented and resolved (ISSUE-001 through ISSUE-011)
- Root cause analysis for each
- Prevention strategies for the future

### [deployment-guide.md](./deployment-guide.md)
Deployment options and operational procedures:
- Streamlit Cloud deployment (current, live)
- Local Docker deployment
- Enterprise cloud considerations (AWS ECS, GCP Cloud Run)
- Testing, security, and monitoring

---

## Project Structure

```
agent-fraud-sentinel/
├── data/
│   ├── raw/                    # IEEE-CIS dataset (not tracked)
│   └── processed/              # train.csv, val.csv, test.csv (not tracked)
├── models/
│   ├── xgboost_final.pkl       # Trained XGBoost model
│   ├── scaler.pkl              # StandardScaler
│   └── threshold_config.pkl    # Production threshold configuration
├── notebooks/
│   ├── exploratory/            # Phase 1-2 notebooks
│   ├── modeling/               # Phase 3-4 notebooks
│   └── dashboard/              # Phase 5 notebook + Streamlit app
│       ├── dashboard_app.py    # Standalone Streamlit application
│       ├── requirements.txt    # Python dependencies
│       └── test_dashboard.csv  # Slim test data for deployment
├── figures/
│   └── shap/                   # 6 SHAP explainability figures
└── docs/                       # This documentation directory
```

---

## How to Use These Docs

### When working on a notebook:
1. Update `notebooks-progress.md` with your progress
2. Commit and push changes

### When making a technical decision:
1. Add entry to `technical-decisions.md`
2. Use next DECISION-XXX number
3. Fill all sections (especially rationale!)

### When hitting a problem:
1. Document in `issues-solutions.md`
2. Update status as you investigate
3. Record solution when resolved

### When deploying:
1. Follow steps in `deployment-guide.md`
2. Document any new issues in `issues-solutions.md`
