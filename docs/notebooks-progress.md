# Agent Fraud Sentinel - Notebooks Progress Log

## 01_eda_fraud_patterns.ipynb

**Date:** 2026-02-06 (updated 2026-02-08)
**Status:** ✅ Completed (32 cells)
**Location:** `notebooks/exploratory/01_eda_fraud_patterns.ipynb`
**Objective:** Identify key fraud signals in the IEEE-CIS dataset to guide feature engineering.

### Focus Areas
- Fraud rate and class distribution
- Top features correlated with fraud
- Missing data patterns
- Transaction amount distribution (stakeholder-friendly + log-scale)
- Temporal fraud patterns
- Cost assumptions for modeling

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Data Loading | Loads `train_transaction.csv` and `train_identity.csv`, merges on TransactionID |
| Data Preview | First 15 rows x 15 columns (first 5 cols + top 10 correlated features) |
| 2. Fraud Rate | Fraud statistics: 20,663 fraud (3.50%), class imbalance ratio 1:27 |
| 3. Class Distribution | Bar chart + pie chart visualization |
| 4. Top 10 Correlations | Features ranked by absolute correlation with isFraud (V257 leads at 0.383) |
| 5. Missing Data | 214 features with >50% missing out of 434 total (49.3%) |
| 6. Amount Distribution | Stakeholder-friendly dollar chart ($0-$500, $25 bins) + log-scale + boxplot |
| 7. Temporal Patterns | Fraud peaks: hours 7-9, days Fri/Sat/Sun |
| 8. Summary Table | Key metrics in tabular format |
| 9. Key Findings | Top 10 features saved to `data/processed/top_features.csv` |
| 10. Cost Assumptions | FN cost: $75 (median fraud), FP cost: $10 (review), ratio 7.5:1 |

### Visualizations (7 total)
1. Class distribution (bar + pie)
2. Top 10 correlation horizontal bar chart
3. Missing data (histogram + top 15 bar)
4. Transaction amount in dollars - stakeholder chart (legit vs fraud, $25 bins)
5. Transaction amount (log-scale histogram + boxplot)
6. Temporal patterns (hourly + daily fraud rates)
7. All saved to `data/processed/*.png`

### Key Findings
- **3.50% fraud rate** with 1:27 class imbalance
- **Top features** are all V-features (V257, V246, V244...) with 76-78% missing data
- **Fraud amounts**: median $75 (vs $68.50 legit), broader distribution
- **Temporal signal**: fraud peaks early morning (7-9 AM) and weekends
- **Cost ratio**: missing fraud costs 7.5x more than a false alarm

### Key Outputs
- All plots saved to `data/processed/` for reference
- `data/processed/top_features.csv` - top 10 correlated features
- Notebook ready to run - execute cells in order (top to bottom)

### Completed Steps
- [x] Feature engineering based on top correlations (done in notebook 02)
- [x] Temporal feature extraction (done in notebook 02)
- [x] Cost assumptions documented for Phase 3 modeling

---

## 02_feature_engineering.ipynb

**Date:** 2026-02-08
**Status:** ✅ Completed (37 cells)
**Location:** `notebooks/exploratory/02_feature_engineering.ipynb`
**Objective:** Transform raw transaction and identity data into engineered features for fraud detection modeling.

### Feature Tiers

| Tier | Features | Description |
|------|----------|-------------|
| Tier 1 - Velocity | `txn_count_1hr`, `txn_count_24hr` | Rolling window count of past transactions per client (no leakage) |
| Tier 2 - Behavioral | `amount_deviation`, `is_first_transaction` | Z-score vs client history (expanding + shift), first-time flag |
| Tier 3 - Temporal | `hour_of_day`, `is_weekend` | Time-based features from TransactionDT |
| Tier 4 - Categorical | `amount_bin` | Small (<$50), Medium ($50-$200), Large (>$200) |

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Setup | Imports, paths (`../../data/raw/`) |
| 2. Load & Merge | Left join on TransactionID, memory cleanup |
| 3. client_id | Composite key: `card1 + addr1 + P_emaildomain` (90,375 unique clients) |
| 4. Data Overview | Shape, memory (2.74 GB), client summary table |
| 5. Tier 1 | Velocity features via time-based rolling windows (~2.5 min runtime) |
| 6. Tier 2 | Amount deviation (expanding Z-score) + first transaction flag |
| 7. Tier 3 | Hour of day + weekend flag from TransactionDT |
| 8. Tier 4 | Amount bins (small/medium/large) |
| 9. Correlation Heatmap | Check engineered features for |r| > 0.95 redundancy |
| 10. Leakage Check | 6 automated tests verifying backward-only lookback |
| 11. Train/Val/Test Split | Temporal 60/20/20 split respecting chronological order |
| 12. Save CSVs | `train.csv`, `val.csv`, `test.csv` to `data/processed/` |
| 13. Summary | Final shapes, fraud rates, feature list |
| Validation Checklist | 8-item checklist for quality assurance |

### Visualizations (5 total)
1. Tier 1 signal: Fraud rate vs transaction velocity (1-hour window) - dual axis
2. Tier 2 signal: fraud rate by amount deviation + first vs returning
3. Tier 3 signal: fraud rate by hour (color-coded) + weekday vs weekend
4. Tier 4 signal: fraud rate by amount category (green/yellow/red)
5. Feature correlation heatmap (engineered features)

### Key Findings
- **Velocity signal**: fraud rate jumps from 2.9% (0 prior txns) to 10.8%+ at high velocity (peaking at 11.4% for 6-10 txns/hr)
- **Amount deviation**: positive Z-scores show strongest fraud signal -- spending *above* client average peaks at 5.2% fraud rate (Z-score 1 to 2), compared to 2.3% for extreme low deviations
- **Returning customers**: higher fraud rate (3.67%) than first-time transactions (2.53%), suggesting compromised accounts are a key fraud vector
- **Temporal**: early morning hours and weekends show elevated fraud
- **Large amounts**: highest fraud rate at 4.41% (>$200), followed by small at 3.83% (<$50), with medium lowest at 2.97% ($50-$200)
- **No high correlations**: all engineered features provide independent signal
- **All 6 leakage tests passed**: no future data contamination

### Data Leakage Prevention
- Data sorted by `TransactionDT` before any feature computation
- `rolling('1H').count() - 1` excludes current transaction
- `expanding().shift(1)` excludes current row in Z-score calculation
- `cumcount().eq(0)` for first transaction is inherently backward-looking
- Temporal split ensures train data is chronologically before val/test

### Key Outputs
- `data/processed/train.csv` - 60% of data (earliest transactions)
- `data/processed/val.csv` - 20% of data (middle period)
- `data/processed/test.csv` - 20% of data (most recent)
- Signal confirmation plots saved to `data/processed/`

### Next Steps
- [x] Phase 3: Baseline model + XGBoost in `03_modeling.ipynb` (completed)
- [x] Use cost ratio (7.5:1 FN:FP) for threshold optimization (completed)
- [ ] Consider adding 6hr/7day velocity if model needs improvement (deferred to future)

---

## 03_model_training.ipynb

**Date:** 2026-02-08 (updated 2026-03-21)
**Status:** ✅ Completed (51 cells)
**Location:** `notebooks/modeling/03_model_training.ipynb`
**Objective:** Train and evaluate machine learning models for real-time fraud detection with production-ready threshold optimization.

### Focus Areas
- Baseline model (Logistic Regression) for interpretable benchmark
- Advanced model (XGBoost) with class imbalance handling
- Advanced model (LightGBM) for apples-to-apples gradient boosting comparison
- Hyperparameter tuning: grid search (6 combinations) + Bayesian optimization via Optuna (30 trials each)
- Dynamic winner selection (best PR-AUC on validation set)
- Cost-based threshold optimization
- Constrained optimization with minimum recall requirement (75%)
- Multi-threshold production strategy (auto-block + manual review)
- Confusion matrix visualizations

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Setup & Data Loading | Load train/val/test splits from Phase 2; imports include `lightgbm` and `optuna` |
| 2. Data Preparation | Define 7 engineered features, cost assumptions, handle infinity/NaN in `amount_deviation` |
| 3. Baseline Model: Logistic Regression | class_weight='balanced', PR-AUC: 0.0821 |
| 4. Advanced Model: XGBoost | scale_pos_weight=28.56, initial PR-AUC: 0.1093 |
| 5. Advanced Model: LightGBM | is_unbalance=True, leaf-wise growth, initial PR-AUC comparison |
| 6. Hyperparameter Tuning | 6.1 Grid search (6 combos, best PR-AUC: 0.1098); 6.2 Bayesian optimization via Optuna (30 trials each for XGBoost and LightGBM) |
| 7. Model Comparison & Threshold Selection | 4-model fair comparison at threshold 0.5; dynamic winner selection; cost-based threshold optimization; constrained (75% recall); multi-threshold production strategy; confusion matrices |
| 8. Model Persistence | Save winner model as `best_model_final.pkl` + `xgboost_final.pkl` (backwards compat); scaler; threshold config with `winning_model` field |
| 9. Summary & Next Steps | Key findings, production strategy, Phase 4 roadmap |

### Visualizations (6 total)
1. LightGBM feature importance (bar chart)
2. XGBoost feature importance (bar chart)
3. Precision-Recall curve comparison (Logistic Regression vs winner model)
4. Cost vs threshold curve (U-shape optimization)
5. Confusion matrix - numbers (heatmap with absolute counts)
6. Confusion matrix - percentages (heatmap with % of total transactions)

### Key Findings

**Model Selection (4-model comparison at threshold 0.5):**

| Model | PR-AUC | Precision | Recall | F1-Score |
|-------|--------|-----------|--------|----------|
| Baseline (Logistic) | 0.0821 | 0.0693 | 0.4279 | 0.1193 |
| XGBoost (grid search) | 0.1098 | 0.0813 | 0.6085 | 0.1435 |
| XGBoost (Bayesian) | 0.1116 | 0.0806 | 0.6194 | 0.1426 |
| LightGBM (Bayesian) | 0.1125 | 0.0844 | 0.6229 | 0.1486 |

- **LightGBM (Bayesian) wins** with PR-AUC 0.1125 (37.1% improvement over baseline)
- Bayesian optimization (30 trials) outperforms grid search (6 trials) for both models
- LightGBM leaf-wise growth provides best PR-AUC and highest recall at 0.5 (0.6229)

**Threshold Optimization (based on winning model):**
- **Pure cost minimization**: threshold 0.720, 17.1% recall, $326K cost (unacceptable for production)
- **Constrained optimization (75% recall floor)**: threshold 0.410, actual recall 76.6%, $578K validation cost (+77.2%)
- **Test set (unbiased)**: 73.8% recall, cost $730,482 ($6.18/txn) -- LightGBM Bayesian at threshold 0.410

**Production Strategy (Multi-Threshold):**
- **Auto-block (>=0.90)**: high-confidence fraud, automated processing
- **Manual review (0.410-0.90)**: human analyst queue
- **Auto-approve (<0.410)**: no review needed
- All downstream cells dynamically use winner model probabilities (`final_proba_val`, `final_proba_test`)

### Key Technical Decisions

1. **Data Cleaning**: Replace `inf` with +/-10, `NaN` with 0 in amount_deviation feature
2. **Class Imbalance**: scale_pos_weight=28.56 (XGBoost), is_unbalance=True (LightGBM)
3. **Evaluation Metric**: PR-AUC preferred over ROC-AUC for imbalanced data
4. **Cost Parameters**: FN=$227 (full economic), FP=$10 (manual review), ratio 22.7:1
5. **Recall Constraint**: 75% minimum (business requirement overrides pure cost minimization)
6. **Multi-Threshold**: Tiered strategy reduces manual review workload while maintaining recall
7. **Dynamic Winner**: `final_proba_val`/`final_proba_test` set by winner selection; all downstream cells use these generic names

### Key Outputs
- `models/best_model_final.pkl` - Winning model (LightGBM Bayesian or XGBoost Bayesian, whichever has higher PR-AUC)
- `models/xgboost_final.pkl` - XGBoost grid search model (kept for backwards compatibility with Phase 4/5)
- `models/scaler.pkl` - StandardScaler fitted on training data only
- `models/threshold_config.pkl` - Production configuration:
  - `auto_block_threshold`: 0.90 (high confidence fraud)
  - `manual_review_threshold`: 0.410 (75% recall target)
  - `min_recall_target`: 0.75
  - `winning_model`: name of winning algorithm
  - Cost parameters and feature list included

### Issues Encountered & Resolved
- [ISSUE-005] ValueError with StandardScaler (infinity in amount_deviation)
- [ISSUE-006] NameError for variables defined out of order (recall_optimal, recall_test)
- [ISSUE-007] F-string backslash syntax error in dictionary access
- [ISSUE-008] Confusion about comparing models at different thresholds
- [ISSUE-012] NameError pr_auc_final (forward reference from Section 7 used in Section 6.2)
- [ISSUE-013] SyntaxError f-string unmatched '[' with dict key in Python 3.11
- [ISSUE-014] NameError baseline_precision_05 / final_precision_05 never defined
- [ISSUE-015] Downstream cells hardcoded to XGBoost probabilities after dynamic winner selection

### Validation Checklist
- [x] Model trains without errors (data cleaning added)
- [x] All cells run sequentially (Run All works)
- [x] Fair model comparison at same threshold (0.5) for all 4 models
- [x] Bayesian optimization runs 30 trials per model without errors
- [x] Dynamic winner selection correctly routes probabilities to downstream cells
- [x] Cost-based threshold optimization implemented
- [x] Constrained optimization with 75% recall constraint
- [x] Multi-threshold strategy evaluated on validation and test sets
- [x] Model artifacts saved with production configuration (winner + XGBoost for compat)
- [x] Confusion matrices visualized (numbers + percentages)

### Next Steps
- [x] Phase 4: SHAP explainability analysis (completed)
- [x] Phase 5: Streamlit dashboard (completed)

---

## 04_shap_explainability.ipynb

**Date:** 2026-02-09 (updated 2026-03-21)
**Status:** ✅ Completed (38 cells)
**Location:** `notebooks/modeling/04_shap_explainability.ipynb`
**Objective:** Explain LightGBM fraud predictions so fraud analysts, business stakeholders, and regulators understand *why* the model flags or approves each transaction.

### Focus Areas
- Global feature importance (SHAP summary and bar charts)
- Local transaction-level explanations (waterfall plots, plain English)
- Business insights for fraud operations
- Regulatory compliance documentation (SR 11-7, OCC 2011-12, FINMA 2023/1, nDSG, EU AI Act, fair lending, right-to-explanation)

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Setup & Model Loading | Load LightGBM (`best_model_final.pkl`), scaler, threshold config from Phase 3 |
| 2. Global Explainability | SHAP TreeExplainer on 2,000-sample subset |
| 2.1 Summary Plot | Beeswarm showing per-feature, per-transaction impact |
| 2.2 Feature Importance Bar | Mean |SHAP| ranked bar chart |
| 2.3 Dependence Plots | Top 4 features: value vs SHAP contribution |
| 2.4 Global Insights | Plain-English summary for fraud analysts |
| 3. Local Explainability | SHAP values computed for full test set (118,108 txns) |
| Case Study Selection | 6 representative transactions selected by score and outcome |
| Plain-English Explanations | `explain_transaction()` helper generates analyst-friendly text |
| 3.1 Waterfall Plots | 3x2 grid of waterfall plots for all 6 cases |
| 4. Business Insights | Fraud vs legitimate SHAP comparison + risk tier decomposition |
| Actionable Insights | 5 operational recommendations for fraud teams |
| 5. Regulatory Compliance | SR 11-7 documentation, fair lending review, audit trail |
| 5.1 Model Documentation | Full model card (inputs, outputs, assumptions, performance); all applicable frameworks (SR 11-7, OCC 2011-12, FINMA 2023/1) |
| 5.2 Fair Lending | Feature-by-feature protected attribute assessment |
| 5.3 Right to Explanation | GDPR Art. 22, nDSG Art. 21, EU AI Act Art. 13/14; human oversight paragraph; dispute resolution workflow; 7/10-year retention |
| 5.4 Governance Summary | Checklist (8 done, 7 pending), monitoring schedule (incl. EU AI Act Art. 43/51) |

### Visualizations (6 total, saved to `figures/shap/`)

| File | Description |
|------|-------------|
| `shap_summary_beeswarm.png` | Overall feature impact (each dot = 1 transaction) |
| `shap_feature_importance_bar.png` | Ranked mean |SHAP| bar chart |
| `shap_dependence_top4.png` | Dependence plots for top 4 features |
| `shap_waterfall_cases.png` | Waterfall plots for 6 case studies (3x2 grid) |
| `shap_fraud_vs_legit.png` | Grouped bar: fraud vs legitimate SHAP comparison |
| `shap_risk_tiers.png` | Feature contribution by risk tier (auto-approve / review / block) |

### 6 Case Studies

| Case | Type | Score | Actual | Key Insight |
|------|------|-------|--------|-------------|
| 1 | True Positive (clear) | 0.9094 | Fraud | Multiple strong indicators, auto-blocked |
| 2 | True Positive (velocity) | 0.7332 | Fraud | Velocity features drove detection |
| 3 | False Negative | 0.0853 | Fraud | All features appeared normal -- model limitation |
| 4 | False Positive | 0.9342 | Legit | Card-testing pattern on legitimate purchase |
| 5 | Auto-Block candidate | 0.9342 | Legit | High-confidence false alarm |
| 6 | Borderline | 0.3648 | Legit | Near threshold, demonstrates sensitivity |

### Key Findings
- **Transaction amount** and **24-hour velocity** are the strongest fraud signals (highest mean |SHAP|)
- **Spending anomaly score** and **time of day** provide strong secondary signals
- **First-time transactions** carry higher uncertainty but lower overall importance
- High-value SHAP contributions are concentrated in velocity + amount features for auto-block tier
- Missed frauds (FN) consistently show zero velocity and normal amounts -- model limitation

### Regulatory Documentation Completed
- [x] SR 11-7 model documentation (purpose, inputs, outputs, assumptions, limitations)
- [x] Fair lending feature review (5 features assessed, risk levels assigned)
- [x] Right-to-explanation capability demonstrated (SHAP waterfall + plain English)
- [x] Audit trail requirements specified (7-year retention GDPR/US; 10-year for FINMA/nDSG scope, per-transaction SHAP logging)
- [x] Model governance checklist (8/15 items complete, 7 pending for production)
- [x] Monitoring schedule (daily/weekly/monthly/quarterly/annual cadence)

### Next Steps
- [x] Phase 5: Streamlit dashboard (completed)

---

## 05_streamlit_dashboard.ipynb

**Date:** 2026-02-09 (updated 2026-03-21)
**Status:** ✅ Completed (18 cells) + deployed to Streamlit Cloud
**Location:** `notebooks/dashboard/05_streamlit_dashboard.ipynb`
**Objective:** Build a professional, interactive dashboard for fraud detection analytics, model explainability, and regulatory compliance -- targeting a Data Scientist portfolio for banking roles.

### Dashboard Architecture

```
+------------------+---------------------------------------------+
| SIDEBAR          | MAIN AREA (st.tabs horizontal navigation)   |
|                  |                                             |
| SAFE             | [Tab 1] Executive Summary                   |
| System for Anti- |   - KPI cards, risk distribution, costs     |
| Fraud Evaluation |                                             |
|                  | [Tab 2] Model Comparison                    |
| Threshold slider |   - Journey table, PR/ROC, confusion matrix |
| Sample size      |                                             |
| Export button    | [Tab 3] Case Study Explorer                 |
|                  |   - 5 cases with individual SHAP waterfalls |
| About & Methods  |                                             |
| (expandable)     | [Tab 4] Client Risk Profile                 |
|                  |   - Flagged watchlist, transaction history  |
|                  |                                             |
|                  | [Tab 5] Regulatory Compliance               |
|                  |   - SR 11-7, fair lending, Swiss/EU regs    |
|                  |                                             |
|                  | [FOOTER on every tab]                       |
+------------------+---------------------------------------------+
```

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Setup & Dependencies | Verify packages (lightgbm) and model artifacts (best_model_final.pkl) |
| 2. Dashboard Architecture | Layout diagram, file dependencies, design principles |
| 3. Streamlit Application | `%%writefile dashboard_app.py` -- complete app (~1,270 lines) |
| 4. Tab Design Documentation | Design rationale for all 5 tabs + code cell parsing journey data from app |
| 5. Deployment | `requirements.txt` (lightgbm), local run instructions, Streamlit Cloud notes |
| Summary | Features table, interactive controls, production readiness checklist |

### Tab Content

**Tab 1 -- Executive Summary:**
- 4 KPI cards: Net Savings vs No Model, Fraud Detected, Fraud Prevented ($), Total Operational Cost
- Performance table (recall, precision, F1, FPR, threshold)
- Risk score distribution histogram (fraud vs legitimate with threshold line)
- Cost analysis: missed fraud cost ($227/FN), false alarm cost ($10/FP), savings vs no-model baseline

**Tab 2 -- Model Comparison:**
- Optimization journey table (4-stage: No Model -> LightGBM winner)
- PR curve comparison (pre-computed image) + cost vs threshold image
- Live confusion matrix with cost overlay ($227 FN, $10 FP), updates with threshold slider
- Live ROC curve with operating point at current threshold
- SHAP feature importance bar chart (from Phase 4)
- Cost-benefit analysis table across 8 threshold values

**Tab 3 -- Case Study Explorer:** (updated 2026-03-13)
- Dropdown to select from 5 case studies (TP clear, TP velocity, FN missed, FP false alarm, Borderline)
- Transaction features in human-readable format
- Individual SHAP waterfall plot per case (fallback to 6-panel grid)
- SHAP-aligned model decision explanations with actual SHAP values cited per feature
- Dynamic section titles per case type: "Key Risk Drivers" (TP), "Key Factors in Missed Detection" (FN), "Key Factors in False Alert" (FP), "Key Factors in Correct Approval" (TN)
- Driver bullet points ordered by absolute SHAP magnitude with SHAP values included

**Tab 4 -- Client Risk Profile:**
- Minimum risk score filter + sort-by selector
- Flagged client watchlist with risk level badge (auto-block / review / low)
- Client summary: total txns, max score, confirmed fraud count, total amount
- Transaction history table + fraud score bar chart

**Tab 5 -- Regulatory Compliance:**
- HTML table of contents with anchor links
- SR 11-7 checklist (8 completed, 6 pending)
- Fair lending review (5 features assessed)
- Model governance framework (identification, risk tier, monitoring schedule)
- Right-to-explanation and dispute resolution workflow
- Data lineage and audit trail documentation
- Swiss/EU regulatory alignment (FINMA Circular 2023/1, nDSG Art. 21, EU AI Act Art. 13/14/43/51)

### Interactive Controls

| Control | Location | Effect |
|---------|----------|--------|
| Risk Threshold slider | Sidebar | Updates KPIs, confusion matrix, cost analysis |
| Sample Size selector | Sidebar | Subsamples test data for faster exploration |
| Export button | Sidebar | Downloads flagged transactions as CSV |
| Case Study dropdown | Tab 3 | Selects individual transaction for detailed SHAP analysis |
| Client filter + sort | Tab 4 | Filters and sorts the flagged account watchlist |

### Artifacts Produced

| File | Description |
|------|-------------|
| `notebooks/dashboard/dashboard_app.py` | Standalone Streamlit application (~1,270 lines) |
| `notebooks/dashboard/requirements.txt` | Python dependencies for deployment |
| `notebooks/dashboard/test_dashboard.csv` | Slim test data (8 columns, 8.3 MB) for Streamlit Cloud |

### Deployment
- **Local:** `cd notebooks/dashboard && streamlit run dashboard_app.py`
- **Cloud:** Deployed at `bankingantifraudsystem.streamlit.app`
- **Repository:** `JuanCRuizA/Agent-Fraud-Sentinel` (main branch)

### Design Principles
- Professional banking aesthetic (blues, grays, no emojis in main content)
- Custom CSS for banking color scheme
- `@st.cache_resource` for model, `@st.cache_data` for data and predictions
- Reusable `render_footer()` function called at the bottom of every tab
- All metrics use actual model outputs (no placeholder data)

### Issues Encountered & Resolved
- [ISSUE-009] `use_container_width` deprecation warning -- replaced with `width="stretch"`
- [ISSUE-010] Streamlit Cloud FileNotFoundError -- model and data files excluded by `.gitignore`
- [ISSUE-011] Test CSV too large for GitHub (145 MB) -- created slim 8.3 MB version
- [ISSUE-016] Tab 3 narratives contradicted SHAP plots -- rewrote all 5 cases with SHAP-aligned text and dynamic section titles
- [ISSUE-017] KPI card 1 duplicated Fraud Prevented ($680K) instead of Net Savings ($192K) -- fixed formula to `no_model - total_cost`

---
