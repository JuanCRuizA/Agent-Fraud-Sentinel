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
- **Amount deviation**: positive Z-scores show strongest fraud signal — spending *above* client average peaks at 5.2% fraud rate (Z-score 1 to 2), compared to 2.3% for extreme low deviations
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

**Date:** 2026-02-08
**Status:** ✅ Completed (41 cells)
**Location:** `notebooks/modeling/03_model_training.ipynb`
**Objective:** Train and evaluate machine learning models for real-time fraud detection with production-ready threshold optimization.

### Focus Areas
- Baseline model (Logistic Regression) for interpretable benchmark
- Advanced model (XGBoost) with class imbalance handling
- Hyperparameter tuning via simple grid search
- Cost-based threshold optimization
- Constrained optimization with minimum recall requirement (75%)
- Multi-threshold production strategy (auto-block + manual review)
- Confusion matrix visualizations

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Setup & Data Loading | Load train/val/test splits from Phase 2, define 7 engineered features |
| Data Cleaning | Handle infinity/NaN in `amount_deviation` feature before StandardScaler |
| 2. Baseline Model | Logistic Regression with class_weight='balanced', PR-AUC: 0.0821 |
| 3. Advanced Model | XGBoost with scale_pos_weight=28.56, initial PR-AUC: 0.1093 |
| 4. Hyperparameter Tuning | Grid search (6 combinations), best PR-AUC: 0.1098 (max_depth=6, n_estimators=200, lr=0.05) |
| 5. Model Comparison | Fair comparison at threshold 0.5 + cost-based threshold optimization |
| Unconstrained Optimization | Optimal threshold 0.740 (14.4% recall, $328K cost) - too low for production |
| Constrained Optimization | 75% minimum recall at threshold 0.410 ($598K cost) |
| Multi-Threshold Strategy | Auto-block (≥0.9) + Manual review (0.41-0.9) + Auto-approve (<0.41) |
| Test Set Evaluation | Final production performance: 76% recall, $598K cost on test set |
| Confusion Matrices | Two visualizations (numbers + percentages) for production strategy |
| Model Persistence | Save XGBoost model, scaler, and multi-threshold config to `models/` |
| Summary & Next Steps | Key findings, production strategy, Phase 4 roadmap |

### Visualizations (5 total)
1. XGBoost feature importance (bar chart)
2. Precision-Recall curve comparison (Logistic vs XGBoost)
3. Cost vs threshold curve (U-shape optimization)
4. Confusion matrix - numbers (heatmap with absolute counts)
5. Confusion matrix - percentages (heatmap with % of total transactions)

### Key Findings

**Model Selection:**
- **XGBoost outperforms Logistic Regression** by 33.8% (PR-AUC: 0.1098 vs 0.0821)
- Fair comparison at threshold 0.5: XGBoost catches 60.9% of fraud vs 42.8% baseline
- Hyperparameter tuning: best config uses max_depth=6, 200 estimators, learning_rate=0.05

**Threshold Optimization:**
- **Pure cost minimization (threshold 0.740)**: Only 14.4% recall — catches just 508 of 4,064 frauds
  - Unacceptable for production despite $328K cost (lowest)
- **Constrained optimization (threshold 0.410)**: 76% recall with 75% minimum requirement
  - Catches 3,505 frauds but costs $598K (82% increase)
  - Cost per additional fraud caught: $94.99

**Production Strategy (Multi-Threshold):**
- **Auto-block (≥0.90)**: 19 transactions, 6 frauds caught, $65 cost (automated processing)
- **Manual review (0.41-0.90)**: 55,010 transactions, 3,499 frauds caught, $515K cost (human analysts)
- **Auto-approve (<0.41)**: 63,079 transactions, 1,106 frauds missed, $83K FN cost
- **Overall**: 76% recall, 6.4% precision, $598K total cost on validation set
- **Test set performance**: 76% recall confirmed, validates production readiness

### Key Technical Decisions

1. **Data Cleaning**: Replace `inf` with ±10, `NaN` with 0 in amount_deviation feature
2. **Class Imbalance**: scale_pos_weight=28.56 in XGBoost (reflects 3.5% fraud rate)
3. **Evaluation Metric**: PR-AUC preferred over ROC-AUC for imbalanced data
4. **Cost Parameters**: FN=$75 (median fraud), FP=$10 (manual review), ratio 7.5:1
5. **Recall Constraint**: 75% minimum (business requirement overrides pure cost minimization)
6. **Multi-Threshold**: Tiered strategy reduces manual review workload while maintaining recall

### Key Outputs
- `models/xgboost_final.pkl` - Trained XGBoost model (best hyperparameters)
- `models/scaler.pkl` - StandardScaler fitted on training data
- `models/threshold_config.pkl` - Production configuration:
  - `auto_block_threshold`: 0.90 (high confidence fraud)
  - `manual_review_threshold`: 0.410 (75% recall target)
  - `min_recall_target`: 0.75
  - Cost parameters and feature list included

### Issues Encountered & Resolved
- [ISSUE-005] ValueError with StandardScaler (infinity in amount_deviation)
- [ISSUE-006] NameError for variables defined out of order (recall_optimal, recall_test)
- [ISSUE-007] F-string backslash syntax error in dictionary access
- [ISSUE-008] Confusion about comparing models at different thresholds

### Validation Checklist
- [x] Model trains without errors (data cleaning added)
- [x] All cells run sequentially (Run All works)
- [x] Fair model comparison at same threshold (0.5)
- [x] Cost-based threshold optimization implemented
- [x] Constrained optimization with 75% recall constraint
- [x] Multi-threshold strategy evaluated on validation and test sets
- [x] Model artifacts saved with production configuration
- [x] Confusion matrices visualized (numbers + percentages)

### Next Steps
- [ ] Phase 4: SHAP explainability analysis (`04_shap_explainability.ipynb`)
  - Global feature importance (which features drive fraud detection?)
  - Local explanations for high-risk transactions (why was transaction X flagged?)
  - Waterfall plots for individual predictions
- [ ] Deployment considerations:
  - Real-time scoring API (<100ms latency)
  - A/B testing framework vs current production model
  - Model drift monitoring dashboard
  - Fallback to rule-based system if model fails

---
