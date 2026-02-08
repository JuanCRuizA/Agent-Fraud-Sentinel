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
- [ ] Phase 3: Baseline model + XGBoost in `03_modeling.ipynb`
- [ ] Use cost ratio (7.5:1 FN:FP) for threshold optimization
- [ ] Consider adding 6hr/7day velocity if model needs improvement

---
