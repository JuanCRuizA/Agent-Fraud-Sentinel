# Agent Fraud Sentinel - Notebooks Progress Log

## 01_eda_fraud_patterns.ipynb

**Date:** 2026-02-06  
**Status:** ✅ Completed (~280 lines)  
**Objective:** Identify key fraud signals in the IEEE-CIS dataset to guide feature engineering.

### Focus Areas
- Fraud rate and class distribution
- Top features correlated with fraud
- Missing data patterns
- Transaction amount distribution
- Temporal fraud patterns

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Data Loading | Loads `train_transaction.csv` and `train_identity.csv`, merges on TransactionID |
| 2. Fraud Rate | Calculates fraud statistics and class imbalance ratio |
| 3. Class Distribution | Bar chart + pie chart visualization |
| 4. Top 10 Correlations | Features ranked by absolute correlation with isFraud |
| 5. Missing Data | Features with >50% missing, histogram + bar chart |
| 6. Amount Distribution | Log-scale histogram + box plot by fraud status |
| 7. Temporal Patterns | Fraud rate by hour and day of week |
| 8. Summary Table | Key metrics in tabular format |
| 9. Feature Engineering List | Top 10 features saved to `data/processed/top_features.csv` |

### Visualizations (5 total)
1. Class distribution (bar + pie)
2. Top 10 correlation bar chart
3. Missing data (histogram + bar)
4. Transaction amount (histogram + boxplot)
5. Temporal patterns (hourly + daily)

### Key Outputs
- All plots saved to `data/processed/` for reference
- Notebook ready to run - execute cells in order

### Next Steps
- [ ] Feature engineering based on top correlations
- [ ] Handle missing data strategy
- [ ] Temporal feature extraction

---