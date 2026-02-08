# Agent Fraud Sentinel - Technical Decisions Log

## Purpose
Document key technical decisions, rationale, and alternatives considered during development.

---

## QUICK REFERENCE

### Active Decisions
- [DECISION-001] IEEE-CIS Dataset Selection
- [DECISION-002] EDA Before Feature Engineering
- [DECISION-003] Composite client_id for Entity Resolution
- [DECISION-004] Leakage-Free Feature Engineering
- [DECISION-005] Temporal Train/Val/Test Split
- [DECISION-006] Asymmetric Cost Assumptions
- [DECISION-007] XGBoost Over Logistic Regression
- [DECISION-008] Constrained Optimization with 75% Minimum Recall
- [DECISION-009] Multi-Threshold Production Strategy

### Pending Review
- None

### Rejected
- None

- - -

## Decision Template

### [DECISION-XXX] Title
**Date:** YYYY-MM-DD  
**Status:** ✅ Implemented | 🔄 In Progress | ❌ Rejected | 🤔 Under Review  
**Context:** What problem are we solving?  
**Decision:** What did we decide?  
**Rationale:** Why this approach?  
**Alternatives Considered:** What else did we evaluate?  
**Consequences:** Trade-offs and implications  
**Related:** Links to notebooks, code, or other decisions

---

## Decisions

### [DECISION-001] IEEE-CIS Dataset Selection
**Date:** 2026-02-06  
**Status:** ✅ Implemented  
**Context:** Need a fraud detection dataset for the Agent Fraud Sentinel project that demonstrates real-world banking scenarios.
**Decision:** Use IEEE-CIS Fraud Detection dataset from Kaggle.
**Rationale:**
- 590,540 transactions with realistic fraud patterns
- Rich feature set (434 features) including transaction details, identity info, and device data
- Industry-standard benchmark dataset
- Highly imbalanced (~3.5% fraud rate) mirrors real banking scenarios
- Temporal component allows fraud pattern analysis over time
**Alternatives Considered:**
- Credit Card Fraud Detection (European dataset): Too small, anonymized features
- Home Credit Default Risk: More about credit risk than fraud detection
- Synthetic data: Less credible for portfolio demonstration
**Consequences:**
- Large dataset requires efficient processing strategies
- Class imbalance needs careful handling (SMOTE, class weights, threshold tuning)
- Multiple files need merging (transaction + identity)
**Related:** `notebooks/01_eda_fraud_patterns.ipynb`

---

### [DECISION-002] EDA Before Feature Engineering
**Date:** 2026-02-06  
**Status:** ✅ Implemented  
**Context:** Need to understand fraud patterns before building features.
**Decision:** Complete comprehensive EDA notebook analyzing fraud signals, correlations, missing data, and temporal patterns.
**Rationale:**
- Data-driven feature engineering based on actual fraud correlations
- Identify which features matter most (top 10 correlations guide feature selection)
- Understand missing data patterns to inform imputation strategy
- Temporal analysis reveals time-based fraud patterns
**Alternatives Considered:**
- Jump directly to modeling: Would miss important insights
- Minimal EDA: Wouldn't identify key fraud signals
**Consequences:**
- Adds one notebook to pipeline but significantly improves feature quality
- Top 10 features saved to `data/processed/top_features.csv` for next stage
- Clear understanding of 3.5% fraud rate informs threshold selection later
**Related:** `notebooks/01_eda_fraud_patterns.ipynb`

---

### [DECISION-003] Composite client_id for Entity Resolution
**Date:** 2026-02-08
**Status:** ✅ Implemented
**Context:** The IEEE-CIS dataset has no explicit customer identifier. Velocity and behavioral features require grouping transactions by customer.
**Decision:** Create composite `client_id` from `card1 + addr1 + P_emaildomain`.
**Rationale:**
- `card1` (card number hash) is the strongest proxy for cardholder identity
- Adding `addr1` (billing address) and `P_emaildomain` further disambiguates shared cards
- Produces 90,375 unique clients from 590,540 transactions (avg 6.5 txns/client)
- NaN handling: `addr1` filled with -1, `P_emaildomain` filled with 'unknown'
**Alternatives Considered:**
- `card1` alone: Too coarse — same card number used at different addresses could be different users
- `card1 + card2 + addr1 + addr2`: Too fine-grained, many NaN combinations create fragmented groups
- `TransactionID`-only analysis: Loses all behavioral/velocity patterns per customer
**Consequences:**
- Enables per-client velocity features (txn_count_1hr, txn_count_24hr)
- Enables behavioral anomaly features (amount_deviation, is_first_transaction)
- Composite key is imperfect (real banks use actual customer IDs), but best available proxy
- 5,126 clients (5.67%) have at least one fraud transaction
**Related:** `notebooks/exploratory/02_feature_engineering.ipynb` (Section 3)

---

### [DECISION-004] Leakage-Free Feature Engineering with Backward-Only Windows
**Date:** 2026-02-08
**Status:** ✅ Implemented
**Context:** Feature engineering for fraud detection must strictly avoid data leakage — no feature at time T should use information from time T+1 or later, as this would inflate model performance unrealistically.
**Decision:** Implement all features with backward-only lookback windows:
- Velocity: `rolling('1H').count() - 1` (subtract 1 to exclude current row)
- Amount deviation: `expanding().mean().shift(1)` (shift excludes current row)
- First transaction: `cumcount().eq(0)` (inherently backward-looking)
**Rationale:**
- In production, a fraud model scores transactions in real-time — it cannot see future data
- `rolling()` in pandas looks backward by default but includes the current row, hence the `- 1`
- `expanding().shift(1)` computes statistics over all prior rows, then shifts to exclude current
- 6 automated leakage tests confirm correctness (all passed)
**Alternatives Considered:**
- `groupby().transform()` with static aggregations: Leaks future data (uses full group)
- `apply()` with manual loops: Correct but extremely slow on 590K rows
- Pre-computed lookback tables: More complex, same result
**Consequences:**
- Velocity computation takes ~2.5 minutes (acceptable for 590K rows x 90K groups)
- First transaction for any client always has velocity=0 and amount_deviation=0 (by design)
- Model performance in validation will realistically reflect production performance
**Related:** `notebooks/exploratory/02_feature_engineering.ipynb` (Sections 5-6, 10)

---

### [DECISION-005] Temporal Train/Val/Test Split Over Random Stratified Split
**Date:** 2026-02-08
**Status:** ✅ Implemented
**Context:** Need to split data for model training, validation, and testing. Fraud detection operates on time-ordered data — the model must predict future fraud based on past patterns.
**Decision:** Temporal 60/20/20 split: sort by `TransactionDT`, then take first 60% as train, next 20% as validation, last 20% as test.
**Rationale:**
- Mirrors production deployment: model trained on past data, evaluated on future data
- Prevents temporal leakage (training on data from the same period as test data)
- Real fraud patterns evolve over time — temporal split tests for concept drift
- Fraud rates across splits will naturally vary, which is more realistic than forced stratification
**Alternatives Considered:**
- Random stratified split (sklearn `train_test_split`): Shuffles time order, creates leakage
- K-fold cross-validation: Not suitable for time series data without modification
- Time-series cross-validation (expanding window): Better but adds complexity for initial baseline
**Consequences:**
- Fraud rates may differ across splits (expected behavior with temporal data)
- More conservative performance estimates compared to random split (which is a feature, not a bug)
- If fraud rate drifts significantly between splits, may need to investigate temporal concept drift
**Related:** `notebooks/exploratory/02_feature_engineering.ipynb` (Section 11)

---

### [DECISION-006] Asymmetric Cost Assumptions for Model Evaluation
**Date:** 2026-02-08
**Status:** ✅ Implemented
**Context:** In banking fraud detection, False Negatives (missed fraud) and False Positives (false alarms) have very different business costs. Standard metrics like accuracy fail to capture this asymmetry.
**Decision:** Define cost parameters derived from EDA:
- **False Negative cost: $75.00** (median fraud transaction amount)
- **False Positive cost: $10.00** (industry benchmark for manual review)
- **Cost ratio: 7.5:1** (missing fraud is 7.5x more costly than a false alarm)
**Rationale:**
- Median ($75) preferred over mean ($149.24) because the mean is inflated by outliers (max $5,191)
- $10 manual review cost is a widely used industry benchmark (analyst time + customer friction)
- The 7.5:1 ratio will directly inform `scale_pos_weight` in XGBoost and threshold optimization
- Documented in EDA notebook for traceability — not a "magic number" pulled from thin air
**Alternatives Considered:**
- Mean fraud amount ($149.24): Skewed by outliers, overestimates typical loss
- Equal costs (1:1 ratio): Ignores business reality, optimizes for accuracy instead of value
- Per-transaction cost (actual amount): More precise but requires custom loss function
**Consequences:**
- Model evaluation will use cost-weighted metrics alongside standard AUC/F1
- Threshold tuning will optimize for minimum total cost, not just classification accuracy
- Stakeholder presentations can frame model value in dollar terms
**Related:** `notebooks/exploratory/01_eda_fraud_patterns.ipynb` (Section 10)

---

### [DECISION-007] XGBoost Over Logistic Regression for Production Model
**Date:** 2026-02-08
**Status:** ✅ Implemented
**Context:** Need to select the best-performing model for production deployment. Baseline (Logistic Regression) provides interpretability, but may lack predictive power for complex fraud patterns.
**Decision:** Deploy XGBoost as the production model after hyperparameter tuning.
**Rationale:**
- **PR-AUC improvement**: XGBoost (0.1098) outperforms Logistic Regression (0.0821) by 33.8%
- **Fair comparison at threshold 0.5**: XGBoost catches 60.9% of fraud vs 42.8% baseline
- **Non-linear patterns**: Fraud detection involves complex interactions that tree-based models capture better
- **Feature importance**: XGBoost provides interpretable tree-based importance metrics
- **Industry standard**: XGBoost is widely used in production fraud detection systems
- **Class imbalance handling**: Built-in `scale_pos_weight` parameter (set to 28.56 for 3.5% fraud rate)
**Alternatives Considered:**
- Logistic Regression: Simpler and more interpretable, but lower PR-AUC (0.0821)
- Random Forest: Similar to XGBoost but typically slower and less performant
- LightGBM/CatBoost: Could be explored in future iterations for potential speed/performance gains
- Neural Networks: Overkill for 7 features, harder to interpret, requires more data
**Consequences:**
- Best hyperparameters: max_depth=6, n_estimators=200, learning_rate=0.05 (from grid search)
- Model file size: ~500KB (xgboost_final.pkl) — small enough for real-time deployment
- Training time: ~30 seconds on full dataset (acceptable for retraining cadence)
- Inference time: <1ms per transaction (meets real-time requirement)
- Trade-off: Less interpretable than Logistic Regression but significantly better performance
**Related:** `notebooks/modeling/03_model_training.ipynb` (Sections 3-5)

---

### [DECISION-008] Constrained Optimization with 75% Minimum Recall Over Pure Cost Minimization
**Date:** 2026-02-08
**Status:** ✅ Implemented
**Context:** Unconstrained cost optimization found threshold 0.740 with lowest total cost ($328K), but this only catches 14.4% of fraud (665 of 4,611 frauds). Real banks prioritize fraud detection over pure cost minimization.
**Decision:** Implement constrained optimization requiring minimum 75% recall, resulting in threshold 0.410.
**Rationale:**
- **Business reality**: Banks cannot tolerate catching only 14% of fraud, even if it minimizes immediate operational cost
- **Regulatory compliance**: Financial institutions face penalties for inadequate fraud prevention
- **Customer trust**: Missing 86% of fraud transactions damages brand reputation and customer confidence
- **Long-term cost**: Reputational damage and regulatory fines exceed short-term operational savings
- **75% target**: Realistic balance between catching most fraud (76% actual) while controlling false positive costs
**Alternatives Considered:**
1. **Unconstrained optimization (threshold 0.740)**: 14.4% recall, $328K cost — rejected as unacceptable
2. **Higher recall targets (85-90%)**: Would require threshold ~0.25-0.30, causing FP explosion (100K+ false alarms)
3. **Fixed threshold (0.5)**: Arbitrary choice, doesn't account for business costs (60.9% recall, moderate cost)
**Consequences:**
- **Threshold shifts**: From 0.740 (unconstrained) to 0.410 (constrained 75%)
- **Recall improvement**: 14.4% → 76.0% (catches 2,840 additional frauds on validation set)
- **Cost increase**: $328K → $598K (82% increase, or +$270K)
- **Cost per additional fraud caught**: $94.99 (vs $75 median fraud amount — slightly negative ROI on cost alone)
- **Trade-off justified**: Preventing $213K in fraud losses (2,840 frauds × $75) costs $270K, but includes intangible benefits (reputation, compliance)
- **False positives**: 3,248 → 51,524 (15.8x increase, significant analyst workload)
**Related:** `notebooks/modeling/03_model_training.ipynb` (Cell 31)

---

### [DECISION-009] Multi-Threshold Production Strategy Over Single Threshold
**Date:** 2026-02-08
**Status:** ✅ Implemented
**Context:** Single threshold (0.410) achieves 75% recall but generates 51,524 manual reviews on validation set (46.6% of all transactions). This is operationally expensive and treats all flagged transactions equally, ignoring confidence levels.
**Decision:** Implement three-tier strategy:
- **Auto-block (score ≥ 0.90)**: Instant fraud block, $5 cost (automated processing)
- **Manual review (0.410 ≤ score < 0.90)**: Human analyst review, $10 cost
- **Auto-approve (score < 0.410)**: No review, $0 cost
**Rationale:**
- **Confidence-based triage**: High-confidence fraud (≥0.90) doesn't need human review
- **Operational efficiency**: Reduces per-transaction cost for obvious fraud cases ($5 vs $10)
- **Same recall target**: Maintains 76% recall (3,505 frauds caught on validation set)
- **Faster fraud blocking**: Auto-block enables instant rejection for score ≥0.90 (no analyst queue)
- **Realistic banking practice**: Real fraud systems use tiered decision rules, not binary threshold
**Alternatives Considered:**
- Single threshold (0.410): 76% recall but all 55,029 flagged txns require manual review
- Four-tier strategy: Adding "auto-approve-with-monitoring" tier (0.30-0.41) adds complexity without clear benefit
- Adaptive thresholds: Dynamic adjustment based on fraud rate trends (future enhancement)
**Consequences:**
- **Auto-block segment**: 19 transactions (0.02%), 6 frauds caught, 13 false positives, $65 cost
- **Manual review segment**: 55,010 transactions (46.6%), 3,499 frauds caught, 51,511 FPs, $515K cost
- **Auto-approve segment**: 63,079 transactions (53.4%), 1,106 frauds missed, $83K FN cost
- **Total cost**: $598K (same as single threshold, slight savings from $5 auto-block)
- **Efficiency gain**: Minimal cost reduction but enables faster blocking of high-confidence fraud
- **Test set validation**: 76% recall confirmed, production strategy is robust
- **Configuration saved**: `threshold_config.pkl` contains all parameters for deployment
**Related:** `notebooks/modeling/03_model_training.ipynb` (Cells 32-33, 37)

---

