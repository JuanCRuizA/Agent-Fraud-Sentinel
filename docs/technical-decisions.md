# Agent Fraud Sentinel - Technical Decisions Log

## Purpose
Document key technical decisions, rationale, and alternatives considered during development.

---

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

### [DECISION-003] Title Here
**Date:** YYYY-MM-DD  
**Status:** 🔄 In Progress  
**Context:** 

**Decision:** 

**Rationale:**

**Alternatives Considered:**

**Consequences:**

**Related:**

---

## Quick Reference

### Active Decisions
- [DECISION-001] IEEE-CIS Dataset Selection
- [DECISION-002] EDA Before Feature Engineering

### Pending Review
- None

### Rejected
- None