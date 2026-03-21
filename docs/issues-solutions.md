# Agent Fraud Sentinel - Issues & Solutions Log

## Purpose
Track technical problems encountered and how they were solved.

---

## Issue Template

### [ISSUE-XXX] Title
**Date:** YYYY-MM-DD  
**Status:** ✅ Resolved | 🔄 Investigating | ⚠️ Blocked  
**Severity:** 🔴 Critical | 🟡 Medium | 🟢 Low  
**Problem:** What went wrong?  
**Root Cause:** Why did it happen?  
**Solution:** How was it fixed?  
**Prevention:** How to avoid in future?

---

## Issues Log

### [ISSUE-001] FileNotFoundError on Notebook Data Paths
**Date:** 2026-02-07
**Status:** ✅ Resolved
**Severity:** 🔴 Critical
**Problem:** `FileNotFoundError: ../data/raw/train_transaction.csv` when running notebook 01 from `notebooks/exploratory/`.
**Root Cause:** Notebooks were moved from `notebooks/` to `notebooks/exploratory/` (one level deeper), but relative paths still used `../data/raw/` instead of `../../data/raw/`.
**Solution:** Updated all `DATA_PATH`, `savefig()`, and `to_csv()` paths from `../` to `../../` across both notebooks.
**Prevention:** Always verify relative paths after moving notebooks to a different directory level. Consider using a project-root-relative path helper.

---

### [ISSUE-002] NameError: 'df' not defined
**Date:** 2026-02-07
**Status:** ✅ Resolved
**Severity:** 🟡 Medium
**Problem:** `NameError: name 'df' is not defined` when running a cell in notebook 01.
**Root Cause:** User ran the data preview cell before executing the data loading and merge cells above it. Jupyter notebooks require sequential execution for variable dependencies.
**Solution:** Ran cells in order from top using "Run All" or Shift+Enter from the first cell.
**Prevention:** Always run notebooks sequentially from the top. Add a note in the first cell reminding users to "Run All" or execute cells in order.

---

### [ISSUE-003] VSCode Pylance Connection Error
**Date:** 2026-02-07
**Status:** ✅ Resolved
**Severity:** 🟢 Low
**Problem:** "Client Pylance: connection to server is erroring" popup in VSCode while working on notebooks.
**Root Cause:** VSCode's Pylance language server crashed due to high memory usage (2.7 GB dataset loaded in notebook kernel). Unrelated to notebook code quality.
**Solution:** Ctrl+Shift+P > "Python: Restart Language Server". The error can also be safely ignored as it does not affect notebook execution.
**Prevention:** This is a known VSCode issue with large datasets. Close unused notebook tabs to reduce memory pressure.

---

### [ISSUE-004] Notebook Not Refreshing After External Edits
**Date:** 2026-02-08
**Status:** ✅ Resolved
**Severity:** 🟢 Low
**Problem:** After adding new cells to the notebook externally (via Claude Code), the changes were not visible in the VSCode Jupyter tab.
**Root Cause:** VSCode caches the notebook in memory. External file changes are not automatically detected by the open editor tab.
**Solution:** Close the notebook tab and reopen the file from the Explorer panel to reload the latest version.
**Prevention:** Always close the notebook tab before requesting external modifications, then reopen after changes are saved.

---

### [ISSUE-005] ValueError: StandardScaler Input Contains Infinity
**Date:** 2026-02-08
**Status:** ✅ Resolved
**Severity:** 🔴 Critical
**Problem:** `ValueError: Input X contains infinity or a value too large for dtype('float64')` when fitting StandardScaler in notebook 03.
**Root Cause:** The `amount_deviation` feature (Z-score) produces infinity when dividing by zero. This occurs for clients with only one transaction (standard deviation = 0).
**Solution:** Added data cleaning function before scaling:
- Replace `inf` with +10, `-inf` with -10 (extreme deviation boundaries)
- Fill `NaN` with 0 (typical for first transactions where deviation is undefined)
- Added assertions to verify no infinity or NaN remain in features
**Prevention:** Always validate features for infinity/NaN after computing Z-scores or ratios. Add data quality checks before feeding data to scikit-learn transformers.

---

### [ISSUE-006] NameError: Variables Defined Out of Order
**Date:** 2026-02-08
**Status:** ✅ Resolved
**Severity:** 🟡 Medium
**Problem:** `NameError: name 'recall_optimal' is not defined` and `NameError: name 'recall_test' is not defined` when running notebook 03 cells sequentially (Run All).
**Root Cause:** Cells 31 and 33 referenced variables (`recall_optimal`, `precision_optimal`, `recall_test`, etc.) that were defined later in Cell 35. This created a dependency violation when executing cells top-to-bottom.
**Solution:** Made Cells 31 and 33 self-contained by computing all required metrics at the START of each cell before using them in comparisons. Each cell now calculates its own unconstrained baseline metrics independently.
**Prevention:** When adding cells that reference metrics, ensure all dependencies are computed within the same cell or in prior cells. Test with "Run All" to verify sequential execution works correctly.

---

### [ISSUE-007] F-string Expression Cannot Include Backslash
**Date:** 2026-02-08
**Status:** ✅ Resolved
**Severity:** 🟢 Low
**Problem:** `SyntaxError: f-string expression part cannot include a backslash` in Cell 37 when accessing dictionary values inside f-strings.
**Root Cause:** Python f-strings do not allow backslashes inside the `{}` expression parts. The code `f"{threshold_config[\"manual_review_threshold\"]:.3f}"` used escaped quotes (backslashes) inside the f-string expression.
**Solution:** Compute the dictionary value outside the f-string first, then reference the variable:
```python
manual_threshold = threshold_config["manual_review_threshold"]
print(f"Manual review threshold: >= {manual_threshold:.3f}")
```
**Prevention:** Extract complex expressions (dictionary access, string operations) into variables before using them in f-strings. This improves both readability and avoids syntax limitations.

---

### [ISSUE-008] Confusion About Model Performance at Different Thresholds
**Date:** 2026-02-08
**Status:** ✅ Resolved
**Severity:** 🟡 Medium
**Problem:** User confused why baseline model showed 42.79% recall at threshold 0.5, XGBoost initial showed 61.05%, but "final" XGBoost only showed 14.42% recall at threshold 0.740.
**Root Cause:** Comparing models at different thresholds (0.5 vs 0.740) is misleading -- it conflates two separate decisions: (1) which model is best, and (2) what threshold to use.
**Solution:** Added Cell 26 (markdown explanation of "TWO SEPARATE STEPS") and Cell 28 (fair comparison showing all models at threshold 0.5). Clearly separated model selection (use PR-AUC) from threshold optimization (use cost analysis).
**Prevention:** Always compare models at the same threshold first. Document that threshold selection is a separate business decision applied AFTER choosing the best model.

---

### [ISSUE-009] Streamlit `use_container_width` Deprecation Warning
**Date:** 2026-02-09
**Status:** ✅ Resolved
**Severity:** 🟢 Low
**Problem:** When running the Streamlit dashboard locally, the terminal showed: `use_container_width will be removed after 2025-12-31. For use_container_width=True, use width='stretch'`.
**Root Cause:** Streamlit deprecated the `use_container_width` parameter in `st.image()` and `st.dataframe()` in favor of the new `width` parameter. The dashboard code used the older API.
**Solution:** Replaced all 5 occurrences of `use_container_width=True` with `width="stretch"` in `dashboard_app.py`.
**Prevention:** Check Streamlit release notes for deprecated parameters when upgrading. Use the latest parameter names from the Streamlit documentation.

---

### [ISSUE-010] Streamlit Cloud FileNotFoundError for Model Artifacts
**Date:** 2026-02-10
**Status:** ✅ Resolved
**Severity:** 🔴 Critical
**Problem:** After deploying to Streamlit Cloud, the app showed: `Failed to load model or data: [Errno 2] No such file or directory: '/mount/src/agent-fraud-sentinel/notebooks/dashboard/../../models/xgboost_final.pkl'`
**Root Cause:** The `.gitignore` file excluded `models/*.pkl` and `data/processed/*`. These files existed locally but were never committed to GitHub, so Streamlit Cloud could not find them.
**Solution:**
1. Commented out `models/*.pkl` in `.gitignore` to allow model files (< 1 MB total) to be tracked
2. Created a slim `test_dashboard.csv` (8.3 MB, 8 columns only) instead of committing the full `test.csv` (145 MB, 442 columns) which exceeded GitHub's 100 MB file limit
3. Updated `dashboard_app.py` to load `test_dashboard.csv` first, falling back to `test.csv` for local development
**Prevention:** When building Streamlit Cloud apps, ensure all required data and model files are committed to the repository. For large files, create slim versions with only the columns needed by the dashboard. Check `.gitignore` rules before deployment.

---

### [ISSUE-011] Test CSV Exceeds GitHub 100 MB File Size Limit
**Date:** 2026-02-10
**Status:** ✅ Resolved
**Severity:** 🟡 Medium
**Problem:** The full `data/processed/test.csv` (145 MB, 118,108 rows x 442 columns) exceeded GitHub's 100 MB per-file limit and could not be committed.
**Root Cause:** The test CSV contained all 442 original + engineered columns, but the dashboard only uses 8 columns (7 features + isFraud). The remaining 434 columns added ~141 MB of unnecessary data.
**Solution:** Created `notebooks/dashboard/test_dashboard.csv` containing only the 8 columns needed by the dashboard. This reduced file size from 145 MB to 8.3 MB (97% reduction). The dashboard app tries the slim file first, then falls back to the full test set if available locally.
**Prevention:** For deployment artifacts, always create minimal data files with only the columns required by the application. Keep full datasets in `.gitignore` and commit only slim versions.

---

### [ISSUE-012] NameError: `pr_auc_final` Not Defined in Bayesian Optimization Cell
**Date:** 2026-02-19
**Status:** ✅ Resolved
**Severity:** 🟡 Medium
**Problem:** `NameError: name 'pr_auc_final' is not defined` when running the Bayesian XGBoost optimization cell (Section 6.2) and the comparison table cell (Section 6.2 summary).
**Root Cause:** `pr_auc_final` was only assigned in the Section 7 model comparison cell, which runs later. The Bayesian optimization cells in Section 6.2 referenced this variable as a forward reference, creating a dependency violation.
**Solution:** Replaced all `pr_auc_final` references in Section 6.2 with `grid_search_pr_auc = best_params["PR-AUC"]`, which is already available from the grid search cell (Section 6.1). This gives the same value (XGBoost grid search best PR-AUC) without the forward dependency.
**Prevention:** When adding new cells above an existing section, verify that all variable references are defined in prior cells. Never reference variables from later sections.

---

### [ISSUE-013] SyntaxError: F-string Unmatched `[` with Dict Key Access in Python 3.11
**Date:** 2026-02-19
**Status:** ✅ Resolved
**Severity:** 🟢 Low
**Problem:** `SyntaxError: f-string: unmatched '['` on the line `print(f'... {best_params['PR-AUC']:.4f}...')`.
**Root Cause:** Python 3.11 does not allow using the same quote style for a dictionary key inside an f-string expression. Single-quoted keys `['key']` cannot appear inside a single-quoted f-string `f'...'`.
**Solution:** Extract the dictionary access to a variable before the f-string: `grid_search_pr_auc = best_params["PR-AUC"]`, then use `{grid_search_pr_auc:.4f}` in the f-string. This is also the fix for ISSUE-012.
**Prevention:** Always extract dict key accesses and complex expressions to named variables before using them in f-strings. This is both a Python 3.11 requirement and a readability improvement. (See also ISSUE-007 for a related f-string backslash issue.)

---

### [ISSUE-014] NameError: `baseline_precision_05` / `final_precision_05` Not Defined
**Date:** 2026-02-19
**Status:** ✅ Resolved
**Severity:** 🟡 Medium
**Problem:** `NameError: name 'baseline_precision_05' is not defined` in the 4-model fair comparison cell (Section 7). The cell also referenced `baseline_recall_05`, `baseline_f1_05`, `final_precision_05`, `final_recall_05`, `final_f1_05`.
**Root Cause:** The new comparison cell was written with placeholder comments ("already computed") but the original notebook used different variable names: `precision_val_bl`, `recall_val_bl` (for Logistic Regression) and never computed threshold-0.5 metrics for the XGBoost grid search model. The `_05` suffix variables were never defined anywhere.
**Solution:** Added explicit computations at the top of the comparison cell:
- Logistic Regression: `baseline_pred_05 = (baseline_proba_val >= 0.5).astype(int)` then `precision_score / recall_score / f1_score`
- XGBoost grid search: `final_pred_05 = (final_xgb_proba_val >= 0.5).astype(int)` then same pattern
**Prevention:** When writing cells that reference variables as "already computed", verify the exact variable names from the source cells. Do not assume naming conventions -- check the actual code.

---

### [ISSUE-015] Downstream Cells Hardcoded to XGBoost Probabilities After Dynamic Winner Selection
**Date:** 2026-02-19
**Status:** ✅ Resolved
**Severity:** 🔴 Critical
**Problem:** After adding dynamic winner selection in Section 7 (LightGBM Bayesian won with PR-AUC 0.1125), all downstream results (threshold optimization, constrained optimization, production strategy, confusion matrices) were identical to the original XGBoost notebook. The winner selection had no effect.
**Root Cause:** 8 cells (39, 41, 42, 43, 44, 45, 46, 47) in Sections 7-8 still referenced `final_xgb_proba_val` and `final_xgb_proba_test` directly instead of the dynamic winner variables `final_proba_val` and `final_proba_test` set by the winner selection cell. The plan assumed these cells would "automatically" use the winner, but they contained hardcoded XGBoost variable names.
**Solution:** Bulk replacement across all 8 cells: `final_xgb_proba_val` → `final_proba_val`, `final_xgb_proba_test` → `final_proba_test`. Verified zero remaining occurrences after fix.
**Prevention:** When implementing dynamic model selection, immediately audit ALL downstream cells for hardcoded model-specific variable references. Variable naming conventions alone do not guarantee correct wiring.

---

### [ISSUE-016] Tab 3 Narratives Contradict SHAP Waterfall Plots
**Date:** 2026-03-13
**Status:** ✅ Resolved
**Severity:** 🔴 Critical
**Problem:** All 5 case studies in Tab 3 (Case Study Explorer) had narrative text (Model Decision Explanation and Key Risk Drivers) that contradicted the actual SHAP waterfall plots. Issues included: wrong feature order (4/5 cases), features with negative SHAP listed as risk drivers (3/5 cases), incorrect section titles for non-TP case types (3/5 cases), and generic archetype descriptions instead of case-specific SHAP-based explanations (5/5 cases).
**Root Cause:** Narratives were semi-hardcoded templates describing generic fraud archetypes rather than being derived from actual SHAP values. For example, Case 1 listed "High 24-hour velocity" as the top driver, but Transaction Amount (SHAP +1.60) was the dominant signal; Case 5 claimed "reasonable amount led to approval" but Transaction Amount had SHAP +0.61 pushing toward fraud.
**Solution:** Rewrote all 5 case explanations and driver lists to match actual SHAP values from the waterfall plots. Added `drivers_title` field to each case for dynamic section titles: "Key Risk Drivers" (TP), "Key Factors in Missed Detection" (FN), "Key Factors in False Alert" (FP), "Key Factors in Correct Approval" (TN). Also added CSS to hide Streamlit anchor link icons on all headings.
**Prevention:** When writing case study narratives for explainable AI dashboards, always derive text directly from model outputs (SHAP values). Cross-reference narrative claims against the actual feature attributions before publishing.

---

### [ISSUE-018] FINMA Circular Number Incorrect in Dashboard and Notebook 05
**Date:** 2026-03-21
**Status:** ✅ Resolved
**Severity:** 🟡 Medium
**Problem:** Tab 5 of the dashboard and notebook 05 referenced "FINMA Circular 2017/1" in the Swiss/EU regulatory alignment table and key takeaways text. FINMA Circular 2017/1 covers a different topic; the correct reference for operational risk and model governance is FINMA Circular 2023/1.
**Root Cause:** Incorrect circular number was used when Tab 5 was first authored (Feb 2026). The error was carried over from `dashboard_app.py` into `05_streamlit_dashboard.ipynb`.
**Solution:** Replaced all three occurrences of 2017/1 with 2023/1 in `dashboard_app.py` and all four occurrences across cells 6 and 17 of `05_streamlit_dashboard.ipynb`.
**Prevention:** When citing specific regulatory circulars, verify the number against the official FINMA website. Operational risk / model governance falls under Circular 2023/1, not the earlier 2017/1 circular.

---

### [ISSUE-017] KPI Card 1 Duplicates Fraud Prevented Instead of Showing Net Savings
**Date:** 2026-03-17
**Status:** ✅ Resolved
**Severity:** 🟡 Medium
**Problem:** The Executive Summary (Tab 1) had two KPI cards showing the same $680K value. Card 1 ("Fraud Savings vs No Model") and Card 3 ("Fraud Prevented") were both displaying `tp * FN_COST` ($680,546). The intended first card should show net savings ($192,046) -- the actual bottom-line impact of deploying the model.
**Root Cause:** The calculation `savings = no_model - missed_fraud` simplifies to `(total_fraud - fn) * FN_COST = tp * FN_COST`, which is identical to `fraud_prevented`. The correct net savings formula is `no_model - total_cost` ($922,528 - $730,482 = $192,046).
**Solution:** Changed card 1 from `savings = no_model - missed_fraud` to `net_savings = no_model - total_cost`, and renamed the label from "Fraud Savings vs No Model" to "Net Savings vs No Model". Updated both `dashboard_app.py` and `05_streamlit_dashboard.ipynb`.
**Prevention:** When building KPI dashboards, verify each card shows a mathematically distinct metric. Cross-check by computing values manually before deploying.

---