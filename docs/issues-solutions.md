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
**Root Cause:** Comparing models at different thresholds (0.5 vs 0.740) is misleading — it conflates two separate decisions: (1) which model is best, and (2) what threshold to use.
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
2. Created a slim `test_dashboard.csv` (4.2 MB, 8 columns only) instead of committing the full `test.csv` (145 MB, 442 columns) which exceeded GitHub's 100 MB file limit
3. Updated `dashboard_app.py` to load `test_dashboard.csv` first, falling back to `test.csv` for local development
**Prevention:** When building Streamlit Cloud apps, ensure all required data and model files are committed to the repository. For large files, create slim versions with only the columns needed by the dashboard. Check `.gitignore` rules before deployment.

---

### [ISSUE-011] Test CSV Exceeds GitHub 100 MB File Size Limit
**Date:** 2026-02-10
**Status:** ✅ Resolved
**Severity:** 🟡 Medium
**Problem:** The full `data/processed/test.csv` (145 MB, 118,108 rows x 442 columns) exceeded GitHub's 100 MB per-file limit and could not be committed.
**Root Cause:** The test CSV contained all 442 original + engineered columns, but the dashboard only uses 8 columns (7 features + isFraud). The remaining 434 columns added ~141 MB of unnecessary data.
**Solution:** Created `notebooks/dashboard/test_dashboard.csv` containing only the 8 columns needed by the dashboard. This reduced file size from 145 MB to 4.2 MB (97% reduction). The dashboard app tries the slim file first, then falls back to the full test set if available locally.
**Prevention:** For deployment artifacts, always create minimal data files with only the columns required by the application. Keep full datasets in `.gitignore` and commit only slim versions.

---