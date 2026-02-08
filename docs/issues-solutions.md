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