# Preprocessing & Data Quality — CLABSI

**Keywords:** preprocessing, cleaning, feature engineering, lag features, data quality, DQ report, SIR

This document describes how we turn raw CDPH/NHSN CLABSI data into a single, analysis-ready CSV for the app and API.

---

## Goals
- **Normalize** column names and numeric types.  
- **Derive** consistently: SIR (if missing), rates per 1,000 line-days, and safe lag features.  
- **Sanitize** for clean JSON responses (no NaN/Inf).  
- **Report** data quality transparently.

---

## Steps (high-level)
1) **Load & rename**  
   - Trim headers; unify variants like `Central_Line_Days` → `Central_line_Days`.  
   - Coerce numeric columns (`Infections_*`, `Central_line_Days`, `SIR`, etc.).

2) **Derive if missing**  
   - `SIR_filled = Infections_Reported / Infections_Predicted` (when SIR is missing).  
   - `rate_per_1000 = Infections_Reported / Central_line_Days * 1000`.

3) **Replace ±Inf / NaN**  
   - Set to `NaN` during processing; downstream API converts to `null` for JSON.  
   - Prevents chart/table issues in the UI.

4) **Feature engineering for models**  
   - Sort by `Facility_ID, Year`.  
   - Create **lag-1**: `SIR_lag1`, `obs_rate_lag1`, `exp_rate_lag1` (per facility).  
   - These are the only predictors used (one-step ahead, no leakage).

5) **Export cleaned file**  
   - Save as `cdph_clabsi_odp_2021_2023_clean.csv` under `demo_data/clabsi/`.  
   - Path is configurable via `CLEAN_CLABSI_CSV`.

6) **Data-quality report (DQ)**  
   - For every column: `nulls`, `non_null`, `null_pct (%)`.  
   - Save as **`clabsi_dq_report.csv`** for transparency and quick checks.

---

## Why a DQ report?
- Helps reviewers confirm **completeness** (missingness by column).  
- Flags fields that might explain model fallbacks (e.g., many missing rates).  
- Useful to track improvements when newer vintages arrive.

---

## JSON-safe API
The API uses a deep sanitizer so values like `NaN`/`Inf` become `null` before sending JSON.  
This avoids UI crashes and makes the app robust.

---

## Repro notes
- The pipeline is implemented in the API loader (`_load_df`) and helpers around it.  
- If you ingest new years:
  - Re-run the preprocessing, regenerate **`clabsi_dq_report.csv`**, and update `CLEAN_CLABSI_CSV`.  
  - The UI will automatically pick up the new **years** from `/debug/columns`.

---

## Related docs
- Data Sources (provenance/quirks): `data_sources.md`  
- Methods (models, intervals, backtests): `methods.md`
