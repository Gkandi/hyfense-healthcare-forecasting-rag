# Data Dictionary — CLABSI Cleaned CSV

This describes the fields the app expects in the cleaned dataset (`cdph_clabsi_odp_2021_2022_2023_clean.csv`). Some columns are optional; the app derives a few when missing.

> **Tip:** Column names sometimes vary by source; the API tries common alternates (e.g., `Central_line_Days` vs `Central_Line_Days`).

---

## Core identifiers
- **Facility_ID** *(string/int)* — hospital identifier. The UI displays it without decimals.  
- **Facility_Name** *(string)* — hospital name.  
- **County** *(string, optional)* — county name.  
- **State** *(string, optional)* — state abbreviation (e.g., CA).  
- **Facility_Type** *(string, optional)* — e.g., Acute Care.  
- **Hospital_Category_RiskAdjustment** *(string, optional)* — risk‑adjustment group label.

## Time
- **Year** *(int)* — reporting year (2021–2023 in the demo).

## Counts & exposure
- **Infections_Reported** *(int)* — observed CLABSI events.  
- **Infections_Predicted** *(float)* — expected events from risk model.  
- **Central_line_Days** *(int)* — exposure denominator.

## Ratios and intervals
- **SIR** *(float, optional)* — provided SIR.  
- **SIR_filled** *(float)* — if `SIR` missing, the app computes **Observed / Predicted**.  
- **SIR_CI_95_Lower_Limit** *(float, optional)* — lower 95% confidence bound.  
- **SIR_CI_95_Upper_Limit** *(float, optional)* — upper 95% confidence bound.

## Flags & misc
- **Met_2020_Goal** *(string/bool, optional)* — “Yes/No” or boolean flag.  
- **Months**, **Notes**, **Comparison** *(optional)* — pass‑through fields from the source.

## App‑derived helper columns
- **rate_per_1000** *(float)* — **Infections_Reported / Central_line_Days × 1000** (set to NaN when denominator ≤ 0).  
- **expected_rate_per_1000** *(float)* — **Infections_Predicted / Central_line_Days × 1000**.  
- **excess_infections** *(float)* — **Observed − Predicted** (counts).  
- **rate_gap** *(float)* — **rate_per_1000 − expected_rate_per_1000**.

## Modeling features (engineered)
Used only in the Forecast (Predictions) tab:
- **SIR_lag1** — last year’s SIR for the same facility.  
- **obs_rate_lag1** — last year’s observed rate per 1,000.  
- **exp_rate_lag1** — last year’s expected rate per 1,000.

> If any of these are missing for a row, the model falls back to **naive** for that row. The API returns a `model_note` describing such fallbacks.
