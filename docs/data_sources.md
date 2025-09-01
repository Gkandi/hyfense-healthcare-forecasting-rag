# Data Sources — CDPH CLABSI

**Keywords:** CDPH, CLABSI, SIR, NHSN, California, dataset, provenance, licensing, line-days

This app uses **public CLABSI data for California hospitals** (2021–2023), compiled from CDPH/NHSN releases.

---

## Provenance
- **Publisher:** California Department of Public Health (CDPH) / NHSN extracts  
- **Scope:** California acute care hospitals reporting CLABSI metrics  
- **Fields used (typical names):**  
  - `Facility_ID`, `Facility_Name`, `County`, `Year`  
  - `Infections_Reported` (Observed), `Infections_Predicted` (Expected),  
  - `Central_line_Days`, `SIR` (and sometimes 2015 re-based SIR),  
  - Confidence interval bounds (when supplied).  
- **Update cadence:** periodic; this demo bundles **2021–2023**.

> Column names may differ subtly across files. The API normalizes common variants (e.g., `Central_line_Days` vs `Central_Line_Days`).

---

## Where the files live
For the demo, a cleaned CSV is placed under `demo_data/clabsi/` and configured via the `CLEAN_CLABSI_CSV` env var.  
See `preprocessing_pipeline.md` for how we standardize and export this cleaned file.

---

## Licensing & Use
- Check the CDPH/NHSN licensing/terms in the original source.  
- The demo is for **education and evaluation**; it is **not a regulatory tool**.

---

## Known quirks & cautions
- **Small denominators** (line-days) can make raw rates unstable; prefer SIR for comparisons.  
- **Missing columns** in some vintages are handled defensively in the API.  
- **Re-based SIR** (e.g., “2015”) may appear; we consistently use one SIR column in the app.  
- Public data may differ from internal hospital dashboards due to timing or definitions.

---

## Related docs
- Data dictionary (columns & derived fields): `data_dictionary.md`  
- Preprocessing & Data Quality: `preprocessing_pipeline.md`
