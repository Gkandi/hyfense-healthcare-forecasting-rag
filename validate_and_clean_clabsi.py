# app/scripts/validate_and_clean_clabsi.py
import os, sys, numpy as np, pandas as pd

# SRC = sys.argv[1] if len(sys.argv) > 1 else "/demo_data/clabsi/clabsi_2021_2023.csv" cdph_clabsi_odp_2021_2022_2023
SRC = sys.argv[1] if len(sys.argv) > 1 else "demo_data/clabsi/cdph_clabsi_odp_2021_2022_2023.csv"
OUT_CLEAN = sys.argv[2] if len(sys.argv) > 2 else "demo_data/clabsi/cdph_clabsi_odp_2021_2022_2023_clean.csv"
OUT_DQ = sys.argv[3] if len(sys.argv) > 3 else "demo_data/clabsi/clabsi_dq_report.csv"
# Normalize Facility_ID
def clean_facility_id(x):
    try:
        f = float(x)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    s = str(x)
    return s[:-2] if s.endswith(".0") else s

def main():
    if not os.path.exists(SRC):
        print(f"Source file {SRC} does not exist.")
        return
    # Read the CSV file  
    df = pd.read_csv(SRC)
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Types
    # Ensure Year is numeric
    if "Year" in df:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    # Ensure numeric columns are converted to float, replace invalid entries with NaN
        num_cols = ["Infections_Reported","Infections_Predicted","Central_line_Days","SIR","SIR_CI_95_Lower_Limit","SIR_CI_95_Upper_Limit","SIR_2015"]
        for c in num_cols:
            if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter scope
    # Ensure keeping only CLABSI related data
    if "HAI" in df:
        df = df[df["HAI"].astype(str).str.contains("CLABSI", case=False, na=False)]
    # Ensure keeping only the years 2021, 2022, and 2023
    if "Year" in df:
        df = df[df["Year"].isin([2021,2022,2023])]

    # Facility_ID normalization
    if "Facility_ID" in df:   
        df["Facility_ID"] = df["Facility_ID"].map(clean_facility_id)

    # Replace negative values in numeric columns with NaN
    for c in num_cols:
        if c in df: df.loc[df[c] < 0, c] = np.nan

     
    # Compute CLABSI per 1,000 line days when denominator > 0
    # Compute for each row (hospital–year), a rate of 'CLABSI infections' by the total 'central-line (aka. central venous catheter) days'
    df["rate_per_1000"] = np.where(df["Central_line_Days"]>0, (df["Infections_Reported"]/df["Central_line_Days"])*1000, np.nan)
    # Recomputes SIR (Standardized Infection Ratio) = Observed/Predicted when when 'Infections_Predicted' > 0, use it as a fallback later.
    df["sir_recalc"] = np.where(df["Infections_Predicted"]>0, df["Infections_Reported"]/df["Infections_Predicted"], np.nan)

    # Fill SIR or Replace SIR column when missing using sir_recalc
    if "SIR" in df:
        df["SIR_filled"] = df["SIR"]
        fill_mask = df["SIR_filled"].isna() & df["sir_recalc"].notna()
        df.loc[fill_mask, "SIR_filled"] = df.loc[fill_mask, "sir_recalc"]
    else:
        df["SIR_filled"] = df["sir_recalc"]

    # Deduplicate (Facility_ID, Year)
    # if df.duplicated(subset=["Facility_ID","Year"]).any():
    #     id_year = ["Facility_ID","Year"]
    #     # Decide what to keep from the non-numeric columns
    #     keep_other = ["State","HAI","Facility_Name","County","Hospital_Category_RiskAdjustment","Facility_Type","Comparison","Met_2020_Goal","Months","Notes"]
    #     # For duplicates, sum the numerators/denominators (observed/predicted/line-days).
    #     sums = df.groupby(id_year, as_index=False)[["Infections_Reported","Infections_Predicted","Central_line_Days"]].sum()
    #     # For metadata, keep the first non-null representative row per group.
    #     firsts = df.sort_values(id_year).groupby(id_year, as_index=False)[[c for c in keep_other if c in df]].first()
    #     # Merge aggregated numerics with representative metadata, one consolidated row per (Facility_ID, Year).
    #     df = sums.merge(firsts, on=id_year, how="left")
    #     # Recomputes derived metrics based on the aggregated values.
    #     df["rate_per_1000"] = np.where(df["Central_line_Days"]>0, (df["Infections_Reported"]/df["Central_line_Days"])*1000, np.nan)
    #     df["SIR_filled"] = np.where(df["Infections_Predicted"]>0, df["Infections_Reported"]/df["Infections_Predicted"], np.nan)
    # Deduplicate (Facility_ID, Year)
    if df.duplicated(subset=["Facility_ID","Year"]).any():
        id_year = ["Facility_ID","Year"]
        keep_other = [
            "State","HAI","Facility_Name","County",
            "Hospital_Category_RiskAdjustment","Facility_Type",
            "Comparison","Met_2020_Goal","Months","Notes",
            # 👇 keep the CI columns too
            "SIR_CI_95_Lower_Limit","SIR_CI_95_Upper_Limit"
        ]
        sums = df.groupby(id_year, as_index=False)[
            ["Infections_Reported","Infections_Predicted","Central_line_Days"]
        ].sum()
        firsts = (df.sort_values(id_year)
                    .groupby(id_year, as_index=False)[[c for c in keep_other if c in df]]
                    .first())
        df = sums.merge(firsts, on=id_year, how="left")

        # recompute derived metrics from the summed numerators/denominators
        df["rate_per_1000"] = np.where(
            df["Central_line_Days"]>0,
            (df["Infections_Reported"]/df["Central_line_Days"])*1000,
            np.nan
        )
        df["SIR_filled"] = np.where(
            df["Infections_Predicted"]>0,
            df["Infections_Reported"]/df["Infections_Predicted"],
            np.nan
        )

    # DQ report
    # -------------------   
    # Build a data-quality report:
    # For each column: number of nulls, non-nulls, and percentage nulls.
    # Save it as a CSV.
    # --------------------
    dq = (df.isna().mean().mul(100).round(2)).rename("null_pct").to_frame()
    dq.insert(0,"non_null", df.notna().sum())
    dq.insert(0,"nulls", df.isna().sum())
    dq.reset_index(names="column").to_csv(OUT_DQ, index=False)

    df.to_csv(OUT_CLEAN, index=False)
    # --------------------
    print(f"Saved clean: {OUT_CLEAN}")
    print(f"Saved DQ:    {OUT_DQ}")

if __name__ == "__main__":
    main()
