# app/scripts/validate_and_clean_clabsi.py
import os, sys, numpy as np, pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "demo_data/clabsi/cdph_clabsi_odp_2021_2022_2023.csv"
OUT_CLEAN = sys.argv[2] if len(sys.argv) > 2 else "demo_data/clabsi/clabsi_2021_2023_clean.csv"
OUT_DQ = sys.argv[3] if len(sys.argv) > 3 else "demo_data/clabsi/clabsi_dq_report.csv"

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
    df = pd.read_csv(SRC)
    df.columns = [c.strip() for c in df.columns]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    num_cols = ["Infections_Reported","Infections_Predicted","Central_line_Days",
                "SIR","SIR_CI_95_Lower_Limit","SIR_CI_95_Upper_Limit","SIR_2015"]
    for c in num_cols:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Facility_ID"] = df["Facility_ID"].map(clean_facility_id)
    df = df[df["HAI"].astype(str).str.contains("CLABSI", case=False, na=False)]
    df = df[df["Year"].isin([2021, 2022, 2023])]
    for c in num_cols:
        if c in df: df.loc[df[c] < 0, c] = np.nan

    df["rate_per_1000"] = np.where(
        df["Central_line_Days"]>0,
        (df["Infections_Reported"]/df["Central_line_Days"])*1000,
        np.nan
    )
    df["sir_recalc"] = np.where(
        df["Infections_Predicted"]>0,
        df["Infections_Reported"]/df["Infections_Predicted"],
        np.nan
    )

    if "SIR" in df:
        df["SIR_filled"] = df["SIR"]
        fill_mask = df["SIR_filled"].isna() & df["sir_recalc"].notna()
        df.loc[fill_mask, "SIR_filled"] = df.loc[fill_mask, "sir_recalc"]
    else:
        df["SIR_filled"] = df["sir_recalc"]

    if df.duplicated(subset=["Facility_ID","Year"]).any():
        id_year = ["Facility_ID","Year"]
        keep_other = ["State","HAI","Facility_Name","County",
                      "Hospital_Category_RiskAdjustment","Facility_Type",
                      "Comparison","Met_2020_Goal","Months","Notes"]
        sums = df.groupby(id_year, as_index=False)[
            ["Infections_Reported","Infections_Predicted","Central_line_Days"]
        ].sum()
        firsts = (df.sort_values(id_year)
                    .groupby(id_year, as_index=False)[[c for c in keep_other if c in df]]
                    .first())
        df = sums.merge(firsts, on=id_year, how="left")
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

    dq = (df.isna().mean().mul(100).round(2)).rename("null_pct").to_frame()
    dq.insert(0, "non_null", df.notna().sum())
    dq.insert(0, "nulls", df.isna().sum())
    dq.reset_index(names="column").to_csv(OUT_DQ, index=False)

    df.to_csv(OUT_CLEAN, index=False)
    print(f"[OK] Clean rows: {len(df)}")
    print(f"[OK] Wrote clean CSV: {OUT_CLEAN}")
    print(f"[OK] Wrote DQ report: {OUT_DQ}")

if __name__ == "__main__":
    main()
