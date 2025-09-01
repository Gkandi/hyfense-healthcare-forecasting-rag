# app/api/main.py

# ---- Imports: web framework & utilities --------------------------------------
from fastapi import FastAPI, HTTPException, Query            # FastAPI app + HTTP errors + typed query params
from fastapi.middleware.cors import CORSMiddleware            # Allow browser UIs (Streamlit) to call this API
from fastapi.encoders import jsonable_encoder                 # Safely convert Python/NumPy/Pandas to JSON-able
from fastapi.responses import JSONResponse                    # Explicit JSON responses
from typing import List, Optional                             # Type hints
from pathlib import Path                                      # Safe filesystem paths
import os, traceback                                          # OS env vars + readable error traces
import pandas as pd                                           # Data wrangling
import numpy as np                                            # Numeric helpers (NaN/Inf handling, arrays)
import math  # <<< ADDED                                        # math.isfinite etc.

# ---- Create the web app ------------------------------------------------------
# Title/version is only for docs (Swagger UI). It doesn't affect behavior.
app = FastAPI(title="Hyfense Demo — Local API (JSON-safe)", version="0.2.4")

# ---- Where to read the cleaned CSV from -------------------------------------
# We check an environment variable first so deployments can override the path.
# If it's not set, we default to the included demo CSV file.
CLEAN_CSV = Path(os.getenv(
    "CLEAN_CLABSI_CSV",
    "demo_data/clabsi/cdph_clabsi_odp_2021_2022_2023_clean.csv"  # keep your filename here
))
# Also write the resolved path back to the environment so other parts of the app can read it.
os.environ["CLEAN_CLABSI_CSV"] = str(CLEAN_CSV.resolve())

# ---- CORS: allow requests from any origin (Streamlit runs in a browser) -----
# This lets the UI (different port) call the API without being blocked by the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- Simple in-memory cache for the loaded DataFrame ------------------------
_df_cache: Optional[pd.DataFrame] = None  # Once we read the CSV, we keep it so repeated calls are fast.

def _safe(v, ndigits: int | None = None):
    """
    Convert a value to a plain float if it's finite; otherwise return None.
    Optional rounding to 'ndigits' places.
    Purpose: JSON can't represent NaN/Inf well; None serializes cleanly.
    """
    try:
        f = float(v)
        if not np.isfinite(f):
            return None
        return round(f, ndigits) if ndigits is not None else f
    except Exception:
        return None

def _resp(payload):
    """
    Wrap a payload into a JSONResponse using FastAPI's encoder.
    The encoder already handles most pandas/numpy types.
    """
    return JSONResponse(content=jsonable_encoder(payload))

# >>>>>>>>>>>>>>>>>>>>>>>>>>> JSON deep sanitizer (extra safety) >>>>>>>>>>>>>>>
# In some corner cases nested dict/list values can still carry NaN/Inf.
# These helpers recursively walk the structure and replace them with None.

def _is_finite_number(x) -> bool:
    """True if x can be read as a finite float; False for NaN/Inf/strings etc."""
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

def _sanitize_for_json(obj):
    """
    Recursively replace NaN/Inf with None in dicts/lists/numbers.
    This helps ensure every response is valid JSON for the frontend.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    # Convert numpy scalar to Python scalar if present
    if isinstance(obj, (np.generic,)):
        obj = obj.item()
    # Handle plain numbers
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, int):
        return obj
    # Treat pandas NA as None
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def _load_df() -> pd.DataFrame:
    """
    Load the cleaned CSV once and cache it. Normalize column names and types.
    Also pre-compute a couple of helpful columns if they are missing.
    """
    global _df_cache
    if _df_cache is not None:
        return _df_cache  # Use cached frame if already loaded

    if not CLEAN_CSV.exists():
        # Fail early with a clear message if the file isn't found.
        raise FileNotFoundError(f"Clean file not found: {CLEAN_CSV}")

    # Read CSV to DataFrame
    df = pd.read_csv(CLEAN_CSV)
    # Strip accidental spaces around column names (common data issue)
    df.columns = [c.strip() for c in df.columns]

    # Normalize a common naming variant for the denominator column.
    if "Central_line_Days" not in df.columns and "Central_Line_Days" in df.columns:
        df = df.rename(columns={"Central_Line_Days": "Central_line_Days"})

    # Ensure Year is numeric nullable integer (keeps NaN but acts like int)
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # Convert known numeric columns to numeric; non-numeric become NaN
    num_cols = [
        "Infections_Reported", "Infections_Predicted", "Central_line_Days",
        "SIR", "SIR_CI_95_Lower_Limit", "SIR_CI_95_Upper_Limit", "SIR_2015",
        "rate_per_1000", "SIR_filled"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If SIR_filled is missing, compute it as Observed / Predicted
    if "SIR_filled" not in df and {"Infections_Reported","Infections_Predicted"}.issubset(df.columns):
        df["SIR_filled"] = df["Infections_Reported"] / df["Infections_Predicted"]

    # If per-1,000 rate is missing, compute Observed / Line-days × 1000
    if "rate_per_1000" not in df and {"Infections_Reported","Central_line_Days"}.issubset(df.columns):
        den = pd.to_numeric(df["Central_line_Days"], errors="coerce")
        df["rate_per_1000"] = (pd.to_numeric(df["Infections_Reported"], errors="coerce") / den) * 1000

    # Replace +/-Inf with NaN so we can later convert them to None on output
    for c in df.select_dtypes(include=[float]).columns:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    # Cache it for next calls
    _df_cache = df
    return _df_cache

def _find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    """
    Try each candidate column name and return the first one that exists.
    If 'required' and none found, raise a clear error listing what's available.
    """
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise HTTPException(500, f"Missing expected column (tried): {candidates}. "
                                 f"Available: {list(df.columns)}")
    return None

# ------------------- Debug endpoints (for health checks) ----------------------

@app.get("/health")
def health():
    """
    Simple 'is the API alive' check.
    Returns CSV path and basic info or an error message.
    """
    try:
        d = _load_df()
        return _resp({"ok": True, "csv": str(CLEAN_CSV), "rows": int(len(d)), "cols": list(d.columns)})
    except Exception as e:
        return _resp({"ok": False, "csv": str(CLEAN_CSV), "error": str(e)})

@app.get("/debug/columns")
def debug_columns():
    """
    Return which columns and years are present in the CSV, plus a tiny sample.
    Helpful to confirm the backend sees what you expect.
    """
    try:
        d = _load_df()
        years = sorted([int(y) for y in d["Year"].dropna().unique().tolist()]) if "Year" in d.columns else []
        # Make the sample JSON-safe by replacing Inf with NaN and then NaN with None
        sample = d.head(3).replace([np.inf, -np.inf], np.nan).where(pd.notna(d.head(3)), None).to_dict(orient="records")
        return _resp({"csv": str(CLEAN_CSV), "columns": list(d.columns), "years": years, "sample": sample})
    except Exception as e:
        # Include a traceback so debugging is easier in development
        return _resp({"csv": str(CLEAN_CSV), "error": str(e), "traceback": traceback.format_exc()})

# ------------------- Summary (state-level rollups for a year) ----------------

@app.get("/benchmark/summary")
def summary(year: int = Query(..., ge=2000, le=2100)):
    """
    Aggregate per-year statewide metrics:
    - observed & predicted totals
    - SIR (observed/predicted)
    - rate per 1,000 line-days
    - average/median facility SIR
    - % facilities with SIR<1, significantly better/worse, met goal
    """
    d = _load_df()
    if "Year" not in d.columns:
        raise HTTPException(500, "Column 'Year' not found.")

    # Filter to the requested year
    dy = d[d["Year"] == year].copy()
    if dy.empty:
        raise HTTPException(404, f"No data for year {year}")

    # Locate commonly-named columns (robust to small naming variations)
    id_col   = _find_col(dy, ["Facility_ID","Facility Id","FacilityId"])
    obs_col  = _find_col(dy, ["Infections_Reported","Observed"])
    pred_col = _find_col(dy, ["Infections_Predicted","Predicted"])
    den_col  = _find_col(dy, ["Central_line_Days","Central Line Days","LineDays"])
    sir_col  = "SIR_filled" if "SIR_filled" in dy.columns else _find_col(dy, ["SIR"])

    # Totals for the year (sum across facilities)
    obs = pd.to_numeric(dy[obs_col], errors="coerce").sum()
    pred = pd.to_numeric(dy[pred_col], errors="coerce").sum()
    line_days = pd.to_numeric(dy[den_col], errors="coerce").sum()

    # Statewide SIR and per-1,000 rate (check for divide-by-zero)
    sir_state = (obs / pred) if pred and np.isfinite(pred) and pred != 0 else np.nan
    rate = (obs / line_days) * 1000 if line_days and np.isfinite(line_days) and line_days != 0 else np.nan

    # Facility-level statistics
    mean_sir = pd.to_numeric(dy[sir_col], errors="coerce").mean()
    median_sir = pd.to_numeric(dy[sir_col], errors="coerce").median()
    pct_lt1 = (pd.to_numeric(dy[sir_col], errors="coerce") < 1).mean() * 100

    # Significance flags if present (confidence interval across facilities)
    u = pd.to_numeric(dy.get("SIR_CI_95_Upper_Limit"), errors="coerce")
    l = pd.to_numeric(dy.get("SIR_CI_95_Lower_Limit"), errors="coerce")
    pct_sig_better = (u < 1).mean() * 100 if u is not None else np.nan
    pct_sig_worse  = (l > 1).mean() * 100 if l is not None else np.nan

    # 2020 goal indicator if present
    met = dy.get("Met_2020_Goal")
    pct_goal = (met.astype(str).str.upper().isin(["YES","Y","TRUE","MET"]).mean() * 100) if met is not None else np.nan

    # Build a JSON-safe payload (numbers get cleaned by _safe)
    payload = {
        "year": year,
        "facilities_reporting": int(dy[id_col].astype(str).nunique()),
        "observed_clabsi": _safe(obs, 2),
        "predicted_clabsi": _safe(pred, 2),
        "statewide_SIR": _safe(sir_state, 3),
        "rate_per_1000_line_days": _safe(rate, 3),
        "mean_SIR": _safe(mean_sir, 3),
        "median_SIR": _safe(median_sir, 3),
        "pct_facilities_SIR_lt_1": _safe(pct_lt1, 1),
        "pct_facilities_significantly_better": _safe(pct_sig_better, 1),
        "pct_facilities_significantly_worse": _safe(pct_sig_worse, 1),
        "pct_met_2020_goal": _safe(pct_goal, 1),
        "total_line_days": int(line_days) if np.isfinite(line_days) else None,
    }
    return _resp(payload)

# ------------------- Facilities list (for dropdowns etc.) ---------------------

@app.get("/benchmark/facilities")
def facilities(year: int):
    """
    Return one row per facility (id, name, county) for the selected year.
    Used by the UI to populate a facility picker.
    """
    d = _load_df()
    if "Year" not in d.columns:
        raise HTTPException(500, "Column 'Year' not found.")
    dy = d[d["Year"] == year].copy()
    if dy.empty:
        return _resp([])

    # Resolve columns for id/name/county
    id_col   = _find_col(dy, ["Facility_ID","Facility Id","FacilityId"])
    name_col = _find_col(dy, ["Facility_Name","Facility Name","Hospital_Name","Hospital Name"])
    county_col = _find_col(dy, ["County","County_Name","County Name"], required=False) or "County"

    # De-duplicate and sort for a clean list
    out = (dy[[id_col, name_col, county_col]]
           .dropna(subset=[id_col, name_col])
           .drop_duplicates()
           .sort_values(name_col))

    # Convert to plain dict rows (safe for JSON)
    rows = [
        {"facility_id": r[id_col],
         "facility_name": r[name_col],
         "county": r[county_col] if county_col in r else None}
        for _, r in out.iterrows()
    ]
    return _resp(rows)

# ------------------- Facility detail / trend over years -----------------------

@app.get("/benchmark/facility/{facility_id}")
def facility_details(facility_id: str):
    """
    For one facility, compute per-year time series (SIR, rates, gaps, excess).
    Also return statewide series for comparison on charts.
    """
    d = _load_df()

    # Flexible column finding
    id_col   = _find_col(d, ["Facility_ID","Facility Id","FacilityId"])
    sir_col  = "SIR_filled" if "SIR_filled" in d.columns else _find_col(d, ["SIR"])
    den_col  = "Central_line_Days" if "Central_line_Days" in d.columns else (
               "Central_Line_Days" if "Central_Line_Days" in d.columns else _find_col(d, ["Central_line_Days","Central Line Days","LineDays"]))

    # Filter rows for this facility id (string match to be robust)
    one = d[d[id_col].astype(str) == str(facility_id)].copy()
    if one.empty:
        raise HTTPException(404, "Facility not found")

    # Only keep the standard three years used in the demo dataset
    one = one[one["Year"].isin([2021, 2022, 2023])]

    # Collapse to one row per year with sums/means as appropriate
    per_year = (one.groupby("Year", as_index=False)
                  .agg(
                      sir=(sir_col, "mean"),
                      obs=("Infections_Reported","sum"),
                      pred=("Infections_Predicted","sum"),
                      line_days=(den_col,"sum"),
                  )).sort_values("Year")

    # Derived rate metrics (guard against divide-by-zero)
    line_days = pd.to_numeric(per_year["line_days"], errors="coerce")
    obs = pd.to_numeric(per_year["obs"], errors="coerce")
    pred = pd.to_numeric(per_year["pred"], errors="coerce")

    per_year["rate_per_1000"] = np.where(line_days > 0, (obs / line_days) * 1000, np.nan)
    per_year["expected_rate_per_1000"] = np.where(line_days > 0, (pred / line_days) * 1000, np.nan)
    per_year["rate_gap"] = per_year["rate_per_1000"] - per_year["expected_rate_per_1000"]
    per_year["excess_infections"] = obs - pred

    # Replace any Infs with NaN to keep JSON clean later
    for c in per_year.select_dtypes(include=[float]).columns:
        per_year[c] = per_year[c].replace([np.inf, -np.inf], np.nan)

    # Build statewide series (aggregated per year) for comparison
    state = (d[d["Year"].isin([2021, 2022, 2023])]
               .groupby("Year", as_index=False)
               .agg(obs=("Infections_Reported","sum"),
                    pred=("Infections_Predicted","sum"),
                    line_days=("Central_line_Days" if "Central_line_Days" in d.columns else
                               ("Central_Line_Days" if "Central_Line_Days" in d.columns else den_col), "sum")))
    state["sir"] = np.where(pd.to_numeric(state["pred"], errors="coerce") > 0,
                            pd.to_numeric(state["obs"], errors="coerce") / pd.to_numeric(state["pred"], errors="coerce"),
                            np.nan)
    state["rate_per_1000"] = np.where(pd.to_numeric(state["line_days"], errors="coerce") > 0,
                                      (pd.to_numeric(state["obs"], errors="coerce") / pd.to_numeric(state["line_days"], errors="coerce")) * 1000,
                                      np.nan)

    # Return JSON-safe objects (replace NaN/Inf with None)
    payload = {
        "facility_id": facility_id,
        "per_year": per_year.replace([np.inf, -np.inf], np.nan).where(pd.notna(per_year), None).to_dict(orient="records"),
        "statewide": state.rename(columns={"Year":"year"})[["year","sir","rate_per_1000"]]
                        .replace([np.inf, -np.inf], np.nan)
                        .where(pd.notna(state), None)
                        .to_dict(orient="records"),
    }
    return _resp(payload)

# ------------------- Top movers between two years -----------------------------

@app.get("/benchmark/top-movers")
def top_movers(year_from: int, year_to: int, n: int = 15):
    """
    Compute change in SIR between two years for each facility.
    Return the top N facilities with the largest improvement (most negative) first.
    """
    d = _load_df()
    sir_col  = "SIR_filled" if "SIR_filled" in d.columns else _find_col(d, ["SIR"])
    rate_col = "rate_per_1000" if "rate_per_1000" in d.columns else None

    # Build a pivot table: rows are facilities, columns are years, values are SIR averages
    idx_cols = ["Facility_ID"]
    if "Facility_Name" in d.columns: idx_cols.append("Facility_Name")
    if "County" in d.columns: idx_cols.append("County")

    a = (d.pivot_table(index=idx_cols, columns="Year", values=sir_col, aggfunc="mean")
           .reset_index())

    # If either comparison year is missing in the data, return empty.
    if (year_from not in a.columns) or (year_to not in a.columns):
        return _resp([])

    # Change = SIR(year_to) - SIR(year_from). Negative means improvement.
    a["delta_sir"] = a[year_to] - a[year_from]

    # Optionally compute change in raw rate if we have it
    if rate_col:
        r = (d.pivot_table(index=["Facility_ID"], columns="Year", values=rate_col, aggfunc="mean")
               .reset_index())
        r["delta_rate"] = r.get(year_to) - r.get(year_from)
        m = a.merge(r[["Facility_ID","delta_rate"]], on="Facility_ID", how="left")
    else:
        a["delta_rate"] = None
        m = a

    # Sort by improvement (most negative) and take top n
    m = m.sort_values("delta_sir", ascending=True).head(n)

    # Helper to round to 3 decimals or return None if invalid
    def _r3(x):
        try:
            v = float(x)
            return round(v, 3) if np.isfinite(v) else None
        except Exception:
            return None

    # Build clean rows for the UI
    out = []
    for _, row in m.iterrows():
        out.append({
            "facility_id": row.get("Facility_ID"),
            "facility_name": row.get("Facility_Name"),
            "county": row.get("County"),
            "delta_sir": _r3(row.get("delta_sir")),
            "delta_rate": _r3(row.get("delta_rate")),
        })
    return _resp(out)

# ------------------- Modeling helpers (panel + features) ----------------------

def _panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a panel (long) dataset with:
    - observed/expected rates per 1,000 (safe division)
    - lag-1 features per facility (previous year's SIR and rates)
    These are the inputs for prediction models.
    """
    # Denominator and numerators as numeric
    den = pd.to_numeric(df["Central_line_Days"], errors="coerce")
    ir = pd.to_numeric(df["Infections_Reported"], errors="coerce")
    ip = pd.to_numeric(df["Infections_Predicted"], errors="coerce")

    # Safe rate computation: if line-days <= 0 or NaN, use NaN instead of Inf
    df["obs_rate"] = np.where(den > 0, (ir / den) * 1000, np.nan)
    df["exp_rate"] = np.where(den > 0, (ip / den) * 1000, np.nan)

    # Sort so groupby().shift(1) creates correct prior-year values
    df = df.sort_values(["Facility_ID", "Year"])

    # Lagged features (previous year per facility)
    df["SIR_lag1"]      = df.groupby("Facility_ID")["SIR_filled"].shift(1)
    df["obs_rate_lag1"] = df.groupby("Facility_ID")["obs_rate"].shift(1)
    df["exp_rate_lag1"] = df.groupby("Facility_ID")["exp_rate"].shift(1)

    # Clean up any Inf and coerce to numeric
    for c in ["obs_rate","exp_rate","SIR_lag1","obs_rate_lag1","exp_rate_lag1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df

def _future_from_last_year(pan: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """
    Build a 'future' panel for forecasting one year ahead:
    - Take the most recent year per facility
    - Copy its last observed features into lag-1 slots
    - Set 'Year' to the target future year
    """
    if pan["Year"].dropna().empty:
        return pd.DataFrame()
    max_year = int(pan["Year"].dropna().max())
    base = pan[pan["Year"] == max_year].copy()
    if base.empty:
        return pd.DataFrame()

    keep = [c for c in ["Facility_ID","Facility_Name","County","SIR_filled",
                        "obs_rate","exp_rate"] if c in base.columns]
    fut = base[keep].copy()
    fut["Year"] = target_year

    # For forecasting, last observed values become "lag-1" predictors
    fut["SIR_lag1"]      = pd.to_numeric(base.get("SIR_filled"), errors="coerce").replace([np.inf, -np.inf], np.nan).values
    fut["obs_rate_lag1"] = pd.to_numeric(base.get("obs_rate"),   errors="coerce").replace([np.inf, -np.inf], np.nan).values
    fut["exp_rate_lag1"] = pd.to_numeric(base.get("exp_rate"),   errors="coerce").replace([np.inf, -np.inf], np.nan).values

    return fut

def _naive_sigma(train: pd.DataFrame) -> float:
    """
    Estimate spread (sigma) of naive residuals:
    residual = actual SIR - previous year's SIR.
    Used to build a 90% prediction interval for the naive model and fallbacks.
    """
    y = pd.to_numeric(train.get("SIR_filled"), errors="coerce")
    ylag = pd.to_numeric(train.get("SIR_lag1"), errors="coerce")
    m = y.notna() & ylag.notna()
    if not m.any():
        # If we can't compute residuals, fall back to the raw std dev of y or 0.2
        s = float(y.std(skipna=True) or 0.2)
        return s if np.isfinite(s) else 0.2
    resid = y[m] - ylag[m]
    s = float(resid.std(skipna=True) or 0.2)
    return s if np.isfinite(s) else 0.2

def _ridge_fit_predict(train: pd.DataFrame, pred: pd.DataFrame, l2: float = 0.5):
    """
    Simple ridge regression (closed-form, NumPy) on features:
      [1, SIR_lag1, obs_rate_lag1, exp_rate_lag1]
    Returns:
      yhat (predictions array aligned to 'pred' rows),
      sigma (std dev of residuals on train, used for PI)
    If a pred row has missing features, we fall back to naive (SIR_lag1).
    """
    feats = ["SIR_lag1", "obs_rate_lag1", "exp_rate_lag1"]

    # Initialize with NaNs; we'll fill what's possible
    yhat = np.full(len(pred), np.nan, dtype=float)
    naive_y = pd.to_numeric(pred.get("SIR_lag1"), errors="coerce") \
                .replace([np.inf, -np.inf], np.nan) \
                .to_numpy()

    # Keep only complete training rows
    t = train.dropna(subset=feats + ["SIR_filled"]).copy()
    if t.empty:
        # Nothing to train → fully naive
        return naive_y, 0.2

    # Build design matrix (add intercept column of ones)
    X = t[feats].to_numpy(dtype=float)
    y = t["SIR_filled"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X)), X])

    # Ridge solution: (X'X + λI)^(-1) X'y ; do not penalize intercept (I[0,0]=0)
    I = np.eye(X.shape[1], dtype=float)
    I[0, 0] = 0.0
    XtX = X.T @ X
    w = np.linalg.pinv(XtX + l2 * I) @ (X.T @ y)

    # Train residual spread (used for a Gaussian 90% PI)
    resid = y - (X @ w)
    sigma = float(np.std(resid, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.2

    # Predict where we have complete pred features
    if not pred.empty:
        mask_ok = pred[feats].notna().all(axis=1)
        if mask_ok.any():
            Xp = pred.loc[mask_ok, feats].to_numpy(dtype=float)
            Xp = np.column_stack([np.ones(len(Xp)), Xp])
            yhat[mask_ok.to_numpy()] = Xp @ w

    # Any remaining NaNs get naive fallback (use SIR_lag1)
    nan_mask = np.isnan(yhat) & ~np.isnan(naive_y)
    if nan_mask.any():
        yhat[nan_mask] = naive_y[nan_mask]

    return yhat, sigma

# ---------- Additional helpers for sklearn-based models -----------------------

def _na_quantile(arr, q: float, fallback: float = 0.3) -> float:
    """
    Compute the q-th quantile of absolute residuals safely (drop NaN/Inf).
    Used for conformal-style 90% intervals (q=0.9).
    """
    try:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return fallback
        q = min(max(q, 0.0), 1.0)
        return float(np.quantile(a, q))
    except Exception:
        return fallback

def _sklearn_fit_predict_generic(train: pd.DataFrame, pred: pd.DataFrame, model_kind: str):
    """
    Fit an sklearn regressor (ElasticNet / HGB) on features:
      [SIR_lag1, obs_rate_lag1, exp_rate_lag1]
    Returns:
      (yhat, width, model_note)
      - yhat: predictions aligned to 'pred'
      - width: half-interval for a 90% PI
          * If trained: q90 of |train residuals|  (conformal-ish)
          * If fallback: 1.64 * naive_sigma      (Gaussian-ish)
      - model_note: short text describing what was used
    We also do per-row fallback to naive when a pred row lacks any feature.
    """
    feats = ["SIR_lag1", "obs_rate_lag1", "exp_rate_lag1"]

    # Naive predictions vector (SIR_lag1) used when features are missing or training fails
    naive_y = pd.to_numeric(pred.get("SIR_lag1"), errors="coerce") \
                 .replace([np.inf, -np.inf], np.nan).to_numpy()
    yhat_default = np.full(len(pred), np.nan, dtype=float)

    # Pick a model by name; if import fails, fall back to naive
    try:
        if model_kind == "elasticnet":
            from sklearn.linear_model import ElasticNet
            model = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True,
                               max_iter=10000, random_state=0)
            base_note = "ElasticNet (sklearn)"
        elif model_kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.08,
                                                  max_iter=300, l2_regularization=1.0,
                                                  random_state=0)
            base_note = "HistGradientBoostingRegressor (sklearn)"
        else:
            # Unknown model name → naive with Gaussian PI
            sigma = _naive_sigma(train)
            return naive_y, 1.64 * sigma, f"Unknown model '{model_kind}'. Fell back to naive."
    except Exception as e:
        # sklearn not installed or import error → naive with Gaussian PI
        sigma = _naive_sigma(train)
        return naive_y, 1.64 * sigma, f"scikit-learn unavailable for '{model_kind}': {e}. Fell back to naive."

    # Build training subset with all features and a target
    t = train.dropna(subset=feats + ["SIR_filled"]).copy()
    if t.empty:
        # Common case when the dataset begins at 2021 and you backtest 2022 (no prior year to learn from)
        sigma = _naive_sigma(train)
        return naive_y, 1.64 * sigma, f"{base_note}: no train rows; fell back to naive."

    # Fit the chosen model
    X = t[feats].to_numpy(dtype=float)
    y = t["SIR_filled"].to_numpy(dtype=float)
    model.fit(X, y)

    # Conformal-ish interval width: 90th percentile of absolute residuals on train
    yhat_tr = model.predict(X)
    q90 = _na_quantile(np.abs(y - yhat_tr), 0.9, fallback=0.3)

    # Predict for rows with complete features
    p = pred.copy()
    mask_ok = p[feats].notna().all(axis=1)
    yhat = np.full(len(p), np.nan, dtype=float)
    if mask_ok.any():
        Xp = p.loc[mask_ok, feats].to_numpy(dtype=float)
        yhat[mask_ok.to_numpy()] = model.predict(Xp)

    # Per-row fallback where features are missing
    nan_mask = np.isnan(yhat) & np.isfinite(naive_y)
    if nan_mask.any():
        yhat[nan_mask] = naive_y[nan_mask]

    return yhat, q90, f"{base_note}; 90% conformal PI; per-row fallback to naive where features missing."
# ------------------------------------------------------------------------------

# ------------------- Predict endpoints (debug + main) -------------------------

@app.get("/predict/_debug")
def predict_debug():
    """
    Quick check for the prediction pipeline:
    - confirms CSV exists
    - lists the available years
    - returns total rows
    """
    try:
        d = _load_df()
        years = sorted([int(y) for y in d["Year"].dropna().unique()])
        return _resp({"csv": str(CLEAN_CSV), "exists": CLEAN_CSV.exists(), "years": years, "rows": int(len(d))})
    except Exception as e:
        return _resp({"csv": str(CLEAN_CSV), "exists": CLEAN_CSV.exists(), "error": str(e), "traceback": traceback.format_exc()})

@app.get("/predict/sir")
def predict_sir(
    target_year: int = Query(..., ge=2000, le=2100),  # which year to predict/backtest
    model: str = Query("naive"),                      # model name: 'naive', 'ridge', 'elasticnet', 'hgb'
    debug: int = 0,                                   # reserved flag (optional)
):
    """
    Core prediction endpoint.
    - If target_year is in the data → backtesting (returns metrics MAE/RMSE when possible)
    - If target_year is the next year → forecasting (no ground-truth metrics)
    Returns rows per facility with prev_year_sir, pred_sir, and 90% intervals.
    """
    try:
        # 1) Prepare panel with features/lagged values
        df = _load_df()
        pan = _panel(df.copy())

        # 2) Know which years exist
        years = sorted([int(y) for y in pan["Year"].dropna().unique()])
        print("years in predict_sit main.py\n", years)
        if not years:
            payload = {"target_year": target_year, "model": model or "naive", "n_pred": 0,
                       "error": "No Year values in dataset."}
            return _resp(_sanitize_for_json(payload))

        max_year = max(years)

        # 3) Training set: everything strictly before target_year (or before max+1 for future)
        # For backtests (e.g., 2023), you train on up to 2022; for forecasts (e.g., 2024), you train on all history.
        train_cutoff = min(target_year, max_year + 1)
        train = pan[pan["Year"] < train_cutoff].copy()
        print("train data\n", train)

        # 4) Prediction set:
        # - Backtest if target_year exists
        # - Otherwise create a synthetic "future" panel using last known year
        if target_year in years:
            pred = pan[pan["Year"] == target_year].copy()
            print("pred1 data\n", pred)
            backtest = True
        else:
            pred = _future_from_last_year(pan, target_year)
            print("pred2 data\n", pred)
            backtest = False

        if pred.empty:
            payload = {"target_year": target_year, "model": model or "naive", "n_pred": 0,
                       "error": "Nothing to predict for target year."}
            return _resp(_sanitize_for_json(payload))

        # Ensure SIR_lag1 exists (safety when external data shape changes)
        if "SIR_lag1" not in pred.columns:
            pred["SIR_lag1"] = pred.groupby("Facility_ID")["SIR_filled"].shift(1)

        # 5) Choose and run the model
        mdl = (model or "naive").lower()
        model_note = None

        if mdl == "ridge":
            # NumPy ridge fit; interval uses train sigma (Gaussian 90% ≈ ±1.64σ)
            yhat, sigma = _ridge_fit_predict(train, pred, l2=0.5)
            lo = yhat - 1.64 * sigma
            hi = yhat + 1.64 * sigma
            model_note = "Ridge (NumPy); 90% PI uses Gaussian sigma from train residuals."

        elif mdl == "elasticnet":
            # sklearn ElasticNet; interval width from q90(|residual|) on train
            yhat, q90, model_note = _sklearn_fit_predict_generic(train, pred, "elasticnet")
            lo, hi = yhat - q90, yhat + q90

        elif mdl == "hgb":
            # sklearn HistGradientBoosting; same conformal-style interval
            yhat, q90, model_note = _sklearn_fit_predict_generic(train, pred, "hgb")
            lo, hi = yhat - q90, yhat + q90

        else:
            # Plain naive: predict next SIR == last year's SIR; Gaussian interval from naive sigma
            yhat = pd.to_numeric(pred["SIR_lag1"], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy()
            sigma = _naive_sigma(train)
            lo, hi = yhat - 1.64 * sigma, yhat + 1.64 * sigma
            model_note = "Naive last-year SIR; 90% PI uses Gaussian sigma from train residuals."

        # 6) Build one output row per facility
        rows = []
        pr = pred.reset_index(drop=True)
        for i, r in pr.iterrows():
            rows.append({
                "facility_id": r.get("Facility_ID"),
                "facility_name": r.get("Facility_Name"),
                "county": r.get("County"),
                "year": int(target_year),
                "prev_year_sir": _safe(r.get("SIR_lag1"), 3),
                "pred_sir": _safe(yhat[i], 3),
                "pi90_lo": _safe(lo[i], 3),
                "pi90_hi": _safe(hi[i], 3),
                "note": "Forecast from last observed year" if not backtest else None
            })

        # 7) Backtest metrics (only if we have ground truth for that year)
        metrics = None
        if backtest and "SIR_filled" in pred.columns:
            truth = pd.to_numeric(pred["SIR_filled"], errors="coerce")
            m = (~np.isnan(yhat)) & truth.notna()
            if m.any():
                y_true = truth[m].to_numpy()
                y_pred = yhat[m.to_numpy()]
                mae = float(np.mean(np.abs(y_true - y_pred)))                 # average error size
                rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))        # penalizes big misses more
                metrics = {"n_eval": int(m.sum()), "mae": _safe(mae, 3), "rmse": _safe(rmse, 3)}

        # 8) Final payload to the UI
        payload = {
            "target_year": int(target_year),
            "model": mdl,
            "model_note": model_note,
            "n_pred": int(len(rows)),
            "metrics": metrics,
            "predictions": rows
        }
        return _resp(_sanitize_for_json(payload))  # Final extra-sanitized return

    except Exception as e:
        # Any error becomes a structured JSON with traceback for easier debugging.
        err_payload = {
            "target_year": int(target_year),
            "model": model or "naive",
            "n_pred": 0,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        return _resp(_sanitize_for_json(err_payload))  # Sanitized error payload

# ------------------- RAG router (optional plug-in) ----------------------------
# If a rag_router exists (separate file), include its endpoints under this app.
try:
    from app.api.rag_router import router as rag_router
    app.include_router(rag_router)
except Exception:
    # If it's not present (e.g., you didn't set up RAG yet), just ignore.
    pass
