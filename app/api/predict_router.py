from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import os
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SK_OK = True
except Exception:
    # If scikit-learn is missing, we’ll stick to naive
    SK_OK = False

router = APIRouter(prefix="/predict", tags=["predict"])

# This is set in main.py before router import; we also read a sensible default.
CLEAN_CSV = Path(os.getenv(
    "CLEAN_CLABSI_CSV",
    "demo_data/clabsi/cdph_clabsi_odp_2021_2022_2023_clean.csv"
))

# ------------------------- IO & feature prep -------------------------
def _read_df() -> pd.DataFrame:
    if not CLEAN_CSV.exists():
        raise HTTPException(500, f"Clean CSV not found: {CLEAN_CSV}")
    df = pd.read_csv(CLEAN_CSV)
    df.columns = [c.strip() for c in df.columns]

    # Normalize known variants
    if "Central_line_Days" not in df.columns and "Central_Line_Days" in df.columns:
        df = df.rename(columns={"Central_Line_Days": "Central_line_Days"})
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # Ensure numeric base columns if present
    for c in ["Infections_Reported", "Infections_Predicted", "Central_line_Days"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Helper columns (if missing)
    if "SIR_filled" not in df.columns and {"Infections_Reported","Infections_Predicted"}.issubset(df.columns):
        pred = pd.to_numeric(df["Infections_Predicted"], errors="coerce")
        with pd.option_context("mode.use_inf_as_na", True):
            df["SIR_filled"] = df["Infections_Reported"] / pred

    if "rate_per_1000" not in df.columns and {"Infections_Reported","Central_line_Days"}.issubset(df.columns):
        den = pd.to_numeric(df["Central_line_Days"], errors="coerce")
        with pd.option_context("mode.use_inf_as_na", True):
            df["rate_per_1000"] = (df["Infections_Reported"] / den) * 1000

    return df

def _build_panel(df: pd.DataFrame) -> pd.DataFrame:
    # Guard required columns
    need = {"Facility_ID","Year","Infections_Reported","Infections_Predicted","Central_line_Days"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise HTTPException(500, f"Missing required columns: {missing}")

    # Derived rates
    den = pd.to_numeric(df["Central_line_Days"], errors="coerce")
    df["obs_rate"] = (pd.to_numeric(df["Infections_Reported"], errors="coerce") / den) * 1000
    df["exp_rate"] = (pd.to_numeric(df["Infections_Predicted"], errors="coerce") / den) * 1000

    # Lag features per facility
    df = df.sort_values(["Facility_ID", "Year"])
    for col in ["SIR_filled", "obs_rate", "exp_rate", "Central_line_Days"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("Facility_ID")[col].shift(1)
        else:
            df[f"{col}_lag1"] = np.nan
    df["log_line_days_lag1"] = np.log1p(pd.to_numeric(df["Central_line_Days_lag1"], errors="coerce"))
    return df

def _future_pred_from_last_year(pan: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Create a synthetic prediction frame (using last observed year as lag1)."""
    if pan["Year"].dropna().empty:
        return pd.DataFrame()
    max_year = int(pan["Year"].dropna().max())
    base = pan[pan["Year"] == max_year].copy()
    if base.empty:
        return pd.DataFrame()

    keep = [c for c in ["Facility_ID", "Facility_Name", "County"] if c in base.columns]
    fut = base[keep].copy()
    fut["Year"] = target_year

    # Fill lag1 features from last observed year contemporaneous values
    fut["SIR_filled_lag1"] = pd.to_numeric(base.get("SIR_filled"), errors="coerce").values
    fut["obs_rate_lag1"] = pd.to_numeric(base.get("obs_rate"), errors="coerce").values
    fut["exp_rate_lag1"] = pd.to_numeric(base.get("exp_rate"), errors="coerce").values
    fut["Central_line_Days_lag1"] = pd.to_numeric(base.get("Central_line_Days"), errors="coerce").values
    fut["log_line_days_lag1"] = np.log1p(fut["Central_line_Days_lag1"])

    # For convenience in UI
    fut["prev_year_sir"] = fut["SIR_filled_lag1"]
    return fut

# ------------------------- models & metrics -------------------------
def _train_ridge(train: pd.DataFrame):
    if not SK_OK:
        return None, ["SIR_filled_lag1","obs_rate_lag1","exp_rate_lag1","log_line_days_lag1"], None

    feats = ["SIR_filled_lag1","obs_rate_lag1","exp_rate_lag1","log_line_days_lag1"]
    # Build X,y with complete rows
    X = train[feats].astype(float)
    y = train["SIR_filled"].astype(float) if "SIR_filled" in train.columns else None
    if y is None:
        return None, feats, None
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    if len(X) < 10:
        return None, feats, None  # not enough data; caller will fallback to naive

    mdl = Ridge(alpha=0.5, random_state=42)
    mdl.fit(X, y)
    yhat = mdl.predict(X)
    resid_std = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return mdl, feats, resid_std

def _naive_sigma(train: pd.DataFrame) -> float:
    """Std of residuals for naive baseline as a quick uncertainty proxy."""
    if "SIR_filled" not in train.columns:
        return 0.2
    y = pd.to_numeric(train["SIR_filled"], errors="coerce")
    ylag = pd.to_numeric(train.get("SIR_filled_lag1"), errors="coerce")
    m = y.notna() & ylag.notna()
    if not m.any():
        s = float(y.std(skipna=True) or 0.2)
        return s if np.isfinite(s) else 0.2
    resid = y[m] - ylag[m]
    s = float(resid.std(skipna=True) or 0.2)
    return s if np.isfinite(s) else 0.2

# ------------------------- diagnostics -------------------------
@router.get("/_debug")
def predict_debug():
    return {"csv": str(CLEAN_CSV), "exists": CLEAN_CSV.exists(), "sklearn_available": SK_OK}

@router.get("/_diag")
def predict_diag():
    try:
        df = _read_df()
        info = {
            "cols": list(df.columns),
            "rows": int(len(df)),
            "years": sorted([int(x) for x in df["Year"].dropna().unique().tolist()]) if "Year" in df.columns else [],
        }
        return info
    except Exception as e:
        return {"error": str(e)}

# ------------------------- main endpoint -------------------------
@router.get("/sir")
def predict_sir(
    target_year: int = Query(..., ge=2000, le=2100),
    model: str = Query("ridge", pattern="^(ridge|naive)$"),
):
    """
    If target_year is in data → backtest against truth.
    If target_year > max(data year) → future forecast using last-year lags.
    Always falls back to naive if ridge can't train.
    """
    df = _read_df()
    if "Year" not in df.columns or "Facility_ID" not in df.columns:
        raise HTTPException(500, "Required columns missing (Year or Facility_ID).")

    pan = _build_panel(df)
    years = sorted([int(y) for y in pan["Year"].dropna().unique()])
    if not years:
        return {"target_year": target_year, "model": model, "n_pred": 0, "error": "No Year values in dataset."}
    max_year = max(years)

    # Train on all rows strictly before target (or up to last available year)
    train_cutoff = min(target_year, max_year + 1)
    train = pan[pan["Year"] < train_cutoff].copy()
    if train.empty:
        return {"target_year": target_year, "model": model, "n_pred": 0, "error": "No training data before target year."}

    # Prediction frame: backtest or future
    if target_year in years:
        pred = pan[pan["Year"] == target_year].copy()
        backtest = True
    else:
        pred = _future_pred_from_last_year(pan, target_year)
        backtest = False

    if pred.empty:
        return {"target_year": target_year, "model": model, "n_pred": 0, "error": "Nothing to predict for target year."}

    # Choose model with graceful fallback
    use_naive = (model == "naive") or (not SK_OK)
    mdl = feats = resid_std = None
    if not use_naive:
        mdl, feats, resid_std = _train_ridge(train)
        if mdl is None:
            use_naive = True

    # Predictions + simple intervals
    if use_naive:
        if "SIR_filled_lag1" not in pred.columns:
            # Backtest case: derive lag1 from panel if missing
            pred["SIR_filled_lag1"] = pred.groupby("Facility_ID")["SIR_filled"].shift(1)
        naive_s = _naive_sigma(train)
        yhat = pd.to_numeric(pred["SIR_filled_lag1"], errors="coerce").to_numpy()
        lo = yhat - 1.64 * naive_s
        hi = yhat + 1.64 * naive_s
        model_used = "naive"
    else:
        feats = feats or ["SIR_filled_lag1","obs_rate_lag1","exp_rate_lag1","log_line_days_lag1"]
        Xp = pred.reindex(columns=feats).astype(float)
        mask = Xp.notna().all(axis=1)
        yhat = np.full(len(pred), np.nan, dtype=float)
        if mask.any():
            yhat[mask.values] = mdl.predict(Xp[mask])
        sigma = resid_std if (resid_std is not None and np.isfinite(resid_std)) else 0.2
        lo = yhat - 1.64 * sigma
        hi = yhat + 1.64 * sigma
        model_used = "ridge"

    # Build response
    out_rows = []
    pred_reset = pred.reset_index(drop=True)
    for i, r in pred_reset.iterrows():
        out_rows.append({
            "facility_id": r.get("Facility_ID"),
            "facility_name": r.get("Facility_Name"),
            "county": r.get("County"),
            "year": int(target_year),
            "prev_year_sir": float(r.get("SIR_filled_lag1")) if pd.notna(r.get("SIR_filled_lag1")) else None,
            "pred_sir": None if np.isnan(yhat[i]) else round(float(yhat[i]), 3),
            "pi90_lo": None if np.isnan(yhat[i]) else round(float(lo[i]), 3),
            "pi90_hi": None if np.isnan(yhat[i]) else round(float(hi[i]), 3),
            "note": "Forecast from last observed year" if not backtest else None
        })

    # Backtest metrics if we have truth in target_year
    metrics = None
    if backtest and "SIR_filled" in pred.columns:
        truth = pd.to_numeric(pred["SIR_filled"], errors="coerce")
        m = (~np.isnan(yhat)) & truth.notna()
        if m.any():
            y_true = truth[m].to_numpy()
            y_pred = yhat[m.to_numpy()]
            if SK_OK:
                mae = float(mean_absolute_error(y_true, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            else:
                # If sklearn not available we can compute simple MAE/RMSE manually
                mae = float(np.mean(np.abs(y_true - y_pred)))
                rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            metrics = {"n_eval": int(m.sum()), "mae": round(mae, 3), "rmse": round(rmse, 3)}

    return {
        "target_year": int(target_year),
        "model": model_used,
        "n_pred": int(len(out_rows)),
        "metrics": metrics,
        "predictions": out_rows
    }
