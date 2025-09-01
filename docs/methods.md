# Methods — SIR Benchmarks & Forecasts

**Keywords:** methods, modeling, naive, ridge, elasticnet, hist gradient boosting, conformal, backtest, metrics

This document explains the modeling choices, what each tab shows, and how to read the outputs.

---

## Data & Features (one-step ahead)
Per facility and year we build **lag-1 features**:
- **SIR_lag1** = last year’s SIR  
- **obs_rate_lag1** = last year’s observed infections per 1,000 line-days  
- **exp_rate_lag1** = last year’s predicted infections per 1,000 line-days  

These avoid leaking future information, so the backtests are fair.

---

## Models
### 1) Naive
- **Idea:** next SIR ≈ last-year SIR.  
- **Why:** strong baseline; useful when data is sparse.  
- **Uncertainty:** Gaussian interval with width from residual spread (train).

### 2) Ridge (NumPy; closed-form)
- **Idea:** linear regression with L2 penalty on coefficients (not intercept).  
- **Features:** SIR_lag1, obs_rate_lag1, exp_rate_lag1.  
- **Uncertainty:** Gaussian using residual std on train.

### 3) ElasticNet (scikit-learn)
- **Idea:** linear model with combined L1/L2, can shrink or zero coefficients.  
- **When helpful:** if some features are weak or noisy.  
- **Uncertainty:** **Conformal** (90th percentile of absolute residuals on train).

### 4) HistGradientBoosting (scikit-learn)
- **Idea:** non-linear, tree-based boosting on histograms; can capture interactions.  
- **Uncertainty:** **Conformal** (same approach as above).

> If training is impossible (e.g., first true backtest year), sklearn models **fall back to Naive** predictions with a proper interval and a clear note.

---

## Backtesting & Metrics
When the target year already has ground truth:
- **MAE** (mean absolute error): average size of mistakes.  
- **RMSE**: like MAE but punishes large misses more.  
More years ⇒ better comparisons.

---

## Reading the Forecast (Predictions) Tab
- **prev_year_sir**: the baseline (what happened last year).  
- **pred_sir**: model’s estimate for the target year.  
- **pi90_lo / pi90_hi**: 90% prediction interval (uncertainty band).  
Charts: scatter (prev vs pred), top bars with error bars, distribution, and delta (pred − prev).

---

## Limitations
- Three historical years means **limited training**; treat intervals and ranks as **directional**.  
- Missing features force **row-wise fallback** to Naive (called out in the model note).  
- Different modeling families will disagree most where the signal is weak.

---

## See also
- Data sources (CDPH CLABSI): `data_sources.md`  
- Preprocessing & Data Quality steps: `preprocessing_pipeline.md`
