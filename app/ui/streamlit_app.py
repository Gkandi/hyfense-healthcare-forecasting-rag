import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px  # for interactive charts

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Wide layout + open sidebar
st.set_page_config(
    page_title="Hyfense Forecast — CLABSI Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- tiny CSS to make metrics pop ----------
st.markdown(
    """
    <style>
      div[data-testid="stMetricValue"] { font-weight: 800; font-size: 2.4rem; line-height: 1.1; }
      div[data-testid="stMetric"] > label { font-weight: 600; opacity: 0.95; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def safe_post(url, json=None):
    try:
        r = requests.post(url, json=json, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def get_years():
    dbg = safe_get(f"{API_URL}/debug/columns")
    if isinstance(dbg, dict) and "years" in dbg and dbg["years"]:
        return sorted(list(set(int(y) for y in dbg["years"])))
    return [2021, 2022, 2023]

def metric_card(col, label, value, fmt=None):
    try:
        if value is None:
            col.metric(label, "—")
        elif fmt:
            col.metric(label, fmt(value))
        else:
            col.metric(label, str(value))
    except Exception:
        col.metric(label, "—")

def fmt_id(x):
    try:
        f = float(x)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    s = str(x)
    return s[:-2] if s.endswith(".0") else s

# ---------------------------------------------------------
# Sidebar — global filters + movers controls
# ---------------------------------------------------------
years = get_years()
last_year = years[-1] if years else 2023

with st.sidebar:
    st.subheader("Filters")

    # Reference year (affects Benchmark metrics and facility list)
    sel_year = st.selectbox(
        "Reference year",
        years,
        index=len(years) - 1 if years else 0,
    )
    st.caption("Applies to Benchmark metrics and the facility list used in Trends.")

    st.divider()

    # --- Top movers controls (kept in sidebar) ---
    st.markdown("**Top movers**")
    mov_target = st.selectbox(
        "Compare year (target)",
        years,
        index=len(years) - 1 if years else 0,
        key="m_target_year",
    )
    base_opts = [y for y in years if y < mov_target]
    if base_opts:
        existing_base = st.session_state.get("m_base_year", base_opts[0])
        base_index = base_opts.index(existing_base) if existing_base in base_opts else 0
        mov_base = st.selectbox("Baseline year", base_opts, index=base_index, key="m_base_year")
    else:
        mov_base = None
        st.info("Pick a target later than the baseline.")

    mov_topn = st.slider("Top N", min_value=3, max_value=50, value=15, step=1, key="m_topn")

# ---------------------------------------------------------
# Tabs (REBRANDED)
# ---------------------------------------------------------
tabs = st.tabs(["Benchmark", "Trends", "Forecast", "Assist"])

# ---------------------------------------------------------
# Benchmark tab (Summary + Top movers)
# ---------------------------------------------------------
with tabs[0]:
    st.title("Hyfense Forecast")
    st.caption("Executive-ready CLABSI benchmarks & SIR movement")

    # ---- Summary (statewide) ----
    summary = safe_get(f"{API_URL}/benchmark/summary", {"year": sel_year})
    if summary and "statewide_SIR" in summary:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        metric_card(c1, "Statewide SIR", summary.get("statewide_SIR"), lambda v: f"{v:.2f}")
        metric_card(c2, "Rate / 1,000 line-days", summary.get("rate_per_1000_line_days"), lambda v: f"{v:.2f}")
        metric_card(c3, "Observed CLABSI", summary.get("observed_clabsi"), lambda v: f"{int(v)}")
        metric_card(c4, "Predicted CLABSI", summary.get("predicted_clabsi"), lambda v: f"{int(v)}")
        metric_card(c5, "Line-days", summary.get("total_line_days"), lambda v: f"{int(v):,}")
        metric_card(c6, "Facilities", summary.get("facilities_reporting"), lambda v: f"{int(v)}")

        d1, d2, d3, d4 = st.columns(4)
        metric_card(d1, "% SIR < 1", summary.get("pct_facilities_SIR_lt_1"), lambda v: f"{v:.1f}%")
        metric_card(d2, "% sig. better", summary.get("pct_facilities_significantly_better"), lambda v: f"{v:.1f}%")
        metric_card(d3, "% sig. worse", summary.get("pct_facilities_significantly_worse"), lambda v: f"{v:.1f}%")
        metric_card(d4, "% met 2020 goal", summary.get("pct_met_2020_goal"), lambda v: f"{v:.1f}%")
    else:
        st.warning("Summary not available. Ensure the API and cleaned CSV are loaded.")

    st.divider()

    # ---- Top movers (SIR change) ----
    st.subheader("Top movers (SIR change)")
    if len(years) >= 2 and mov_base is not None and mov_base < mov_target:
        movers = safe_get(
            f"{API_URL}/benchmark/top-movers",
            {"year_from": mov_base, "year_to": mov_target, "n": int(mov_topn)},
        )
        if isinstance(movers, list) and movers:
            dfm = pd.DataFrame(movers)
            if "facility_id" in dfm:
                dfm["facility_id"] = dfm["facility_id"].map(fmt_id)
            for c in ("delta_sir", "delta_rate"):
                if c in dfm:
                    dfm[c] = pd.to_numeric(dfm[c], errors="coerce").round(3)
            st.dataframe(dfm, use_container_width=True)
        else:
            st.info("No movers data for that comparison.")
    else:
        st.info("Choose a baseline earlier than the target year (see sidebar).")

# ---------------------------------------------------------
# Trends tab (Facility trend)
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Trends — Facility time series")
    facs = safe_get(f"{API_URL}/benchmark/facilities", {"year": sel_year})
    if isinstance(facs, list) and facs:
        options = {
            f"{f.get('facility_name','(Unnamed)')} ({f.get('county','')})": str(f["facility_id"])
            for f in facs
        }
        pick = st.selectbox("Choose facility", sorted(options.keys()))
        facility_id = options[pick]
        det = safe_get(f"{API_URL}/benchmark/facility/{facility_id}")
        if det and "per_year" in det:
            per = pd.DataFrame(det["per_year"])

            # Year label for charts (avoid 2,021 formatting)
            if "Year" in per.columns:
                per["YearLabel"] = per["Year"].astype("Int64").astype(str)
                per = per.sort_values("Year")

            # SIR chart
            if "sir" in per.columns:
                st.line_chart(per.set_index("YearLabel")[["sir"]], height=240)

            # Observed vs expected rate chart
            show_cols = [c for c in ["rate_per_1000", "expected_rate_per_1000"] if c in per.columns]
            if show_cols:
                st.line_chart(per.set_index("YearLabel")[show_cols], height=240)

            # ---------------- Raw facility data (SAFE VERSION) ----------------
            with st.expander("Raw facility data", expanded=True):
                want = [
                    "Year","sir","rate_per_1000","expected_rate_per_1000",
                    "rate_gap","obs","pred","line_days","excess_infections",
                ]
                per_disp = per.reindex(columns=want).copy()

                if per_disp.empty:
                    st.info("No rows to display for this facility.")
                else:
                    num_cols = [
                        "sir","rate_per_1000","expected_rate_per_1000",
                        "rate_gap","obs","pred","line_days","excess_infections"
                    ]
                    for c in num_cols:
                        if c in per_disp.columns:
                            per_disp[c] = pd.to_numeric(per_disp[c], errors="coerce")

                    if "sir" in per_disp: per_disp["sir"] = per_disp["sir"].round(3)
                    if "rate_per_1000" in per_disp: per_disp["rate_per_1000"] = per_disp["rate_per_1000"].round(3)
                    if "expected_rate_per_1000" in per_disp: per_disp["expected_rate_per_1000"] = per_disp["expected_rate_per_1000"].round(3)
                    if "rate_gap" in per_disp: per_disp["rate_gap"] = per_disp["rate_gap"].round(3)
                    if "pred" in per_disp: per_disp["pred"] = per_disp["pred"].round(2)

                    if "Year" in per_disp:
                        per_disp["Year"] = per_disp["Year"].astype("Int64").astype(str)

                    per_disp = per_disp.rename(columns={
                        "Year": "Year",
                        "sir": "SIR",
                        "rate_per_1000": "Rate / 1,000 line-days",
                        "expected_rate_per_1000": "Expected rate / 1,000",
                        "rate_gap": "Rate gap (obs - exp)",
                        "obs": "Observed",
                        "pred": "Predicted",
                        "line_days": "Line-days",
                        "excess_infections": "Excess infections (Obs - Pred)",
                    }).fillna("")

                    st.dataframe(per_disp, use_container_width=True)

                    csv_cols = [
                        "Year","SIR","Rate / 1,000 line-days","Expected rate / 1,000",
                        "Rate gap (obs - exp)","Observed","Predicted","Line-days",
                        "Excess infections (Obs - Pred)"
                    ]
                    csv_df = per_disp[csv_cols].copy()
                    st.download_button(
                        "Download facility data (CSV)",
                        data=csv_df.to_csv(index=False).encode("utf-8"),
                        file_name="facility_trend.csv",
                        mime="text/csv",
                    )
        else:
            st.info("No details for that facility.")
    else:
        st.info("No facilities for that year, or API error.")

# ---------------------------------------------------------
# Forecast tab — controls + table + charts
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Forecast — Next-year SIR per facility")

    # Build target-year options:
    all_years = years or [2021, 2022, 2023]
    if len(all_years) >= 2:
        # drop the first year (no lag) AND hide 2022 from backtest options
        usable_backtest_years = [y for y in all_years[1:] if y != 2022]
    else:
        usable_backtest_years = []

    next_year = all_years[-1] + 1
    pred_options = usable_backtest_years + [next_year]
    default_index = len(usable_backtest_years) - 1 if usable_backtest_years else 0

    pred_year = st.selectbox(
        "Target year",
        pred_options,
        index=default_index,
        help="Backtest years need a prior baseline; the first dataset year is hidden.",
    )
    pred_model = st.radio("Model", ["ridge", "naive", "elasticnet", "hgb"], index=0, horizontal=True)

    preds = safe_get(f"{API_URL}/predict/sir", {"target_year": pred_year, "model": pred_model})

    if isinstance(preds, dict):
        if preds.get("error"):
            st.warning(f"Prediction engine: {preds['error']}")

        rows = preds.get("predictions") or []
        dfp = pd.DataFrame(rows)

        if preds.get("model_note"):
            st.caption(f"Note: {preds['model_note']}")

        if preds.get("metrics"):
            m = preds["metrics"]
            st.caption(f"Backtest: n={m.get('n_eval','-')} • MAE={m.get('mae','-')} • RMSE={m.get('rmse','-')}")

        if not dfp.empty:
            # ---- tidy display table (NO 'note' column) ----
            if "facility_id" in dfp:
                dfp["facility_id"] = dfp["facility_id"].apply(
                    lambda v: str(int(float(v))) if pd.notna(v) and str(v).replace('.', '', 1).isdigit() else str(v)
                )
            if "year" in dfp:
                dfp["year"] = pd.to_numeric(dfp["year"], errors="coerce").astype("Int64").astype(str)
            for c in ["prev_year_sir", "pred_sir", "pi90_lo", "pi90_hi"]:
                if c in dfp:
                    dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

            show_cols = ["facility_id","facility_name","county","year","prev_year_sir","pred_sir","pi90_lo","pi90_hi"]
            dfp = dfp.reindex(columns=[c for c in show_cols if c in dfp.columns])

            st.dataframe(
                dfp.sort_values("pred_sir", ascending=False, na_position="last").head(25),
                use_container_width=True
            )

            # -------------------- Charts --------------------
            st.markdown("### Visualizations")

            dfv = dfp.copy()
            # numeric for charts
            for c in ["prev_year_sir", "pred_sir", "pi90_lo", "pi90_hi"]:
                if c in dfv:
                    dfv[c] = pd.to_numeric(dfv[c], errors="coerce")
            # label for hover / axes
            def _label(r):
                name = r.get("facility_name") or r.get("facility_id")
                county = r.get("county") or ""
                lbl = f"{name} ({county})" if county else str(name)
                return (lbl[:29] + "…") if len(lbl) > 32 else lbl
            dfv["label"] = dfv.apply(_label, axis=1)

            # 1) Scatter: previous vs predicted SIR
            if {"prev_year_sir","pred_sir"}.issubset(dfv.columns):
                fig1 = px.scatter(
                    dfv.dropna(subset=["prev_year_sir","pred_sir"]),
                    x="prev_year_sir", y="pred_sir",
                    hover_name="facility_name",
                    hover_data={
                        "county": True,
                        "year": True,
                        "prev_year_sir": ":.3f",
                        "pred_sir": ":.3f",
                        "pi90_lo": ":.3f",
                        "pi90_hi": ":.3f",
                    },
                    labels={"prev_year_sir":"Previous year SIR", "pred_sir":"Predicted SIR"},
                    title="Prev vs Pred SIR",
                )
                # y=x reference line
                dd = dfv[["prev_year_sir","pred_sir"]].dropna()
                if not dd.empty:
                    lo = float(min(dd.min()))
                    hi = float(max(dd.max()))
                    fig1.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash"))
                st.plotly_chart(fig1, use_container_width=True)

            # 2) Bars with 90% PI for top 20 predicted SIR
            if {"pred_sir","pi90_lo","pi90_hi"}.issubset(dfv.columns):
                top = dfv.dropna(subset=["pred_sir"]).sort_values("pred_sir", ascending=False).head(20).copy()
                if not top.empty:
                    top["err_plus"] = top["pi90_hi"] - top["pred_sir"]
                    top["err_minus"] = top["pred_sir"] - top["pi90_lo"]
                    fig2 = px.bar(
                        top,
                        x="label", y="pred_sir",
                        error_y="err_plus", error_y_minus="err_minus",
                        hover_data={
                            "facility_name": True,
                            "county": True,
                            "pred_sir": ":.3f",
                            "pi90_lo": ":.3f",
                            "pi90_hi": ":.3f",
                        },
                        labels={"label":"Facility", "pred_sir":"Predicted SIR"},
                        title="Top facilities by predicted SIR (with 90% PI)",
                    )
                    fig2.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig2, use_container_width=True)

            # 3) Distribution of predicted SIR
            if "pred_sir" in dfv:
                fig3 = px.histogram(
                    dfv.dropna(subset=["pred_sir"]),
                    x="pred_sir", nbins=30,
                    labels={"pred_sir":"Predicted SIR"},
                    title="Distribution of predicted SIR",
                )
                fig3.add_vline(x=1.0, line_dash="dash")
                st.plotly_chart(fig3, use_container_width=True)

            # 4) Change vs previous (pred − prev)
            if {"prev_year_sir","pred_sir"}.issubset(dfv.columns):
                dfv["delta"] = dfv["pred_sir"] - dfv["prev_year_sir"]
                delta_top = dfv.dropna(subset=["delta"]).sort_values("delta", ascending=False).head(20)
                if not delta_top.empty:
                    delta_top["direction"] = np.where(delta_top["delta"] >= 0, "Higher vs prev", "Lower vs prev")
                    fig4 = px.bar(
                        delta_top,
                        x="label", y="delta", color="direction",
                        hover_data={
                            "facility_name": True,
                            "county": True,
                            "prev_year_sir": ":.3f",
                            "pred_sir": ":.3f",
                            "delta": ":.3f",
                        },
                        labels={"label":"Facility", "delta":"Δ SIR (pred − prev)"},
                        title="Change from previous year (Δ SIR)",
                    )
                    fig4.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig4, use_container_width=True)

        else:
            st.info("No predictions returned for the selected year/model.")
    else:
        st.error("Unexpected response from the predictions API.")

# ---------------------------------------------------------
# Assist (RAG) tab
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Assist — Ask the docs")
    st.caption("Search your local project docs (README, FAQ, Quickstart, Methods, Data Dictionary, etc.).")

    col_a, col_b = st.columns([0.75, 0.25])
    with col_a:
        q = st.text_input("Your question", placeholder="e.g., What does SIR mean and how is it computed?")
    with col_b:
        topk = st.slider("Results", 1, 8, 4)

    c1, c2 = st.columns([0.25, 0.75])
    with c1:
        if st.button("Refresh index"):
            out = safe_post(f"{API_URL}/rag/refresh", {})
            if out: st.success(f"Indexed {out.get('files',0)} files, {out.get('chunks',0)} chunks.")
    with c2:
        srcs = safe_get(f"{API_URL}/rag/sources")
        if isinstance(srcs, dict) and srcs.get("files_indexed", 0) >= 0:
            st.caption(f"Indexed dir: {srcs.get('docs_dir','?')} • Files: {srcs.get('files_indexed',0)} • Chunks: {srcs.get('chunks',0)}")

    if q:
        res = safe_post(f"{API_URL}/rag/ask", {"question": q, "k": int(topk)})
        if isinstance(res, dict):
            st.markdown("#### Answer")
            st.write(res.get("answer", ""))
            cits = res.get("citations", [])
            if cits:
                st.markdown("#### Sources")
                dfc = pd.DataFrame(cits)[["rank","title","score","path"]]
                dfc["score"] = pd.to_numeric(dfc["score"], errors="coerce").round(3)
                st.dataframe(dfc, use_container_width=True)
            # Expanders with matched chunks
            chunks = res.get("chunks", [])
            for i, ch in enumerate(chunks, start=1):
                with st.expander(f"[{i}] {ch.get('title')}  •  score={ch.get('score'):.3f}"):
                    st.markdown(ch.get("text",""))
        else:
            st.info("No answer returned.")

    # Move the old Legend here for convenience
    with st.expander("Legend & formulas", expanded=False):
        st.markdown(r"""
**SIR (Standardized Infection Ratio)**  
:  \( \text{SIR} = \frac{\text{Observed infections}}{\text{Predicted infections}} \).  
Interpretation: **< 1** better-than-expected; **> 1** worse-than-expected.

**Rate / 1,000 line-days**  
:  \( \frac{\text{Observed}}{\text{Central line-days}} \times 1000 \).

**Expected rate / 1,000**  
:  \( \frac{\text{Predicted}}{\text{Central line-days}} \times 1000 \).

**Rate gap (obs − exp)**  
:  Observed rate − Expected rate (per 1,000 line-days).

**Excess infections (Obs − Pred)**  
:  Count difference (not rate).

**% met 2020 goal**  
:  Share of facilities marked as having met the state 2020 CLABSI goal in the source.
""")
