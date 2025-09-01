# FAQ — Hyfense CLABSI

### What does SIR mean in plain English?
**SIR = Observed infections ÷ Predicted infections.**  
Below 1 means fewer infections than expected after risk adjustment; above 1 means more than expected.

---

### Why do the two facility trend charts sometimes look similar?
- The **SIR** chart is risk‑adjusted.  
- The **Observed vs Expected rate** chart is raw incidence per 1,000 line‑days.  
If exposure (line‑days) changes modestly, both can have similar shapes. Still, the **level** of SIR (<1 vs >1) and the **gap** between observed and expected rate tell the meaningful story.

---

### Why doesn’t the Top movers table change?
Make sure **target year** is later than **baseline year**. Also verify both years exist in the data: check `GET /debug/columns`.

---

### What is “% met 2020 goal”?
A flag from the source indicating whether a facility met the **state’s 2020 CLABSI goal**. It’s a quick, historical success indicator.

---

### Why can’t I predict 2021?
Predictions need a **previous year** to build lagged features. 2021 is the first year in the dataset, so there’s no prior year to learn from.

---

### Why do 2022 predictions sometimes look like a copy of 2021?
Sparse rows (missing features or first available year per facility) cause a **fallback to naive**, which uses last year’s SIR. We always disclose this in the **model note**.

---

### How are the prediction intervals computed?
We use the model’s residual standard deviation (**σ**) from training and build a **90% interval**:  
**PI = prediction ± 1.64 × σ**.  
It’s a rough “uncertainty band”—wide when the model is noisy, narrower when stable.

---

### Why did facility IDs show a decimal (.0)?
They sometimes come as floats in CSVs. We format them as integers in the UI so they look like `60012345` instead of `60012345.0`.

---

### How do I test the API quickly?
Use `curl` or a browser:
```bash
curl "http://localhost:8000/benchmark/summary?year=2023"
curl "http://localhost:8000/predict/sir?target_year=2023&model=hgb"
```
See **README** for more endpoints.

---

### How do I add docs to power Q&A?
Put Markdown files in the project’s `/docs` folder (e.g., `README.md`, `faq.md`, `data_dictionary.md`). Use your RAG indexer has a “refresh” endpoint, call it after changes.

---

### Why do I see “No predictions available”?
Most common reasons:
- You picked the first dataset year (no lag available).  
- Features are missing for all rows and every model fell back to naive, but even naive had no prior value.  
- Dataset path is wrong (see `/health` and `/debug/columns`).

---

### Can I change the theme?
Yes. Streamlit supports a theme in `.streamlit/config.toml` (e.g., dark mode).

---

### Is the UI mobile‑friendly?
It’s responsive, but complex tables are easiest on a laptop/desktop.
