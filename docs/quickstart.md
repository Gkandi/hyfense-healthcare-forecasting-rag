# Quickstart (1 page)

1) **Create a virtual environment & install**

```bash
python -m venv .venv
# Windows:
. .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
 
```

2) **Run the API**
```bash
uvicorn app.api.main:app --reload --port 8000
```
Optional: `set CLEAN_CLABSI_CSV=demo_data/clabsi/cdph_clabsi_odp_2021_2022_2023_clean.csv`

3) **Run the UI**
```bash
streamlit run app/ui/streamlit_app.py  --server.port 8501   
```
Optional: `set API_URL=http://localhost:8000`

4) **Use the app**
- **Benchmark** → statewide metrics for the selected year.  
- **Top movers** → pick two years; see biggest SIR changes.  
- **Trends** → pick a hospital; view SIR and rates over time; export CSV.  
- **Forecast** → choose year & model; sort facilities by predicted SIR; see uncertainty.  

5) **Assist** Q&A (RAG)
- **Legend** → simple definitions & formulas.
The assistant can answer questions based on the docs in `docs/`. If you change these files, re-index them.