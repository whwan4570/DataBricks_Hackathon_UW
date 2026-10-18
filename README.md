# TerraCast

TerraCast is a team project from the Databricks × University of Washington hackathon, where it won first place. We explored how weather and climate data can help forecast crop yields and identify unusual county-year outcomes.

## My contribution

I led the data pipeline and predictive modeling work on the Databricks platform. This included integrating crop-yield and weather data, developing features, using Delta Lake in the data workflow, tracking modeling experiments with MLflow, and training and evaluating the crop-yield models. Other team members contributed to the broader project; the award was a team result.

A LightGBM/Random Forest ensemble achieved a mean R² of approximately **0.88** in five-fold validation that held out counties. This is a cross-validation result, not a production performance claim.

## What is in this repository

- `app.py` — Streamlit prototype with yield/weather maps, simple anomaly views, and an interactive interface.
- `weather_yield_dashboard_databricks.py` — a second copy of the prototype dashboard script.
- `merge_datasets.py` — a local data-preparation prototype.
- `requirements.txt` and CSV files — dependencies and data used by the local prototype.

**Scope of the public code:** The files here show the local Streamlit prototype and a simple CSV merge. In particular, `merge_datasets.py` aggregates weather by year and joins it to yield records on year; it does not reproduce the full Databricks feature pipeline. The Databricks notebooks, Delta Lake tables, MLflow run artifacts, and code needed to independently reproduce the reported cross-validation result are not included in this repository.

## Run the local prototype

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard reads `merged_weather_yields.csv` from the repository root and retrieves a county GeoJSON file when the map is loaded, so that view needs an internet connection.
