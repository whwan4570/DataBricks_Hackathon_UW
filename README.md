# TerraCast

TerraCast is a team project from the Databricks × University of Washington hackathon in 2026, where it won first place. We explored how historical crop yield, weather, and climate data can help forecast corn and soybean yields and surface unusual county-year outcomes. Presented at Data + AI Summit 2026.

## My contribution

I built Python, SQL, Spark, and Databricks pipelines to integrate crop, weather, and climate data, and contributed to LightGBM/Random Forest modeling and model evaluation. I also presented the team project at Data + AI Summit 2026. The hackathon award was a team result.

## Evaluation record

| Evaluation | Mean R² | Scope |
| --- | --- | --- |
| **Historical county-held-out 5-fold CV** | **0.8815** | Historical MLflow runs; LightGBM 60% + Random Forest 40% ensemble |
| **Later walk-forward time CV** | **0.8483** | A separate evaluation across time; not combined with county CV |

Dataset: **6,823 county-year/crop observations**. These are cross-validation scores, not test-set results or claims about the latest deployed model.

An October 2026 audit traced a previously displayed 0.6440 value to a hard-coded dashboard constant without evaluation-run provenance. It also identified an index-alignment issue in out-of-fold prediction storage that affected stored predictions, residuals, and downstream anomaly counts, but not the fold-level CV arrays used to calculate R². The previously reported 31-anomaly / 0.45% figure is withdrawn pending a canonical rerun.

[Read the methodology and full audit](https://whwan4570.github.io/terracast-evaluation.html).

## What is in this repository

- `app.py` — Streamlit prototype with yield/weather maps, simple exploratory anomaly views, and an interactive interface.
- `weather_yield_dashboard_databricks.py` — a second copy of the prototype dashboard script.
- `merge_datasets.py` — a local data-preparation prototype.
- `requirements.txt` and CSV files — dependencies and data used by the local prototype.

**Scope of the public code:** The files here show the local Streamlit prototype and a simple CSV merge. In particular, `merge_datasets.py` aggregates weather by year and joins it to yield records on year; it does not reproduce the full Databricks feature pipeline. The Databricks notebooks, Delta Lake tables, MLflow run artifacts, and code needed to independently reproduce the reported cross-validation result are not included in this repository. The prototype's exploratory flags are not validated model-anomaly counts.

## Run the local prototype

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard reads `merged_weather_yields.csv` from the repository root and retrieves a county GeoJSON file when the map is loaded, so that view needs an internet connection.
