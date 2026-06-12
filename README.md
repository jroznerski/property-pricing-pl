# Real Estate ML System

Production-style machine learning system for predicting apartment prices in Warsaw. Trained on ~19 months of Polish real estate listings (Aug 2023 – Jun 2024) and refreshed weekly with live Otodom data.

## What it does

- Predicts apartment sale prices based on size, floor, build year, location, and distances to amenities
- Explains each prediction with SHAP feature contributions (XGBoost native SHAP)
- Logs every prediction to SQLite for real drift monitoring over time
- Monitors model quality and feature drift via Evidently (uses logged predictions once 50+ rows exist, falls back to test split)
- Tracks experiments with MLflow
- Retrains automatically every Sunday via GitHub Actions

## Architecture

```
┌─────────────────┐        ┌──────────────────────────────┐
│  Streamlit UI   │──────▶ │  FastAPI  (port 8000)         │
│  (port 8501)    │        │  POST /predict                │
│                 │        │  POST /batch-predict          │
│  Monitoring tab │        │  GET  /model-info             │
│  (Evidently)    │        │  GET  /health                 │
└─────────────────┘        └──────────────────────────────┘
                                        │
                               SQLite prediction log
                               (data/predictions.db)
```

## Running with Docker

```bash
docker compose up --build
```

| Service   | URL                        |
|-----------|----------------------------|
| Streamlit | http://localhost:8501       |
| API docs  | http://localhost:8000/docs  |

## Running locally

```bash
pip install -r requirements.txt

# Start the API
uvicorn main:app --reload

# Start the UI (separate terminal)
streamlit run streamlit_app.py
```

## API

### `POST /predict`

Returns the predicted price, price per sqm, and per-feature SHAP contributions.

```json
{
  "squareMeters": 55.0,
  "floor": 3,
  "floorCount": 5,
  "buildYear": 2005,
  "latitude": 52.23,
  "longitude": 21.01,
  "centreDistance": 5.0,
  "poiCount": 20,
  "schoolDistance": 0.3,
  "clinicDistance": 0.5,
  "postOfficeDistance": 0.4,
  "kindergartenDistance": 0.2,
  "restaurantDistance": 0.2,
  "collegeDistance": 1.5,
  "pharmacyDistance": 0.3,
  "date": 5
}
```

`date` encodes the listing month: `0` = January 2024 … `10` = December 2023 (see `streamlit_app.py` for the full mapping).  
`floor`, `floorCount`, and `buildYear` are optional — pass `null` if unknown.

Response includes `predicted_price`, `predicted_price_formatted`, `price_per_sqm`, `base_price`, and `contributions` (SHAP values per feature).

### `POST /batch-predict`

Same schema as `/predict` but accepts a JSON array of apartments. Returns a list of prediction responses.

### `GET /model-info`

Returns metadata about the currently loaded model: training date, data source, sample count, RMSE, MAPE, and feature list.

### `GET /health`

Returns `{"status": "ok"}`. Used for container liveness probes.

## Retraining

A GitHub Actions workflow (`.github/workflows/retrain.yml`) runs every Sunday at 03:00 UTC:

1. Scrapes fresh listings from Otodom (15 pages ≈ 540 listings)
2. Enriches with POI distances from OpenStreetMap
3. Retrains XGBoost on the fresh data
4. Commits the updated model and `model_info.json` back to `main`

Can also be triggered manually from the GitHub Actions UI with a configurable page count.

To collect data manually:

```bash
python -m src.data_collection.collect_data --pages 10
```

To retrain manually:

```bash
# On Kaggle data
python train.py --data data/raw

# On fresh Otodom data
python train.py --data data/raw/otodom_collected.parquet

# Combine both
python train.py --data data/raw --supplement data/raw/otodom_collected.parquet

# With Optuna hyperparameter search
python train.py --data data/raw --tune --trials 50
```

## Tests

```bash
pytest tests/ -v
```

CI runs on every push and PR to `main` via GitHub Actions.

## Tech Stack

| Layer            | Tool                              |
|------------------|-----------------------------------|
| Model            | XGBoost                           |
| Explainability   | SHAP (XGBoost native)             |
| Hyperparameters  | Optuna                            |
| API              | FastAPI + Uvicorn                 |
| UI               | Streamlit                         |
| Monitoring       | Evidently                         |
| Experiment tracking | MLflow                         |
| Prediction logging | SQLite                          |
| Data collection  | Otodom scraper + Overpass API     |
| Containers       | Docker Compose                    |
| CI / Retraining  | GitHub Actions                    |
| Data             | Kaggle Warsaw apartments + Otodom |
