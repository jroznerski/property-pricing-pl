# Real Estate ML System

Production-style machine learning system for predicting apartment prices in Warsaw. Trained on ~19 months of Polish real estate listings (Aug 2023 – Jun 2024).

## What it does

- Predicts apartment sale prices based on size, floor, build year, location, and distances to amenities
- Explains each prediction with SHAP feature contributions (powered by XGBoost's native SHAP)
- Monitors model quality and data drift via an Evidently report (regression metrics + drift detection)
- Tracks experiments with MLflow

## Architecture

```
┌─────────────────┐        ┌──────────────────────┐
│  Streamlit UI   │──────▶ │  FastAPI  (port 8000) │
│  (port 8501)    │        │  /predict             │
│                 │        │  XGBoost + SHAP       │
│  Monitoring tab │        └──────────────────────┘
│  (Evidently)    │
└─────────────────┘
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

`POST /predict` — returns the predicted price and per-feature SHAP contributions.

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

## Tests

```bash
pytest tests/ -v
```

CI runs on every push and PR to `main` via GitHub Actions.

## Tech Stack

| Layer        | Tool                        |
|--------------|-----------------------------|
| Model        | XGBoost                     |
| Explainability | SHAP (XGBoost native)     |
| API          | FastAPI + Uvicorn           |
| UI           | Streamlit                   |
| Monitoring   | Evidently                   |
| Experiments  | MLflow                      |
| Containers   | Docker Compose              |
| CI           | GitHub Actions              |
| Data         | Polish real estate listings (Kaggle) |
