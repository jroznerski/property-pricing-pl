import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
model = joblib.load(BASE_DIR / "models" / "xgboost_model.joblib")
with open(BASE_DIR / "models" / "feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

booster = model.get_booster()

app = FastAPI(
    title="Warsaw Apartment Price Predictor",
    version="1.0.0",
    description="Predicts apartment prices in Warsaw based on location and property features.",
)


class ApartmentFeatures(BaseModel):
    squareMeters: float = Field(..., gt=0, json_schema_extra={"example": 55.0})
    floor: float | None = Field(None, json_schema_extra={"example": 3.0})
    floorCount: float | None = Field(None, json_schema_extra={"example": 5.0})
    buildYear: float | None = Field(None, json_schema_extra={"example": 2005.0})
    latitude: float = Field(..., json_schema_extra={"example": 52.23})
    longitude: float = Field(..., json_schema_extra={"example": 21.01})
    centreDistance: float = Field(..., json_schema_extra={"example": 5.0})
    poiCount: float = Field(..., json_schema_extra={"example": 20.0})
    schoolDistance: float = Field(..., json_schema_extra={"example": 0.3})
    clinicDistance: float = Field(..., json_schema_extra={"example": 0.5})
    postOfficeDistance: float = Field(..., json_schema_extra={"example": 0.4})
    kindergartenDistance: float = Field(..., json_schema_extra={"example": 0.2})
    restaurantDistance: float = Field(..., json_schema_extra={"example": 0.2})
    collegeDistance: float = Field(..., json_schema_extra={"example": 1.5})
    pharmacyDistance: float = Field(..., json_schema_extra={"example": 0.3})
    date: int = Field(..., description="Encoded month: 0=01.2024 ... 10=12.2023", json_schema_extra={"example": 5})


class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    base_price: float
    contributions: dict[str, float]


@app.get("/")
def root():
    return {"status": "ok", "model_features": FEATURE_NAMES}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ApartmentFeatures):
    data = features.model_dump()

    try:
        input_values = [data[f] if data[f] is not None else np.nan for f in FEATURE_NAMES]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")

    dmatrix = xgb.DMatrix(np.array([input_values]), feature_names=FEATURE_NAMES)
    prediction = float(booster.predict(dmatrix)[0])

    # pred_contribs shape: (1, n_features + 1) — last column is the bias (base value)
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]
    shap_values = contribs[:-1]
    base_value = contribs[-1]

    contributions = {
        name: round(float(val)) for name, val in zip(FEATURE_NAMES, shap_values)
    }

    return PredictionResponse(
        predicted_price=round(prediction),
        predicted_price_formatted=f"{round(prediction):,} PLN",
        base_price=round(float(base_value)),
        contributions=contributions,
    )
