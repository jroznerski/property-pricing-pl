import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
model = joblib.load(BASE_DIR / "models" / "xgboost_model.joblib")
with open(BASE_DIR / "models" / "feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

app = FastAPI(
    title="Warsaw Apartment Price Predictor",
    version="1.0.0",
    description="Predicts apartment prices in Warsaw based on location and property features.",
)


class ApartmentFeatures(BaseModel):
    squareMeters: float = Field(..., gt=0, example=55.0)
    floor: float | None = Field(None, example=3.0)
    floorCount: float | None = Field(None, example=5.0)
    buildYear: float | None = Field(None, example=2005.0)
    latitude: float = Field(..., example=52.23)
    longitude: float = Field(..., example=21.01)
    centreDistance: float = Field(..., example=5.0)
    poiCount: float = Field(..., example=20.0)
    schoolDistance: float = Field(..., example=0.3)
    clinicDistance: float = Field(..., example=0.5)
    postOfficeDistance: float = Field(..., example=0.4)
    kindergartenDistance: float = Field(..., example=0.2)
    restaurantDistance: float = Field(..., example=0.2)
    collegeDistance: float = Field(..., example=1.5)
    pharmacyDistance: float = Field(..., example=0.3)
    date: int = Field(..., example=5, description="Encoded month: 0=01.2024 ... 10=12.2023")


class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str


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

    prediction = model.predict([input_values])[0]

    return PredictionResponse(
        predicted_price=round(float(prediction)),
        predicted_price_formatted=f"{round(float(prediction)):,} PLN"
    )
