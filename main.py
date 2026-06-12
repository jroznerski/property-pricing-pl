from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference.logger import PredictionLogger
from src.inference.predictor import ApartmentPricePredictor

BASE_DIR = Path(__file__).parent
predictor = ApartmentPricePredictor(BASE_DIR / "models")
pred_logger = PredictionLogger(BASE_DIR / "data" / "predictions.db")

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
    return {"status": "ok", "model_features": predictor.feature_names}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ApartmentFeatures):
    feature_dict = features.model_dump()
    result = predictor.predict(feature_dict)
    pred_logger.log(feature_dict, result)
    return PredictionResponse(
        predicted_price=result.predicted_price,
        predicted_price_formatted=f"{result.predicted_price:,} PLN",
        base_price=result.base_price,
        contributions=result.contributions,
    )
