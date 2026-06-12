import json
import numpy as np
import pytest
import xgboost as xgb
import joblib
from pathlib import Path

from src.inference.predictor import ApartmentPricePredictor


@pytest.fixture()
def predictor(tmp_path):
    feature_names = [
        "squareMeters", "floor", "floorCount", "buildYear",
        "latitude", "longitude", "centreDistance", "poiCount",
        "schoolDistance", "clinicDistance", "postOfficeDistance",
        "kindergartenDistance", "restaurantDistance", "collegeDistance",
        "pharmacyDistance", "date",
    ]
    X = np.random.rand(50, len(feature_names))
    y = np.random.rand(50) * 500_000 + 200_000
    model = xgb.XGBRegressor(n_estimators=3, tree_method="hist")
    model.fit(X, y)

    joblib.dump(model, tmp_path / "xgboost_model.joblib")
    (tmp_path / "feature_names.json").write_text(json.dumps(feature_names))
    return ApartmentPricePredictor(tmp_path)


SAMPLE_FEATURES = {
    "squareMeters": 55.0, "floor": 3.0, "floorCount": 5.0, "buildYear": 2005.0,
    "latitude": 52.23, "longitude": 21.01, "centreDistance": 5.0, "poiCount": 20.0,
    "schoolDistance": 0.3, "clinicDistance": 0.5, "postOfficeDistance": 0.4,
    "kindergartenDistance": 0.2, "restaurantDistance": 0.2, "collegeDistance": 1.5,
    "pharmacyDistance": 0.3, "date": 5,
}


def test_predict_returns_result(predictor):
    result = predictor.predict(SAMPLE_FEATURES)
    assert result.predicted_price > 0
    assert isinstance(result.contributions, dict)
    assert set(result.contributions.keys()) == set(predictor.feature_names)


def test_contributions_sum_to_prediction(predictor):
    result = predictor.predict(SAMPLE_FEATURES)
    total = result.base_price + sum(result.contributions.values())
    assert abs(total - result.predicted_price) < 10


def test_predict_with_null_optional_fields(predictor):
    features = {**SAMPLE_FEATURES, "floor": None, "floorCount": None, "buildYear": None}
    result = predictor.predict(features)
    assert result.predicted_price > 0
