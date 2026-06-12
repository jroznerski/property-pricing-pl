import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "squareMeters": 55.0,
    "floor": 3.0,
    "floorCount": 5.0,
    "buildYear": 2005.0,
    "latitude": 52.23,
    "longitude": 21.01,
    "centreDistance": 5.0,
    "poiCount": 20.0,
    "schoolDistance": 0.3,
    "clinicDistance": 0.5,
    "postOfficeDistance": 0.4,
    "kindergartenDistance": 0.2,
    "restaurantDistance": 0.2,
    "collegeDistance": 1.5,
    "pharmacyDistance": 0.3,
    "date": 5,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert isinstance(data["features"], list)


def test_root_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_features" in response.json()


def test_predict_returns_price():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert "predicted_price_formatted" in data
    assert "PLN" in data["predicted_price_formatted"]


def test_predict_returns_shap_contributions():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "contributions" in data
    assert "base_price" in data
    assert set(data["contributions"].keys()) == set(VALID_PAYLOAD.keys()) - {"date"} | {"date"}


def test_predict_contributions_sum_to_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    total = data["base_price"] + sum(data["contributions"].values())
    assert abs(total - data["predicted_price"]) < 10


def test_predict_optional_fields_can_be_null():
    payload = {**VALID_PAYLOAD, "floor": None, "floorCount": None, "buildYear": None}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_missing_required_field_returns_422():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "squareMeters"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_area_returns_422():
    payload = {**VALID_PAYLOAD, "squareMeters": -10.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_predict_returns_list():
    response = client.post("/batch-predict", json=[VALID_PAYLOAD, VALID_PAYLOAD])
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2


def test_batch_predict_each_item_has_price():
    second = {**VALID_PAYLOAD, "squareMeters": 80.0}
    response = client.post("/batch-predict", json=[VALID_PAYLOAD, second])
    data = response.json()
    for item in data:
        assert "predicted_price" in item
        assert "PLN" in item["predicted_price_formatted"]
        assert "contributions" in item


def test_batch_predict_empty_list_returns_422():
    response = client.post("/batch-predict", json=[])
    assert response.status_code == 422


def test_batch_predict_invalid_item_returns_422():
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "squareMeters"}
    response = client.post("/batch-predict", json=[VALID_PAYLOAD, bad])
    assert response.status_code == 422
