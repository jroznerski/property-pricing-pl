import sqlite3

import pytest

from src.inference.logger import PredictionLogger
from src.inference.predictor import PredictionResult

FEATURES = {
    "squareMeters": 55.0, "floor": 3.0, "floorCount": 5.0, "buildYear": 2005.0,
    "latitude": 52.23, "longitude": 21.01, "centreDistance": 5.0, "poiCount": 20.0,
    "schoolDistance": 0.3, "clinicDistance": 0.5, "postOfficeDistance": 0.4,
    "kindergartenDistance": 0.2, "restaurantDistance": 0.2, "collegeDistance": 1.5,
    "pharmacyDistance": 0.3, "date": 5,
}

RESULT = PredictionResult(
    predicted_price=650_000,
    base_price=420_000,
    contributions={"squareMeters": 80_000, "latitude": 50_000},
)


@pytest.fixture()
def logger(tmp_path):
    return PredictionLogger(tmp_path / "predictions.db")


def test_log_creates_row(logger, tmp_path):
    logger.log(FEATURES, RESULT)

    conn = sqlite3.connect(tmp_path / "predictions.db")
    rows = conn.execute("SELECT * FROM predictions").fetchall()
    conn.close()

    assert len(rows) == 1


def test_log_stores_correct_values(logger, tmp_path):
    logger.log(FEATURES, RESULT)

    conn = sqlite3.connect(tmp_path / "predictions.db")
    row = conn.execute(
        "SELECT squareMeters, predicted_price, base_price FROM predictions"
    ).fetchone()
    conn.close()

    assert row[0] == 55.0
    assert row[1] == 650_000
    assert row[2] == 420_000


def test_log_multiple_rows(logger, tmp_path):
    logger.log(FEATURES, RESULT)
    logger.log(FEATURES, RESULT)

    conn = sqlite3.connect(tmp_path / "predictions.db")
    count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()

    assert count == 2


def test_log_stores_timestamp(logger, tmp_path):
    logger.log(FEATURES, RESULT)

    conn = sqlite3.connect(tmp_path / "predictions.db")
    ts = conn.execute("SELECT ts FROM predictions").fetchone()[0]
    conn.close()

    assert ts is not None and len(ts) > 0
