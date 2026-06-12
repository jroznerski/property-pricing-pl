import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.inference.predictor import PredictionResult

_lock = threading.Lock()

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    squareMeters      REAL,
    floor             REAL,
    floorCount        REAL,
    buildYear         REAL,
    latitude          REAL,
    longitude         REAL,
    centreDistance    REAL,
    poiCount          REAL,
    schoolDistance    REAL,
    clinicDistance    REAL,
    postOfficeDistance REAL,
    kindergartenDistance REAL,
    restaurantDistance REAL,
    collegeDistance   REAL,
    pharmacyDistance  REAL,
    date              INTEGER,
    predicted_price   REAL NOT NULL,
    base_price        REAL NOT NULL
)
"""

INSERT = """
INSERT INTO predictions (
    ts, squareMeters, floor, floorCount, buildYear,
    latitude, longitude, centreDistance, poiCount,
    schoolDistance, clinicDistance, postOfficeDistance,
    kindergartenDistance, restaurantDistance, collegeDistance,
    pharmacyDistance, date, predicted_price, base_price
) VALUES (
    :ts, :squareMeters, :floor, :floorCount, :buildYear,
    :latitude, :longitude, :centreDistance, :poiCount,
    :schoolDistance, :clinicDistance, :postOfficeDistance,
    :kindergartenDistance, :restaurantDistance, :collegeDistance,
    :pharmacyDistance, :date, :predicted_price, :base_price
)
"""


class PredictionLogger:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, check_same_thread=False)

    def log(self, features: dict, result: PredictionResult) -> None:
        row = {**features, "ts": datetime.now(timezone.utc).isoformat(),
               "predicted_price": result.predicted_price,
               "base_price": result.base_price}
        with _lock, self._connect() as conn:
            conn.execute(INSERT, row)
