import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb


@dataclass
class PredictionResult:
    predicted_price: float
    base_price: float
    contributions: dict[str, float]


class ApartmentPricePredictor:
    def __init__(self, models_dir: Path):
        model = joblib.load(models_dir / "xgboost_model.joblib")
        with open(models_dir / "feature_names.json") as f:
            self.feature_names: list[str] = json.load(f)
        self._booster = model.get_booster()

    def predict(self, feature_values: dict[str, float | None]) -> PredictionResult:
        inputs = [
            feature_values[f] if feature_values.get(f) is not None else np.nan
            for f in self.feature_names
        ]
        dmatrix = xgb.DMatrix(np.array([inputs]), feature_names=self.feature_names)

        price = float(self._booster.predict(dmatrix)[0])

        # pred_contribs shape: (1, n_features + 1) — last column is bias (base value)
        contribs = self._booster.predict(dmatrix, pred_contribs=True)[0]
        contributions = {
            name: round(float(val))
            for name, val in zip(self.feature_names, contribs[:-1])
        }

        return PredictionResult(
            predicted_price=round(price),
            base_price=round(float(contribs[-1])),
            contributions=contributions,
        )
