import json
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from src.inference.predictor import ApartmentPricePredictor

FEATURE_NAMES = json.loads(
    (Path(__file__).parent.parent / "models" / "feature_names.json").read_text()
)


@pytest.fixture(autouse=True)
def patch_predictor(monkeypatch, tmp_path):
    rng = np.random.default_rng(42)
    X = rng.random((100, len(FEATURE_NAMES)))
    y = rng.random(100) * 500_000 + 200_000

    import joblib
    tiny_model = xgb.XGBRegressor(n_estimators=5, tree_method="hist")
    tiny_model.fit(X, y)
    joblib.dump(tiny_model, tmp_path / "xgboost_model.joblib")
    (tmp_path / "feature_names.json").write_text(json.dumps(FEATURE_NAMES))

    import main
    monkeypatch.setattr(main, "predictor", ApartmentPricePredictor(tmp_path))
