import json
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

FEATURE_NAMES = json.loads(
    (Path(__file__).parent.parent / "models" / "feature_names.json").read_text()
)


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    rng = np.random.default_rng(42)
    X = rng.random((100, len(FEATURE_NAMES)))
    y = rng.random(100) * 500_000 + 200_000

    tiny_model = xgb.XGBRegressor(n_estimators=5, tree_method="hist")
    tiny_model.fit(X, y)

    import main
    monkeypatch.setattr(main, "model", tiny_model)
    monkeypatch.setattr(main, "FEATURE_NAMES", FEATURE_NAMES)
