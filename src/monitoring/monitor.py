import json
import sqlite3
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from evidently import Report, Regression, Dataset, DataDefinition
from evidently.presets import RegressionPreset, DataDriftPreset

BASE_DIR = Path(__file__).parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports"
PREDICTIONS_DB = BASE_DIR / "data" / "predictions.db"
MIN_LOGGED_ROWS = 50


def _load_logged_predictions(feature_names: list[str]) -> pd.DataFrame | None:
    if not PREDICTIONS_DB.exists():
        return None
    conn = sqlite3.connect(PREDICTIONS_DB)
    df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()
    if len(df) < MIN_LOGGED_ROWS:
        return None
    df = df.rename(columns={"predicted_price": "prediction"})
    df["target"] = df["prediction"]  # no ground truth yet; use prediction as proxy
    return df[feature_names + ["prediction", "target"]]


def generate_report(output_path: Path = None) -> Path:
    df = pd.read_parquet(BASE_DIR / "data/processed/warsaw_apartments.parquet")
    model = joblib.load(BASE_DIR / "models/xgboost_model.joblib")

    with open(BASE_DIR / "models/feature_names.json") as f:
        feature_names = json.load(f)

    X = df[feature_names]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ref = X_train.copy()
    ref["target"] = y_train.values
    ref["prediction"] = model.predict(X_train)

    logged = _load_logged_predictions(feature_names)
    if logged is not None:
        cur = logged
    else:
        cur = X_test.copy()
        cur["target"] = y_test.values
        cur["prediction"] = model.predict(X_test)

    data_def = DataDefinition(
        regression=[Regression(target="target", prediction="prediction")]
    )
    ref_ds = Dataset.from_pandas(ref, data_definition=data_def)
    cur_ds = Dataset.from_pandas(cur, data_definition=data_def)

    report = Report([RegressionPreset(), DataDriftPreset()])
    run = report.run(reference_data=ref_ds, current_data=cur_ds)

    if output_path is None:
        REPORTS_DIR.mkdir(exist_ok=True)
        output_path = REPORTS_DIR / "monitoring_report.html"

    run.save_html(str(output_path))
    return output_path
