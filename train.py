"""
Training pipeline — Warsaw apartment price prediction.

Usage:
    # Train on original Kaggle CSV directory (reproduces existing model)
    python train.py --data data/raw

    # Train on fresh Otodom data
    python train.py --data data/raw/otodom_collected.parquet

    # Combine Kaggle + fresh Otodom data
    python train.py --data data/raw --supplement data/raw/otodom_collected.parquet

    # Run Optuna hyperparameter search (slower, but finds better params)
    python train.py --data data/raw --tune --trials 50
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import optuna
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.preprocessing.preprocess import FEATURES, load_and_clean

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Best params found by Optuna in notebooks/02_modeling.ipynb
BEST_PARAMS = {
    "n_estimators": 893,
    "learning_rate": 0.0276,
    "max_depth": 12,
    "min_child_weight": 3,
    "subsample": 0.832,
    "colsample_bytree": 0.904,
    "tree_method": "hist",
    "random_state": 42,
}

MODELS_DIR = Path("models")
EXPERIMENT = "warsaw-apartment-price"


def tune(X_train, y_train, X_test, y_test, n_trials: int) -> dict:
    log.info("Running Optuna (%d trials)...", n_trials)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "tree_method": "hist",
            "random_state": 42,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        return root_mean_squared_error(y_test, model.predict(X_test))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    log.info("Best RMSE from tuning: %,.0f PLN", study.best_value)
    return study.best_params | {"tree_method": "hist", "random_state": 42}


def train(
    data: Path,
    supplement: Path = None,
    run_tune: bool = False,
    n_trials: int = 50,
    model_out: Path = None,
):
    if model_out is None:
        model_out = MODELS_DIR / "xgboost_model.joblib"

    # -----------------------------------------------------------------------
    # Load & preprocess
    # -----------------------------------------------------------------------
    log.info("Loading data from %s...", data)
    df = load_and_clean(data, supplement=supplement)
    log.info("Dataset: %d rows, features: %s", len(df), FEATURES)

    features_in_data = [f for f in FEATURES if f in df.columns]
    X = df[features_in_data]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log.info("Train: %d  Test: %d", len(X_train), len(X_test))

    # -----------------------------------------------------------------------
    # Hyperparameters
    # -----------------------------------------------------------------------
    params = tune(X_train, y_train, X_test, y_test, n_trials) if run_tune else BEST_PARAMS

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    mlflow.set_experiment(EXPERIMENT)
    run_name = "tuned" if run_tune else "best-params"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("data_source", str(data))
        mlflow.log_params(params)

        log.info("Training XGBoost...")
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        mlflow.log_metrics({"rmse": rmse, "mape": round(mape, 2)})
        mlflow.xgboost.log_model(model, name="xgboost_model")

        log.info("RMSE: %s PLN  |  MAPE: %.2f%%", f"{rmse:,.0f}", mape)

    # -----------------------------------------------------------------------
    # Save model + feature list + metadata
    # -----------------------------------------------------------------------
    from datetime import datetime, timezone

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, model_out)
    features_path = MODELS_DIR / "feature_names.json"
    features_path.write_text(json.dumps(features_in_data))

    model_info = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": str(data),
        "n_samples": len(df),
        "rmse": round(rmse),
        "mape": round(mape, 2),
        "n_features": len(features_in_data),
    }
    (MODELS_DIR / "model_info.json").write_text(json.dumps(model_info, indent=2))

    log.info("Model saved to %s", model_out)
    log.info("Features saved to %s", features_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Warsaw apartment price model")
    parser.add_argument("--data", type=Path, required=True,
                        help="Kaggle CSV directory or Otodom .parquet file")
    parser.add_argument("--supplement", type=Path, default=None,
                        help="Extra Otodom .parquet to merge with --data")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter search")
    parser.add_argument("--trials", type=int, default=50,
                        help="Optuna trial count (default: 50)")
    parser.add_argument("--model-out", type=Path, default=None,
                        help="Output path for the model (default: models/xgboost_model.joblib)")
    args = parser.parse_args()

    train(
        data=args.data,
        supplement=args.supplement,
        run_tune=args.tune,
        n_trials=args.trials,
        model_out=args.model_out,
    )
