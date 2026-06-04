"""
Preprocessing for two data sources:

  Kaggle CSVs  — raw monthly files downloaded from Kaggle (apartments_pl_*.csv)
  Otodom parquet — collected via src/data_collection/collect_data.py

Both paths return a DataFrame with FEATURES + 'price' columns, cleaned and
ready for train/test split.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "squareMeters", "floor", "floorCount", "buildYear",
    "latitude", "longitude", "centreDistance", "poiCount",
    "schoolDistance", "clinicDistance", "postOfficeDistance",
    "kindergartenDistance", "restaurantDistance", "collegeDistance",
    "pharmacyDistance", "date",
]

# Features the notebook identified as below 0.5% importance — dropped for Kaggle source
_LOW_IMPORTANCE = [
    "ownership", "hasSecurity", "hasParkingSpace", "hasBalcony",
    "hasStorageRoom", "rooms", "hasElevator", "type", "buildingMaterial",
]

# date string → integer encoding used in the original dataset
_DATE_MAP = {
    "08.2023": 6, "09.2023": 7, "10.2023": 8, "11.2023": 9, "12.2023": 10,
    "01.2024": 0, "02.2024": 1, "03.2024": 2, "04.2024": 3,
    "05.2024": 4, "06.2024": 5,
}

_DISTANCE_COLS = [
    "schoolDistance", "clinicDistance", "postOfficeDistance",
    "kindergartenDistance", "restaurantDistance", "collegeDistance",
    "pharmacyDistance",
]


# ---------------------------------------------------------------------------
# Kaggle CSV source
# ---------------------------------------------------------------------------

def _load_kaggle_csvs(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("apartments_pl_*.csv"))
    if not files:
        raise FileNotFoundError(f"No apartments_pl_*.csv files found in {data_dir}")

    chunks = []
    for f in files:
        month = f.stem.replace("apartments_pl_", "").replace("_", ".")  # 2023_08 → 2023.08
        # Normalise to MM.YYYY format expected by _DATE_MAP
        parts = month.split(".")
        if len(parts) == 2:
            month = f"{parts[1].zfill(2)}.{parts[0]}"  # 2023.08 → 08.2023
        chunks.append(pd.read_csv(f).assign(date=month))

    df = pd.concat(chunks, ignore_index=True)
    df = df[df["city"] == "warszawa"].copy()
    df.drop(columns=["city", "id"], errors="ignore", inplace=True)

    # Encode date string → int
    df["date"] = df["date"].map(_DATE_MAP)

    # Drop condition (76% missing) and low-importance features
    df.drop(columns=["condition"] + _LOW_IMPORTANCE, errors="ignore", inplace=True)

    return df


# ---------------------------------------------------------------------------
# Otodom parquet source
# ---------------------------------------------------------------------------

def _load_otodom_parquet(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    # Otodom data already has numeric features; date is already encoded
    return df


# ---------------------------------------------------------------------------
# Shared cleaning
# ---------------------------------------------------------------------------

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing distances with column median
    for col in _DISTANCE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df["pricePerSqm"] = df["price"] / df["squareMeters"]
    low = df["pricePerSqm"].quantile(0.01)
    high = df["pricePerSqm"].quantile(0.99)
    df = df[(df["pricePerSqm"] >= low) & (df["pricePerSqm"] <= high)].copy()
    df.drop(columns=["pricePerSqm"], inplace=True)

    # Keep only model features + target
    cols = [c for c in FEATURES if c in df.columns] + ["price"]
    df = df[cols].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_clean(source: Path, supplement: Path = None) -> pd.DataFrame:
    """
    Load and clean data from a Kaggle CSV directory or an Otodom parquet file.

    Args:
        source:     path to a directory of Kaggle CSVs  OR  a single .parquet file
        supplement: optional extra parquet to merge with the primary source

    Returns:
        DataFrame with FEATURES + 'price' columns.
    """
    if source.is_dir():
        df = _load_kaggle_csvs(source)
    elif source.suffix == ".parquet":
        df = _load_otodom_parquet(source)
    else:
        raise ValueError(f"source must be a directory (Kaggle CSVs) or a .parquet file, got: {source}")

    if supplement is not None:
        extra = _load_otodom_parquet(supplement)
        df = pd.concat([df, extra], ignore_index=True)

    return _clean(df)
