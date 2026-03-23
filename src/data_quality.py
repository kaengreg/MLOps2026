from pathlib import Path
from datetime import datetime

import pandas as pd

from config import (
    PICKUP_COL,
    DROPOFF_COL,
    TARGET_COL,
    METRICS_DIR,
)
from src.features import make_time_features


DATA_QUALITY_LOG_FILE = METRICS_DIR/"data_quality_log.csv"


def count_invalid_coords(df) -> int:
    coordinate_columns = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]

    if not all(col in df.columns for col in coordinate_columns):
        return 0

    invalid_mask = (
        ~df["pickup_longitude"].between(-180, 180)
        | ~df["dropoff_longitude"].between(-180, 180)
        | ~df["pickup_latitude"].between(-90, 90)
        | ~df["dropoff_latitude"].between(-90, 90)
    )

    return int(invalid_mask.fillna(True).sum())


def count_invalid_duration(df) -> int:
    if "trip_duration_min" not in df.columns:
        return 0

    invalid_mask = (df["trip_duration_min"].isna()) | (df["trip_duration_min"] <= 0)
    return int(invalid_mask.sum())


def compute_batch_quality_metrics(df, batch_name):
    local_df = df.copy()
    local_df = make_time_features(local_df)

    metrics = {
        "batch_name": batch_name,
        "row_count": int(len(local_df)),
        "missing_total": int(local_df.isna().sum().sum()),
        "duplicate_count": int(local_df.duplicated().sum()),
        "negative_trip_distance_count": int(((local_df["trip_distance"] < 0).fillna(False)).sum()) 
                                        if "trip_distance" in local_df.columns else 0,
        "negative_target_count": int(((local_df[TARGET_COL] < 0).fillna(False)).sum()) 
                                 if TARGET_COL in local_df.columns else 0,
        "invalid_duration_count": count_invalid_duration(local_df),
        "invalid_coordinate_count": count_invalid_coords(local_df),
        "trip_distance_q99": float(local_df["trip_distance"].dropna().quantile(0.99))
                             if "trip_distance" in local_df.columns and not local_df["trip_distance"].dropna().empty
                             else None,
        "target_q99": float(local_df[TARGET_COL].dropna().quantile(0.99))
                      if TARGET_COL in local_df.columns and not local_df[TARGET_COL].dropna().empty
                      else None,
        "duration_q99": float(local_df["trip_duration_min"].dropna().quantile(0.99))
                        if "trip_duration_min" in local_df.columns and not local_df["trip_duration_min"].dropna().empty
                        else None,
        "created_at": datetime.now().isoformat(),
    }

    return metrics


def append_data_quality_log(metrics, log_file=DATA_QUALITY_LOG_FILE):
    log_file.parent.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([metrics])

    if log_file.exists():
        existing_df = pd.read_csv(log_file)
        result_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    else:
        result_df = metrics_df

    result_df.to_csv(log_file, index=False)