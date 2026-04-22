from pathlib import Path
from datetime import datetime

import pandas as pd

from config import SOURCE_FILE, RAW_BATCHES_DIR, REQUIRED_COLUMNS, PICKUP_COL, DROPOFF_COL, BATCH_FREQ, MIN_ROWS_IN_BATCH, METRICS_DIR

BATCH_META_LOG_FILE = METRICS_DIR / "batch_meta_log.csv"

def normalize_column_names(df):
    df = df.copy()

    rename_map = {"RateCodeID": "RatecodeID"}

    columns_to_rename = {
        old_name: new_name
        for old_name, new_name in rename_map.items()
        if old_name in df.columns and new_name not in df.columns
    }

    if columns_to_rename:
        df = df.rename(columns=columns_to_rename)

    return df


def load_source_data(source_file):
    if not source_file.exists():
        raise FileNotFoundError(f"No source file '{source_file}' was found, check the path/filename")
    
    df = pd.read_csv(source_file)
    df = normalize_column_names(df)


    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df[PICKUP_COL] = pd.to_datetime(df[PICKUP_COL],errors='coerce')
    df[DROPOFF_COL] = pd.to_datetime(df[DROPOFF_COL],errors='coerce')

    df = df.dropna(subset=[PICKUP_COL, DROPOFF_COL]).copy()
    df = df.sort_values(PICKUP_COL).reset_index(drop=True)

    return df 

def split_into_batches(df, freq=BATCH_FREQ, min_rows=MIN_ROWS_IN_BATCH):
    if df.empty:
        return []
    
    local_df = df.copy()
    local_df["batch_period"] = local_df[PICKUP_COL].dt.to_period(freq)

    batches = []
    for _, batch_df in local_df.groupby("batch_period"):
        batch_df = batch_df.drop(columns="batch_period").reset_index(drop=True)
        if len(batch_df) >= min_rows:
            batches.append(batch_df)

    return batches

def save_batches(batches, output_dir=RAW_BATCHES_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for idx, batch_df in enumerate(batches, start=1):
        batch_start = batch_df[PICKUP_COL].min().strftime("%Y-%m-%d")
        batch_end = batch_df[PICKUP_COL].max().strftime("%Y-%m-%d")
        filename = f"batch_{idx:04d}_{batch_start}_{batch_end}.csv"
        file_path = output_dir/filename

        batch_df.to_csv(file_path, index=False)
        saved_paths.append(file_path)

    return saved_paths


def compute_batch_meta(batch_df, batch_path):
    pickup_min = batch_df[PICKUP_COL].min()
    pickup_max = batch_df[PICKUP_COL].max()

    return {
        "batch_name": batch_path.name,
        "row_count": int(len(batch_df)),
        "column_count": int(batch_df.shape[1]),
        "pickup_min": pickup_min.isoformat() if pd.notna(pickup_min) else None,
        "pickup_max": pickup_max.isoformat() if pd.notna(pickup_max) else None,
        "missing_total": int(batch_df.isna().sum().sum()),
        "duplicate_count": int(batch_df.duplicated().sum()),
    }


def append_batch_meta_log(meta_records, log_file = BATCH_META_LOG_FILE):
    if not meta_records:
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    meta_df = pd.DataFrame(meta_records)

    if log_file.exists():
        existing_df = pd.read_csv(log_file)
        result_df = pd.concat([existing_df, meta_df], ignore_index=True)
    else:
        result_df = meta_df

    result_df.to_csv(log_file, index=False)


def prepare_raw_batches(path=SOURCE_FILE, output_dir=RAW_BATCHES_DIR):
    df = load_source_data(path)
    batches = split_into_batches(df)
    saved_paths = save_batches(batches, output_dir)

    meta_records = [compute_batch_meta(batch_df=batch_df, batch_path=batch_path)
                    for batch_df, batch_path in zip(batches, saved_paths)]
    
    append_batch_meta_log(meta_records)

    return saved_paths

def list_batch_files(raw_batch_dir=RAW_BATCHES_DIR):
    if not raw_batch_dir.exists():
        return []

    return sorted(raw_batch_dir.glob('batch_*.csv'))

def load_batch(batch_path):
    df = pd.read_csv(batch_path)
    df = normalize_column_names(df)

    df[PICKUP_COL] = pd.to_datetime(df[PICKUP_COL], errors='coerce')
    df[DROPOFF_COL] = pd.to_datetime(df[DROPOFF_COL], errors='coerce')
    return df
        