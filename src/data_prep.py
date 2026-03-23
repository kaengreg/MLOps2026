from pathlib import Path 
import pandas as pd 

from config import SOURCE_FILE, RAW_BATCHES_DIR, REQUIRED_COLUMNS, PICKUP_COL, DROPOFF_COL, BATCH_FREQ, MIN_ROWS_IN_BATCH

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
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

def prepare_raw_batches(path=SOURCE_FILE,output_dir=RAW_BATCHES_DIR):
    df = load_source_data(path)
    batches = split_into_batches(df)
    return save_batches(batches,output_dir)

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
        