import pandas as pd
from config import (PICKUP_COL, DROPOFF_COL, TARGET_COL, BANNED_COLUMNS, FEATURE_COLUMNS, 
                    GENERATED_FEATURE_COLUMNS, TRIP_DISTANCE_UPPER_QUANTILE, TARGET_UPPER_QUANTILE, TRIP_DURATION_UPPER_QUANTILE)

def make_time_features(df):
    df = df.copy()
    df[PICKUP_COL] = pd.to_datetime(df[PICKUP_COL],errors='coerce')
    df[DROPOFF_COL] = pd.to_datetime(df[DROPOFF_COL],errors='coerce')

    df["pickup_hour"] = df[PICKUP_COL].dt.hour
    df["pickup_weekday"] = df[PICKUP_COL].dt.weekday
    df["pickup_month"] = df[PICKUP_COL].dt.month
    df["trip_duration_min"] = (df[DROPOFF_COL] - df[PICKUP_COL]).dt.total_seconds() / 60.0

    return df 

def apply_quantile_filter(df, column, upper_quantile):
    if column not in df.columns:
        return df
    
    non_null_vals = df[column].dropna()
    if non_null_vals.empty:
        return df
    
    upper_bound = non_null_vals.quantile(upper_quantile)
    return df[df[column] <= upper_bound] 

def df_cleaning(df):
    df = df.copy()
    df = df.dropna(subset=[PICKUP_COL, DROPOFF_COL, TARGET_COL])

    if "trip_distance" in df.columns:
        # Filtering negative distance and extremal values
        df = df[df['trip_distance'].notna()]
        df = df[df['trip_distance'] >= 0]
        df = apply_quantile_filter(df=df, column='trip_distance', upper_quantile=TRIP_DISTANCE_UPPER_QUANTILE)

    df = df[df[TARGET_COL] >= 0]
    df = apply_quantile_filter(df=df, column=TARGET_COL, upper_quantile=TARGET_UPPER_QUANTILE)

    coordinates = ["pickup_longitude",
                   "pickup_latitude",
                   "dropoff_longitude",
                   "dropoff_latitude"]
    
    existing_coords = [col for col in coordinates if col in df.columns]
    if existing_coords:
        # Filtering incorrect coordinates
        df = df.dropna(subset=existing_coords)
        df = df[df['pickup_longitude'].between(-180, 180)]
        df = df[df['dropoff_longitude'].between(-180, 180)]
        df = df[df['pickup_latitude'].between(-90, 90)]
        df = df[df['dropoff_latitude'].between(-90, 90)]

    if "trip_duration_min" in df.columns:
        # Filtering negative duration trips
        df = df[df["trip_duration_min"] > 0]
        df = apply_quantile_filter(df=df, column="trip_duration_min", upper_quantile=TRIP_DURATION_UPPER_QUANTILE)

    df = df.drop_duplicates()

    return df.reset_index(drop=True)
    
def prepare_dataset(df):
    df = make_time_features(df)
    df = df_cleaning(df)

    missing = [col for col in GENERATED_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing additional feature columns: {missing}")
    return df 


def model_data(df):
    df = prepare_dataset(df)
    
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns after feature tuning: {missing}")
    
    banned = sorted(set(FEATURE_COLUMNS) & set(BANNED_COLUMNS))
    if banned:
        raise ValueError(f"Banned columns found in feature set: {banned}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].copy()

    return X, y

