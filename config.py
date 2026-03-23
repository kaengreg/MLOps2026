from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR/"data"
SOURCE_DIR = DATA_DIR/"source"
RAW_BATCHES_DIR = DATA_DIR/"raw_batches"
EXTERNAL_DIR = DATA_DIR/"external"

DEV_DIR = BASE_DIR/"dev"
MODELS_DIR = DEV_DIR/"models"
METRICS_DIR = DEV_DIR/"metrics"
REPORTS_DIR = DEV_DIR/"reports"
LOGS_DIR = DEV_DIR/"logs"

BEST_MODEL_FILE = MODELS_DIR/"best_model.pkl"
BEST_MODEL_META_FILE = MODELS_DIR/"best_model_meta.json"
PREDICTIONS_DIR = DEV_DIR/"predictions"

STATE_DIR = BASE_DIR/"state"

SOURCE_FILE = SOURCE_DIR/"yellow_tripdata_2015-01.csv"
PIPELINE_STATE_FILE = STATE_DIR/"pipeline_state.json"

TARGET_COL = "total_amount"
PICKUP_COL = "tpep_pickup_datetime"
DROPOFF_COL = "tpep_dropoff_datetime"

RAW_FEATURE_COLUMNS = [
    "VendorID",
    "passenger_count",
    "trip_distance",
    "pickup_longitude",
    "pickup_latitude",
    "RatecodeID",
    "store_and_fwd_flag",
    "dropoff_longitude",
    "dropoff_latitude",
]

GENERATED_FEATURE_COLUMNS = [
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
    "trip_duration_min",
]

FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + GENERATED_FEATURE_COLUMNS

PRICE_COMPONENT_COLUMNS = [
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
]

BANNED_COLUMNS = PRICE_COMPONENT_COLUMNS + [TARGET_COL]

REQUIRED_COLUMNS = [
    PICKUP_COL,
    DROPOFF_COL,
    TARGET_COL,
    *RAW_FEATURE_COLUMNS,
]

BATCH_FREQ = "W" # Weekly by default, might be Daily (D), Hourly (H), Monthly (M)
MIN_ROWS_IN_BATCH = 100

RANDOM_STATE = 42
TEST_SIZE = 0.2

TRIP_DISTANCE_UPPER_QUANTILE = 0.99
TARGET_UPPER_QUANTILE = 0.99
TRIP_DURATION_UPPER_QUANTILE = 0.99