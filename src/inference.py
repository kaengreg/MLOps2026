from pathlib import Path
from datetime import datetime
import json
import pickle
import math

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    TARGET_COL,
    BEST_MODEL_FILE,
    BEST_MODEL_META_FILE,
    PREDICTIONS_DIR,
)
from src.features import prepare_dataset
from src.data_prep import normalize_column_names


def load_saved_model():
    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(f"Saved model not found: {BEST_MODEL_FILE}")

    if not BEST_MODEL_META_FILE.exists():
        raise FileNotFoundError(f"Saved model metadata not found: {BEST_MODEL_META_FILE}")

    with open(BEST_MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    with open(BEST_MODEL_META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, meta


def prepare_catboost_inference_data(df, categorical_features, numerical_features):
    local_df = df.copy()

    for col in categorical_features:
        if col in local_df.columns:
            local_df[col] = local_df[col].astype("string").fillna("missing")

    for col in numerical_features:
        if col in local_df.columns:
            median_value = local_df[col].median()
            local_df[col] = local_df[col].fillna(median_value)

    return local_df


def inference_metrics(df, predictions):
    if TARGET_COL not in df.columns:
        return None

    valid_mask = df[TARGET_COL].notna()
    if valid_mask.sum() == 0:
        return None

    y_true = df.loc[valid_mask, TARGET_COL]
    y_pred = pd.Series(predictions, index=df.index).loc[valid_mask]

    metrics = {
        "rows_with_target": int(valid_mask.sum()),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    return metrics


def predict_from_file(input_file):
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_df = pd.read_csv(input_path)
    raw_df = normalize_column_names(raw_df)

    prepared_df = prepare_dataset(raw_df)

    model, meta = load_saved_model()
    feature_columns = meta["feature_columns"]

    missing_features = [col for col in feature_columns if col not in prepared_df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns for inference: {missing_features}")

    X = prepared_df[feature_columns].copy()
    model_type = meta["model_type"]

    if model_type == "catboost":
        X_for_pred = prepare_catboost_inference_data(
            X,
            categorical_features=meta["categorical_features"],
            numerical_features=meta["numerical_features"],
        )
        predictions = model.predict(X_for_pred)

    elif model_type.startswith("sklearn_pipeline"):
        predictions = model.predict(X)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    result_df = prepared_df.copy()
    result_df["predict"] = predictions

    metrics = inference_metrics(result_df, predictions)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = PREDICTIONS_DIR/f"predictions_{timestamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    if metrics is not None:
        metrics_path = PREDICTIONS_DIR/f"predictions_{timestamp}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Inference metrics saved to: {metrics_path}")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    return output_path

