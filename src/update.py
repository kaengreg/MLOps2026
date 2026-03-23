from datetime import datetime
from pathlib import Path
import json
import pickle

import pandas as pd
from tqdm import tqdm

from config import (
    BEST_MODEL_FILE,
    BEST_MODEL_META_FILE,
    METRICS_DIR,
    MODELS_DIR,
)
from src.data_prep import list_batch_files, load_batch
from src.data_quality import compute_batch_quality_metrics, append_data_quality_log
from src.features import model_data
from src.train import (
    evaluate_regression,
    save_best_model,
    save_model,
    build_model_registry,
    build_tree_pipeline,
    load_batch_splits,
)
from src.utils import save_pipeline_state, load_pipeline_state

UPDATE_LOG_FILE = METRICS_DIR/"update_log.csv"


def load_model_by_key(model_key):
    model_file = MODELS_DIR/f"current_{model_key}.pkl"
    meta_file = MODELS_DIR/f"current_{model_key}_meta.json"

    if not model_file.exists():
        raise FileNotFoundError(f"Saved model not found for model_key='{model_key}': {model_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Saved model metadata not found for model_key='{model_key}': {meta_file}")

    print(f"Loading saved model for model_key='{model_key}' from: {model_file}")
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, meta


def load_saved_model():
    print(f"Loading saved best model from: {BEST_MODEL_FILE}")
    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(f"Saved model not found: {BEST_MODEL_FILE}")
    if not BEST_MODEL_META_FILE.exists():
        raise FileNotFoundError(f"Saved model metadata not found: {BEST_MODEL_META_FILE}")

    with open(BEST_MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(BEST_MODEL_META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, meta


def prepare_catboost_inference_data(X, categorical_features, numerical_features):
    X_local = X.copy()

    for col in categorical_features:
        if col in X_local.columns:
            X_local[col] = X_local[col].astype("string").fillna("missing")

    for col in numerical_features:
        if col in X_local.columns:
            median_value = X_local[col].median()
            X_local[col] = X_local[col].fillna(median_value)

    return X_local


def prepare_batch_for_model(batch_df, meta):
    print("Preparing batch features and target for model update...")
    X_batch, y_batch = model_data(batch_df)

    feature_columns = meta["feature_columns"]
    missing_features = [col for col in feature_columns if col not in X_batch.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns for update: {missing_features}")

    X_batch = X_batch[feature_columns].copy()
    return X_batch, y_batch


def update_single_model(model_key, batch_df, available_batch_files):
    model, meta = load_model_by_key(model_key)

    X_batch, y_batch = prepare_batch_for_model(batch_df, meta)
    print(f"Starting model update for model_key={model_key}, model_type={meta['model_type']} on {len(X_batch)} rows...")
    model_type = meta["model_type"]

    if model_type == "catboost":
        X_batch_prepared = prepare_catboost_inference_data(
            X_batch,
            categorical_features=meta["categorical_features"],
            numerical_features=meta["numerical_features"],
        )

        updated_model = model.copy()
        updated_model.fit(
            X_batch_prepared,
            y_batch,
            cat_features=meta["categorical_features"],
            init_model=model,
            use_best_model=False,
            verbose=False,
        )
        batch_predictions = updated_model.predict(X_batch_prepared)
        update_strategy = "catboost_continue_fit"

        meta_to_save = {
            **meta,
            "updated_at": datetime.now().isoformat(),
            "update_strategy": update_strategy,
            "last_incremental_batch_rows": int(len(X_batch)),
            "last_incremental_batch_metrics": evaluate_regression(y_batch, batch_predictions),
        }
        saved_best = save_model(updated_model, meta_to_save, model_key=model_key, make_best_alias=False)

        return {
            "model_key": model_key,
            "model_name": meta["model_name"],
            "model_type": model_type,
            "update_strategy": update_strategy,
            "updated_batch_rows": int(len(X_batch)),
            "model_version": saved_best["version"],
            "saved_model_path": saved_best["current_model_path"],
            "saved_meta_path": saved_best["current_meta_path"],
            "versioned_model_path": saved_best["versioned_model_path"],
            "versioned_meta_path": saved_best["versioned_meta_path"],
            "batch_metrics": meta_to_save["last_incremental_batch_metrics"],
        }

    elif model_type == "sklearn_pipeline_mlp":
        updated_model = model
        preprocessor = updated_model.named_steps["preprocessor"]
        mlp_model = updated_model.named_steps["model"]

        if getattr(mlp_model, "early_stopping", False):
            print("Disabling early_stopping for MLPRegressor before partial_fit...")
            mlp_model.set_params(early_stopping=False)

        if hasattr(mlp_model, "best_loss_") and mlp_model.best_loss_ is None:
            print("Resetting MLPRegressor early-stopping state before partial_fit...")
            mlp_model.best_loss_ = float("inf")

        if hasattr(mlp_model, "validation_scores_"):
            mlp_model.validation_scores_ = []

        if hasattr(mlp_model, "_no_improvement_count"):
            mlp_model._no_improvement_count = 0

        X_batch_prepared = preprocessor.transform(X_batch)
        mlp_model.partial_fit(X_batch_prepared, y_batch)
        batch_predictions = updated_model.predict(X_batch)
        update_strategy = "mlp_partial_fit"

        meta_to_save = {
            **meta,
            "updated_at": datetime.now().isoformat(),
            "update_strategy": update_strategy,
            "last_incremental_batch_rows": int(len(X_batch)),
            "last_incremental_batch_metrics": evaluate_regression(y_batch, batch_predictions),
        }
        saved_best = save_model(updated_model, meta_to_save, model_key=model_key, make_best_alias=False)

        return {
            "model_key": model_key,
            "model_name": meta["model_name"],
            "model_type": model_type,
            "update_strategy": update_strategy,
            "updated_batch_rows": int(len(X_batch)),
            "model_version": saved_best["version"],
            "saved_model_path": saved_best["current_model_path"],
            "saved_meta_path": saved_best["current_meta_path"],
            "versioned_model_path": saved_best["versioned_model_path"],
            "versioned_meta_path": saved_best["versioned_meta_path"],
            "batch_metrics": meta_to_save["last_incremental_batch_metrics"],
        }

    elif model_key == "decision_tree":
        print("DecisionTreeRegressor does not support incremental learning. Falling back to full retraining of the decision tree on accumulated batches using saved hyperparameters...")

        train_df, test_df = load_batch_splits(batch_files=available_batch_files)
        X_train_full, y_train_full = model_data(train_df)
        X_test_full, y_test_full = model_data(test_df)

        feature_columns = meta["feature_columns"]
        X_train_full = X_train_full[feature_columns].copy()
        X_test_full = X_test_full[feature_columns].copy()

        tree_params = meta.get("params", {
            "max_depth": 12,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
        })
        print(f"Retraining decision_tree with saved params: {tree_params}")
        updated_model = build_tree_pipeline(params=tree_params)
        updated_model.fit(X_train_full, y_train_full)

        batch_predictions = updated_model.predict(X_batch)
        test_predictions = updated_model.predict(X_test_full)
        test_metrics = evaluate_regression(y_test_full, test_predictions)

        metrics = {
            "val_mae": meta.get("metrics", {}).get("val_mae"),
            "val_rmse": meta.get("metrics", {}).get("val_rmse"),
            "val_r2": meta.get("metrics", {}).get("val_r2"),
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        }

        meta_to_save = {
            **meta,
            "updated_at": datetime.now().isoformat(),
            "update_strategy": "tree_full_retrain",
            "train_rows": int(len(X_train_full)),
            "test_rows": int(len(X_test_full)),
            "train_batches": len(available_batch_files) - 1,
            "test_batch": available_batch_files[-1].name,
            "last_incremental_batch_rows": int(len(X_batch)),
            "last_incremental_batch_metrics": evaluate_regression(y_batch, batch_predictions),
            "metrics": metrics,
        }
        saved_best = save_model(updated_model, meta_to_save, model_key=model_key, make_best_alias=False)

        return {
            "model_key": "decision_tree",
            "model_name": meta_to_save["model_name"],
            "model_type": meta_to_save["model_type"],
            "update_strategy": "tree_full_retrain",
            "updated_batch_rows": int(len(X_batch)),
            "model_version": saved_best["version"],
            "saved_model_path": saved_best["current_model_path"],
            "saved_meta_path": saved_best["current_meta_path"],
            "versioned_model_path": saved_best["versioned_model_path"],
            "versioned_meta_path": saved_best["versioned_meta_path"],
            "batch_metrics": meta_to_save["last_incremental_batch_metrics"],
        }

    else:
        raise ValueError(f"Unsupported model type for update: {model_type}")


def update_all_models(batch_df, available_batch_files):
    registry = build_model_registry()
    updated_results = []

    for model_key in registry.keys():
        print(f"Updating candidate model: {model_key}")
        updated_results.append(update_single_model(model_key, batch_df, available_batch_files))

    return updated_results


def select_and_save_best_updated_model(updated_results):
    if not updated_results:
        raise ValueError("No updated model results were produced.")

    best_result = min(updated_results, key=lambda item: item["batch_metrics"]["rmse"])
    best_model, best_meta = load_model_by_key(best_result["model_key"])
    best_saved = save_best_model(best_model, best_meta)

    best_result = {
        **best_result,
        "saved_model_path": best_saved["best_model_path"],
        "saved_meta_path": best_saved["best_meta_path"],
        "versioned_model_path": best_saved["versioned_model_path"],
        "versioned_meta_path": best_saved["versioned_meta_path"],
        "model_version": best_saved["version"],
    }
    return best_result, updated_results


def batch_predict_saved_model(batch_df):
    model, meta = load_saved_model()

    X_batch, y_batch = prepare_batch_for_model(batch_df, meta)
    print(f"Scoring current saved model on incoming batch with {len(X_batch)} rows...")
    model_type = meta["model_type"]

    if model_type == "catboost":
        X_batch_prepared = prepare_catboost_inference_data(
            X_batch,
            categorical_features=meta["categorical_features"],
            numerical_features=meta["numerical_features"],
        )
        predictions = model.predict(X_batch_prepared)

    elif model_type.startswith("sklearn_pipeline"):
        predictions = model.predict(X_batch)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    metrics = evaluate_regression(y_batch, predictions)
    return {
        "model_name": meta["model_name"],
        "model_key": meta.get("model_key"),
        "model_type": model_type,
        "batch_rows": int(len(X_batch)),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
    }


def update_log(record, log_file=UPDATE_LOG_FILE):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    record_df = pd.DataFrame([record])

    if log_file.exists():
        existing_df = pd.read_csv(log_file)
        result_df = pd.concat([existing_df, record_df], ignore_index=True)
    else:
        result_df = record_df

    result_df.to_csv(log_file, index=False)


def get_next_batch_to_process(all_batch_files, last_processed_batch):
    next_batch_index = last_processed_batch + 1

    if next_batch_index > len(all_batch_files):
        return next_batch_index, None

    return next_batch_index, all_batch_files[next_batch_index - 1]


def update_pipeline():
    state = load_pipeline_state()
    print("Starting update pipeline...")
    all_batch_files = list_batch_files()
    print(f"Discovered {len(all_batch_files)} batch files in storage.")

    if not all_batch_files:
        raise ValueError("No batch files found. Run prepare_data first.")

    if not state.get("initialized", False):
        raise ValueError("Pipeline is not initialized. Run train first.")

    last_processed_batch = int(state.get("last_processed_batch", 0))
    next_batch_index, next_batch_path = get_next_batch_to_process(all_batch_files=all_batch_files, last_processed_batch=last_processed_batch)

    if next_batch_path is None:
        return {
            "status": "no_new_batches",
            "message": "No new batches available for update.",
            "last_processed_batch": last_processed_batch,
            "available_batches": len(all_batch_files),
        }

    batch_df = load_batch(next_batch_path)
    print(f"Loaded next batch for update: {next_batch_path.name}")
    print("Running data quality checks on the new batch...")
    for _ in tqdm(range(1), desc="Data quality", leave=False):
        quality_metrics = compute_batch_quality_metrics(batch_df, batch_name=next_batch_path.name)
    append_data_quality_log(quality_metrics)
    print("Data quality checks completed.")

    pre_update_metrics = None
    if BEST_MODEL_FILE.exists() and BEST_MODEL_META_FILE.exists():
        print("Evaluating current production model on the new batch before update...")
        pre_update_metrics = batch_predict_saved_model(batch_df)

    available_batch_files = all_batch_files[:next_batch_index]
    training_result = None
    update_error = None
    all_update_results = []

    if BEST_MODEL_FILE.exists() and BEST_MODEL_META_FILE.exists():
        print("Updating all candidate models using the new batch...")
        try:
            for _ in tqdm(range(1), desc="Model update", leave=False):
                all_update_results = update_all_models(batch_df, available_batch_files)
                training_result, all_update_results = select_and_save_best_updated_model(all_update_results)
            print("Model update finished.")
        except ValueError as exc:
            update_error = str(exc)
            print(f"Model update skipped with error: {update_error}")

    state["last_processed_batch"] = next_batch_index
    state["last_update_at"] = datetime.now().isoformat()
    state["last_processed_batch_name"] = next_batch_path.name
    print("Saving pipeline state after update...")
    save_pipeline_state(state)

    update_record = {
        "created_at": datetime.now().isoformat(),
        "processed_batch_index": next_batch_index,
        "processed_batch_name": next_batch_path.name,
        "available_batches_after_update": len(available_batch_files),
        "quality_row_count": quality_metrics["row_count"],
        "quality_missing_total": quality_metrics["missing_total"],
        "quality_duplicate_count": quality_metrics["duplicate_count"],
        "quality_negative_target_count": quality_metrics["negative_target_count"],
        "quality_invalid_duration_count": quality_metrics["invalid_duration_count"],
        "quality_invalid_coordinate_count": quality_metrics["invalid_coordinate_count"],
        "incremental_update_attempted": BEST_MODEL_FILE.exists() and BEST_MODEL_META_FILE.exists(),
        "updated_candidate_models": len(all_update_results),
    }

    if pre_update_metrics is not None:
        update_record["pre_update_model_name"] = pre_update_metrics["model_name"]
        update_record["pre_update_model_key"] = pre_update_metrics.get("model_key")
        update_record["pre_update_batch_rows"] = pre_update_metrics["batch_rows"]
        update_record["pre_update_mae"] = pre_update_metrics["mae"]
        update_record["pre_update_rmse"] = pre_update_metrics["rmse"]
        update_record["pre_update_r2"] = pre_update_metrics["r2"]

    if training_result is not None:
        update_record["post_update_model_name"] = training_result["model_name"]
        update_record["post_update_model_key"] = training_result["model_key"]
        update_record["post_update_model_type"] = training_result["model_type"]
        update_record["post_update_strategy"] = training_result["update_strategy"]
        update_record["post_update_batch_rows"] = training_result["updated_batch_rows"]
        update_record["post_update_model_version"] = training_result["model_version"]
        update_record["post_update_batch_mae"] = training_result["batch_metrics"]["mae"]
        update_record["post_update_batch_rmse"] = training_result["batch_metrics"]["rmse"]
        update_record["post_update_batch_r2"] = training_result["batch_metrics"]["r2"]

    if update_error is not None:
        update_record["update_error"] = update_error

    print("Writing update record to log...")
    update_log(update_record)

    print("Update pipeline completed.")
    return {
        "status": "updated",
        "processed_batch_index": next_batch_index,
        "processed_batch_name": next_batch_path.name,
        "available_batches": len(available_batch_files),
        "quality_metrics": quality_metrics,
        "pre_update_metrics": pre_update_metrics,
        "training_result": training_result,
        "all_update_results": all_update_results,
        "update_error": update_error,
    }