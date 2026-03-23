from pathlib import Path
from datetime import datetime
import json
import pickle
import shutil
from itertools import product

import pandas as pd

from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    FEATURE_COLUMNS,
    MODELS_DIR,
    METRICS_DIR,
    BEST_MODEL_FILE,
    BEST_MODEL_META_FILE,
    RANDOM_STATE,
)
from src.data_prep import list_batch_files, load_batch
from src.features import model_data
from src.utils import load_pipeline_state, save_pipeline_state


TRAINING_LOG_FILE = METRICS_DIR/"training_log.csv"

CATEGORICAL_FEATURES = ["VendorID", "RatecodeID", "store_and_fwd_flag"]


NUMERICAL_FEATURES = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES]


def load_batch_splits(batch_files):
    if batch_files is None:
        batch_files = list_batch_files()

    if len(batch_files) < 2:
        raise ValueError(
            "At least 2 batch files are required for train/test split by batches."
        )

    train_batch_files = batch_files[:-1]
    test_batch_file = batch_files[-1]

    train_dfs = []
    for batch_path in train_batch_files:
        batch_df = load_batch(batch_path)
        batch_df["source_batch"] = batch_path.name
        train_dfs.append(batch_df)

    train_df = pd.concat(train_dfs, ignore_index=True)

    test_df = load_batch(test_batch_file)
    test_df["source_batch"] = test_batch_file.name

    return train_df, test_df


def split_train_validation(X_train, y_train, val_fraction=0.2, min_val_rows=1000):
    if len(X_train) < min_val_rows * 2:
        split_idx = max(1, int(len(X_train) * (1 - val_fraction)))
    else:
        split_idx = len(X_train) - min_val_rows

    split_idx = min(max(split_idx, 1), len(X_train) - 1)

    X_subtrain = X_train.iloc[:split_idx].copy()
    y_subtrain = y_train.iloc[:split_idx].copy()
    X_val = X_train.iloc[split_idx:].copy()
    y_val = y_train.iloc[split_idx:].copy()

    return X_subtrain, X_val, y_subtrain, y_val


def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {"mae": float(mae),"rmse": float(rmse),"r2": float(r2)}


def write_training_log(metrics, log_file=TRAINING_LOG_FILE):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([metrics])

    if log_file.exists():
        existing_df = pd.read_csv(log_file)
        result_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    else:
        result_df = metrics_df

    result_df.to_csv(log_file, index=False)


def save_model(model_object, meta, model_key, make_best_alias=False):
    BEST_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_model_file = MODELS_DIR/f"{model_key}_{version}.pkl"
    versioned_meta_file = MODELS_DIR/f"{model_key}_{version}_meta.json"
    current_model_file = MODELS_DIR/f"current_{model_key}.pkl"
    current_meta_file = MODELS_DIR/f"current_{model_key}_meta.json"

    meta_to_save = {
        **meta,
        "version": version,
        "model_key": model_key,
        "saved_at": datetime.now().isoformat(),
        "versioned_model_path": str(versioned_model_file),
        "versioned_meta_path": str(versioned_meta_file),
        "current_model_path": str(current_model_file),
        "current_meta_path": str(current_meta_file),
        "is_best": bool(make_best_alias),
    }

    with open(versioned_model_file, "wb") as f:
        pickle.dump(model_object, f)

    with open(versioned_meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_to_save, f, ensure_ascii=False, indent=2)

    shutil.copy2(versioned_model_file, current_model_file)
    shutil.copy2(versioned_meta_file, current_meta_file)

    result = {
        "version": version,
        "versioned_model_path": str(versioned_model_file),
        "versioned_meta_path": str(versioned_meta_file),
        "current_model_path": str(current_model_file),
        "current_meta_path": str(current_meta_file),
    }

    if make_best_alias:
        shutil.copy2(versioned_model_file, BEST_MODEL_FILE)
        shutil.copy2(versioned_meta_file, BEST_MODEL_META_FILE)
        result.update({
            "best_model_path": str(BEST_MODEL_FILE),
            "best_meta_path": str(BEST_MODEL_META_FILE),
        })

    return result


def save_best_model(model_object, meta):
    model_key = meta.get("model_key")
    if model_key is None:
        raise ValueError("Model metadata must contain 'model_key' to save best model artifacts.")
    return save_model(model_object, meta, model_key=model_key, make_best_alias=True)


def build_tree_pipeline(params=None):
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                              ("encoder", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, NUMERICAL_FEATURES),
                                                   ("cat", categorical_transformer, CATEGORICAL_FEATURES)])
    if params is None:
        params = {
            "max_depth": 12,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": RANDOM_STATE,
        }

    pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                               ("model", DecisionTreeRegressor(**params))])
    return pipeline


def build_mlp_pipeline(params=None):
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                          ("scaler", StandardScaler())])

    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                              ("encoder", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, NUMERICAL_FEATURES),
                                                   ("cat", categorical_transformer, CATEGORICAL_FEATURES)])
    if params is None:
        params = {
            "hidden_layer_sizes": (256, 128, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "learning_rate_init": 5e-4,
            "batch_size": 256,
            "max_iter": 200,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 15,
            "random_state": RANDOM_STATE,
        }

    pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                ("model", MLPRegressor(**params))])
    return pipeline


def prepare_catboost_data(X_train, X_other):
    X_train_cb = X_train.copy()
    X_other_cb = X_other.copy()

    for col in CATEGORICAL_FEATURES:
        if col in X_train_cb.columns:
            X_train_cb[col] = X_train_cb[col].astype("string").fillna("missing")
            X_other_cb[col] = X_other_cb[col].astype("string").fillna("missing")

    for col in NUMERICAL_FEATURES:
        if col in X_train_cb.columns:
            median_value = X_train_cb[col].median()
            X_train_cb[col] = X_train_cb[col].fillna(median_value)
            X_other_cb[col] = X_other_cb[col].fillna(median_value)

    return X_train_cb, X_other_cb


def search_catboost_hyperparams(X_train, y_train):
    X_subtrain, X_val, y_subtrain, y_val = split_train_validation(X_train, y_train)

    candidate_grid = list(product([200, 300], [6, 8], [0.03, 0.1]))
    best_result = None

    for iterations, depth, learning_rate in candidate_grid:
        params = {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": RANDOM_STATE,
            "verbose": False,
        }

        X_subtrain_cb, X_val_cb = prepare_catboost_data(X_subtrain, X_val)
        model = CatBoostRegressor(**params)
        model.fit(X_subtrain_cb, y_subtrain, cat_features=CATEGORICAL_FEATURES)
        val_predictions = model.predict(X_val_cb)
        val_metrics = evaluate_regression(y_val, val_predictions)

        current_result = {
            "params": params,
            "val_metrics": val_metrics,
        }

        if best_result is None or current_result["val_metrics"]["rmse"] < best_result["val_metrics"]["rmse"]:
            best_result = current_result

    return best_result


def search_decision_tree_hyperparams(X_train, y_train):
    X_subtrain, X_val, y_subtrain, y_val = split_train_validation(X_train, y_train)

    candidate_grid = list(product([8, 12, 16], [10, 20], [5, 10]))
    best_result = None

    for max_depth, min_samples_split, min_samples_leaf in candidate_grid:
        params = {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": RANDOM_STATE,
        }

        pipeline = build_tree_pipeline(params=params)
        pipeline.fit(X_subtrain, y_subtrain)
        val_predictions = pipeline.predict(X_val)
        val_metrics = evaluate_regression(y_val, val_predictions)

        current_result = {
            "params": params,
            "val_metrics": val_metrics,
        }

        if best_result is None or current_result["val_metrics"]["rmse"] < best_result["val_metrics"]["rmse"]:
            best_result = current_result

    return best_result


def search_mlp_hyperparams(X_train, y_train):
    X_subtrain, X_val, y_subtrain, y_val = split_train_validation(X_train, y_train)

    candidate_grid = list(product([(128, 64), (256, 128, 64)], [1e-4, 1e-3], [1e-3, 5e-4]))
    best_result = None

    for hidden_layer_sizes, alpha, learning_rate_init in candidate_grid:
        params = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": "relu",
            "solver": "adam",
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
            "batch_size": 256,
            "max_iter": 200,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 15,
            "random_state": RANDOM_STATE,
        }

        pipeline = build_mlp_pipeline(params=params)
        pipeline.fit(X_subtrain, y_subtrain)
        val_predictions = pipeline.predict(X_val)
        val_metrics = evaluate_regression(y_val, val_predictions)

        current_result = {
            "params": params,
            "val_metrics": val_metrics,
        }

        if best_result is None or current_result["val_metrics"]["rmse"] < best_result["val_metrics"]["rmse"]:
            best_result = current_result

    return best_result

def train_catboost(X_train, X_test, y_train, y_test):
    print("Searching CatBoost hyperparameters...")
    search_result = search_catboost_hyperparams(X_train, y_train)
    best_params = search_result["params"]
    val_metrics = search_result["val_metrics"]

    X_train_cb, X_test_cb = prepare_catboost_data(X_train, X_test)

    print(f"Training CatBoostRegressor with params: {best_params}")
    model = CatBoostRegressor(**best_params)

    print("Fitting CatBoost on train split...")
    model.fit(
        X_train_cb,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
    )

    print("Evaluating CatBoost on test split...")
    test_predictions = model.predict(X_test_cb)
    test_metrics = evaluate_regression(y_test, test_predictions)

    metrics = {
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
    }

    meta = {
        "model_name": "CatBoostRegressor",
        "model_type": "catboost",
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "preprocessing": "native_catboost",
        "params": best_params,
        "val_metrics": val_metrics,
        "selection_strategy": "mini_grid_search",
    }

    return metrics, model, meta


def train_decision_tree(X_train, X_test, y_train, y_test):
    print("Searching DecisionTree hyperparameters...")
    search_result = search_decision_tree_hyperparams(X_train, y_train)
    best_params = search_result["params"]
    val_metrics = search_result["val_metrics"]

    print(f"Training DecisionTreeRegressor with params: {best_params}")
    pipeline = build_tree_pipeline(params=best_params)

    print("Fitting DecisionTree pipeline on train split...")
    pipeline.fit(X_train, y_train)

    print("Evaluating DecisionTree on test split...")
    test_predictions = pipeline.predict(X_test)
    test_metrics = evaluate_regression(y_test, test_predictions)

    metrics = {
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
    }

    meta = {
        "model_name": "DecisionTreeRegressor",
        "model_type": "sklearn_pipeline_tree",
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "preprocessing": "sklearn_pipeline_ohe",
        "params": best_params,
        "val_metrics": val_metrics,
        "selection_strategy": "mini_grid_search",
    }

    return metrics, pipeline, meta


def train_mlp(X_train, X_test, y_train, y_test):
    print("Searching MLP hyperparameters...")
    search_result = search_mlp_hyperparams(X_train, y_train)
    best_params = search_result["params"]
    val_metrics = search_result["val_metrics"]

    print(f"Training MLPRegressor with params: {best_params}")
    pipeline = build_mlp_pipeline(params=best_params)

    print("Fitting MLP pipeline on train split...")
    pipeline.fit(X_train, y_train)

    print("Evaluating MLP on test split...")
    test_predictions = pipeline.predict(X_test)
    test_metrics = evaluate_regression(y_test, test_predictions)

    metrics = {
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
    }

    meta = {
        "model_name": "MLPRegressor",
        "model_type": "sklearn_pipeline_mlp",
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "preprocessing": "sklearn_pipeline_ohe_scale",
        "params": {
            **best_params,
            "hidden_layer_sizes": list(best_params["hidden_layer_sizes"]),
        },
        "val_metrics": val_metrics,
        "selection_strategy": "mini_grid_search",
    }

    return metrics, pipeline, meta


def build_model_registry():
    return {
        "catboost": {
            "display_name": "CatBoostRegressor",
            "trainer": train_catboost,
        },
        "decision_tree": {
            "display_name": "DecisionTreeRegressor",
            "trainer": train_decision_tree,
        },
        "mlp": {
            "display_name": "MLPRegressor",
            "trainer": train_mlp,
        },
    }


def run_single_model_training(model_key, trainer, X_train, X_test, y_train, y_test, batch_files):
    metrics, model_object, meta = trainer(X_train, X_test, y_train, y_test)

    result = {
        "created_at": datetime.now().isoformat(),
        "model_name": meta["model_name"],
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_batches": len(batch_files) - 1,
        "test_batch": batch_files[-1].name,
        "model_key": model_key,
        "selection_strategy": meta.get("selection_strategy"),
        **metrics,
    }

    meta = {
        **meta,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_batches": len(batch_files) - 1,
        "test_batch": batch_files[-1].name,
        "model_key": model_key,
        "metrics": metrics,
    }

    write_training_log(result)
    print(f"{meta['model_name']} finished: test_rmse={result['test_rmse']:.4f}")
    print(f"Saving artifacts for model='{model_key}'...")
    saved_artifacts = save_model(model_object, meta, model_key=model_key, make_best_alias=False)

    print(f"Current artifacts saved for model='{model_key}': {saved_artifacts['current_model_path']}")

    result["saved_model"] = model_object
    result["saved_meta"] = meta
    result["saved_artifacts"] = saved_artifacts
    return result


def train_models(batch_files=None, selected_models=None, save_as_best=True, update_state=True):
    if batch_files is None:
        batch_files = list_batch_files()

    model_registry = build_model_registry()

    if selected_models is None:
        selected_models = list(model_registry.keys())

    unknown_models = [model_name for model_name in selected_models if model_name not in model_registry]
    if unknown_models:
        raise ValueError(f"Unknown model names requested for training: {unknown_models}")

    print(f"Models selected for training: {selected_models}")

    train_df, test_df = load_batch_splits(batch_files=batch_files)

    print("Preparing train and test datasets...")
    X_train, y_train = model_data(train_df)
    X_test, y_test = model_data(test_df)
    print(f"Prepared datasets: train={len(X_train)} rows, test={len(X_test)} rows")

    best_result = None

    for model_key in selected_models:
        trainer = model_registry[model_key]["trainer"]
        print(f"Starting training for model='{model_key}'...")
        current_result = run_single_model_training(
            model_key=model_key,
            trainer=trainer,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            batch_files=batch_files
        )

        if best_result is None or current_result["test_rmse"] < best_result["test_rmse"]:
            best_result = current_result

    saved_best = None
    if save_as_best:
        print(f"Best model selected: {best_result['model_name']}")
        print("Saving best model artifacts...")
        saved_best = save_best_model(best_result["saved_model"], best_result["saved_meta"])

        write_training_log({
            **best_result,
            "is_best": True,
            "model_version": saved_best["version"] if saved_best else None,
        })

        print("Best model artifacts saved.")

    if update_state:
        state = load_pipeline_state()
        state["initialized"] = True
        state["last_processed_batch"] = len(batch_files)
        state["last_train_at"] = datetime.now().isoformat()
        state["last_train_batch_name"] = batch_files[-1].name
        save_pipeline_state(state)
        print("Pipeline state saved.")

    result = {
        "best_model_name": best_result["model_name"],
        "best_model_key": best_result["model_key"],
        "test_mae": best_result["test_mae"],
        "test_rmse": best_result["test_rmse"],
        "test_r2": best_result["test_r2"],
        "trained_models": selected_models,
    }

    if saved_best is not None:
        result.update({
            "saved_model_path": saved_best["best_model_path"],
            "saved_meta_path": saved_best["best_meta_path"],
            "model_version": saved_best["version"],
            "versioned_model_path": saved_best["versioned_model_path"],
            "versioned_meta_path": saved_best["versioned_meta_path"],
        })

    result["current_model_artifacts"] = {model_key: model_registry[model_key]["display_name"]
                                            for model_key in selected_models}

    return result
