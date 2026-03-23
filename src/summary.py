import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    METRICS_DIR,
    REPORTS_DIR,
    BEST_MODEL_META_FILE,
)


DATA_QUALITY_LOG_FILE = METRICS_DIR/"data_quality_log.csv"
TRAINING_LOG_FILE = METRICS_DIR/"training_log.csv"
SUMMARY_REPORT_FILE = REPORTS_DIR/"summary_report.md"
UPDATE_LOG_FILE = METRICS_DIR/"update_log.csv"


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_value(value):
    if pd.isna(value):
        return "n/a"
    return value


def build_overview_section(data_quality_df, training_df, update_df, best_model_meta):
    total_batches = int(len(data_quality_df)) if not data_quality_df.empty else 0
    total_training_records = int(len(training_df)) if not training_df.empty else 0
    total_updates = int(len(update_df)) if not update_df.empty else 0

    lines = [
        "## Overview",
        "",
        f"- Logged data-quality batches: {total_batches}",
        f"- Logged training records: {total_training_records}",
        f"- Logged updates: {total_updates}",
    ]

    if best_model_meta:
        lines.extend([
            f"- Current best model: {best_model_meta.get('model_name', 'n/a')}",
            f"- Current best model key: {best_model_meta.get('model_key', 'n/a')}",
            f"- Current best model version: {best_model_meta.get('version', 'n/a')}",
        ])

    lines.append("")
    return "\n".join(lines)


def build_data_quality_section(df):
    if df.empty:
        return "## Data Quality\n\nNo data quality logs found.\n"

    total_batches = len(df)
    total_rows = int(df["quality_row_count"].sum()) if "quality_row_count" in df.columns else 0
    total_missing = int(df["quality_missing_total"].sum()) if "quality_missing_total" in df.columns else 0
    total_duplicates = int(df["quality_duplicate_count"].sum()) if "quality_duplicate_count" in df.columns else 0
    total_negative_target = (
        int(df["quality_negative_target_count"].sum())
        if "quality_negative_target_count" in df.columns
        else 0
    )
    total_invalid_duration = (
        int(df["quality_invalid_duration_count"].sum())
        if "quality_invalid_duration_count" in df.columns
        else 0
    )
    total_invalid_coordinates = (
        int(df["quality_invalid_coordinate_count"].sum())
        if "quality_invalid_coordinate_count" in df.columns
        else 0
    )

    lines = [
        "## Data Quality",
        "",
        f"- Processed batches: {total_batches}",
        f"- Total rows across batches: {total_rows}",
        f"- Total missing values: {total_missing}",
        f"- Total duplicate rows: {total_duplicates}",
        f"- Total rows with negative target: {total_negative_target}",
        f"- Total rows with invalid duration: {total_invalid_duration}",
        f"- Total rows with invalid coordinates: {total_invalid_coordinates}",
        "",
    ]

    if "batch_name" in df.columns and "quality_row_count" in df.columns:
        lines.append("### Per-batch overview")
        lines.append("")
        preview_cols = [
            col
            for col in [
                "batch_name",
                "quality_row_count",
                "quality_missing_total",
                "quality_duplicate_count",
                "quality_negative_target_count",
                "quality_invalid_duration_count",
                "quality_invalid_coordinate_count",
            ]
            if col in df.columns
        ]
        lines.append(df[preview_cols].to_markdown(index=False))
        lines.append("")

    return "\n".join(lines)


def build_training_section(df):
    if df.empty:
        return "## Training\n\nNo training logs found.\n"

    lines = [
        "## Training",
        "",
        f"- Total training records: {len(df)}",
        "",
    ]

    best_records_df = df.copy()
    if "is_best" in best_records_df.columns:
        best_records_df = best_records_df[best_records_df["is_best"] == True]

    if not best_records_df.empty:
        latest_best = best_records_df.iloc[-1].to_dict()
        lines.append("### Latest selected best model")
        lines.append("")
        for key in [
            "created_at",
            "model_name",
            "model_key",
            "model_version",
            "selection_strategy",
            "train_rows",
            "test_rows",
            "train_batches",
            "test_batch",
            "val_rmse",
            "val_mae",
            "val_r2",
            "test_rmse",
            "test_mae",
            "test_r2",
        ]:
            if key in latest_best:
                lines.append(f"- {key}: {format_value(latest_best[key])}")
        lines.append("")

    history_cols = [
        col
        for col in [
            "created_at",
            "model_name",
            "model_key",
            "selection_strategy",
            "val_rmse",
            "val_mae",
            "val_r2",
            "test_rmse",
            "test_mae",
            "test_r2",
            "is_best",
        ]
        if col in df.columns
    ]
    if history_cols:
        lines.append("### Training history")
        lines.append("")
        lines.append(df[history_cols].to_markdown(index=False))
        lines.append("")

    return "\n".join(lines)


def build_update_section(df):
    if df.empty:
        return "## Updates\n\nNo update logs found.\n"

    lines = [
        "## Updates",
        "",
        f"- Total update records: {len(df)}",
        "",
    ]

    latest_update = df.iloc[-1].to_dict()
    lines.append("### Latest update")
    lines.append("")
    for key in [
        "created_at",
        "processed_batch_name",
        "quality_row_count",
        "incremental_update_attempted",
        "updated_candidate_models",
        "post_update_model_name",
        "post_update_model_key",
        "post_update_model_type",
        "post_update_strategy",
        "post_update_model_version",
        "post_update_batch_mae",
        "post_update_batch_rmse",
        "post_update_batch_r2",
        "update_error",
    ]:
        if key in latest_update:
            lines.append(f"- {key}: {format_value(latest_update[key])}")
    lines.append("")

    history_cols = [
        col
        for col in [
            "created_at",
            "processed_batch_name",
            "post_update_model_name",
            "post_update_model_key",
            "post_update_strategy",
            "post_update_batch_rmse",
            "update_error",
        ]
        if col in df.columns
    ]
    if history_cols:
        lines.append("### Update history")
        lines.append("")
        lines.append(df[history_cols].to_markdown(index=False))
        lines.append("")

    return "\n".join(lines)


def build_best_model_section(meta):
    if not meta:
        return "## Best Model\n\nNo saved model metadata found.\n"

    lines = [
        "## Best Model",
        "",
        f"- model_name: {meta.get('model_name', 'n/a')}",
        f"- model_key: {meta.get('model_key', 'n/a')}",
        f"- model_type: {meta.get('model_type', 'n/a')}",
        f"- version: {meta.get('version', 'n/a')}",
        f"- preprocessing: {meta.get('preprocessing', 'n/a')}",
        f"- saved_at: {meta.get('saved_at', 'n/a')}",
        f"- updated_at: {meta.get('updated_at', 'n/a')}",
        f"- update_strategy: {meta.get('update_strategy', 'n/a')}",
        f"- selection_strategy: {meta.get('selection_strategy', 'n/a')}",
        "",
    ]

    metrics = meta.get("metrics", {})
    if metrics:
        lines.append("### Metrics")
        lines.append("")
        for key in ["val_mae", "val_rmse", "val_r2", "test_mae", "test_rmse", "test_r2"]:
            if key in metrics:
                lines.append(f"- {key}: {metrics[key]}")
        lines.append("")

    params = meta.get("params", {})
    if params:
        lines.append("### Hyperparameters")
        lines.append("")
        for key, value in params.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    for key in ["train_rows", "test_rows", "train_batches", "test_batch", "last_incremental_batch_rows"]:
        if key in meta:
            lines.append(f"- {key}: {meta[key]}")
    lines.append("")

    last_incremental_batch_metrics = meta.get("last_incremental_batch_metrics", {})
    if last_incremental_batch_metrics:
        lines.append("### Last incremental batch metrics")
        lines.append("")
        for key, value in last_incremental_batch_metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    feature_columns = meta.get("feature_columns", [])
    categorical_features = meta.get("categorical_features", [])
    numerical_features = meta.get("numerical_features", [])

    if feature_columns:
        lines.append("### Feature columns")
        lines.append("")
        for col in feature_columns:
            lines.append(f"- {col}")
        lines.append("")

    if categorical_features:
        lines.append("### Categorical features")
        lines.append("")
        for col in categorical_features:
            lines.append(f"- {col}")
        lines.append("")

    if numerical_features:
        lines.append("### Numerical features")
        lines.append("")
        for col in numerical_features:
            lines.append(f"- {col}")
        lines.append("")

    return "\n".join(lines)


def generate_summary_report(output_file=SUMMARY_REPORT_FILE):
    data_quality_df = read_csv(DATA_QUALITY_LOG_FILE)
    training_df = read_csv(TRAINING_LOG_FILE)
    update_df = read_csv(UPDATE_LOG_FILE)
    best_model_meta = read_json(BEST_MODEL_META_FILE)

    sections = [
        "# MVP Summary Report",
        "",
        f"Generated at: {datetime.now().isoformat()}",
        "",
        build_overview_section(data_quality_df, training_df, update_df, best_model_meta),
        build_data_quality_section(data_quality_df),
        build_training_section(training_df),
        build_update_section(update_df),
        build_best_model_section(best_model_meta),
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(sections), encoding="utf-8")

    return output_file