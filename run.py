import argparse

from src.data_prep import list_batch_files, load_batch, prepare_raw_batches
from src.data_quality import compute_batch_quality_metrics, append_data_quality_log
from src.train import train_models
from src.inference import predict_from_file
from src.summary import generate_summary_report
from src.update import update_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["prepare_data", "raw_batches_stats", "train", "inference", "summary", "update"],
    )
    parser.add_argument("--file", type=str)

    args = parser.parse_args()

    if args.mode == "inference" and not args.file:
        parser.error("--file is required when --mode inference")

    if args.mode == "prepare_data":
        saved_paths = prepare_raw_batches()
        print(f"Prepared {len(saved_paths)} raw batches")
        for path in saved_paths[:5]:
            print(path)

    if args.mode == "raw_batches_stats":
        batch_files = list_batch_files()
        if not batch_files:
            print("No batch files found. Run prepare_data.py first.")
            return

        print(f"Found {len(batch_files)} batch files.")

        for batch_path in batch_files:
            df = load_batch(batch_path)
            metrics = compute_batch_quality_metrics(df, batch_name=batch_path.name)
            append_data_quality_log(metrics)

            print(
                f"[OK] {batch_path.name}: "
                f"rows={metrics['row_count']}, "
                f"missing_total={metrics['missing_total']}, "
                f"duplicates={metrics['duplicate_count']}, "
                f"negative_trip_distance={metrics['negative_trip_distance_count']}, "
                f"negative_target={metrics['negative_target_count']}, "
                f"invalid_duration={metrics['invalid_duration_count']}, "
                f"invalid_coordinates={metrics['invalid_coordinate_count']}"
            )

        print("Data quality log updated successfully.")


    if args.mode == "train":
        result = train_models()

        print("Training finished.")
        print(f"Best model: {result['best_model_name']}")
        print(f"Best model key: {result['best_model_key']}")
        print(f"Test MAE:        {result['test_mae']:.4f}")
        print(f"Test RMSE:       {result['test_rmse']:.4f}")
        print(f"Test R2:         {result['test_r2']:.4f}")
        if "model_version" in result:
            print(f"Model version: {result['model_version']}")
        if "saved_model_path" in result:
            print(f"Model saved to: {result['saved_model_path']}")
        if "saved_meta_path" in result:
            print(f"Meta saved to: {result['saved_meta_path']}")
        if "versioned_model_path" in result:
            print(f"Versioned model: {result['versioned_model_path']}")
        if "versioned_meta_path" in result:
            print(f"Versioned meta: {result['versioned_meta_path']}")


    if args.mode == "inference":
        output_path = predict_from_file(args.file)
        print(f"Predictions saved to: {output_path}")

    if args.mode == "summary": 
        output_path = generate_summary_report()
        print(f"Summary report saved to: {output_path}")

    if args.mode == "update":
        result = update_pipeline()

        if result["status"] == "no_new_batches":
            print(result["message"])
            return

        print("Update finished.")
        print(f"Processed batch: {result['processed_batch_name']}")
        print(f"Available batches: {result['available_batches']}")

        qm = result["quality_metrics"]
        print(
            f"Data quality: rows={qm['row_count']}, "
            f"missing_total={qm['missing_total']}, "
            f"duplicates={qm['duplicate_count']}, "
            f"negative_target={qm['negative_target_count']}, "
            f"invalid_duration={qm['invalid_duration_count']}, "
            f"invalid_coordinates={qm['invalid_coordinate_count']}"
        )

        if result["training_result"] is None:
            if result.get("update_error"):
                print(f"Update skipped: {result['update_error']}")
            else:
                print("No model update was performed.")
        else:
            tr = result["training_result"]
            print(f"Updated model: {tr['model_name']}")
            print(f"Model type: {tr['model_type']}")
            print(f"Update strategy: {tr['update_strategy']}")
            print(f"Updated batch rows: {tr['updated_batch_rows']}")
            print(f"Model version: {tr['model_version']}")
            batch_metrics = tr["batch_metrics"]
            print(f"Batch MAE: {batch_metrics['mae']:.4f}")
            print(f"Batch RMSE: {batch_metrics['rmse']:.4f}")
            print(f"Batch R2: {batch_metrics['r2']:.4f}")
            print(f"Model saved to: {tr['saved_model_path']}")
            print(f"Meta saved to: {tr['saved_meta_path']}")
            print(f"Versioned model: {tr['versioned_model_path']}")
            print(f"Versioned meta: {tr['versioned_meta_path']}")
            if result.get("update_error"):
                print(f"Update warning: {result['update_error']}")






if __name__ == "__main__":
    main()