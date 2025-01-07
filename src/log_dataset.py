import mlflow
import mlflow.artifacts
import pandas as pd

def log_dataset(dataset_path):
    # Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")  # Sesuaikan dengan port yang kamu pakai
    
    # Load dataset
    data = pd.read_csv(dataset_path)
    
    # Start MLflow run
    with mlflow.start_run(run_name="dataset_logging"):
        # Log dataset as artifact
        mlflow.log_artifact(dataset_path, "dataset")
        
        # Log basic dataset information
        mlflow.log_params({
            'num_samples': len(data),
            'num_features': len(data.columns),
            'dataset_columns': str(data.columns.tolist())
        })
        
        # You can also log other metrics/statistics, such as missing values
        missing_values = data.isnull().sum().to_dict()
        mlflow.log_dict(missing_values, "missing_values.json")

        # Optionally, log summary statistics
        summary_stats = data.describe().to_dict()
        mlflow.log_dict(summary_stats, "summary_stats.json")
        
    print("Dataset logged successfully.")

if __name__ == "__main__":
    dataset_path = "C:/Users/user/tugas/Semester_V/data_models/mlops/data/data_train.csv"  # Ganti dengan path dataset yang kamu pilih
    log_dataset(dataset_path)