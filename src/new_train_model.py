import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

def train_and_log_model(data_path, model_type):
    # Load dataset
    data = pd.read_csv(data_path)
    
    X = data.drop('pIC50', axis=1)
    y = data['pIC50']
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"model_{model_type}") as run:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == 'LinearRegression':
            model = LinearRegression()
            params = {}
        
        elif model_type == 'RandomForest':
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            params = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
        
        elif model_type == 'NeuralNetwork':
            model = Sequential([
                Dense(64, input_dim=X_train.shape[1], activation='relu'),
                Dense(32, activation='relu'),
                Dense(1)  # No activation for regression
            ])
            model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
            params = {'optimizer': 'Adam', 'loss': 'mse'}
        
        # Train model
        if model_type == 'NeuralNetwork':
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        results_df = pd.DataFrame({
            'Actual_pIC50': y_test,
            'Predicted_pIC50': y_pred.flatten(),
            'Absolute_Error': abs(y_test - y_pred.flatten())
        })
        
        # Reset index untuk mendapatkan nomor sampel
        results_df = results_df.reset_index()
        results_df = results_df.rename(columns={'index': 'Sample_Number'})
        
        # Menyimpan ke CSV
        csv_filename = f"predictions_{model_type}.csv"
        results_df.to_csv(csv_filename, index=False)
        
        # Log file sebagai artifact di MLflow
        mlflow.log_artifact(csv_filename)
        
        # Log parameters, metrics, and model
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({'model_type': model_type})
        
        if model_type == 'NeuralNetwork':
            mlflow.tensorflow.log_model(model, f"model_{model_type}")
        else:
            mlflow.sklearn.log_model(model, f"model_{model_type}")
        
        print(f"Model {model_type} logged successfully with run_id: {run.info.run_id}")
        print(f"Predictions saved to {csv_filename}")
        return run.info.run_id

if __name__ == "__main__":
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("pIC50_prediction")
    
    dataset_path = "C:/Users/user/tugas/Semester_V/data_models/mlops/data/data_train.csv"
    
    run_id_lr = train_and_log_model(dataset_path, 'LinearRegression')
    run_id_rf = train_and_log_model(dataset_path, 'RandomForest')
    run_id_nn = train_and_log_model(dataset_path, 'NeuralNetwork')
    
    print(f"Linear Regression run_id: {run_id_lr}")
    print(f"Random Forest run_id: {run_id_rf}")
    print(f"Neural Network run_id: {run_id_nn}")