import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def train_and_log_model(data_path, model_type):
    # Load dataset
    data = pd.read_csv(data_path)
    
    # Diskritisasi target jika perlu
    threshold = data['pIC50'].median()  # Ambang batas
    data['pIC50_binary'] = (data['pIC50'] >= threshold).astype(int)
    
    X = data.drop(['pIC50', 'pIC50_binary'], axis=1)
    y = data['pIC50_binary'] if model_type != 'LinearRegression' else data['pIC50']
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"model_{model_type}") as run:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000)
            params = {'max_iter': 1000}
        
        elif model_type == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            params = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
        
        elif model_type == 'NeuralNetwork':
            model = Sequential([
                Dense(64, input_dim=X_train.shape[1], activation='relu'),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
            params = {'optimizer': 'Adam', 'loss': 'binary_crossentropy'}
        
        elif model_type == 'LinearRegression':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            params = {}
        
        # Train model
        if model_type == 'NeuralNetwork':
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            y_pred = model.predict(X_test)
            y_pred = (y_pred > 0.5).astype(int)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_score': roc_auc_score(y_test, y_prob)
        }
        
        # Log parameters, metrics, and model
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({'model_type': model_type})
        
        if model_type == 'NeuralNetwork':
            mlflow.tensorflow.log_model(model, f"model_{model_type}")
        else:
            mlflow.sklearn.log_model(model, f"model_{model_type}")
        
        print(f"Model {model_type} logged successfully with run_id: {run.info.run_id}")
        return run.info.run_id

if __name__ == "__main__":
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("pIC50_prediction")
    
    # Ganti path dataset sesuai dengan dataset Anda
    dataset_path = "C:/Users/user/tugas/Semester_V/data_models/mlops/data/data_train.csv"
    
    run_id_lr = train_and_log_model(dataset_path, 'LogisticRegression')
    run_id_rf = train_and_log_model(dataset_path, 'RandomForest')
    run_id_nn = train_and_log_model(dataset_path, 'NeuralNetwork')
    
    print(f"Logistic Regression run_id: {run_id_lr}")
    print(f"Random Forest run_id: {run_id_rf}")
    print(f"Neural Network run_id: {run_id_nn}")