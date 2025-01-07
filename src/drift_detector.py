import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow Tracking URI

# Fungsi untuk menghitung PSI
def calculate_psi(expected, actual, bins=10):
    # Discretize kedua distribusi
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    expected_binned = discretizer.fit_transform(expected.reshape(-1, 1)).astype(int)
    actual_binned = discretizer.transform(actual.reshape(-1, 1)).astype(int)

    # Hitung frekuensi
    expected_freq = np.bincount(expected_binned.flatten(), minlength=bins) / len(expected)
    actual_freq = np.bincount(actual_binned.flatten(), minlength=bins) / len(actual)

    # Tambahkan epsilon untuk mencegah pembagian nol
    epsilon = 1e-10
    expected_freq = np.maximum(expected_freq, epsilon)
    actual_freq = np.maximum(actual_freq, epsilon)

    # Hitung PSI
    psi_value = np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))
    return psi_value

# Fungsi untuk mendeteksi drift pada model
def detect_drift(dataset_path):
    # Membaca dataset
    data = pd.read_csv(dataset_path)
    
    # Misalnya kita ingin mendeteksi drift pada kolom 'Feature'
    expected_feature = data['Actual_pIC50'].values  # Data training atau distribusi ekspektasi
    actual_feature = data['Predicted_pIC50'].values  # Data baru (data yang diuji untuk drift)

    # Hitung PSI
    psi_value = calculate_psi(expected_feature, actual_feature)
    
    # Log PSI ke MLflow sebagai metrik
    mlflow.log_metric('PSI_Value', psi_value)

    # Tentukan tingkat drift
    if psi_value < 0.1:
        drift_status = 'No drift'
    elif 0.1 <= psi_value < 0.2:
        drift_status = 'Moderate drift'
    else:
        drift_status = 'High drift'

    print(f"PSI: {psi_value}, Drift Status: {drift_status}")
    return psi_value, drift_status

if __name__ == "__main__":
    # Path dataset
    dataset_path = 'C:/Users/user/tugas/Semester_V/data_models/mlops/src/predictions_RandomForest.csv'  # Sesuaikan path dataset Anda
    mlflow.set_experiment("pIC50_prediction")
    # Mulai MLflow Run
    with mlflow.start_run():
        psi_value, drift_status = detect_drift(dataset_path)
        print(f"Drift detection completed with PSI: {psi_value} and status: {drift_status}")
