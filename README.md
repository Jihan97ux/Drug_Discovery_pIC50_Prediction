# Model Logging and Tracking with MLflow

## Author  
**Jihan Fadila**   
📧 Email: jihan4han97@gmail.com  

## Overview  
This project focuses on two key machine learning experiments:  
 **pIC50 Prediction** using regression models for drug discovery.  
Experiments are logged and monitored using **MLflow**.  

## Dataset Preparation  
- The dataset consists of **3,068 aromatase inhibitor compounds**.  
- Features are generated from **canonical SMILES** using **PaDEL Descriptor**, resulting in **307 molecular fingerprints**.  

### **Tools Used**  
- **Google Colab** for model training  
- **PaDEL Descriptor** for feature extraction  
- **VS Code & MLflow** for logging and monitoring  

## pIC50 Prediction  
pIC50 is a **continuous variable**, so **regression models** were used:  
- **Linear Regression**  
- **Random Forest Regressor**  
- **Neural Network**  

### **Evaluation Metrics**  
- **Mean Squared Error (MSE)**  
- **R-Squared (R²)**  
- **Root Mean Squared Error (RMSE)**  

### **Model Comparison**  
| Model  | MSE  | R² | RMSE |
|--------|------|------|------|
| **Random Forest** | **1.261** | **0.328** | **1.233** |
| Neural Network | 1.565 | 0.166 | 1.251 |
| Linear Regression | **16.7575** | **-7.9294** | **4.0936** |

**Findings:**  
- **Random Forest performed the best**, explaining more variance in the data.  
- **Neural Network showed stability**, but slightly underperformed compared to Random Forest.  
- **Linear Regression had the worst performance**, indicating the dataset requires more complex models.  

## Drift Detection & Model Selection  
To ensure long-term reliability, **Population Stability Index (PSI)** was used:  
- **Neural Network PSI: 0.26** (Moderate drift, stable)  
- **Random Forest PSI: 4.85** (High drift, significant data shift)  
- **Linear Regression PSI: 0.39** (Moderate drift)  

**Conclusion:**  
- **Random Forest was chosen as the best model** due to superior performance.  
- To mitigate drift, regular model updates and data monitoring are required.  
- **MLflow Drift Detector** was used to analyze prediction stability over time.

## Prerequisites
- Python 3.9 or higher
- Git
- Basic understanding of Machine Learning concepts
- Basic understanding of command line operations

## Setup Instructions

### 1. Clone the Repository
```bash
# Clone this repository
git clone https://github.com/Jihan97ux/Drug_Discovery_pIC50_Prediction.git
cd mlops

# Create project directories if they don't exist
mkdir -p data models
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv .

# Activate virtual environment
# For Windows:
mlops\Scripts\activate
# For Unix or MacOS:
source mlops/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

## Start 

1. Start MLflow UI server:
```bash
mlflow server --host 127.0.0.1 --port 5000
```

2. Generate synthetic data:
```bash
python data_generator.py
```
This will create:
- `data/data_train.csv`: Initial training data
- `data/new_data.csv`: Data with drift for later use

3. Train and log models:
```bash
python new_train_model.py
```
This will:
- Train a Neural Network model
- Train a Random Forest model
- Train a Linear Regression model
- Log models with their metrics to MLflow

4. Access MLflow UI:
- Open your browser and navigate to `http://127.0.0.1:5000`
- Compare the models by:
  - Checking accuracy and AUC scores
  - Looking at model parameters
  - Examining run metadata

## Troubleshooting

Common issues and solutions:

1. MLflow UI not starting:
   - Check if port 5000 is available
   - Ensure MLflow is installed correctly
   - Try a different port

2. Model logging fails:
   - Check file paths
   - Verify data format
   - Look for missing dependencies

3. Dataset logging issues:
   - Check file permissions
   - Verify data types
   - Ensure consistent column names

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Dataset Sources](https://www.kaggle.com/datasets)
- [Git Tutorial](https://git-scm.com/docs/gittutorial)
