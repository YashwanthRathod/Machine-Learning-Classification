# Supervised Machine Learning Classification Benchmarks

A comprehensive machine learning repository containing implementations, preprocessing workflows, model training, and performance evaluations across four supervised classification tasks:

1. **Random Forest Classifier** — Breast Cancer Dataset (`sklearn`)
2. **Logistic Regression Classifier** — Pima Indians Diabetes Dataset
3. **XGBoost Classifier** — Titanic Passenger Survival Dataset
4. **Decision Tree Classifier** — Pima Indians Diabetes Dataset (Unrestricted vs. Pruned `max_depth=3`)

---

## 📌 Project Overview

This repository demonstrates practical workflows in machine learning classification:
- **Preprocessing & Cleaning**: Handling missing values, imputing physiological zero anomalies, feature scaling (`StandardScaler`), categorical one-hot encoding.
- **Model Evaluation**: Comprehensive benchmarking using Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and ROC-AUC scores.
- **Interpretability**: Extracting feature importances and interpreting Logistic Regression coefficients (Odds Ratios).

---

## 📁 Repository Structure

```
ml-classification-benchmark/
│
├── task1_random_forest.py       # Random Forest model on Breast Cancer dataset
├── task2_logistic_regression.py # Logistic Regression model on Diabetes dataset
├── task3_xgboost.py            # XGBoost model on Titanic Survival dataset
├── task4_decision_tree.py      # Decision Tree comparison on Diabetes dataset
├── main.py                     # Execution entry-point for running all tasks
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ml-classification-benchmark.git
   cd ml-classification-benchmark
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Models

Run any individual task script:
```bash
python task1_random_forest.py
python task2_logistic_regression.py
python task3_xgboost.py
python task4_decision_tree.py
```

Or run all 4 classification tasks sequentially:
```bash
python main.py
```

---

## 📊 Summary of Benchmark Results

### 1. Task 1: Random Forest (Breast Cancer Dataset)
- **Dataset**: `sklearn.datasets.load_breast_cancer` (569 samples, 30 features)
- **Train/Test Split**: 80% / 20% (Stratified)
- **Key Metrics**:
  - **Accuracy**: `95.61%`
  - **Precision**: `95.89%`
  - **Recall**: `97.22%`
  - **F1-Score**: `96.55%`
  - **ROC-AUC Score**: `99.37%`

---

### 2. Task 2: Logistic Regression (Pima Indians Diabetes Dataset)
- **Preprocessing**: Feature scaling applied using `StandardScaler`.
- **Key Metrics**:
  - **Accuracy**: `71.43%`
  - **Precision**: `60.87%`
  - **Recall**: `51.85%`
  - **F1-Score**: `56.00%`
  - **ROC-AUC Score**: `82.30%`
- **Coefficient Interpretation (Top Predictors)**:
  - `Glucose` ($\beta = 1.1442$, $\text{Odds Ratio} = 3.14$): 1 SD increase increases diabetes odds by ~214%.
  - `BMI` ($\beta = 0.7139$, $\text{Odds Ratio} = 2.04$): Doubles diabetes odds per 1 SD increase.

---

### 3. Task 3: XGBoost Classifier (Titanic Dataset)
- **Preprocessing**: Imputed missing `Age`, `Fare`, and `Embarked` values; categorical features one-hot encoded.
- **Hyperparameters**: `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Key Metrics**:
  - **Accuracy**: `79.89%`
  - **Precision**: `78.95%`
  - **Recall**: `65.22%`
  - **F1-Score**: `71.43%`
  - **ROC-AUC Score**: `81.71%`
- **Top Feature Importances**:
  1. `Sex_male`: **48.53%**
  2. `Pclass`: **16.95%**
  3. `Embarked_Q`: **6.96%**

---

### 4. Task 4: Decision Tree Comparison (Pima Indians Diabetes Dataset)
- **Preprocessing**: Unrealistic zero values in biological fields (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) handled via median imputation.
- **Model Comparison**:

| Metric | Unrestricted Decision Tree | Restricted Decision Tree (`max_depth=3`) |
| :--- | :---: | :---: |
| **Accuracy** | 72.73% | **75.97%** |
| **Precision** | 61.40% | **68.00%** |
| **Recall** | **63.64%** | 61.82% |
| **F1-Score** | 62.50% | **64.76%** |

- **Top Features (`max_depth=3`)**: `Glucose` (59.66%), `BMI` (26.33%), `Age` (14.01%). Restricting depth effectively prevented overfitting on noise.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
