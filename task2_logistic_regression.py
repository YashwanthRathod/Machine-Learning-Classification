import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

def load_diabetes_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    return pd.read_csv(url, names=columns)

def run_task2():
    # 1 & 2. Load dataset into pandas DataFrame and assign column names
    df = load_diabetes_data()
    print("Dataset Preview:")
    print(df.head())

    # 3. Check for missing or zero values
    print("\nMissing values check:")
    print(df.isnull().sum())

    print("\nZero values count per column:")
    for col in df.columns:
        zeros = (df[col] == 0).sum()
        print(f"  {col}: {zeros}")

    # Separate features and target label
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # 4. Split dataset into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Apply feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train Logistic Regression model
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    # 7. Evaluate model performance
    y_pred = log_reg.predict(X_test_scaled)
    y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

    print("\n=== Task 2: Logistic Regression Results ===")
    print(f"Accuracy:       {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision:      {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:         {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:       {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score:  {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 8. Interpret model coefficients
    coef_summary = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': log_reg.coef_[0],
        'Odds Ratio': np.exp(log_reg.coef_[0])
    }).sort_values(by='Coefficient', ascending=False)

    print("\nModel Coefficients & Odds Ratios:")
    print(coef_summary.to_string(index=False))

if __name__ == "__main__":
    run_task2()
