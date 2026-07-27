import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

def load_titanic_dataset():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

def preprocess_titanic(data):
    # Select features of interest
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
    df = data[features].copy()

    # Impute missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df

def run_task3():
    # 1 & 2. Load dataset
    raw_df = load_titanic_dataset()
    print("Titanic Dataset Sample:")
    print(raw_df.head())

    # 3. Check for missing values
    print("\nMissing values check:")
    print(raw_df.isnull().sum())

    # Preprocess dataset
    clean_df = preprocess_titanic(raw_df)
    X = clean_df.drop(columns=['Survived'])
    y = clean_df['Survived']

    # 4. Split dataset into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Apply feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train XGBoost classifier with custom parameters
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    xgb.fit(X_train_scaled, y_train)

    # 7. Evaluate model performance
    y_pred = xgb.predict(X_test_scaled)
    y_proba = xgb.predict_proba(X_test_scaled)[:, 1]

    print("\n=== Task 3: XGBoost Classifier Results ===")
    print(f"Accuracy:       {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision:      {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:         {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:       {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score:  {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 8. Feature Importance Interpretation
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importances:")
    print(importance_df.to_string(index=False))

if __name__ == "__main__":
    run_task3()
