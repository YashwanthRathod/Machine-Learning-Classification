import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def load_pima_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    return pd.read_csv(url, names=cols)

def run_task4():
    # 1 & 2. Load dataset into pandas DataFrame with appropriate column names
    df = load_pima_dataset()
    print("Dataset Overview:")
    print(df.head())

    # 3. Check for missing or unrealistic zero values and handle them
    unrealistic_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    print("\nUnrealistic Zero Counts:")
    for col in unrealistic_zero_cols:
        zeros = (df[col] == 0).sum()
        print(f"  {col}: {zeros} zero values")

    # Replace zero values with NaN and impute using median
    df_clean = df.copy()
    for col in unrealistic_zero_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 5. Define features (X) and target variable (y)
    X = df_clean.drop(columns=['Outcome'])
    y = df_clean['Outcome']

    # 4. Split dataset into 80% train / 20% test with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 6. Train standard DecisionTreeClassifier (Unrestricted Depth)
    dt_unrestricted = DecisionTreeClassifier(random_state=42)
    dt_unrestricted.fit(X_train, y_train)

    # 7. Evaluate Unrestricted Decision Tree
    y_pred_unrestricted = dt_unrestricted.predict(X_test)
    print("\n=== Task 4: Unrestricted Decision Tree Results ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_unrestricted):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_unrestricted):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_unrestricted):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred_unrestricted):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_unrestricted))

    # 8. Train restricted Decision Tree (max_depth=3)
    dt_restricted = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_restricted.fit(X_train, y_train)

    y_pred_restricted = dt_restricted.predict(X_test)
    print("\n=== Task 4: Restricted Decision Tree (max_depth=3) Results ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_restricted):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_restricted):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_restricted):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred_restricted):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_restricted))

    # Compare feature importances of both models
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Unrestricted Importance': dt_unrestricted.feature_importances_,
        'Restricted (max_depth=3) Importance': dt_restricted.feature_importances_
    }).sort_values(by='Restricted (max_depth=3) Importance', ascending=False)

    print("\nFeature Importances Comparison:")
    print(feature_importance_df.to_string(index=False))

if __name__ == "__main__":
    run_task4()
