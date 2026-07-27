import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from task1_random_forest import run_task1
from task2_logistic_regression import run_task2
from task3_xgboost import run_task3
from task4_decision_tree import run_task4

def main():
    print("==================================================")
    print("      SUPERVISED CLASSIFICATION BENCHMARK        ")
    print("==================================================")

    print("\nRunning Task 1: Random Forest Classifier...")
    run_task1()

    print("\n" + "-"*50 + "\n")
    print("Running Task 2: Logistic Regression...")
    run_task2()

    print("\n" + "-"*50 + "\n")
    print("Running Task 3: XGBoost Classifier...")
    run_task3()

    print("\n" + "-"*50 + "\n")
    print("Running Task 4: Decision Tree Classifier...")
    run_task4()

    print("\n==================================================")
    print("  ALL TASKS COMPLETED SUCCESSFULLY!              ")
    print("==================================================")

if __name__ == "__main__":
    main()
