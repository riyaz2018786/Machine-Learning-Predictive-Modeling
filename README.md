# Machine Learning Predictive Modeling

An end-to-end supervised machine learning pipeline built to simulate customer behavior data, train a predictive model, and evaluate performance classification metrics.

## Project Workflow
1. Data Generation (generate_ml_data.py): Simulates a customer dataset tracking age, monthly spend, and session metrics while generating a binary outcome target.
2. Model Training (train_model.py): Splits the dataset into training and testing sets, applies standard feature scaling, and trains a Random Forest Classifier.
3. Model Evaluation (evaluate_model.py): Tests the trained pipeline against unseen data to compute accuracy scores and outputs a performance evaluation dashboard.

## Tech Stack
- Language: Python
- Libraries: Pandas, NumPy, Scikit-Learn, Joblib, Matplotlib

## Core Evaluation Outputs
- Confusion Matrix: Tracks true versus predicted classifications to measure exact decision boundaries.
- ROC Curve: Maps the true positive rate against the false positive rate, confirming strong predictive capabilities with an AUC of 0.86.
