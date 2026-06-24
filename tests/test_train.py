import os
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def get_dummy_processed_data():
    np.random.seed(42)
    # Generate 50 rows of processed numerical features
    num_features = 25
    X_train_arr = np.random.randn(40, num_features)
    X_test_arr = np.random.randn(10, num_features)
    
    # Class labels
    y_train = np.random.randint(0, 2, size=40)
    y_test = np.random.randint(0, 2, size=10)
    
    cols = [f"feat_{i}" for i in range(num_features)]
    train_df = pd.DataFrame(X_train_arr, columns=cols)
    test_df = pd.DataFrame(X_test_arr, columns=cols)
    
    return train_df, test_df, y_train, y_test

def test_model_training_and_metrics():
    train_df, test_df, y_train, y_test = get_dummy_processed_data()
    
    # Train Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(train_df, y_train)
    
    y_pred = lr.predict(test_df)
    y_pred_proba = lr.predict_proba(test_df)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= roc_auc <= 1.0

def test_best_model_selection_logic():
    # Simulate scores from different models
    model_scores = {
        "Logistic_Regression": {"roc_auc": 0.72, "f1": 0.65},
        "Random_Forest": {"roc_auc": 0.78, "f1": 0.69},
        "XGBoost": {"roc_auc": 0.81, "f1": 0.73}
    }
    
    best_roc_auc = 0.0
    best_model_name = None
    
    for name, metrics in model_scores.items():
        if metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = metrics["roc_auc"]
            best_model_name = name
            
    assert best_model_name == "XGBoost"
    assert best_roc_auc == 0.81
