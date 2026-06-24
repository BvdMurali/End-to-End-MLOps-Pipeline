import os
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from src.preprocess import load_config

# Helper function to create a dummy Bank Marketing dataset
def get_dummy_data():
    return pd.DataFrame({
        "age": [30, 40, 50, 25, 45],
        "job": ["blue-collar", "technician", "management", "blue-collar", "admin."],
        "marital": ["married", "single", "married", "single", "divorced"],
        "education": ["basic.9y", "university.degree", "high.school", "basic.4y", "high.school"],
        "default": ["no", "no", "unknown", "no", "no"],
        "housing": ["yes", "no", "yes", "yes", "no"],
        "loan": ["no", "no", "no", "yes", "no"],
        "contact": ["cellular", "telephone", "cellular", "cellular", "telephone"],
        "month": ["may", "jul", "aug", "may", "jun"],
        "day_of_week": ["mon", "wed", "fri", "tue", "thu"],
        "duration": [200, 150, 300, 100, 400],
        "campaign": [2, 1, 3, 1, 2],
        "pdays": [999, 999, 12, 999, 999],
        "previous": [0, 0, 1, 0, 0],
        "poutcome": ["nonexistent", "nonexistent", "success", "nonexistent", "nonexistent"],
        "emp.var.rate": [-1.8, 1.4, -0.1, -1.8, 1.4],
        "cons.price.idx": [92.893, 93.994, 93.444, 92.893, 93.994],
        "cons.conf.idx": [-46.2, -36.4, -36.1, -46.2, -36.4],
        "euribor3m": [1.299, 4.857, 1.344, 1.299, 4.857],
        "nr.employed": [5099.1, 5191.0, 5076.2, 5099.1, 5191.0],
        "y": ["yes", "no", "yes", "no", "no"]
    })

def test_load_config():
    config = load_config("config/config.yaml")
    assert "data" in config
    assert "model" in config
    assert "mlflow" in config
    assert "logging" in config

def test_target_mapping():
    df = get_dummy_data()
    # Map target
    df["y"] = df["y"].map({"yes": 1, "no": 0})
    assert set(df["y"].unique()).issubset({0, 1})
    assert df["y"].tolist() == [1, 0, 1, 0, 0]

def test_column_transformer_fitting():
    df = get_dummy_data()
    df["y"] = df["y"].map({"yes": 1, "no": 0})
    
    X = df.drop(columns=["y"])
    
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    assert len(numerical_cols) == 10
    assert len(categorical_cols) == 10
    
    # Fit StandardScaler and OneHotEncoder
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    # Verify shape
    # 5 rows, 10 numerical features, and encoded categories
    assert X_processed.shape[0] == 5
    assert X_processed.shape[1] > 10  # Encoded categories should expand the size
    
    # Test that preprocessor can be saved and loaded
    joblib.dump(preprocessor, "temp_preprocessor.joblib")
    assert os.path.exists("temp_preprocessor.joblib")
    
    loaded_preprocessor = joblib.load("temp_preprocessor.joblib")
    X_loaded_processed = loaded_preprocessor.transform(X)
    
    np.testing.assert_array_equal(X_processed, X_loaded_processed)
    
    # Cleanup
    os.remove("temp_preprocessor.joblib")
