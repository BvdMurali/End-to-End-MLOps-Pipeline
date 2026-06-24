import pytest
import pandas as pd
import numpy as np

# Column lists
expected_columns = [
    "age", "job", "marital", "education", "default", "housing", "loan", 
    "contact", "month", "day_of_week", "duration", "campaign", "pdays", 
    "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", 
    "euribor3m", "nr.employed", "y"
]

def check_schema(df, expected_cols):
    actual_cols = df.columns.tolist()
    missing_cols = [col for col in expected_cols if col not in actual_cols]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    return True, "Passed"

def check_missing_values(df, threshold):
    null_ratios = df.isnull().mean()
    excessive_cols = null_ratios[null_ratios > threshold].to_dict()
    if excessive_cols:
        return False, excessive_cols
    return True, "Passed"

def test_schema_check_success():
    # Construct complete dataframe
    data = {col: [1, 2, 3] for col in expected_columns}
    df = pd.DataFrame(data)
    
    passed, msg = check_schema(df, expected_columns)
    assert passed
    assert msg == "Passed"

def test_schema_check_failure():
    # Remove 'y' and 'age'
    incomplete_cols = [col for col in expected_columns if col not in ["y", "age"]]
    data = {col: [1, 2, 3] for col in incomplete_cols}
    df = pd.DataFrame(data)
    
    passed, msg = check_schema(df, expected_columns)
    assert not passed
    assert "Missing columns" in msg
    assert "y" in msg
    assert "age" in msg

def test_missing_values_check_success():
    data = {col: [1, 2, 3] for col in expected_columns}
    df = pd.DataFrame(data)
    
    passed, msg = check_missing_values(df, 0.05)
    assert passed
    assert msg == "Passed"

def test_missing_values_check_failure():
    data = {col: [1, 2, np.nan] for col in expected_columns}  # 33% missing values in all columns
    df = pd.DataFrame(data)
    
    passed, msg = check_missing_values(df, 0.05)
    assert not passed
    assert "age" in msg  # age has 33% nulls which is > 5%
