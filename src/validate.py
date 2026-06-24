import os
import sys
import json
import logging
import joblib
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Loads configurations from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def validate_pipeline(config_path="config/config.yaml"):
    """Validates schema, missing values, and model performance."""
    config = load_config(config_path)
    
    raw_file_path = config["data"]["raw_file_path"]
    test_path = config["data"]["test_path"]
    target_col = config["data"]["target_col"]
    model_path = config["artifacts"]["model_path"]
    
    roc_auc_threshold = config["model"]["thresholds"]["roc_auc"]
    missing_ratio_threshold = config["model"]["thresholds"]["missing_ratio"]
    
    validation_status = {
        "schema_validation": "FAILED",
        "missing_values_validation": "FAILED",
        "model_performance_validation": "FAILED",
        "overall_status": "FAILED",
        "details": {}
    }
    
    report_dir = config["artifacts"]["dir"]
    def save_report(status_dict):
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "validation_report.json")
        with open(report_path, "w") as f:
            json.dump(status_dict, f, indent=4)
        logger.info(f"Validation report saved to {report_path}")
        
    # 1. Schema Validation
    logger.info("Starting schema validation...")
    expected_columns = [
        "age", "job", "marital", "education", "default", "housing", "loan", 
        "contact", "month", "day_of_week", "duration", "campaign", "pdays", 
        "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", 
        "euribor3m", "nr.employed", "y"
    ]
    
    if not os.path.exists(raw_file_path):
        msg = f"Raw dataset not found at {raw_file_path}"
        logger.error(msg)
        validation_status["details"]["schema"] = msg
        save_report(validation_status)
        sys.exit(1)
        
    raw_df = pd.read_csv(raw_file_path, sep=";")
    actual_columns = raw_df.columns.tolist()
    
    missing_cols = [col for col in expected_columns if col not in actual_columns]
    if missing_cols:
        msg = f"Schema mismatch: Missing columns {missing_cols}"
        logger.error(msg)
        validation_status["details"]["schema"] = msg
        save_report(validation_status)
        sys.exit(1)
    else:
        logger.info("Schema validation passed.")
        validation_status["schema_validation"] = "PASSED"
        validation_status["details"]["schema"] = "All expected columns are present."
        
    # 2. Missing Values Validation
    logger.info("Starting missing values validation...")
    # standard pandas missing values
    null_ratios = raw_df.isnull().mean()
    excessive_null_cols = null_ratios[null_ratios > missing_ratio_threshold].to_dict()
    
    if excessive_null_cols:
        msg = f"Excessive missing values detected in columns: {excessive_null_cols}"
        logger.error(msg)
        validation_status["details"]["missing_values"] = excessive_null_cols
        save_report(validation_status)
        sys.exit(1)
    else:
        logger.info("Missing values validation passed (no columns exceed threshold).")
        validation_status["missing_values_validation"] = "PASSED"
        validation_status["details"]["missing_values"] = "All columns are within null threshold."
        
    # 3. Model Performance Validation
    logger.info("Starting model performance validation...")
    if not os.path.exists(model_path):
        msg = f"Trained model not found at {model_path}"
        logger.error(msg)
        validation_status["details"]["model_performance"] = msg
        save_report(validation_status)
        sys.exit(1)
        
    if not os.path.exists(test_path):
        msg = f"Processed test set not found at {test_path}"
        logger.error(msg)
        validation_status["details"]["model_performance"] = msg
        save_report(validation_status)
        sys.exit(1)
        
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    model = joblib.load(model_path)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"Model ROC-AUC on test set: {roc_auc:.4f} (Threshold: {roc_auc_threshold:.4f})")
    
    validation_status["details"]["roc_auc"] = roc_auc
    
    if roc_auc < roc_auc_threshold:
        msg = f"Model performance failed: ROC-AUC {roc_auc:.4f} is below threshold {roc_auc_threshold:.4f}"
        logger.error(msg)
        validation_status["details"]["model_performance"] = msg
        save_report(validation_status)
        sys.exit(1)
    else:
        logger.info("Model performance validation passed.")
        validation_status["model_performance_validation"] = "PASSED"
        validation_status["details"]["model_performance"] = f"ROC-AUC {roc_auc:.4f} meets threshold {roc_auc_threshold:.4f}."
        
    # Set overall status to PASSED if all checks passed
    if (validation_status["schema_validation"] == "PASSED" and 
        validation_status["missing_values_validation"] == "PASSED" and 
        validation_status["model_performance_validation"] == "PASSED"):
        validation_status["overall_status"] = "PASSED"
        logger.info("All pipeline validation checks completed successfully!")
        
    save_report(validation_status)
    
def save_report(status_dict):
    """Saves validation report as a JSON file."""
    report_dir = "artifacts"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(status_dict, f, indent=4)
    logger.info(f"Validation report saved to {report_path}")

if __name__ == "__main__":
    validate_pipeline()
