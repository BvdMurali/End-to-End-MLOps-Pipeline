import os
import logging
import joblib
import pandas as pd
import yaml
from evidently import Report, Dataset, DataDefinition, BinaryClassification
from evidently.presets import DataDriftPreset, ClassificationPreset

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Loads configurations from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_monitoring(config_path="config/config.yaml"):
    """Generates Evidently AI reports for data drift, prediction drift, and feature drift."""
    config = load_config(config_path)
    
    train_path = config["data"]["train_path"]
    test_path = config["data"]["test_path"]
    target_col = config["data"]["target_col"]
    model_path = config["artifacts"]["model_path"]
    report_path = config["monitoring"]["report_path"]
    
    logger.info("Loading reference (train) and current (test) datasets for monitoring...")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed datasets not found. Run preprocessing first.")
        
    reference_df = pd.read_csv(train_path)
    current_df = pd.read_csv(test_path)
    
    # Load model to generate predictions
    logger.info("Loading best model to compute predictions for drift analysis...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}. Run training first.")
    model = joblib.load(model_path)
    
    # Features (exclude target)
    X_ref = reference_df.drop(columns=[target_col])
    X_cur = current_df.drop(columns=[target_col])
    
    # Generate predictions
    logger.info("Computing predictions...")
    reference_df["prediction"] = model.predict(X_ref)
    current_df["prediction"] = model.predict(X_cur)
    
    # Define DataDefinition for Evidently AI
    # Identify numerical features (all columns except target and prediction)
    numerical_features = X_ref.columns.tolist()
    
    data_definition = DataDefinition(
        numerical_columns=numerical_features,
        classification=[
            BinaryClassification(
                target=target_col,
                prediction_labels="prediction"
            )
        ]
    )
    
    # Wrap in Evidently Dataset objects
    reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)
    
    logger.info("Running data drift, target drift, and feature drift analysis...")
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])
    
    snapshot = report.run(
        reference_data=reference_dataset,
        current_data=current_dataset
    )
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    snapshot.save_html(report_path)
    logger.info(f"Evidently AI Monitoring Report successfully saved to: {report_path}")

if __name__ == "__main__":
    run_monitoring()
