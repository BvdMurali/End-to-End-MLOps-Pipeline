import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import shutil
import pytest
import pandas as pd
import numpy as np
import yaml
import mlflow
from src.preprocess import preprocess_data
from src.train import train_and_evaluate
from src.validate import validate_pipeline
from src.register import register_best_model
from monitoring.monitoring import run_monitoring

# Create a fixture to set up and tear down a full test workspace
@pytest.fixture(scope="module")
def setup_test_pipeline():
    import uuid
    exp_name = f"test-experiment-{uuid.uuid4().hex[:8]}"
    test_dir = "tests/test_workspace"
    
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(f"{test_dir}/raw", exist_ok=True)
    os.makedirs(f"{test_dir}/processed", exist_ok=True)
    os.makedirs(f"{test_dir}/artifacts", exist_ok=True)
    os.makedirs(f"{test_dir}/monitoring", exist_ok=True)
    os.makedirs(f"{test_dir}/logs", exist_ok=True)
    
    # 1. Create a dummy raw dataset (50 rows)
    np.random.seed(42)
    dummy_df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=50),
        "job": np.random.choice(["blue-collar", "technician", "management", "admin."], size=50),
        "marital": np.random.choice(["married", "single", "divorced"], size=50),
        "education": np.random.choice(["basic.9y", "university.degree", "high.school"], size=50),
        "default": np.random.choice(["no", "unknown"], size=50),
        "housing": np.random.choice(["yes", "no"], size=50),
        "loan": np.random.choice(["yes", "no"], size=50),
        "contact": np.random.choice(["cellular", "telephone"], size=50),
        "month": np.random.choice(["may", "jun", "jul", "aug"], size=50),
        "day_of_week": np.random.choice(["mon", "tue", "wed", "thu", "fri"], size=50),
        "duration": np.random.randint(10, 1000, size=50),
        "campaign": np.random.randint(1, 5, size=50),
        "pdays": np.random.choice([999, 10, 5], size=50),
        "previous": np.random.randint(0, 3, size=50),
        "poutcome": np.random.choice(["nonexistent", "success", "failure"], size=50),
        "emp.var.rate": np.random.randn(50),
        "cons.price.idx": np.random.uniform(92.0, 95.0, size=50),
        "cons.conf.idx": np.random.uniform(-50.0, -30.0, size=50),
        "euribor3m": np.random.uniform(0.5, 5.0, size=50),
        "nr.employed": np.random.uniform(4900, 5200, size=50),
        "y": np.random.choice(["yes", "no"], size=50)
    })
    
    # Save raw data with semicolon separator
    raw_path = f"{test_dir}/raw/bank-additional-full.csv"
    dummy_df.to_csv(raw_path, sep=";", index=False)
    
    # 2. Create test config yaml file
    config_dict = {
        "data": {
            "raw_url": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip", # Dummy
            "raw_dir": f"{test_dir}/raw",
            "raw_file_path": raw_path,
            "processed_dir": f"{test_dir}/processed",
            "train_path": f"{test_dir}/processed/train.csv",
            "test_path": f"{test_dir}/processed/test.csv",
            "target_col": "y"
        },
        "artifacts": {
            "dir": f"{test_dir}/artifacts",
            "preprocessor_path": f"{test_dir}/artifacts/preprocessor.joblib",
            "model_path": f"{test_dir}/artifacts/best_model.joblib"
        },
        "model": {
            "test_size": 0.2,
            "random_state": 42,
            "thresholds": {
                "roc_auc": 0.30, # Low threshold so dummy model passes
                "missing_ratio": 0.05
            },
            "logistic_regression": {
                "max_iter": 50,
                "C": 1.0
            },
            "random_forest": {
                "n_estimators": 5,
                "max_depth": 3,
                "random_state": 42
            },
            "xgboost": {
                "n_estimators": 5,
                "max_depth": 3,
                "learning_rate": 0.1,
                "random_state": 42
            }
        },
        "mlflow": {
            "tracking_uri": "mlruns",
            "experiment_name": exp_name,
            "model_name": "test-bank-marketing-model"
        },
        "monitoring": {
            "report_path": f"{test_dir}/monitoring/drift_report.html"
        },
        "logging": {
            "log_file": f"{test_dir}/logs/test.log",
            "level": "INFO"
        }
    }
    
    config_path = f"{test_dir}/test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)
        
    yield config_path
    
    # Tear down
    shutil.rmtree(test_dir, ignore_errors=True)
    # Delete mlflow test runs if needed
    try:
        client = mlflow.client.MlflowClient()
        experiment = client.get_experiment_by_name(exp_name)
        if experiment:
            client.delete_experiment(experiment.experiment_id)
    except Exception:
        pass

def test_full_pipeline_flow(setup_test_pipeline):
    config_path = setup_test_pipeline
    
    # 1. Run preprocessing stage
    preprocess_data(config_path=config_path)
    
    # Assert processed train/test datasets exist
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    assert os.path.exists(config["data"]["train_path"])
    assert os.path.exists(config["data"]["test_path"])
    assert os.path.exists(config["artifacts"]["preprocessor_path"])
    
    # 2. Run model training stage
    best_model_name, best_metrics = train_and_evaluate(config_path=config_path)
    assert best_model_name in ["Logistic_Regression", "Random_Forest", "XGBoost"]
    assert "roc_auc" in best_metrics
    assert os.path.exists(config["artifacts"]["model_path"])
    
    # 3. Run model validation stage
    # This should complete successfully and not exit because ROC-AUC is above 0.30
    validate_pipeline(config_path=config_path)
    assert os.path.exists(os.path.join(config["artifacts"]["dir"], "validation_report.json"))
    
    # 4. Run model registration stage
    register_best_model(config_path=config_path)
    
    # 5. Run drift monitoring stage
    run_monitoring(config_path=config_path)
    assert os.path.exists(config["monitoring"]["report_path"])
