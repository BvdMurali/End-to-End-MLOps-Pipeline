import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import logging
import yaml
import mlflow
from mlflow.client import MlflowClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Loads configurations from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def register_best_model(config_path="config/config.yaml"):
    """Registers the best model from the experiment in the MLflow Model Registry."""
    config = load_config(config_path)
    
    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]
    model_name = config["mlflow"]["model_name"]
    
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    # Get experiment
    logger.info(f"Retrieving experiment: {experiment_name}")
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found. Ensure training has run.")
        
    # Search runs for the best model (highest ROC-AUC)
    # We exclude the aggregate 'Best_Model' run name to find the actual individual model run, 
    # or we can use the 'Best_Model' run itself. Let's find the best run based on metric.
    logger.info("Searching for the best run based on ROC-AUC metric...")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.roc_auc > 0",
        order_by=["metrics.roc_auc DESC"],
        max_results=5
    )
    
    if not runs:
        raise ValueError("No runs found in the experiment with valid ROC-AUC metrics.")
        
    # Filter out aggregate run and pick the actual trained model run
    best_run = None
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "")
        # We can register the individual model run (e.g. Logistic_Regression, Random_Forest, XGBoost)
        # because they have the 'model' artifact.
        if run_name in ["Logistic_Regression", "Random_Forest", "XGBoost"]:
            best_run = run
            break
            
    if not best_run:
        # Fallback to the first run if filter did not match
        best_run = runs[0]
        
    best_run_id = best_run.info.run_id
    best_roc_auc = best_run.data.metrics["roc_auc"]
    best_model_run_name = best_run.data.tags.get("mlflow.runName", "Unknown")
    
    logger.info(f"Best run ID: {best_run_id} ({best_model_run_name}) with ROC-AUC: {best_roc_auc:.4f}")
    
    # Register the model
    model_uri = f"runs/{best_run_id}/model"
    logger.info(f"Registering model under name '{model_name}' using URI '{model_uri}'...")
    
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info(f"Model registered successfully. Version: {model_version.version}")
    
    # Transition to Production stage
    logger.info(f"Transitioning model version {model_version.version} to Production stage...")
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"Model version {model_version.version} successfully transitioned to Production.")
    except Exception as e:
        logger.warning(f"Could not transition model version stage: {e}. (This can happen if using newer MLflow versions that deprecate Stages in favor of aliases).")
        
if __name__ == "__main__":
    register_best_model()
