import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import logging
import joblib
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try importing XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logger.info("XGBoost is available.")
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost is not installed or import failed. Will fallback to Scikit-Learn models.")

def load_config(config_path="config/config.yaml"):
    """Loads configurations from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generates and saves confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(importances, feature_names, save_path, model_name):
    """Generates and saves feature importance plot."""
    indices = np.argsort(importances)[::-1]
    # Keep top 15 features for readability
    top_n = min(15, len(feature_names))
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Feature Importances - {model_name}")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_and_evaluate(config_path="config/config.yaml"):
    """Loads processed data, trains candidate models, logs to MLflow, and saves the best model."""
    config = load_config(config_path)
    
    train_path = config["data"]["train_path"]
    test_path = config["data"]["test_path"]
    target_col = config["data"]["target_col"]
    model_path = config["artifacts"]["model_path"]
    random_state = config["model"]["random_state"]
    
    # Load processed data
    logger.info("Loading processed datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    feature_names = X_train.columns.tolist()
    
    # Configure MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    # Define models dictionary
    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=config["model"]["logistic_regression"]["max_iter"],
            C=config["model"]["logistic_regression"]["C"],
            random_state=random_state
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=config["model"]["random_forest"]["n_estimators"],
            max_depth=config["model"]["random_forest"]["max_depth"],
            random_state=random_state
        )
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=config["model"]["xgboost"]["n_estimators"],
            max_depth=config["model"]["xgboost"]["max_depth"],
            learning_rate=config["model"]["xgboost"]["learning_rate"],
            random_state=random_state,
            eval_metric="logloss"
        )
        
    best_roc_auc = 0.0
    best_model_name = None
    best_model_obj = None
    best_metrics = {}
    
    os.makedirs(config["artifacts"]["dir"], exist_ok=True)
    os.makedirs("temp_plots", exist_ok=True)
    
    for name, model in models.items():
        logger.info(f"Training model: {name}...")
        
        with mlflow.start_run(run_name=name) as run:
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Compute metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_pred_proba)
            }
            
            logger.info(f"{name} Metrics: {metrics}")
            
            # Log params & metrics to MLflow
            # Extract parameters for logging
            if name == "Logistic_Regression":
                mlflow.log_params({
                    "C": model.C,
                    "max_iter": model.max_iter
                })
            elif name == "Random_Forest":
                mlflow.log_params({
                    "n_estimators": model.n_estimators,
                    "max_depth": model.max_depth
                })
            elif name == "XGBoost":
                mlflow.log_params({
                    "n_estimators": model.n_estimators,
                    "max_depth": model.max_depth,
                    "learning_rate": model.learning_rate
                })
                
            mlflow.log_metrics(metrics)
            
            # Save and log Confusion Matrix
            cm_path = f"temp_plots/confusion_matrix_{name}.png"
            plot_confusion_matrix(y_test, y_pred, cm_path)
            mlflow.log_artifact(cm_path)
            
            # Save and log Classification Report
            report_text = classification_report(y_test, y_pred, target_names=["No", "Yes"])
            report_path = f"temp_plots/classification_report_{name}.txt"
            with open(report_path, "w") as f:
                f.write(report_text)
            mlflow.log_artifact(report_path)
            
            # Save and log Feature Importance (for RF and XGBoost)
            if hasattr(model, "feature_importances_"):
                fi_path = f"temp_plots/feature_importance_{name}.png"
                plot_feature_importance(model.feature_importances_, feature_names, fi_path, name)
                mlflow.log_artifact(fi_path)
                
            # Log sklearn model
            mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="cloudpickle")
            
            # Keep track of the best model based on ROC-AUC
            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_model_name = name
                best_model_obj = model
                best_metrics = metrics
                
    logger.info(f"Model selection complete. Best model: {best_model_name} with ROC-AUC {best_roc_auc:.4f}")
    
    # Save best model locally
    joblib.dump(best_model_obj, model_path)
    logger.info(f"Saved best model locally to {model_path}")
    
    # Log best model as a separate run or tag it
    with mlflow.start_run(run_name="Best_Model") as run:
        mlflow.set_tag("best_model_name", best_model_name)
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(best_model_obj, artifact_path="best_model", serialization_format="cloudpickle")
        
    # Clean up temp plots
    import shutil
    shutil.rmtree("temp_plots", ignore_errors=True)
    
    return best_model_name, best_metrics

if __name__ == "__main__":
    train_and_evaluate()
