import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import sys
import logging
import time

# Configure logging for pipeline orchestration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PipelineOrchestrator")

def run_pipeline():
    start_time = time.time()
    logger.info("=========================================")
    logger.info("Starting End-to-End MLOps Pipeline Run...")
    logger.info("=========================================")
    
    # 1. Data Ingestion
    logger.info("--- STAGE 1: Data Ingestion ---")
    try:
        from src.data_ingestion import ingest_data
        ingest_data()
        logger.info("Data Ingestion stage completed successfully.")
    except Exception as e:
        logger.exception("Data Ingestion failed!")
        sys.exit(1)
        
    # 2. Data Preprocessing
    logger.info("--- STAGE 2: Preprocessing ---")
    try:
        from src.preprocess import preprocess_data
        preprocess_data()
        logger.info("Preprocessing stage completed successfully.")
    except Exception as e:
        logger.exception("Preprocessing failed!")
        sys.exit(1)
        
    # 3. Model Training & MLflow Logging
    logger.info("--- STAGE 3: Model Training ---")
    try:
        from src.train import train_and_evaluate
        best_model_name, best_metrics = train_and_evaluate()
        logger.info(f"Model Training completed. Best model: {best_model_name}")
        logger.info(f"Best model metrics: {best_metrics}")
    except Exception as e:
        logger.exception("Model Training failed!")
        sys.exit(1)
        
    # 4. Model Validation
    logger.info("--- STAGE 4: Model Validation ---")
    try:
        from src.validate import validate_pipeline
        validate_pipeline()
        logger.info("Model Validation completed successfully. Metrics meet quality bar.")
    except SystemExit as se:
        # validate_pipeline uses sys.exit(1) to fail
        if se.code != 0:
            logger.error("Model Validation failed!")
            sys.exit(se.code)
    except Exception as e:
        logger.exception("Model Validation failed with an error!")
        sys.exit(1)
        
    # 5. Model Registration
    logger.info("--- STAGE 5: Model Registration ---")
    try:
        from src.register import register_best_model
        register_best_model()
        logger.info("Model Registration completed successfully.")
    except Exception as e:
        logger.exception("Model Registration failed!")
        # We don't want to completely block the build if there's local registry issues,
        # but for MLOps pipeline correctness, we log it.
        logger.warning("Continuing pipeline despite registration failure.")
        
    # 6. Drift Monitoring
    logger.info("--- STAGE 6: Drift Monitoring ---")
    try:
        from monitoring.monitoring import run_monitoring
        run_monitoring()
        logger.info("Drift Monitoring report generated successfully.")
    except Exception as e:
        logger.exception("Drift Monitoring failed!")
        sys.exit(1)
        
    end_time = time.time()
    duration = end_time - start_time
    logger.info("=========================================")
    logger.info(f"End-to-End Pipeline Completed in {duration:.2f} seconds!")
    logger.info("=========================================")

if __name__ == "__main__":
    run_pipeline()
