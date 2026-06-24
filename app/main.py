import os
import logging
import yaml
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Setup logging function to be shared or configured on startup
def configure_logging(config_path="config/config.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        log_file = config["logging"]["log_file"]
        log_level = config["logging"]["level"]
    except Exception:
        log_file = "logs/app.log"
        log_level = "INFO"
        
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

configure_logging()
logger = logging.getLogger("app.main")

def load_config(config_path="config/config.yaml"):
    """Loads configurations from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Global dictionary to hold model artifacts
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load config
    config = load_config()
    preprocessor_path = config["artifacts"]["preprocessor_path"]
    model_path = config["artifacts"]["model_path"]
    
    logger.info("Initializing FastAPI Lifespan. Loading ML artifacts...")
    if ml_models.get("preprocessor") is not None and ml_models.get("model") is not None:
        logger.info("ML artifacts already initialized (mocked for testing). Skipping disk load.")
    else:
        if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
            logger.error("ML artifacts not found. Please train and validate the model first.")
            # We don't crash startup during build/testing environments, 
            # but in production we want to log the error.
            ml_models["preprocessor"] = None
            ml_models["model"] = None
        else:
            ml_models["preprocessor"] = joblib.load(preprocessor_path)
            ml_models["model"] = joblib.load(model_path)
            logger.info("ML artifacts loaded successfully.")
    yield
    # Cleanup
    logger.info("Cleaning up ML artifacts on shutdown...")
    ml_models.clear()


app = FastAPI(
    title="Bank Marketing Subscription Prediction API",
    description="MLOps API for predicting if a customer will subscribe to a term deposit.",
    version="1.0.0",
    lifespan=lifespan
)

class CustomerData(BaseModel):
    age: int = Field(..., description="Age of the customer")
    job: str = Field(..., description="Type of job")
    marital: str = Field(..., description="Marital status")
    education: str = Field(..., description="Education level")
    default: str = Field(..., description="Has credit in default?")
    housing: str = Field(..., description="Has housing loan?")
    loan: str = Field(..., description="Has personal loan?")
    contact: str = Field(..., description="Contact communication type")
    month: str = Field(..., description="Last contact month of year")
    day_of_week: str = Field(..., description="Last contact day of the week")
    duration: int = Field(..., description="Last contact duration, in seconds")
    campaign: int = Field(..., description="Number of contacts performed during this campaign")
    pdays: int = Field(..., description="Number of days that passed after the customer was last contacted")
    previous: int = Field(..., description="Number of contacts performed before this campaign")
    poutcome: str = Field(..., description="Outcome of the previous marketing campaign")
    emp_var_rate: float = Field(..., alias="emp.var.rate", description="Employment variation rate - quarterly indicator")
    cons_price_idx: float = Field(..., alias="cons.price.idx", description="Consumer price index - monthly indicator")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx", description="Consumer confidence index - monthly indicator")
    euribor3m: float = Field(..., alias="euribor3m", description="Euribor 3 month rate - daily indicator")
    nr_employed: float = Field(..., alias="nr.employed", description="Number of employees - quarterly indicator")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "age": 30,
                "job": "blue-collar",
                "marital": "married",
                "education": "basic.9y",
                "default": "no",
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "month": "may",
                "day_of_week": "mon",
                "duration": 200,
                "campaign": 2,
                "pdays": 999,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp.var.rate": -1.8,
                "cons.price.idx": 92.893,
                "cons.conf.idx": -46.2,
                "euribor3m": 1.299,
                "nr.employed": 5099.1
            }
        }
    }

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Bank Marketing Subscription Prediction API!",
        "status": "online",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    # Verify artifacts are loaded
    if ml_models.get("preprocessor") is None or ml_models.get("model") is None:
        return {
            "status": "unhealthy",
            "reason": "ML artifacts not loaded correctly."
        }
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict(customer: CustomerData):
    logger.info("Received a prediction request.")
    
    # Check if models are loaded
    preprocessor = ml_models.get("preprocessor")
    model = ml_models.get("model")
    
    if preprocessor is None or model is None:
        logger.error("Prediction failed: Model or Preprocessor not loaded.")
        raise HTTPException(
            status_code=503,
            detail="Model is not ready. Preprocessor or model is not loaded."
        )
        
    try:
        # Convert Pydantic model to DataFrame using alias names (dots)
        input_data = customer.model_dump(by_alias=True)
        input_df = pd.DataFrame([input_data])
        
        # Apply preprocessing
        input_processed = preprocessor.transform(input_df)
        
        # Preprocessor transforms output to dense array, but it might not have column names.
        # Scikit-learn models fit on pandas df containing feature names expect features names 
        # or matching shape. Since train.py outputs processed df columns, we need to pass 
        # feature names or array to the model. Best model trained on DataFrame expects names.
        # Let's check feature names in preprocessor.
        cat_encoder = preprocessor.named_transformers_["cat"]
        # Find numerical/categorical columns
        # To make it simple, we can reconstruct the DataFrame columns:
        numerical_cols = [c for c in input_data.keys() if not isinstance(input_data[c], str)]
        categorical_cols = [c for c in input_data.keys() if isinstance(input_data[c], str)]
        
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        feature_names = numerical_cols + cat_feature_names
        
        input_processed_df = pd.DataFrame(input_processed, columns=feature_names)
        
        # Predict
        prediction_val = int(model.predict(input_processed_df)[0])
        prediction_proba = float(model.predict_proba(input_processed_df)[0][1])
        
        prediction_str = "yes" if prediction_val == 1 else "no"
        
        response = {
            "prediction": prediction_str,
            "probability": round(prediction_proba, 4)
        }
        
        logger.info(f"Prediction result: {response}")
        return response
        
    except Exception as e:
        logger.exception("Error occurred during prediction.")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during model prediction: {str(e)}"
        )
