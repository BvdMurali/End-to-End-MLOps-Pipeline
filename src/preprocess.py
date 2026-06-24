import os
import logging
import joblib
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Loads configurations from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def preprocess_data(config_path="config/config.yaml"):
    """Main function to run the preprocessing pipeline."""
    config = load_config(config_path)
    
    raw_file_path = config["data"]["raw_file_path"]
    processed_dir = config["data"]["processed_dir"]
    target_col = config["data"]["target_col"]
    preprocessor_path = config["artifacts"]["preprocessor_path"]
    test_size = config["model"]["test_size"]
    random_state = config["model"]["random_state"]
    
    logger.info(f"Loading raw data from {raw_file_path}...")
    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_file_path}. Run ingestion first.")
        
    # UCI Bank Marketing dataset uses ";" as separator
    df = pd.read_csv(raw_file_path, sep=";")
    logger.info(f"Loaded dataset of shape {df.shape}")
    
    # Drop duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    duplicate_count = initial_shape[0] - df.shape[0]
    if duplicate_count > 0:
        logger.info(f"Dropped {duplicate_count} duplicate rows. New shape: {df.shape}")
        
    # Encode target variable
    logger.info("Encoding target column 'y'...")
    df[target_col] = df[target_col].map({"yes": 1, "no": 0})
    if df[target_col].isnull().any():
        raise ValueError("Target column contains values other than 'yes' and 'no'.")
        
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split train/test
    logger.info(f"Splitting data into train and test sets (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    logger.info(f"Identified {len(numerical_cols)} numerical features: {numerical_cols}")
    logger.info(f"Identified {len(categorical_cols)} categorical features: {categorical_cols}")
    
    # Build ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )
    
    # Fit and transform training data
    logger.info("Fitting and applying preprocessor to training data...")
    X_train_processed_arr = preprocessor.fit_transform(X_train)
    
    # Transform test data
    logger.info("Applying preprocessor to test data...")
    X_test_processed_arr = preprocessor.transform(X_test)
    
    # Reconstruct Column Names
    # Get feature names from encoders
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    feature_names = numerical_cols + cat_feature_names
    
    # Reconstruct DataFrames
    X_train_processed = pd.DataFrame(X_train_processed_arr, columns=feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed_arr, columns=feature_names, index=X_test.index)
    
    # Add target back
    train_processed_df = X_train_processed.copy()
    train_processed_df[target_col] = y_train
    
    test_processed_df = X_test_processed.copy()
    test_processed_df[target_col] = y_test
    
    # Ensure artifact and processed directories exist
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save processed datasets
    train_path = config["data"]["train_path"]
    test_path = config["data"]["test_path"]
    train_processed_df.to_csv(train_path, index=False)
    test_processed_df.to_csv(test_path, index=False)
    logger.info(f"Saved processed train data to {train_path}")
    logger.info(f"Saved processed test data to {test_path}")
    
    # Persist the preprocessor
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved preprocessor pipeline to {preprocessor_path}")

if __name__ == "__main__":
    preprocess_data()
