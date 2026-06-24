import os
import zipfile
import logging
import requests
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Loads configurations from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_file(url, destination):
    """Downloads a file from a URL to a destination path."""
    logger.info(f"Downloading dataset from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logger.info(f"Download complete: {destination}")

def extract_nested_zip(zip_path, raw_dir):
    """Extracts the bank-additional-full.csv from nested zips."""
    logger.info(f"Extracting outer zip file: {zip_path}")
    temp_dir = Path(raw_dir) / "temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as outer_zip:
        outer_zip.extractall(temp_dir)
        
    # bank-additional.zip should be inside temp_dir
    inner_zip_path = temp_dir / "bank-additional.zip"
    if not inner_zip_path.exists():
        # Let's search recursively for bank-additional.zip
        for p in temp_dir.rglob("bank-additional.zip"):
            inner_zip_path = p
            break
            
    if not inner_zip_path.exists():
        raise FileNotFoundError("Could not find bank-additional.zip inside the extracted files.")
        
    logger.info(f"Extracting inner zip file: {inner_zip_path}")
    with zipfile.ZipFile(inner_zip_path, 'r') as inner_zip:
        # List files in inner zip
        file_list = inner_zip.namelist()
        logger.info(f"Files in bank-additional.zip: {file_list}")
        
        # Find bank-additional-full.csv
        csv_file = None
        for f in file_list:
            if f.endswith("bank-additional-full.csv"):
                csv_file = f
                break
                
        if not csv_file:
            raise FileNotFoundError("Could not find bank-additional-full.csv in bank-additional.zip")
            
        inner_zip.extract(csv_file, raw_dir)
        
        # The file is extracted as raw_dir/bank-additional/bank-additional-full.csv or similar.
        # Move it directly to raw_dir/bank-additional-full.csv for simplicity.
        extracted_csv_path = Path(raw_dir) / csv_file
        target_csv_path = Path(raw_dir) / "bank-additional-full.csv"
        
        if extracted_csv_path.exists():
            if target_csv_path.exists():
                os.remove(target_csv_path)
            os.rename(extracted_csv_path, target_csv_path)
            logger.info(f"Successfully saved csv to: {target_csv_path}")
        else:
            raise FileNotFoundError(f"Extracted file not found at expected path: {extracted_csv_path}")
            
    # Clean up temporary directories and zip files
    logger.info("Cleaning up temporary download artifacts...")
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    # Remove nested folders inside data/raw if any
    shutil.rmtree(Path(raw_dir) / "bank-additional", ignore_errors=True)

def ingest_data(config_path="config/config.yaml"):
    """Main function to run the ingestion pipeline."""
    config = load_config(config_path)
    url = config["data"]["raw_url"]
    raw_dir = config["data"]["raw_dir"]
    zip_destination = os.path.join(raw_dir, "bank_marketing.zip")
    
    # Download
    download_file(url, zip_destination)
    
    # Extract nested CSV
    extract_nested_zip(zip_destination, raw_dir)

if __name__ == "__main__":
    ingest_data()
