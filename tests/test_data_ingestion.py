import os
import shutil
import zipfile
import io
import pytest
import yaml
from unittest.mock import patch, MagicMock
from src.data_ingestion import ingest_data, download_file, extract_nested_zip

@pytest.fixture
def temp_ingest_workspace():
    test_dir = "tests/test_ingest_workspace"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test config
    config_dict = {
        "data": {
            "raw_url": "http://dummy.url/bank.zip",
            "raw_dir": test_dir,
            "raw_file_path": f"{test_dir}/bank-additional-full.csv"
        }
    }
    
    config_path = f"{test_dir}/test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)
        
    yield test_dir, config_path
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)

def test_ingest_data_success(temp_ingest_workspace):
    test_dir, config_path = temp_ingest_workspace
    
    # 1. Create a mock inner zip containing bank-additional/bank-additional-full.csv
    inner_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_zip_buffer, "w") as inner_zip:
        inner_zip.writestr("bank-additional/bank-additional-full.csv", "col1;col2;y\n1;2;no\n")
    inner_zip_bytes = inner_zip_buffer.getvalue()
    
    # 2. Create mock outer zip containing bank-additional.zip
    outer_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(outer_zip_buffer, "w") as outer_zip:
        outer_zip.writestr("bank-additional.zip", inner_zip_bytes)
    outer_zip_bytes = outer_zip_buffer.getvalue()
    
    # 3. Mock requests.get
    mock_response = MagicMock()
    mock_response.iter_content = lambda chunk_size: [outer_zip_bytes]
    mock_response.raise_for_status = MagicMock()
    
    with patch("requests.get", return_value=mock_response) as mock_get:
        ingest_data(config_path=config_path)
        
        mock_get.assert_called_once_with("http://dummy.url/bank.zip", stream=True)
        
        # Verify that bank-additional-full.csv was extracted and sits in the raw directory
        target_csv_path = os.path.join(test_dir, "bank-additional-full.csv")
        assert os.path.exists(target_csv_path)
        
        # Verify CSV content
        with open(target_csv_path, "r") as f:
            content = f.read()
        assert "col1;col2;y" in content
        
        # Verify that zip file and temporary folder were cleaned up
        assert not os.path.exists(os.path.join(test_dir, "bank_marketing.zip"))
        assert not os.path.exists(os.path.join(test_dir, "temp_extract"))
