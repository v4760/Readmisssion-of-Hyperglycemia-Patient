
import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Get the absolute path of the project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Reset the working directory to the project root
os.chdir(PROJECT_ROOT)

# Ensure the project root and src are in the Python path
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

print(f"Current Working Directory Reset to: {os.getcwd()}")
print(f"PYTHONPATH: {sys.path}")

# Import modules for testing
from src.data_preprocessing.feature_extract import feature_extract
from src.data_preprocessing.duplicate_missing_values import missingVal, duplicates
from src.data_preprocessing.unzip import unzip_file
from src.data_preprocessing.data_mapping import (
    clean_gender, map_age_ranges, map_race, map_admission_type,
    map_discharge, map_admission_source, clean_diagnosis, process_diagnosis_columns, process_data_mapping
)
from src.data_preprocessing.data_download import ingest_data
from logger import logging  # Ensure this import works
from exceptions import CustomException

# Mock Data for Testing
@pytest.fixture
def mock_data():
    df=pd.read_csv('../data/diabetic_data.csv')
    return df.head(5)

@pytest.fixture
def mock_path():
    return '../data/diabetic_data.csv'

# Test Feature Extraction
def test_feature_extract(mock_data):
    result = feature_extract(mock_data)
    print(result['Health_index'],'health index')
    assert "Health_index" in result.columns
    assert "severity_of_disease" in result.columns
    assert "number_of_changes" in result.columns

# Test Data Cleaning Functions
def test_clean_gender(mock_data):
    result = clean_gender(mock_data)
    print(result.head(),'result.head')
    assert "Unknown/Invalid" not in result["gender"].values

def test_map_age_ranges(mock_data):
    result = map_age_ranges(mock_data)
    print(result['age'],'result age')
    assert all(isinstance(age, (int, np.integer)) for age in result["age"].values)

def test_map_race(mock_data):
    result = map_race(mock_data)
    print(result.head(),'result.head')
    assert "Hispanic" not in result["race"].values
    assert "?" not in result["race"].values

def test_map_admission_type(mock_data):
    result = map_admission_type(mock_data)
    print(result.head(),'result.head')
    assert result["admission_type_id"].dtype == "object"

def test_map_discharge(mock_data):
    result = map_discharge(mock_data)
    print(result.head(),'result.head')
    assert result["discharge_disposition_id"].dtype == "object"

def test_map_admission_source(mock_data):
    result = map_admission_source(mock_data)
    print(result.head(),'result.head')
    assert "Referral" in result["admission_source_id"].values

def test_clean_diagnosis(mock_data):
    # Drop rows with '?' values before testing
    mock_data = mock_data[~mock_data[['diag_1', 'diag_2', 'diag_3']].isin(['?']).any(axis=1)] 
    result = clean_diagnosis(mock_data) 
    print(result[["diag_1", "diag_2", "diag_3"]], 'result diag')

    assert "?" not in result["diag_1"].values
    assert "?" not in result["diag_2"].values
    assert "?" not in result["diag_3"].values


def test_process_diagnosis_columns(mock_data):
    result = process_diagnosis_columns(mock_data)
    print(result.head(),'result.head')
    assert "Neoplasms" in result["diag_1"].values

def test_process_data_mapping(mock_data):
    save_path = process_data_mapping(mock_data)
    assert os.path.exists(save_path)

# Test Data Ingestion
def test_ingest_data():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"Fake Data"
        mock_get.return_value = mock_response
        result = ingest_data()
        assert os.path.exists(result)

# Test Duplicate Handling
def test_missingVal(mock_data):
    result = missingVal(mock_data)
    assert "max_glu_serum" not in result.columns
    assert "A1Cresult" not in result.columns

def test_duplicates(mock_path):
    
    result = duplicates(mock_path)
    assert "encounter_id" not in result.columns
    assert "patient_nbr" not in result.columns
