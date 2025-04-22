import os
import mlflow
import logging
from mlflow.tracking import MlflowClient
from google.cloud import storage
from logger import logging

# Paths
PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
LOG_FILE_PATH = os.path.join("/opt/airflow/logs", "gcs_model_push.log")

#key path
KEY_PATH = "/opt/airflow/config/key.json" # Replace with your key path
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH

#set mlflow url
mlflow.set_tracking_uri("http://mlflow:5000")  # Replace with your MLflow server URI

# Create necessary directories
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)


def get_latest_model_version(model_name, stage="Staging"):    #Note: production level is also there
    """Get the latest version of the model in the specified stage."""
    client = MlflowClient()
    latest_versions = client.get_latest_versions(model_name, stages=[stage])
    if latest_versions:
        latest_version = latest_versions[0].version
        logging.info(f"Latest version of model '{model_name}' in stage '{stage}' is {latest_version}")
        return latest_version
    else:
        logging.error(f"No model versions found in stage '{stage}' for model '{model_name}'")
        raise ValueError(f"No model versions found in stage '{stage}' for model '{model_name}'")

def download_model_from_mlflow(model_name, stage="Staging", local_path=""):#add local path
    """Download the latest model version from MLflow by stage."""
    try:
        # Create local directory if it doesn't exist
        os.makedirs(local_path, exist_ok=True)
        
        # Get the latest version in the specified stage
        latest_version = get_latest_model_version(model_name, stage)
        
        # Download model as an artifact
        model_uri = f"models:/{model_name}/{latest_version}"
        logging.info(f"Downloading model {model_name} version {latest_version} from MLflow to {local_path}")
        return mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=local_path)
    except Exception as e:
        logging.error(f"Failed to download model {model_name} from MLflow: {e}")
        raise

def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    """Upload a local directory to Google Cloud Storage."""
    bucket_name="readmission_prediction"
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Check if bucket exists
        if not bucket.exists():
            raise ValueError(f"Bucket {bucket_name} does not exist.")

        for root, _, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                blob_path = os.path.join(destination_blob_name, os.path.relpath(file_path, local_path))
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(file_path)
                logging.info(f"Uploaded {file_path} to gs://{bucket_name}/{blob_path}")
    except Exception as e:
        logging.error(f"Failed to upload model to GCS: {e}")
        raise
#
def push_to_gcp():
    logging.info("Starting the model download and upload process.")
    
    # Configuration variables
    model_name = "best_xgboost_model" #the best_model name is replaced
    stage = "Staging"  
    local_model_path = "best_final_model" #replace path
    bucket_name = "readmission_prediction"  #my buck name 
    destination_blob_name = "models/best_xgboost_model"# destiation buck name 

    # Step 1: Check if the model is registered
    try:
        download_model_from_mlflow(model_name, stage=stage, local_path=local_model_path)
    except ValueError as e:
        logging.error(f"Model '{model_name}' not found in the MLflow registry at stage '{stage}'. Please verify the model name and stage.")
        return  # Exit if the model is not found
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        return  # Exit if any other error occurs during download

    # Step 2: Upload model to GCS
    try:
        upload_to_gcs(local_model_path, bucket_name, destination_blob_name)
    except Exception as e:
        logging.error(f"Failed to upload model to GCS: {e}")
        return  # Exit if the upload fails

    logging.info("Model download and upload process completed successfully.")


if __name__ == "__main__":
    push_to_gcp()