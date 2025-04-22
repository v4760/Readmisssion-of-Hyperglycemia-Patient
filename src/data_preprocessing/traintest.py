import os
import pandas as pd
import pickle
from logger import logging
import sys
import time
from google.cloud import storage
from sklearn.model_selection import train_test_split

KEY_PATH = "/opt/airflow/config/key.json" #replace own json path
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH #env create

ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
LOCAL_PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

if not os.path.exists(LOCAL_PROCESSED_DIR):
    os.makedirs(LOCAL_PROCESSED_DIR, exist_ok=True)

 
def upload_to_gcs(data, bucket_name, destination_blob_name, as_pickle=False):
    """Uploads CSV or Pickle file to Google Cloud Storage with retry on failure."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        if not bucket.exists():
            raise ValueError(f"Bucket {bucket_name} does not exist.")

        blob = bucket.blob(destination_blob_name)

        if blob.exists():
            logging.info(f"File {destination_blob_name} already exists. Overwriting...")
        else:
            logging.info(f"File {destination_blob_name} does not exist. Creating new file...")

        if as_pickle:
            blob.upload_from_string(pickle.dumps(data), content_type="application/octet-stream")
            logging.info(f"Uploaded pickle data to gs://{bucket_name}/{destination_blob_name}")
        else:
            temp_path = f"/tmp/{destination_blob_name}"
            data.to_csv(temp_path, index=False)
            blob.upload_from_filename(temp_path, content_type="text/csv")
            logging.info(f"Uploaded CSV data to gs://{bucket_name}/{destination_blob_name}")

    except Exception as e:
        logging.error(f"Failed to upload to GCS: {e}")
        time.sleep(2)
        try:
            blob.upload_from_filename(temp_path, content_type="text/csv")
            logging.info(f"Retry successful: Uploaded {destination_blob_name}")
        except Exception as e:
            logging.error(f"Final attempt to upload {destination_blob_name} failed: {e}")
            raise

def train_test_upload(df):
    """Splits data into train-test sets and uploads them to GCS."""
    bucket_name = "readmission_prediction"#--> our bucket

    try:
        #  Train-Test Split
        X = df.drop('readmitted', axis=1)
        y = df['readmitted']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        local_train_path = os.path.join(LOCAL_PROCESSED_DIR, "train_data.csv")
        local_test_path = os.path.join(LOCAL_PROCESSED_DIR, "test_data.csv")
 
        train_df.to_csv(local_train_path, index=False)
        test_df.to_csv(local_test_path, index=False)
 
        logging.info(f" Train data saved locally at: {local_train_path}")
        logging.info(f" Test data saved locally at: {local_test_path}")

        train_filename = "train_data.csv"
        test_filename = "test_data.csv"

        try:
            upload_to_gcs(train_df, bucket_name, train_filename)
            logging.info(" Train data uploaded successfully.")
        except Exception as e:
            logging.error(f" Error uploading train data: {e}")
            time.sleep(2)
            upload_to_gcs(train_df, bucket_name, train_filename)

        try:
            upload_to_gcs(test_df, bucket_name, test_filename)
            logging.info(" Test data uploaded successfully.")
        except Exception as e:
            logging.error(f" Error uploading test data: {e}")
            time.sleep(2)
            upload_to_gcs(test_df, bucket_name, test_filename)

        logging.info("Train-test split completed. Data uploaded to GCS.")

    except Exception as e:
        logging.error(" An error occurred during train-test split or upload.")
        raise
    