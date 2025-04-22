from google.cloud import storage
import os

def download_model_from_gcs(bucket_name, source_blob_name, destination_file_name):
    print(f"[INFO] Downloading model from GCS: {bucket_name}/{source_blob_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob.download_to_filename(destination_file_name)
    print("[INFO] Model downloaded successfully.")
