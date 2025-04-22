"""
Function to download and ingest the data file
"""
import os
import requests
from logger import logging
from exceptions import CustomException
import sys

DEFAULT_FILE_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

def ingest_data(file_url=DEFAULT_FILE_URL):
    """
    Function to download file from URL
    Args:
        file_url: URL of the file, A default is used if not specified
    Returns:
        zipfile_path: The zipped file path to the data
    """
    try:
        logging.info('Begin data ingestion')
        # Send an HTTP GET request to the URL
        response = requests.get(file_url, timeout=30)

        logging.info("Downloading Data")

        # Set the root directory variable using a relative path
        root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
        logging.info(f"Root directory: {root_dir}")

        # Path to store the zipfile
        zipfile_path=os.path.join(root_dir, 'data','data.zip')
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save file to data
            with open(zipfile_path, "wb") as file:
                file.write(response.content)
            logging.info(f"File downloaded successfully. Zip file available under {zipfile_path}")
        else:
            logging.info(f"Failed to download the file. Status code: {response.status_code}")

        return zipfile_path
    except Exception as e:
        logging.info('__.__Error occoured__.__')
        raise CustomException(e,sys)

if __name__ == "__main__":
    ZIPFILE_PATH = ingest_data("https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip")