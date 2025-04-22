import pandas as pd
from sqlalchemy import create_engine, text
from logger import logging
import os
from dotenv import load_dotenv

def upload_patient_data(df):

    # === 2. Add missing nullable columns ===
    for col in ["f_name", "l_name", "dob", "predict"]:
        if col not in df.columns:
            df[col] = None

    # === 3. Normalize column names ===
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # === 4. Optional: Rename specific columns if needed ===
    column_map = {
        "diabetesmed_yes": "diabetesmed_yes",
        "readmitted": "readmitted",
        "dob": "dob",
        "f_name": "f_name",
        "l_name": "l_name",
        "predict": "predict",
    }
    df.rename(columns=column_map, inplace=True)


    # Load environment variables from the .env file
    load_dotenv()

    # Retrieve the environment variables
    DB_NAME = os.getenv('DB_NAME')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASS')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')

    engine = create_engine(
        f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    )

    # === 6. Delete existing NULL-marker rows before inserting ===
    with engine.begin() as conn:
        delete_query = text("""
            DELETE FROM patients_data
            WHERE f_name IS NULL
              AND l_name IS NULL
              AND dob IS NULL
              AND predict IS NULL;
        """)
        conn.execute(delete_query)
        logging.info("Deleted existing rows with NULL in f_name, l_name, dob, and predict.")

    # === 7. Upload data ===
    df.to_sql('patients_data', engine, if_exists='append', index=False)
    logging.info("Data successfully uploaded to patients_data table.")

