# import pandas as pd
# import numpy as np
import sys
import os
from logger import logging
from exceptions import CustomException

def feature_extraction(df):
    """
    Extracts three new features: Health Index, Severity of Disease, and Number of Changes.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing patient records.

    Returns:
    pd.DataFrame: DataFrame with new extracted features.
    """
    try:
        logging.info("Starting feature extraction process...")

        # 1st Feature: Health Index
        df['health_index'] = df['number_emergency'] + df['number_inpatient'] + df['number_outpatient']
        logging.info("Feature 'Health_index' added successfully.")

        # 2nd Feature: Severity of Disease
        df['severity_of_disease'] = (df['time_in_hospital'] + df['num_procedures'] + 
                                     df['num_medications'] + df['num_lab_procedures'] + df['number_diagnoses'])
        logging.info("Feature 'severity_of_disease' added successfully.")

        # 3rd Feature: Number of Changes
        if not {'metformin', 'metformin-pioglitazone'}.issubset(df.columns):
            logging.warning("Medication columns missing. 'number_of_changes' might not be calculated correctly.")

        medication_columns = df.loc[:, 'metformin':'metformin-pioglitazone']
        number_of_changes = []

        for i in range(len(df)):
            change_count = 0
            for col in medication_columns.columns: 
                if df.iloc[i][col] in [10, -10]:  # Checking for medication changes
                    change_count += 1
            number_of_changes.append(change_count)

        df['number_of_changes'] = number_of_changes
        logging.info("Feature 'number_of_changes' added successfully.")


        logging.info("Feature extraction completed successfully.")
        return df

    except Exception as e:
        logging.error("An error occurred during feature extraction.")
        raise CustomException(e, sys)
