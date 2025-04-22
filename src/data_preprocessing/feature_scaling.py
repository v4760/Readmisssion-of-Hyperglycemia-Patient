import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os
import joblib
from logger import logging
from exceptions import CustomException
import sys


PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
models_dir = os.path.join(PROJECT_DIR, "airflow", "final_model")

def scaling(df, model_output_dir="models/best_final_model"):
    try:
        logging.info("Starting feature scaling process...")

        if 'readmitted' not in df.columns:
            raise CustomException("Target variable 'readmitted' not found in DataFrame.", sys)

        # Target extraction
        df_target = df['readmitted'].map({'Yes': 1, 'No': 0})
        df = df.drop(columns=['readmitted'])

        logging.info(f"[DEBUG] Columns BEFORE SCALING: {list(df.columns)}")

        # Separate numeric and categorical
        df_num = df.select_dtypes(include=['number'])
        df_cat = df.select_dtypes(include=['object'])

        logging.info(f"[DEBUG] Numeric columns for scaling: {list(df_num.columns)}")
        logging.info(f"[DEBUG] Categorical columns: {list(df_cat.columns)}")

        # Apply scaling
        scaler = RobustScaler()
        df_num_scaled = pd.DataFrame(
            scaler.fit_transform(df_num),
            columns=df_num.columns,
            index=df.index
        )

        # One-hot encode categorical if present
        if not df_cat.empty:
            df_cat_dummy = pd.get_dummies(df_cat, drop_first=True).astype(int)
            df_cat_dummy.index = df.index
            df_scaled = pd.concat([df_num_scaled, df_cat_dummy], axis=1)
        else:
            logging.warning("No categorical features found. Skipping one-hot encoding.")
            df_scaled = df_num_scaled

        logging.info(f"[DEBUG] Columns AFTER encoding and scaling: {list(df_scaled.columns)}")
        logging.info(f"[DEBUG] Total number of columns: {df_scaled.shape[1]}")

        # Add back target
        df_scaled["readmitted"] = df_target

        # Ensure output dir exists
        os.makedirs(models_dir, exist_ok=True)

        # Save scaler
        scaler_path = os.path.join(models_dir, "scalar_weight.pkl")
        joblib.dump(scaler, scaler_path)
        logging.info(f"Scaler saved at: {scaler_path}")

        logging.info("Feature scaling completed successfully.")
        return df_scaled

    except Exception as e:
        logging.error("An error occurred during feature scaling.")
        raise CustomException(e, sys)
