import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import sys
from xgboost import XGBClassifier
from logger import logging
from google.cloud import storage
from sqlalchemy import create_engine
from io import BytesIO
from exceptions import CustomException
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio
from dotenv import load_dotenv
import os



def bias_Evaluation():
    try:
 
        KEY_PATH = "/opt/airflow/config/key.json" 
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH

        load_dotenv()
        DB_NAME = os.getenv('DB_NAME')
        DB_USER = os.getenv('DB_USER')
        DB_PASSWORD = os.getenv('DB_PASS')
        DB_HOST = os.getenv('DB_HOST')
        DB_PORT = os.getenv('DB_PORT')

        engine = create_engine(
            f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
        )

        #import model.pkl from GCP

        def load_model_from_gcs(bucket_name, blob_path):
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                model_bytes = blob.download_as_bytes()
                model = pickle.loads(model_bytes)

                return model

        bucket_name = 'readmission_prediction'
        blob_path = 'models/best_xgboost_model/model.pkl'
        model=load_model_from_gcs(bucket_name,blob_path)

        ##Connecting to PostGres DB to fetch Patients Table

        df = pd.read_sql(sql='SELECT * FROM public.patients_data',con=engine)
        logging.info(f"Data loaded successfully, number of records: {len(df)}")

        df=df.drop(columns=['predict','f_name','l_name','dob'])
        df=df.dropna()

        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        logging.info(f"Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")

        train_df['race'] = np.where(train_df['race_caucasian'] == 1, 1, 
                            np.where(train_df['race_other'] == 1, 0, np.nan))

        test_df['race'] = np.where(test_df['race_caucasian'] == 1, 1, 
                                np.where(test_df['race_other'] == 1, 0, np.nan))

        train_df = train_df.dropna(subset=['race']) ## dropping africanamerican race
        test_df = test_df.dropna(subset=['race']) ## dropping africanamerican race

        target_labels_train = train_df['readmitted']
        sensitive_features_train = train_df[['gender_male','race']]

        target_labels_test = test_df['readmitted']
        sensitive_features_test = test_df[['gender_male','race']]

        # Split data into features (X) and labels (y)
        X_train = train_df.drop(columns=['readmitted'])
        X_test = test_df.drop(columns=['readmitted'])
        y_train = target_labels_train
        y_test = target_labels_test
        A_train = sensitive_features_train
        A_test = sensitive_features_test
        
        # added to handle model and test train df features mismatch
        expected_features = model.get_booster().feature_names   
        X_test = test_df[expected_features] 
        X_train = train_df[expected_features]

        y_pred = model.predict(X_test)                                    
        #model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        def disparate_impact_ratio(y_true, y_pred, sensitive_feature):
            try:
                group_1 = (sensitive_feature == 1)
                group_0 = (sensitive_feature == 0)

                rate_1 = np.mean(y_pred[group_1]) if np.sum(group_1) > 0 else np.nan
                rate_0 = np.mean(y_pred[group_0]) if np.sum(group_0) > 0 else np.nan

                if np.isnan(rate_1) or np.isnan(rate_0) or rate_0 == 0:
                    return np.nan 

            except Exception as e:
                logging.info('__.__Error occoured__.__')
                raise CustomException(e,sys)
            return rate_1 / rate_0

        dir_gender = disparate_impact_ratio(y_test, y_pred, test_df['gender_male'])
        dir_race = disparate_impact_ratio(y_test, y_pred, test_df['race'])
        logging.info(f'Value of disparate_impact_gender: {round(dir_gender, 2)}') 
        logging.info(f'Value of disparate_impact_race: {round(dir_race, 2)}')  

        #0.89 - Gender Slight Imbalance 
        #1..91 - High Imbalance however our datasets have higher rates of Caucasian Race
       
        m_dpr = demographic_parity_ratio(y_test, y_pred, sensitive_features=A_test)
        m_eqo = equalized_odds_ratio(y_test, y_pred, sensitive_features=A_test)
        logging.info(f'Value of demographic parity ratio: {round(m_dpr, 2)}')  # e.g., 0.49
        logging.info(f'Value of equal odds ratio: {round(m_eqo, 2)}')  # e.g., 0.39
        #Because of Higher distribution of Caucasian Race we got 0.49 and .39
        

        #Retraining
        threshold_optimizer = ThresholdOptimizer(
        estimator=model,  # Base model
        constraints="equalized_odds",  # You can also try "equalized_odds"
        prefit=True  # Since the model is already trained
        )

        # Fit ThresholdOptimizer on training data
        threshold_optimizer.fit(X_train, y_train, sensitive_features=A_train)

        # Make fair predictions
        y_pred_fair = threshold_optimizer.predict(X_test, sensitive_features=A_test)
        m_dpr_fair = demographic_parity_ratio(y_test, y_pred_fair, sensitive_features=A_test)
        m_eqo_fair = equalized_odds_ratio(y_test, y_pred_fair, sensitive_features=A_test)
        logging.info(f'Value of demographic parity ratio: {round(m_dpr_fair, 2)}')
        logging.info(f'Value of equal odds ratio: {round(m_eqo_fair, 2)}')

        def evaluate_slices(X_test, y_test, y_pred, feature_name):
            try:
                results = []
                for group in X_test[feature_name].unique():
                    mask = X_test[feature_name] == group 
                   
                    if mask.sum() == 0:
                        continue
                    acc = accuracy_score(y_test[mask], y_pred[mask])
                    prec, rec, f1, _ = precision_recall_fscore_support(y_test[mask], y_pred[mask], average='weighted')

                    results.append({
                        'Group': group,
                        'Accuracy': acc,
                        'Precision': prec,
                        'Recall': rec,
                        'F1-score': f1
                    })
            except Exception as e:
                logging.info('__.__Error occoured__.__')
                raise CustomException(e,sys)
            return pd.DataFrame(results)
        
        gender_results = evaluate_slices(test_df, y_test, y_pred, 'gender_male')
        race_results = evaluate_slices(test_df, y_test, y_pred, 'race')
        

        logging.info(f'gender_results: {round(gender_results, 2)}')  # e.g., 1.0
        logging.info(f'race_results: {round(race_results, 2)}')  # e.g., 1.0

        def plot_bias_results(results, feature_name):
            try:
                save_folder = os.path.abspath("Bias_Plots")
                os.makedirs(save_folder, exist_ok=True)
                plt.figure(figsize=(10, 6))
                results.set_index("Group")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(kind="bar")
                plt.title(f"Model Performance across {feature_name}")
                plt.ylabel("Score")
                plt.xticks(rotation=45)
                plt.legend(loc="lower right")
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.tight_layout()

                save_path_gender = os.path.join(save_folder, "Bias_Gender.png")
                save_path_race = os.path.join(save_folder, "Bias_Race.png")
                # Save gender plot
                plt.savefig(save_path_gender)

                # Save race plot
                plt.savefig(save_path_race)
                #plt.show()

                

            except Exception as e:
                logging.info('__.__Error occoured__.__')
                raise CustomException(e,sys)
            return 0
            
        plot_bias_results(gender_results, "Gender")
        plot_bias_results(race_results, "Race")

    except Exception as e:
        logging.info('__.__Error occoured__.__')
        raise CustomException(e,sys)
    return 0

