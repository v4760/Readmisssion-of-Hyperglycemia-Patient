import os
import io
from logger import logging
from exceptions import CustomException
from google.cloud import storage
import json
from datetime import datetime
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope
import pickle 
import numpy as np
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv



PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "airflow", "final_model")


SPACE = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 50)),  
    'learning_rate': hp.loguniform('learning_rate', -5, 0),  
    'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),  
    'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 10, 1)),  
    'gamma': hp.uniform('gamma', 0, 5),  
    'booster': hp.choice('booster', ['gbtree', 'gblinear', 'dart']),  
}


def log_metrics_to_file(metrics, model_name):
    log_dir = os.path.join(PROJECT_DIR, "airflow","logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "ml_metrics.log")
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        **metrics
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def setup_mlflow():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("readmission_prediction")
    
KEY_PATH = "/opt/airflow/config/key.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH
bucket_name = "readmission_prediction"

def load_data():
    try:
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

        # === 2. Load data from patients_data ===
        df = pd.read_sql('SELECT * FROM patients_data WHERE readmitted IS NOT NULL', con=engine)
        logging.info(f"Loaded {len(df)} records from patients_data table.")

        # === 3. Drop unwanted columns ===
        drop_cols = ['f_name', 'l_name', 'dob', 'predict']
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        logging.info(f"Dropped columns: {drop_cols}")

        # === 4. Split into features and target ===
        if 'readmitted' not in df.columns:
            raise ValueError("Target column 'readmitted' not found in data.")
        
        X = df.drop(columns=['readmitted'])
        y = df['readmitted']

        # === 5. Train-test split ===
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logging.info(f"Data split into train and test sets. Train size: {len(X_train)}, Test size: {len(X_test)}")

        frac = 0.9
        X_train = X_train.sample(frac=frac, random_state=42)
        y_train = y_train.loc[X_train.index]

        X_test = X_test.sample(frac=frac, random_state=42)
        y_test = y_test.loc[X_test.index] 

        return X_train, y_train, X_test, y_test

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def objective(params, X, y):
    """Objective function for hyperopt to minimize"""
    clf = xgb.XGBClassifier(**params, n_jobs=4)
    score = cross_val_score(clf, X, y, cv=3, scoring='accuracy', n_jobs=4).mean()
    return {'loss': -score, 'status': STATUS_OK}

def save_model_and_results(model, results, run_name, X_test, y_test):
    """Save the model and results as JSON in the models folder and return the JSON path."""
    models_dir = DATA_DIR
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, f"xgboost_{run_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save X_test and y_test to CSV
    X_test_path = os.path.join(models_dir,f"xgboost_{run_name}_X_test.csv")
    y_test_path = os.path.join(models_dir,f"xgboost_{run_name}_y_test.csv")
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    # Convert int64 to regular int for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Save results
    results['timestamp'] = run_name
    results['model_path'] = str(model_path)
    results['X_test_path'] = str(X_test_path)
    results['y_test_path'] = str(y_test_path)
    serializable_results = json.loads(json.dumps(results, default=convert_to_serializable))
    results_path = os.path.join(models_dir, f"results_{run_name}.json")
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    logging.info(f"Model saved to {model_path}")
    logging.info(f"X_test saved to {X_test_path}")
    logging.info(f"y_test saved to {y_test_path}")
    logging.info(f"Results saved to {results_path}")
    
    return results_path  # Return the path of the JSON file

def evaluate_model_performance(y_test, y_pred, threshold=0.7):
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
    
    logging.info(f"Model performance metrics: {json.dumps(metrics)}")
    
    return all(metric >= threshold for metric in metrics.values()), metrics

def train_and_log_model(X_train, y_train, X_test, y_test):
    """Train multiple XGBoost models and log results with MLflow"""
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.set_tag("run_name", run_name)

        trials = Trials()

        # Perform hyperparameter optimization using Hyperopt
        best_params = fmin(fn=lambda params: objective(params, X_train, y_train),
                           space=SPACE,
                           algo=tpe.suggest,
                           max_evals=10,
                           trials=trials)

        best_model = None
        best_performance = None
        best_metrics = None

        # Train and log each model from the trials
        for i in range(len(trials.trials)):
            trial = trials.trials[i]
            params = {
                'n_estimators': int(trial['misc']['vals']['n_estimators'][0]),
                'learning_rate': float(trial['misc']['vals']['learning_rate'][0]),
                'max_depth': int(trial['misc']['vals']['max_depth'][0]),
                'min_child_weight': int(trial['misc']['vals']['min_child_weight'][0]),
                'gamma': float(trial['misc']['vals']['gamma'][0]),
                'booster': ['gbtree', 'gblinear', 'dart'][trial['misc']['vals']['booster'][0]],
            }

            # Start a nested run for each model training
            with mlflow.start_run(run_name=f"{run_name}_model_{i}", nested=True) as child_run:
                # Train the XGBoost model with current parameters
                model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
                model.fit(X_train, y_train)

                # Evaluate on test set
                y_pred = model.predict(X_test)
                performance_ok, metrics = evaluate_model_performance(y_test, y_pred)

                # Log parameters and metrics for this model in MLflow directly
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                # Log the model to MLflow
                signature = infer_signature(X_test, y_pred)
                mlflow.xgboost.log_model(model, f"model_{i}", signature=signature)

                log_metrics_to_file(metrics, f"model_{i}")
                logging.info(f"Logged model {i} with parameters: {params}")

                # Update best model if this one performs better
                if best_model is None or metrics['accuracy'] > best_metrics['accuracy']:
                    best_model = model
                    best_performance = performance_ok
                    best_metrics = metrics

        # Save the best trained model and results locally after logging metrics
        log_metrics_to_file(best_metrics, "best_model")
        results = {
            **best_metrics,
            "best_params": best_params,
            "run_id": parent_run.info.run_id,
            "timestamp": run_name
        }
        save_model_and_results(best_model, results, run_name, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        return best_performance, best_metrics


def run_model_development(max_attempts=3):
    setup_mlflow()
    X_train, y_train, X_test, y_test = load_data()
    attempt = 0
    while attempt < max_attempts:
        logging.info(f"Starting model development attempt {attempt + 1}")
        performance_ok, metrics = train_and_log_model(X_train, y_train, X_test, y_test)
        if performance_ok:
            logging.info("Model performance meets the threshold. Process complete.")
            log_metrics_to_file(metrics, f"final_model_attempt_{attempt + 1}")
            return metrics
        else:
            logging.warning("Model performance below threshold. Rerunning the process.")
            attempt += 1
    
    logging.error(f"Failed to achieve desired performance after {max_attempts} attempts.")
    log_metrics_to_file(metrics, f"final_model_attempt_{attempt}")
    return metrics

if __name__ == "__main__":
    # Example usage 
    PAR_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), ".."))
    TRAIN_PATH = os.path.join(PAR_DIRECTORY, "data", "processed", "train_data.csv")
    TEST_PATH = os.path.join(PAR_DIRECTORY, "data", "processed", "test_data.csv")
    final_metrics = run_model_development()
    print("Final model metrics:", final_metrics)