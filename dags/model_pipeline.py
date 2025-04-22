from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.email import send_email
from datetime import datetime, timedelta
import os
import logging
import sys

# Airflow root directory (where the project is mounted)
AIRFLOW_ROOT = "/opt/airflow"

# Add src to Python path
SRC_PATH = os.path.join(AIRFLOW_ROOT, "src")
sys.path.append(SRC_PATH)

from model_development.model_development_evalution import run_model_development
from model_development.best_model_selection import compare_and_select_best
from model_development.bias_model_evaluation import bias_Evaluation
from model_development.gcs_model_push import push_to_gcp

default_args = {
    'owner': 'MLopsProjectGroup14',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

# Define function to notify failure or sucess via an email
def notify_success(context):
    success_email = EmailOperator(
        task_id='success_email',
        to='mlopsneu2025@gmail.com',
        subject='Success Notification from Airflow',
        html_content='<p>The task succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

def notify_failure(context):
    failure_email = EmailOperator(
        task_id='failure_email',
        to='mlopsneu2025@gmail.com',
        subject='Failure Notification from Airflow',
        html_content='<p>The task failed.</p>',
        dag=context['dag']
    )
    failure_email.execute(context=context)


dag_2 = DAG(
    'ModelDevelopmentPipeline',
    default_args=default_args,
    description='DAG for running model development with MLflow',
    schedule_interval=None,
    catchup=False,
    is_paused_upon_creation=False,
)

def run_model_development_task(**kwargs):
    
    final_metrics = run_model_development(max_attempts=3)
    
    logging.info(f"Final model metrics: {final_metrics}")
    kwargs['ti'].xcom_push(key='final_metrics', value=final_metrics)
    
    if all(metric >= 0.5 for metric in final_metrics.values()):
        logging.info("Model development successful: All metrics are above 0.5")
    else:
        logging.warning("Model development completed, but not all metrics are above 0.7")


email_notification_start_task = EmailOperator(
    task_id='dag_started_email',
    to='mlopsneu2025@gmail.com',
    subject='Model Pipeline Dag Started',
    html_content='<p> Model Pipeline Dag Started</p>',
    dag=dag_2,
)

model_development_task = PythonOperator(
    task_id='run_model_development',
    python_callable=run_model_development_task,
    provide_context=True,
    dag=dag_2,
)

def find_best_model(**kwargs):
    best_model_info = compare_and_select_best()
    if best_model_info:
        best_model_path, X_test_path, y_test_path = best_model_info
        logging.info(f"Best model path: {best_model_path}")
        logging.info(f"X_test path: {X_test_path}")
        logging.info(f"y_test path: {y_test_path}")
        kwargs['ti'].xcom_push(key='best_model_path', value=best_model_path)
        kwargs['ti'].xcom_push(key='X_test_path', value=X_test_path)
        kwargs['ti'].xcom_push(key='y_test_path', value=y_test_path)
    else:
        logging.error("Failed to find the best model paths.")

compare_best_model_task = PythonOperator(
    task_id='compare_best_models',
    python_callable=find_best_model,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag_2,
)

bias_eval_task = PythonOperator(
    task_id='bias_evaluation',
    python_callable=bias_Evaluation,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag_2,
)

upload_to_gcs_task = PythonOperator(
    task_id='upload_to_gcs',
    python_callable=push_to_gcp,
    provide_context=True,
    on_failure_callback=notify_failure,
    dag=dag_2,
)

email_notification_end_task = EmailOperator(
    task_id='dag_completed_email',
    to='mlopsneu2025@gmail.com',
    subject='Model Pipeline Completed Successfully',
    html_content='<p> Model Pipeline Dag Completed</p>',
    dag=dag_2,
)

# model_development_task
email_notification_start_task >> model_development_task >> compare_best_model_task >> bias_eval_task >> upload_to_gcs_task >> email_notification_end_task