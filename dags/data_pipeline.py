import sys
import os
from airflow import DAG
# Airflow root directory (where the project is mounted)
AIRFLOW_ROOT = "/opt/airflow"

# Add src to Python path
SRC_PATH = os.path.join(AIRFLOW_ROOT, "src")
sys.path.append(SRC_PATH)

# Now import from src.data_preprocessing
from data_preprocessing.data_download import ingest_data
from data_preprocessing.unzip import unzip_file
from data_preprocessing.duplicate_missing_values import duplicates, missingVal
from data_preprocessing.data_mapping import process_data_mapping
from data_preprocessing.encoder import target_encoding
from data_preprocessing.feature_extract import feature_extraction
from data_preprocessing.data_schema_statistics_generation import validate_data_schema
from data_preprocessing.feature_selection import feature_selection
from data_preprocessing.feature_scaling import scaling
from data_preprocessing.traintest import train_test_upload
from data_preprocessing.bias import Bias_Dataset_Evaluation
from data_preprocessing.push_to_database import upload_patient_data 
# Define data path
DATA_PATH = os.path.join(AIRFLOW_ROOT, "data", "diabetic_data.csv")

from datetime import datetime, timedelta
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule


# Define default_args
default_args = {
    'owner': 'MLOps_Team14',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Define function to notify failure or sucess via an email
def notify_success(context):
    task_id = context.get('task_instance').task_id
    dag_id = context.get('dag').dag_id
    success_email = EmailOperator(
        task_id='success_email_' + task_id,
        to='mlopsneu2025@gmail.com',
        subject=f'Success: Task {task_id} in DAG {dag_id}',
        html_content=f'<p>✅ Task <strong>{task_id}</strong> in DAG <strong>{dag_id}</strong> succeeded.</p>',
        dag=context['dag']
    )
    success_email.execute(context=context)

def notify_failure(context):
    task_id = context.get('task_instance').task_id
    dag_id = context.get('dag').dag_id
    failure_email = EmailOperator(
        task_id='failure_email_' + task_id,
        to='mlopsneu2025@gmail.com',
        subject=f'Failure: Task {task_id} in DAG {dag_id}',
        html_content=f'<p>❌ Task <strong>{task_id}</strong> in DAG <strong>{dag_id}</strong> failed.</p>',
        dag=context['dag']
    )
    failure_email.execute(context=context)

#INITIALIZE THE DAG INSTANCE
dag = DAG(
    'DataPipeline',
    default_args = default_args,
    description = 'MLOps Data pipeline',
    schedule_interval = None,  # Set the schedule interval or use None for manual triggering
    catchup = False,
)

email_notification_task = EmailOperator(
    task_id='dag_started_email',
    to='mlopsneu2025@gmail.com',
    subject='Data Pipeline Dag Started',
    html_content='<p> ✅ Data Pipeline Dag Started</p>',
    dag=dag,
)

ingest_data_task = PythonOperator(
    task_id='ingest_data_task',
    python_callable=ingest_data,
    op_args=["https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"],
    on_failure_callback=notify_failure,
    dag=dag,
)

unzip_file_task = PythonOperator(
    task_id='unzip_file_task',
    python_callable=unzip_file,
    op_args=[ingest_data_task.output],
    on_failure_callback=notify_failure,
    dag=dag,
)

remove_duplicates_task = PythonOperator(
    task_id='remove_duplicates_task',
    python_callable= duplicates,
    op_args=[DATA_PATH],
    on_failure_callback=notify_failure,
    dag=dag,
)

missing_value_task = PythonOperator(
    task_id='missing_value_task',
    python_callable= missingVal,
    op_args= [remove_duplicates_task.output],
    on_failure_callback=notify_failure,
    dag=dag,
)

data_mapping_task = PythonOperator(
    task_id='data_mapping_task',
    python_callable= process_data_mapping,
    op_args= [missing_value_task.output],
    on_failure_callback=notify_failure,
    dag=dag,
)

schema_validation_task = PythonOperator(
    task_id='schema_validation_task',
    python_callable= validate_data_schema,
    op_args=[data_mapping_task.output],
    on_failure_callback=notify_failure,
    dag=dag,
)

data_bias_task = PythonOperator(
    task_id='data_bias_task',
    python_callable= Bias_Dataset_Evaluation,
    op_args=[data_mapping_task.output],
    on_failure_callback=notify_failure,
    dag=dag,
)

encoding_task = PythonOperator(
    task_id='encoding_task',
    python_callable= target_encoding,
    op_args= [data_mapping_task.output],
    on_failure_callback=notify_failure,
    dag=dag,
)

feature_extract_task = PythonOperator(
    task_id='feature_extract_task',
    python_callable= feature_extraction,
    op_args= [encoding_task.output],
    on_failure_callback=notify_failure,
    dag=dag,
)

feature_selection_task = PythonOperator(
    task_id='feature_selection_task',
    python_callable= feature_selection,
    op_args= [feature_extract_task.output],
    on_failure_callback=notify_failure,
    dag=dag,
)

feature_scaling_task = PythonOperator(
    task_id='feature_scaling_task',
    python_callable= scaling,
    op_args= [feature_selection_task.output],
    on_failure_callback=notify_failure,
    dag=dag,
)

gcp_upload_task = PythonOperator(
    task_id='upload_patient_data_task',
    python_callable= upload_patient_data,
    op_args= [feature_scaling_task.output],
    on_failure_callback=notify_failure,
    dag=dag,
)


email_notification_task = EmailOperator(
    task_id='dag_completed_email',
    to='mlopsneu2025@gmail.com',
    subject='Dag Completed Successfully',
    html_content='<p> ✅ Data Pipeline Dag Completed</p>',
    dag=dag,
)

# Task to trigger the ModelPipeline DAG
# trigger_model_pipeline_task = TriggerDagRunOperator(
#     task_id='trigger_model_pipeline_task',
#     trigger_dag_id='ModelDevelopmentPipeline',
#     trigger_rule=TriggerRule.ALL_DONE,  # Ensure this task runs only if all upstream tasks succeed
#     dag=dag,
# )

ingest_data_task >> unzip_file_task >> remove_duplicates_task >> missing_value_task >> data_mapping_task >> encoding_task >> feature_extract_task  >> schema_validation_task >> data_bias_task >>feature_selection_task  >> feature_scaling_task >> gcp_upload_task >> email_notification_task #>> trigger_model_pipeline_task

if __name__ == "__main__":
    dag.cli()