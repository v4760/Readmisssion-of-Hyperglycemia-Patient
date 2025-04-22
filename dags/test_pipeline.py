from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

# Define the path to your local virtual environment and pytest script
VENV_PATH = "/opt/airflow/venv"  # Update with your actual venv path
PYTEST_SCRIPT = "/opt/airflow/tests/test_data.py"  # Update with actual test file path

# Define default DAG arguments
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 2, 28),
    "retries": 1,
}

# Define the DAG
dag = DAG(
    "test_data_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)

# Task to trigger pytest using BashOperator
run_tests_task = BashOperator(
    task_id="run_tests",
    bash_command=f"cd /opt/airflow/tests && source {VENV_PATH}/bin/activate && pytest -s test_data.py",
    dag=dag,
)
