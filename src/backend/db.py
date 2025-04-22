from sqlalchemy import create_engine, text
import pandas as pd
import os


# === Database Connection Setup ===
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASS')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT', '5432')

def get_engine():
    return create_engine(
        f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    )

def insert_into_patients_data(data):
    """
    Accepts either a dict (single row) or a DataFrame and inserts it into the DB.
    """
    # 👇 Normalize input to a DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Expected dict or DataFrame")

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM patients_data LIMIT 0"))
        db_columns = [col.lower() for col in result.keys()]
        df_columns = list(df.columns)
        unmatched = set(df_columns) - set(db_columns)
        if unmatched:
            print("Columns in DataFrame but not in DB:", unmatched)
        else:
            print(" All DataFrame columns match the DB")
        df.to_sql('patients_data', engine, if_exists='append', index=False)
        print("Data inserted into patients_data table.")

def get_patient_by_identity(fname: str, lname: str, dob: str):
    """
    Fetches the most recent patient record matching fname, lname, and dob.
    Returns None if no match is found.
    """
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT * FROM patients_data
                WHERE f_name = :fname AND l_name = :lname AND dob = :dob
                ORDER BY patient_id DESC LIMIT 1
            """),
            {"fname": fname, "lname": lname, "dob": dob}
        )
        row = result.fetchone()
        return dict(row._mapping) if row else None

def update_actual_result_in_db(fname, lname, dob, actual_result):
    engine = get_engine()

    with engine.begin() as conn:
        result = conn.execute(
            text("""
                UPDATE patients_data
                SET readmitted = :actual_result
                WHERE f_name = :fname AND l_name = :lname AND dob = :dob
            """),
            {"actual_result": actual_result, "fname": fname, "lname": lname, "dob": dob}
        )
        print(f"[DB DEBUG] Rows affected: {result.rowcount}")

