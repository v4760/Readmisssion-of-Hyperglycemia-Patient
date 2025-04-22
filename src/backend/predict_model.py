import joblib
import pandas as pd
from models import PredictionRequest
from db import insert_into_patients_data
from datetime import datetime

model = None
scaler = None

def load_model(path: str):
    global model
    print(f"[INFO] Loading model from {path}")
    model = joblib.load(path)
    print("[INFO] Model loaded successfully.")

def load_scaler(path: str):
    global scaler
    print(f"[INFO] Loading scaler from {path}")
    scaler = joblib.load(path)
    print("[INFO] Scaler loaded successfully.")

def calculate_age(dob_str: str) -> int:
    dob = datetime.strptime(dob_str, "%Y-%m-%d")
    today = datetime.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

def normalize_diabetes_med(val):
    return "Yes" if str(val).lower() in ["1", "yes"] else "No"

def transform_input(data: PredictionRequest):
    global scaler
    features = {}

    # --- 1. Compute age from DOB ---
    features["age"] = calculate_age(data.dob)

    # --- 2. Direct numeric fields ---
    features["time_in_hospital"] = data.time_in_hospital
    features["num_lab_procedures"] = data.num_lab_procedures
    features["num_procedures"] = data.num_procedures
    features["num_medications"] = data.num_medications
    features["number_outpatient"] = data.number_outpatient
    features["number_emergency"] = data.number_emergency
    features["number_inpatient"] = data.number_inpatient
    features["number_diagnoses"] = data.number_diagnoses
    features["patient_id"] = 0
    features["number_of_changes"] = data.change_num
    features["health_index"] = (
        features["number_emergency"] +
        features["number_inpatient"] +
        features["number_outpatient"]
    )
    features["severity_of_disease"] = (
        features["time_in_hospital"] +
        features["num_procedures"] +
        features["num_medications"] +
        features["num_lab_procedures"] +
        features["number_diagnoses"]
    )

    # --- 3. Medications ---
    all_meds = [
        "metformin", "repaglinide", "glipizide", "glyburide", "pioglitazone",
        "rosiglitazone", "acarbose", "insulin"
    ]
    meds_set = set(m.lower() for m in data.meds)
    for med in all_meds:
        features[med] = 1 if med in meds_set else 0

    # --- 4. One-hot encodings ---
    features["race_caucasian"] = 1 if data.race == "Caucasian" else 0
    features["race_other"] = 1 if data.race == "Other" else 0
    features["gender_male"] = 1 if data.gender == "Male" else 0

    adm_map = ["Emergency", "New Born", "Not Available", "Trauma Center", "Urgent"]
    for val in adm_map:
        key = f"admission_type_id_{val.lower().replace(' ', '_')}"
        features[key] = 1 if data.admission_type == val else 0

    discharges = ["Discharged to Home", "Other", "Unknown"]
    for val in discharges:
        key = f"discharge_disposition_id_{val.lower().replace(' ', '_')}"
        features[key] = 1 if data.discharge_disposition == val else 0

    features["admission_source_id_referral"] = 1 if data.admission_source_id == "Referral" else 0
    features["admission_source_id_others"] = 1 if data.admission_source_id == "Others" else 0

    # --- 5. Diagnoses ---
    diag_cats = ["diabetes", "genitourinary", "injury", "musculoskelatal", "neoplasms", "others", "respiratory"]
    for i in ["diag_1", "diag_2", "diag_3"]:
        value = getattr(data, i, "").lower()
        for cat in diag_cats:
            features[f"{i}_{cat}"] = 1 if value == cat else 0

    # --- 6. Special fields ---
    features["change_no"] = 1 if data.change_num == 0 else 0
    diabetic_med_str = normalize_diabetes_med(data.diabetic_medication)
    features["diabetesmed_yes"] = 1 if diabetic_med_str.lower() == "yes" else 0

    # --- 7. Final structure ---
    final_columns = [
        "patient_id", "age", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
        "metformin", "repaglinide", "glipizide", "glyburide", "pioglitazone",
        "rosiglitazone", "acarbose", "insulin", "health_index", "severity_of_disease",
        "number_of_changes", "race_caucasian", "race_other", "gender_male",
        "admission_type_id_emergency", "admission_type_id_new_born", "admission_type_id_not_available",
        "admission_type_id_trauma_center", "admission_type_id_urgent",
        "discharge_disposition_id_discharged_to_home", "discharge_disposition_id_other",
        "discharge_disposition_id_unknown", "admission_source_id_others",
        "admission_source_id_referral",
        "diag_1_diabetes", "diag_1_genitourinary", "diag_1_injury", "diag_1_musculoskelatal",
        "diag_1_neoplasms", "diag_1_others", "diag_1_respiratory",
        "diag_2_diabetes", "diag_2_genitourinary", "diag_2_injury", "diag_2_musculoskelatal",
        "diag_2_neoplasms", "diag_2_others", "diag_2_respiratory",
        "diag_3_diabetes", "diag_3_genitourinary", "diag_3_injury", "diag_3_musculoskelatal",
        "diag_3_neoplasms", "diag_3_others", "diag_3_respiratory",
        "change_no", "diabetesmed_yes"
    ]

    for col in final_columns:
        if col not in features:
            features[col] = 0

    df = pd.DataFrame([features])[final_columns]

    # Apply RobustScaler to numeric features
    numeric_columns = [
        "age", "time_in_hospital", "num_lab_procedures", "num_procedures", 
        "num_medications", "number_outpatient", "number_emergency", 
        "number_inpatient", "number_diagnoses", "health_index", 
        "severity_of_disease", "number_of_changes"
    ]

    print("[DEBUG] Columns passed to scaler:", list(df[scaler.feature_names_in_].columns))
    print("[DEBUG] Scaler expects:", list(scaler.feature_names_in_))
    print("[DEBUG] Shape of transformed data:", scaler.transform(df[scaler.feature_names_in_]).shape)

    if scaler:
        missing_cols = set(scaler.feature_names_in_) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing expected features: {missing_cols}")

        # Use scaler.feature_names_in_ as both index and assignment
        df[scaler.feature_names_in_] = scaler.transform(df[scaler.feature_names_in_])

    else:
        print("[WARNING] Scaler not loaded. Skipping input scaling!")

    return df

def normalize_keys(data: dict) -> dict:
    return {k.replace(" ", "_").replace("-", "_").lower(): v for k, v in data.items()}

def make_prediction(data: PredictionRequest):
    global model
    if model is None:
        raise ValueError("Model is not loaded.")

    df = transform_input(data)
    prediction = model.predict(df)[0]

    df = df.drop(columns=["patient_id"])
    record = df.to_dict(orient="records")[0]
    record["predict"] = float(prediction)
    record["f_name"] = data.fname
    record["l_name"] = data.lname
    record["dob"] = data.dob
    record["age"] = calculate_age(data.dob)

    insert_into_patients_data(normalize_keys(record))
    print(f"[INFO] Prediction made: {prediction}")
    return int(prediction)
