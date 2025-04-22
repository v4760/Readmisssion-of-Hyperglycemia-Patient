from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import PredictionRequest, ActualResultUpdate
from gcs_utils import download_model_from_gcs
from predict_model import load_model, make_prediction, load_scaler
from db import get_patient_by_identity, update_actual_result_in_db
from decode import decode_one_hot_record
from datetime import datetime
import traceback

app = FastAPI()

# === CORS for frontend ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://term-deposit-frontend-149146997593.us-east1.run.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Startup: Load model and scaler ===
@app.on_event("startup")
def startup_event():
    bucket_name = "readmission_prediction"
    model_blob = "models/best_xgboost_model/model.pkl"
    scaler_blob = "models/best_xgboost_model/scalar_weight.pkl"
    model_path = "models/my_model.pkl"
    scaler_path = "models/scalar_weight.pkl"

    try:
        print("[INFO] Downloading model and scaler from GCS...")
        download_model_from_gcs(bucket_name, model_blob, model_path)
        download_model_from_gcs(bucket_name, scaler_blob, scaler_path)
        load_model(model_path)
        load_scaler(scaler_path)
        print("[INFO] Model and scaler loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize app: {e}")
        traceback.print_exc()

# === Predict Route ===
@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
        print("[DEBUG] Raw request body:", body)

        data = PredictionRequest(**body)
        prediction = make_prediction(data)
        return {"prediction": prediction}
    except Exception as e:
        print("[ERROR] Exception in /predict:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# === Update Actual Result ===
@app.post("/update-actual-result")
def update_actual_result(payload: ActualResultUpdate):
    try:
        print(f"[INFO] Received actual result payload: {payload}")
        dob_parsed = datetime.strptime(payload.dob, "%Y-%m-%d").date()
        update_actual_result_in_db(payload.fname, payload.lname, dob_parsed, payload.actual_result)
        return {"message": "Actual result updated successfully."}
    except Exception as e:
        print("[ERROR] Exception during actual result update:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to update result.")

# === Search Patient ===
@app.post("/search-patient")
async def search_patient(request: Request):
    try:
        body = await request.json()
        print(f"[DEBUG] Search patient request body: {body}")
        fname, lname, dob = body.get("fname"), body.get("lname"), body.get("dob")

        if not (fname and lname and dob):
            return JSONResponse(status_code=400, content={"error": "Missing input fields"})

        record = get_patient_by_identity(fname, lname, dob)
        print(f"[DEBUG] Retrieved patient record: {record}")
        if not record:
            return JSONResponse(status_code=404, content={"error": "Patient not found"})

        parsed = decode_one_hot_record(record)
        print(f"[DEBUG] Decoded patient data: {parsed}")

        dob_str = parsed.get("dob")
        age = None
        if dob_str:
            try:
                dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
                today = datetime.today().date()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            except ValueError:
                print(f"[WARNING] Invalid DOB format: {dob_str}")

        filtered_data = {
            "fname": parsed.get("fname"),
            "lname": parsed.get("lname"),
            "dob": parsed.get("dob"),
            "gender": parsed.get("gender"),
            "race": parsed.get("race"),
            "admission_type": parsed.get("admission_type"),
            "admission_source_id": parsed.get("admission_source_id"),
            "discharge_disposition": parsed.get("discharge_disposition"),
            "diag_1": parsed.get("diag_1"),
            "diag_2": parsed.get("diag_2"),
            "diag_3": parsed.get("diag_3"),
            "diabetic_medication": parsed.get("diabetic_medication"),
            "change_num": parsed.get("change_num"),
            "meds": parsed.get("meds"),
            "age": age,
            "time_in_hospital": parsed.get("time_in_hospital"),
            "num_lab_procedures": parsed.get("num_lab_procedures"),
            "num_procedures": parsed.get("num_procedures"),
            "num_medications": parsed.get("num_medications"),
            "number_outpatient": parsed.get("number_outpatient"),
            "number_emergency": parsed.get("number_emergency"),
            "number_inpatient": parsed.get("number_inpatient"),
            "number_diagnoses": parsed.get("number_diagnoses"),
            "predicted_result": int(record.get("predict", 0)),
            "actual_result": int(record.get("readmitted", -1)) if record.get("readmitted") is not None else -1,
        }

        return filtered_data
        # return {
        #     "predicted_result": int(record.get("predict", 0)),
        #     "actual_result": int(record.get("readmitted", -1)) if record.get("readmitted") is not None else -1,
        #     **parsed
        # }



    except Exception as e:
        print("[ERROR] Exception during patient search:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal error during search.")
