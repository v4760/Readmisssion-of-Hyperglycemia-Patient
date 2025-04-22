from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    fname: str
    lname: str
    dob: str
    meds: List[str]
    gender: str
    race: str
    admission_type: str
    discharge_disposition: str
    diag_1: str
    diag_2: str
    diag_3: str
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    admission_source_id: str
    diabetic_medication: int
    change_num: int


class ActualResultUpdate(BaseModel):
    fname: str
    lname: str
    dob: str  # or datetime.date if you're parsing dates
    actual_result: int  # should be 0 or 1
