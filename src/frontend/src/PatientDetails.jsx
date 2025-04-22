import React, { useEffect, useState } from "react";
import { Switch } from "@mui/material";
import { useLocation } from "react-router-dom";

const fieldLabels = {
  fname: "First Name",
  lname: "Last Name",
  age: "Age",
  meds: "Medications",
  gender: "Gender",
  race: "Race",
  admission_type: "Admission Type",
  discharge_disposition: "Discharge Disposition",
  admission_source_id: "Admission Source",
  diag_1: "Diagnosis 1",
  diag_2: "Diagnosis 2",
  diag_3: "Diagnosis 3",
  time_in_hospital: "Time in Hospital",
  num_lab_procedures: "Number of Lab Procedures",
  num_procedures: "Number of Procedures",
  num_medications: "Number of Medications",
  number_outpatient: "Number of Outpatient Visits",
  number_emergency: "Number of Emergency Visits",
  number_inpatient: "Number of Inpatient Visits",
  number_diagnoses: "Number of Diagnoses",
  diabetic_medication: "Diabetic Medication",
  change_num: "Change Number",
  dob: "Date of Birth",
  predicted_result: "Predicted Result",
};

const PatientDetailsPage = () => {
  const location = useLocation();
  const passedPatient = location.state;

  const [patient, setPatient] = useState(null);
  const [actualResult, setActualResult] = useState(null);
  const [predictedResult, setPredictedResult] = useState("No");

  useEffect(() => {
    if (passedPatient) {
      setPatient(passedPatient);
  
      if (
        passedPatient.predicted_result === 1 ||
        passedPatient.predicted_result === "Yes"
      ) {
        setPredictedResult("Yes");
      } else {
        setPredictedResult("No");
      }
  
      if (
        passedPatient.actual_result === 1 ||
        passedPatient.actual_result === "Yes"
      ) {
        setActualResult(1);
      } else if (
        passedPatient.actual_result === 0 ||
        passedPatient.actual_result === "No"
      ) {
        setActualResult(0);
      }
    }
  }, [passedPatient]);  

  const handleUpdate = async () => {
    if (!patient) return;

    try {
      const response = await fetch("https://term-deposit-backend-149146997593.us-east1.run.app/update-actual-result", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          fname: patient.fname,
          lname: patient.lname,
          dob: patient.dob,
          actual_result: actualResult ? 1 : 0,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to update result.");
      }

      const result = await response.json();
      alert("Actual result updated successfully.");
      console.log(result);
    } catch (err) {
      console.error("Update error:", err);
    }
  };

  if (!patient) return <div>Loading...</div>;

  return (
    <div className="flex items-center justify-center py-20 bg-gradient-to-br from-blue-100 to-white">
      <div className="w-3/4 bg-white p-10 drop-shadow-xl rounded-3xl">
        <div className="text-2xl font-bold mb-6">Patient Details</div>

        <div className="grid grid-cols-2 gap-4 mb-8">
          {Object.entries(patient).map(([key, value]) =>
            key === "predicted_result" || key === "actual_result" ? null : (
              <div key={key} className="flex flex-row">
                <label className="text-md text-gray-600 font-bold mb-1">
                  {fieldLabels[key] || key}:
                </label>
                <div className="text-base text-gray-800 ml-2">
                  {Array.isArray(value) ? value.join(", ") : value}
                </div>
              </div>
            )
          )}
        </div>

        <div className="flex flex-row mb-6">
          <div className="text-md font-semibold text-gray-700 mb-1">
            Predicted Result:
          </div>
          <div
            className={`text-md font-bold ml-2 ${
              predictedResult === "Yes" ? "text-red-600" : "text-green-600"
            }`}
          >
            {predictedResult === "Yes"
              ? "Patient is at risk of readmission"
              : "Patient is not at risk of readmission"}
          </div>
        </div>

        <div className="mb-6">
  <div className="text-md font-semibold text-gray-700 mb-2">
    Mark Actual Result:
  </div>
  <div className="flex items-center gap-6">
    <label className="flex items-center gap-2">
      <input
        type="radio"
        name="actualResult"
        value="0"
        checked={actualResult === 0}
        onChange={() => setActualResult(0)}
      />
      <span>No</span>
    </label>
    <label className="flex items-center gap-2">
      <input
        type="radio"
        name="actualResult"
        value="1"
        checked={actualResult === 1}
        onChange={() => setActualResult(1)}
      />
      <span>Yes</span>
    </label>
  </div>
</div>


        <button
          className="border-2 border-teal px-6 py-2 rounded-full text-xs bg-teal hover:bg-white transition sm:text-sm"
          onClick={handleUpdate}
          disabled={actualResult === null}
        >
          Update Actual Result
        </button>
      </div>
    </div>
  );
};

export default PatientDetailsPage;
