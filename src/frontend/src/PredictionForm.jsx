import React, { useState } from "react";
import {
  TextField,
  MenuItem,
  Box,
  Modal,
  FormControl,
  InputLabel,
  Select,
} from "@mui/material";

const initialFormData = {
  fname: "",
  lname: "",
  dob: "",
  meds: [],
  gender: "",
  race: "",
  admission_type: "",
  discharge_disposition: "",
  diag_1: "",
  diag_2: "",
  diag_3: "",
  time_in_hospital: "",
  num_lab_procedures: "",
  num_procedures: "",
  num_medications: "",
  number_outpatient: "",
  number_emergency: "",
  number_inpatient: "",
  number_diagnoses: "",
  admission_source_id: "",
  diabetic_medication: "",
  change_num: "",
};

const PredictionForm = () => {
  const [formData, setFormData] = useState(initialFormData);
  const [open, setOpen] = useState(false);
  const [predictedResult, setPredictedResult] = useState(null);
  const [step, setStep] = useState(1);

  const sendToFastAPI = async () => {
    const payload = transformDataForBackend();

    try {
      const response = await fetch("https://term-deposit-backend-149146997593.us-east1.run.app/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const result = await response.json();
      setPredictedResult(result.prediction === 1 ? "Yes" : "No");
      setOpen(true);
    } catch (error) {
      console.error("Prediction API error:", error.message);
    }
  };

  const transformDataForBackend = () => ({
    ...formData,
    dob: String(formData.dob),
    time_in_hospital: Number(formData.time_in_hospital),
    num_lab_procedures: Number(formData.num_lab_procedures),
    num_procedures: Number(formData.num_procedures),
    num_medications: Number(formData.num_medications),
    number_outpatient: Number(formData.number_outpatient),
    number_emergency: Number(formData.number_emergency),
    number_inpatient: Number(formData.number_inpatient),
    number_diagnoses: Number(formData.number_diagnoses),
    diabetic_medication: formData.diabetic_medication === "Yes" ? 1 : 0,
    change_num: formData.change_num === "Yes" ? 1 : 0,
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const validateStep1 = () => {
    const requiredFields = [
      "fname",
      "lname",
      "dob",
      "meds",
      "gender",
      "race",
      "admission_type",
      "discharge_disposition",
      "admission_source_id",
    ];
    return requiredFields.every(
      (field) =>
        formData[field] !== "" &&
        formData[field] !== null &&
        (!Array.isArray(formData[field]) || formData[field].length > 0)
    );
  };

  const numericField = (label, name, min, max) => (
    <TextField
      label={label}
      name={name}
      value={formData[name]}
      type="number"
      variant="outlined"
      fullWidth
      onChange={(e) => {
        const value = e.target.value;
        if (value === "" || (Number(value) >= min && Number(value) <= max)) {
          handleChange(e);
        }
      }}
      inputProps={{
        min,
        max,
        inputMode: "numeric",
        pattern: "[0-9]*",
      }}
      required
    />
  );

  const handleSubmit = (e) => {
    e.preventDefault();
    sendToFastAPI(formData);
  };

  const diag = [
    "Diabetes",
    "Genitourinary",
    "Injury",
    "Musculoskeletal",
    "Neoplasms",
    "Respiratory",
    "Others",
  ];

  return (
    <div className="flex items-center justify-center py-20">
      <div className="w-3/4 bg-white flex items-center justify-center p-6 drop-shadow-2xl rounded-3xl">
        <div className="p-6 w-full">
          <div className="text-2xl font-bold mb-6">Prediction Form</div>
          <form onSubmit={handleSubmit}>
            <div className="grid grid-cols-2 gap-6">
              {step === 1 && (
                <>
                  <TextField label="First Name" name="fname" value={formData.fname} variant="outlined" fullWidth onChange={handleChange} required />
                  <TextField label="Last Name" name="lname" value={formData.lname} variant="outlined" fullWidth onChange={handleChange} required />
                  <TextField label="Date of Birth" name="dob" type="date" value={formData.dob} onChange={handleChange} variant="outlined" fullWidth InputLabelProps={{ shrink: true }} required />
                  <FormControl fullWidth required>
                    <InputLabel id="meds">Medications</InputLabel>
                    <Select
                      labelId="meds"
                      name="meds"
                      multiple
                      value={formData.meds}
                      onChange={(e) => setFormData({ ...formData, meds: e.target.value })}
                      renderValue={(selected) => selected.join(", ")}
                    >
                      {[
                        "Metformin",
                        "Repaglinide",
                        "Glipizide",
                        "Glyburide",
                        "Pioglitazone",
                        "Rosiglitazone",
                        "Acarbose",
                        "Insulin",
                      ].map((item) => (
                        <MenuItem key={item} value={item}>{item}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  <TextField label="Gender" name="gender" value={formData.gender} select variant="outlined" fullWidth onChange={handleChange} required>
                    {["Male", "Female"].map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
                  </TextField>
                  <TextField label="Race" name="race" value={formData.race} select variant="outlined" fullWidth onChange={handleChange} required>
                    {["Caucasian", "AfricanAmerican", "Other"].map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
                  </TextField>
                  <TextField label="Admission Type" name="admission_type" value={formData.admission_type} select variant="outlined" fullWidth onChange={handleChange} required>
                    {["Emergency", "Urgent", "Elective", "New Born", "Trauma Center", "Not Available"].map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
                  </TextField>
                  <TextField label="Discharge Disposition" name="discharge_disposition" value={formData.discharge_disposition} select variant="outlined" fullWidth onChange={handleChange} required>
                    {["Discharged to Home", "Care/Nursing", "Other", "Unknown"].map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
                  </TextField>
                  <TextField label="Admission Source" name="admission_source_id" value={formData.admission_source_id} select variant="outlined" fullWidth onChange={handleChange} required>
                    {["Referral", "Emergency room", "Others"].map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
                  </TextField>
                  {["diag_1", "diag_2", "diag_3"].map((diagno, idx) => (
                    <TextField key={diagno} label={`Diagnosis ${idx + 1}`} name={diagno} value={formData[diagno]} select variant="outlined" fullWidth onChange={handleChange}>
                      {diag.map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
                    </TextField>
                  ))}
                </>
              )}
              {step === 2 && (
                <>
                  {numericField("Time in Hospital", "time_in_hospital", 1, 14)}
                  {numericField("Number of Lab Procedures", "num_lab_procedures", 1, 132)}
                  {numericField("Number of Procedures", "num_procedures", 0, 10)}
                  {numericField("Number of Medications", "num_medications", 1, 100)}
                  {numericField("Number of Outpatient Visits", "number_outpatient", 0, 50)}
                  {numericField("Number of Emergency Visits", "number_emergency", 0, 100)}
                  {numericField("Number of Inpatient Visits", "number_inpatient", 0, 50)}
                  {numericField("Number of Diagnoses", "number_diagnoses", 1, 20)}
                  <TextField label="Change in Diabetic Medication" name="change_num" value={formData.change_num} select variant="outlined" fullWidth onChange={handleChange} required>
                    {["Yes", "No"].map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
                  </TextField>
                  <TextField label="Diabetic Medication" name="diabetic_medication" value={formData.diabetic_medication} select variant="outlined" fullWidth onChange={handleChange} required>
                    {["Yes", "No"].map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
                  </TextField>
                </>
              )}
            </div>
            <Box className="flex justify-between mt-8">
              {step > 1 && <button type="button" onClick={() => setStep(step - 1)} className="border-2 border-teal px-6 py-2 rounded-full font-medium hover:bg-teal hover:opacity-95 transition">Back</button>}
              {step < 2 && <button type="button" onClick={() => validateStep1() ? setStep(step + 1) : alert("Please fill all required fields")} className="border-2 border-teal px-6 py-2 rounded-full font-medium bg-teal hover:bg-white transition">Next</button>}
              {step === 2 && <button type="submit" className="border-2 border-teal px-6 py-2 rounded-full font-medium bg-teal hover:bg-white transition">Predict</button>}
            </Box>
          </form>

          <Modal open={open && predictedResult !== null} onClose={() => {
            setOpen(false);
            setFormData(initialFormData);
            setPredictedResult(null);
            setStep(1);
          }}>
            <Box className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white/90 backdrop-blur-md border border-gray-200 rounded-3xl shadow-2xl p-8 w-[440px] max-w-[90%] animate-fadeInUp">
              <div className={`w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6 text-5xl animate-bounceSlow shadow-md ${predictedResult === "Yes" ? "bg-red-100 text-red-500" : "bg-green-100 text-green-500"}`}>{predictedResult === "Yes" ? "🚨" : "🎉"}</div>
              <div className={`text-2xl font-bold text-center font-bold mb-2 ${predictedResult === "Yes" ? "text-red-600" : "text-green-600"}`}>{predictedResult === "Yes" ? "Readmission Risk Detected" : "All Clear!"}</div>
              <div className="text-center text-gray-700 text-md mb-8 leading-relaxed">{predictedResult === "Yes" ? "Our prediction model indicates a high risk of readmission. Please take action." : "No risk detected. Continue with standard care and follow-up."}</div>
              <div className="flex justify-center mt-2">
                <button onClick={() => {
                  setOpen(false);
                  setFormData(initialFormData);
                  setPredictedResult(null);
                  setStep(1);
                }} className={`px-6 py-2 rounded-full font-semibold text-white transition duration-300 ${predictedResult === "Yes" ? "bg-red-500 hover:bg-red-600" : "bg-green-500 hover:bg-green-600"}`}>
                  Close
                </button>
              </div>
            </Box>
          </Modal>
        </div>
      </div>
    </div>
  );
};

export default PredictionForm;