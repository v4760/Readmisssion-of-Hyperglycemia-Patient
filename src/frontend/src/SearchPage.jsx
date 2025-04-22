import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { TextField, Box } from "@mui/material";

const SearchPatient = () => {
  const [searchData, setSearchData] = useState({
    fname: "",
    lname: "",
    dob: "",
  });

  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setSearchData({ ...searchData, [name]: value });
  };

  const isFormValid =
    searchData.fname.trim() !== "" &&
    searchData.lname.trim() !== "" &&
    searchData.dob.trim() !== "";

  const handleSearch = async () => {
    if (!isFormValid) return;

    setLoading(true);
    try {
      const response = await fetch("https://term-deposit-backend-149146997593.us-east1.run.app/search-patient", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(searchData),
      });

      if (!response.ok) {
        throw new Error("Patient not found");
      }

      const data = await response.json();

      if (data) {
        navigate("/patient-details", { state: data });
      } else {
        alert("Patient not found");
      }
    } catch (error) {
      console.error("Search error:", error);
      alert("Patient not found");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center h-screen bg-gradient-to-tr from-blue-100 to-white">
      <div className="w-1/2 bg-white p-8 rounded-3xl drop-shadow-2xl">
        <h2 className="text-2xl font-bold mb-6 text-center">Search Patient</h2>
        <Box className="flex flex-col gap-6">
          <TextField
            label="First Name"
            name="fname"
            value={searchData.fname}
            onChange={handleChange}
            variant="outlined"
            fullWidth
            required
          />
          <TextField
            label="Last Name"
            name="lname"
            value={searchData.lname}
            onChange={handleChange}
            variant="outlined"
            fullWidth
            required
          />
          <TextField
            label="Date of Birth"
            name="dob"
            type="date"
            value={searchData.dob}
            onChange={handleChange}
            variant="outlined"
            fullWidth
            InputLabelProps={{ shrink: true }}
            required
          />
          <button
            className={`border-2 border-teal px-6 py-2 rounded-full font-medium bg-teal text-black hover:bg-white transition ${
              !isFormValid || loading ? "opacity-50 cursor-not-allowed" : ""
            }`}
            onClick={handleSearch}
            disabled={!isFormValid || loading}
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </Box>
      </div>
    </div>
  );
};

export default SearchPatient;
