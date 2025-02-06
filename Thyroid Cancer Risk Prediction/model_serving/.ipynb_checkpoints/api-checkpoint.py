import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Load model, scaler, encoders, and feature columns
model = joblib.load(os.path.join(r"../saved_params", "model.pkl"))
scaler = joblib.load(os.path.join(r"../saved_params", "scaler.pkl"))
encoders = joblib.load(os.path.join(r"../saved_params", "encoders.pkl"))
feature_columns = joblib.load(os.path.join(r"../saved_params", "feature_columns.pkl"))
risk_mapping = joblib.load(os.path.join(r"../saved_params", "risk_mapping.pkl"))

# FastAPI setup
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic model for input validation
class PatientData(BaseModel):
    Age: int
    Gender: str
    Country: str
    Ethnicity: str
    Family_History: str
    Radiation_Exposure: str
    Iodine_Deficiency: str
    Smoking: str
    Obesity: str
    Diabetes: str
    TSH_Level: float
    T3_Level: float
    T4_Level: float
    Nodule_Size: float
    Thyroid_Cancer_Risk: str

@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert categorical features using saved encoders
        categorical_values = [encoders[col].transform([getattr(data, col)])[0] for col in encoders]

        # Convert ordinal feature "Thyroid_Cancer_Risk"
        risk_value = risk_mapping[data.Thyroid_Cancer_Risk]  # Convert string to numerical value
        
        # Prepare input data
        input_data = np.array([[data.Age] + categorical_values + [
            data.TSH_Level, data.T3_Level, data.T4_Level, data.Nodule_Size, risk_value]])

        # # Prepare input data
        # input_data = pd.DataFrame([[
        #     data.Age] + categorical_values + [
        #     data.TSH_Level, data.T3_Level, data.T4_Level, data.Nodule_Size,
        #     risk_value]
        # ], columns=feature_columns)  # Ensure correct feature names

        # Prepare input data
        input_data = pd.DataFrame([[
            data.Age] + categorical_values + [
            data.TSH_Level, data.T3_Level, data.T4_Level, data.Nodule_Size,
            risk_mapping[data.Thyroid_Cancer_Risk]
        ]], columns=feature_columns)  # Ensure correct feature names

        print("Processed Input Data:", input_data)  # Debugging
        
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)
        return {"Diagnosis": "Malignant" if prediction[0] == 1 else "Benign"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000)
    