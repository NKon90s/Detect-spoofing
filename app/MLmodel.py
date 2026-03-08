from fastapi import FastAPI, UploadFile, HTTPException, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, Field
from typing import Annotated
from predict import predict
import uvicorn
from pathlib import Path
import pandas as pd
import os
from io import StringIO
 
# Creating a FastAPI app
app = FastAPI(
    title="GNSS Signal Spoofing Detector",
    description="Predicts if a GNSS signal is spoofed/manipulated",
    version="1.0.0"
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
ALLOWED_EXTENSIONS = {".csv"} # in future it will accept rinex files directly as well
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))  # 50MB

# Defining input schema using Pydantic
class DataInput(BaseModel):
    time_utc: Annotated[object | None, Field(description="Data and time of recording in UTC")]
    time: Annotated[object | None, Field(description="Data and time of recording based on system time")]
    sys: Annotated[object | None, Field(description="Satellite system identifier")]
    sv: Annotated[object | None, Field(description="Space-Vehicle-Number: Serial number of the satellite")]
    prn: Annotated[int | None, Field(description="Pseudo-Random-Noise: The signal ID used to identify the satellite")]
    pseudorange: Annotated[float | None, Field(description="Pseudo distance between satellite and reciever")]
    phase: Annotated[float | None, Field(description="carrier phase of the given satellite")]
    doppler: Annotated[float | None, Field(description="Doppler shift of the signal")]
    snr: Annotated[float | None, Field(description="Signal to Noise ratio")]
    time_s: Annotated[int | None, Field(description="Time in seconds")] 
    delta_t: Annotated[float | None, Field(description="Time differenece between epochs")]
    delta_pr: Annotated[float | None, Field(description="Change of the pseudorange")]
    pr_rate: Annotated[float | None, Field(description="The rate of change of the distance between the satellite and the reciever")]
    wavelength: Annotated[float | None, Field(description="Wavelength of the signal")]
    doppler_vs_prrate: Annotated[float | None, Field(description="Rate to measure consistency between doppler and pseudorange")]
    snr_mean_5: Annotated[float | None, Field(description="Mean Signal to Noise ratio accross the last five epochs")]
    snr_std_5: Annotated[float | None, Field(description="Standard deviation of Signal to Noise ratio accross the last five epochs")]
    sat_count: Annotated[int | None, Field(description="Number of satellites visible in the given epoch")]
    n_missing_pr: Annotated[int | None, Field(description="Number of missing pseudoranges (loss of signal)")]


    class Config:
        schema_extra = {
            "example": {
                "time_utc": "2025-10-27T21:15:42",
                "time": "2025.10.27 21:15",
                "sys": "G",
                "sv": "G05",
                "prn": 5,
                "pseudorange": 19999103.134,
                "phase": 104845425.919,
                "doppler": -475.204,
                "snr": 21.0,
                "time_s": 1761599743,
                "delta_t": 1.0,
                "delta_pr": 92.10201301303,
                "pr_rate": 93.1201201,
                "wavelength": 0.190293672798365,
                "doppler_vs_prrate": 4.206735463463,
                "snr_mean_5": 18.0,
                "snr_std_5": 1.25677,
                "sat_count": 19,
                "n_missing_pr": 15
            }
        }
 
# Defining Response Model with pydantic
class PredictionResponse(BaseModel):
    total_samples: int
    spoofed_count: int
    spoofing_ratio: float
    avg_spoofing_probability: float
    is_spoofed: str


# Class for validating uploaded file. File should be csv file and match with 'DataInput' schema.
class ValidateFile:

    def __init__(self, max_size: int = MAX_FILE_SIZE, allowed_extensions: set = ALLOWED_EXTENSIONS):
        self.max_size = max_size
        self.allowed_extensions = allowed_extensions

    async def __call__(self, file: UploadFile = File(...)) -> UploadFile:

        """
        Validate file when used as a dependency.
        """
        file_extension = Path(file.filename).suffix.lower()

        # Check extension
        if file_extension not in self.allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File extension: {file_extension} not allowed")
        
        # Load content 
        content = await file.read()
        if len(content) > self.max_size:
            raise HTTPException(status_code=400, detail=f"File exceeds maximum size of {self.max_size} bytes")
        
        # Validate file format. We are using pandas. It is faster if we have tens of thousands or more rows.
        try:
            text = content.decode("utf-8") 
            df = pd.read_csv(StringIO(text))
            records = df.to_dict(orient = "records")
        except Exception as e:
                raise HTTPException(400, f"Invalid CSV format: {str(e)}")

        validated_rows = []
        invalid_rows = []
        for i, row in enumerate(records):
            try:
                validated_rows.append(DataInput(**row))
            except ValidationError as e:
                    invalid_rows.append({f"row": row, "errors": e.errors()})
                
        if invalid_rows:
            raise HTTPException(status_code=422, detail={"message": "Invalid rows found",
                                                         "invalid_rows": invalid_rows[:10]})
        
        #await file.seek(0)
        return df


# Defining the threshold for the data to be considered spoofed. 
SPOOFING_THRESHOLD = 0.6
# To avoid small amount of outliers, which could be caused by anomalies we will set a minimum amount for spoofed samples.
MIN_SPOOFED_SAMPLE = 10

@app.post("/predict-spoofing", response_model=PredictionResponse)
async def start_prediction(df: pd.DataFrame = Depends(ValidateFile())):
    prediction = predict(df)

    spoofed_count = len(prediction[prediction["pred_attack"] == 1])
    total_samples = len(prediction)
    if total_samples == 0:
        raise HTTPException(400, "CSV file contains no data")
    spoofing_ratio = round(spoofed_count/total_samples, 2)
    spoofed_samples = prediction[prediction["pred_attack"] == 1]
    avg_spoofing_probability = spoofed_samples["attack_probability"].mean()

    return PredictionResponse(
        total_samples = total_samples,
        spoofed_count = spoofed_count,
        spoofing_ratio = spoofing_ratio,
        avg_spoofing_probability = avg_spoofing_probability,
        is_spoofed = "Signal is likely spoofed" if avg_spoofing_probability > SPOOFING_THRESHOLD and spoofed_count > MIN_SPOOFED_SAMPLE 
        else "Signal is likely not spoofed"
    )


@app.get("/")
def read_root():
    return {"message": "You are using GNSS spoofing detection API"}


@app.get("/health")
def health_check():
    """Health check endpoint to verify API is running"""
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    return {
        "model": "GNSS Spoofing Detector",
        "version": "1.0",
        "threshold": SPOOFING_THRESHOLD,
        "min-spoofed data": MIN_SPOOFED_SAMPLE
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


