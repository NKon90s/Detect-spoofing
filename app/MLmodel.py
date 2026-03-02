from fastapi import FastAPI
import joblib
from pydantic import BaseModel, ValidationError, Field
from typing import Annotated
from predict import predict
import uvicorn
from src import rinex_conversion as rx
from pathlib import Path


# Creating a FastAPI app
app = FastAPI(
    title="GNSS Signal Spoofing Detector",
    description="Predicts if a GNSS signal is spoofed/manipulated",
    version="1.0.0"
)

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
    delta_t: Annotated[float | None, Field(description="Change in time")]
    delta_pr: Annotated[float | None, Field(description="Change of the pseudorange")]
    pr_rate: Annotated[float | None, Field(description="The rate of change of the distance between the satellite and the reciever")]
    wavelength: Annotated[float | None, Field(description="Wavelength of the signal")]
    doppler_vs_prrate: Annotated[float | None, Field(description="Rate to measure consistency between doppler and pseudorange")]
    snr_mean_5: Annotated[float | None, Field(description="Mean Signal to Noise ratio accross the last five epochs")]
    snr_std_5: Annotated[float | None, Field(description="Standard deviation of Signal to Noise ratio accross the last five epochs")]
    sat_count: Annotated[int | None, Field(description="Number of satellites visible in the given epoch")]
    n_missing_pr: Annotated[int | None, Field(description="Number of missing pseudoranges")]


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


@app.get("/")
def read_root():
    return {"message": "Welcome to the Signal Spoofing Detection API."}


@app.post("/predict", response_model=DataInput)
def start_prediction():
    
    predict()


@app.get("/health")
def health_check():
    """Health check endpoint to verify API is running"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)