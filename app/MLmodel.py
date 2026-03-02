from fastapi import FastAPI
import joblib
from pydantic import BaseModel, ValidationError, Field
from typing import Annotated
from predict import predict
import uvicorn

#####################################################
#
#  
#
#####################################################

app = FastAPI(
    title="GNSS Signal Spoofing Detector",
    description="Predicts if a GNSS signal is spoofed/manipulated",
    version="1.0.0"
)

# Creating a class to validate the inputs for our prediction model
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




@app.post("/predict", response_model=DataInput)
def start_prediction():
    predict()
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)