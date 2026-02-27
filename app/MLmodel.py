from fastapi import FastAPI
import joblib
from preprocess import preProcess
from pydantic import BaseModel


app = FastAPI()

@app.post("/predict")
def predict():
    pass
