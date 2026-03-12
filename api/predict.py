import pandas as pd 
import joblib
from preprocess import preProcess

MODEL_PATH = "gnss_xgboost_model.joblib"   # exchange it with the model you generate with detectionML_model.py in src lib
THRESHOLD = 0.4

model = joblib.load(MODEL_PATH)

def predict(df):
 
    X, df_proc = preProcess(df)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba > THRESHOLD).astype(int)

    df_proc["pred_attack"] = pred
    df_proc["attack_probability"] = proba

    return df_proc


