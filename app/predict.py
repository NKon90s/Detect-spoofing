import pandas as pd 
import joblib
from preprocess import preProcess

MODEL_PATH = "gnss_xgboost_model.joblib"
THRESHOLD = 0.4

def predict(csv_path, out_path):
    df = pd.read_csv(csv_path)

    X, df_proc = preProcess(df)

    model = joblib.load(MODEL_PATH)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba > THRESHOLD).astype(int)

    df_proc["pred_attack"] = pred
    df_proc["attack_probability"] = proba

    df_proc.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    predict("obs_Oct21log8_spoofed.csv", "predictions.csv")
