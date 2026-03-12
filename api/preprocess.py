import pandas as pd
 
# Add a few extra parameters that helps classification
def add_extra_features(df):
    df['doppler_prrate_nonlinear'] = (df['doppler'] - df['pr_rate'])**2
    df['snr_z'] = (df['snr'] - df['snr_mean_5']) / (df['snr_std_5'] + 1e-6)
    return df


def preProcess(df):
    df = df.copy()

    # handling time
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    # Clipping anomalies/too large values
    df['doppler_vs_prrate'] = df['doppler_vs_prrate'].clip(-2000, 2000)

    # add_extra_features
    df = add_extra_features(df)

    #loading label encoders:
    df["sys"] = df["sys"].map({"G": 0, "R": 1, "E": 2})

    # Fill missing
    df = df.fillna(-9999)

    drop_cols = ["time", "attack_label", "attack_type", "time_utc","sv", "prn"]
    X = df.drop(columns=drop_cols, errors="ignore")

    return X, df