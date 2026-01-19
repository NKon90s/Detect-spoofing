
from typing import Optional
import numpy as np
import pandas as pd
import georinex as gr
import re

class RinexToCSV_Converter:

    def __init__(self, obs_path: str, nav_path: Optional[str] = None):
        """
        Initialize the RINEX converter with observation and navigation files.
        """
        self.obs_path = obs_path
        self.nav_path = nav_path
        self.obs_ds = None
        self.nav_ds = None

    def load_obs(self, obs_path):
        print(f"[+] Loading RINEX observation: {obs_path}")
        obs = gr.load(
            obs_path,
            use=['G', 'R', 'E'],
            meas=['C1C', 'L1C', 'D1C', 'S1C', 'C1X', 'L1X', 'D1X', 'S1X']
        )
        available = list(obs.data_vars.keys())
        print(f"Loaded observables: {available}")
        if not any(k.startswith('C1X') for k in available):
            print("No Galileo data found — proceeding with GPS/GLONASS only.")
        return obs


    def get_system_obs_names(self, sys_letter):
        """
        Return observation variable names for each system.

        Assumes consistent RINEX naming from RTKCONV:
        G/R : C1C, L1C, D1C, S1C
        E   : C1X, L1X, D1X, S1X
        """
        if sys_letter == 'E':   # Galileo
            return 'C1X', 'L1X', 'D1X', 'S1X'
        else:                   # GPS or GLONASS
            return 'C1C', 'L1C', 'D1C', 'S1C'
        
    
        
    def extract_features(self, obs_ds):
        """
        Convert the RINEX xarray dataset into a flat DataFrame.
        Each row = (time, satellite) pair with observables.
        """
        times = obs_ds.time.values
        sv_list = list(obs_ds.sv.values)
        rows = []

        for sv in sv_list:
            sv_str = sv.decode() if isinstance(sv, bytes) else str(sv)
            sys = sv_str[0]  #e.g: 'G', 'R', 'E'
            prn = sv_str[1:] # numeric part of satellite ID
            pr_var, ph_var, dp_var, sn_var = self.get_system_obs_names(sys)

            # safe access function
            def safe(var):
                return obs_ds[var].sel(sv=sv).values if var in obs_ds else np.full(len(times), np.nan)

            pr_arr = safe(pr_var)
            ph_arr = safe(ph_var)
            dp_arr = safe(dp_var)
            sn_arr = safe(sn_var)

            # append rows by epoch
            for i, t in enumerate(times):
                rows.append({
                    "time_utc": np.datetime_as_string(t, unit="s"),
                    "time": pd.to_datetime(np.datetime_as_string(t, unit="s")),
                    "sys": sys,
                    "sv": sv_str,
                    "prn": prn,
                    "pseudorange": float(pr_arr[i]) if not np.isnan(pr_arr[i]) else np.nan,
                    "phase": float(ph_arr[i]) if not np.isnan(ph_arr[i]) else np.nan,
                    "doppler": float(dp_arr[i]) if not np.isnan(dp_arr[i]) else np.nan,
                    "snr": float(sn_arr[i]) if not np.isnan(sn_arr[i]) else np.nan,
                })

        df = pd.DataFrame(rows)
        print(f"[+] Extracted {len(df)} total rows for {len(df['sv'].unique())} satellites")
        return df


    def add_derived_features(self, df):
        """Add derived features to the DataFrame for anomaly detection."""

        df2 = df.copy()
        df2["time"] = pd.to_datetime(df2["time"])
        df2["time_s"] = df["time"].astype("int64") // 10**9

        # vectorized diff calculation per satellite
        df2["delta_t"] = df2.groupby("sv")["time_s"].diff()           # seconds
        df2["delta_pr"] = df2.groupby("sv")["pseudorange"].diff()     # meters

        # pseudorange rate (m/s) - derivative of pseudorange
        df2["pr_rate"] = df2["delta_pr"] / df2["delta_t"]
        df2.loc[~np.isfinite(df2["pr_rate"]), "pr_rate"] = np.nan

        lambda_L1 = 0.190293672798365  # L1 wavelength in meters for GPS and Galileo
        fcn_dict = self.extract_glonass_fcn(self.obs_path)  # PRN to FCN mapping for GLONASS

        df2['wavelength'] = df2.apply(
            lambda row: lambda_L1 if row['sys'] != 'R' else self.glonass_wavelength(fcn_dict.get(row['sv'],0)),
            axis=1)

        # Dopple vs Pseudorange-Rate consistency feature
        df2["doppler_vs_prrate"] = (df2[ "doppler"] * df2['wavelength']) + df2["pr_rate"]
    
        # SNR rolling stats (window=5 epochs)
        df2["snr_mean_5"] = (df2.groupby("sv")["snr"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True))
        df2["snr_std_5"] = (df2.groupby("sv")["snr"].rolling(5, min_periods=1).std().reset_index(level=0, drop=True))

        # contextual features: satellite count and missing pseudoranges per epoch
        epoch_stats = (df2.groupby("time").agg(sat_count=("sv", "count"), n_missing_pr=("pseudorange", lambda x: x.isna().sum())))

        df2 = df2.merge(epoch_stats, left_on="time", right_index=True, how="left")
        return df2

<<<<<<< HEAD

=======
<<<<<<< HEAD

=======
 
>>>>>>> 0f3322b (Added rinex conversion)
>>>>>>> 674b53d (Added rinex conversion)
    # Load the NAV file for potential future use
    def load_nav(self, nav_path):
        print(f"Loading RINEX navigation: {nav_path}")
        if nav_path is None:
            print("No navigation file provided.")
            return None
        else:
            nav_ds = gr.load(nav_path)
        return nav_ds
    

    def extract_glonass_fcn(self, obs_path):
        fcn_dict = {}
        with open(obs_path, 'r') as f:
            for line in f:
                if "GLONASS SLOT / FRQ #" in line:
                    # extract PRN and FCN pairs with regex
                    matches = re.findall(r'(R\d{1,2})\s+(-?\d+)', line)
                    for prn, fcn in matches:
                        fcn_dict[prn] = int(fcn)
                elif "END OF HEADER" in line:
                    break  # stop after header
        return fcn_dict
    
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
    # Calculate wavelenght for GLONASS
>>>>>>> 0f3322b (Added rinex conversion)
>>>>>>> 674b53d (Added rinex conversion)
    def glonass_wavelength(self, K):
        c = 299792458.0
        f0 = 1602.0e6
        df = 0.5625e6
        f = f0 + K * df
        return c / f




if __name__ == "__main__":
    converter = RinexToCSV_Converter(obs_path ="Oct27log1.obs", nav_path="Oct27log1.nav")
    obs = converter.load_obs(converter.obs_path)
    df = converter.extract_features(obs)
    df_obs = converter.add_derived_features(df)

    df_obs.to_csv("obs_Oct27log1.csv", index=False)

    # Convert to DataFrame for inspection
    nav_ds = converter.load_nav(converter.nav_path)
    df_nav = nav_ds.to_dataframe().reset_index()
    df_nav.to_csv("nav_Oct27log1.csv", index=False)
  




