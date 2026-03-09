
from rinex_conversion import RinexToCSV_Converter
import spoofing_simulation as sp

# instead of the example files insert your files

if __name__ == "__main__":
    converter = RinexToCSV_Converter(obs_path ="examplefile.obs", nav_path="examplefile.nav")

    # Convert to DataFrame for inspection
    obs = converter.load_obs(converter.obs_path)
    df = converter.extract_features(obs)
    df_obs = converter.add_derived_features(df)

    df_obs.to_csv("obs_example.csv", index=False)

    # Convert to DataFrame for inspection
    nav_ds = converter.load_nav(converter.nav_path)
    df_nav = nav_ds.to_dataframe().reset_index()
    df_nav.to_csv("nav_example.csv", index=False)


    # Creating spoofed samples example with 'spoofing_simulation.py'
    # instead of the example files insert your files
    converter2 = RinexToCSV_Converter(obs_path ="examplefile2.obs", nav_path="examplefile2.nav")
    obs2 = converter2.load_obs(converter.obs_path)
    nav_ds2 = converter2.load_nav(converter.nav_path)

    df2 = converter2.extract_features(obs2)
    df2 = sp.add_common_offset(df2, "2025-10-27 21:22:40", "2025-10-27 21:23:40", 180, sats=None)
    df2 = sp.add_common_offset(df2, "2025-10-27 21:25:00", "2025-10-27 21:26:00", 120, sats=None)
    df2 = sp.add_common_offset(df2, "2025-10-27 21:32:00", "2025-10-27 21:33:00", 150, sats=None)
    df2 = sp.change_snr(df2, "2025-10-27 21:30:00", "2025-10-27 21:31:20", 15, sats=None)
    df2 = sp.change_snr(df2, "2025-10-27 21:34:00", "2025-10-27 21:35:20", 9, sats=None)
    df2 = sp.inject_doppler_offset(df2, "2025-10-27 21:16:00", "2025-10-27 21:21:20", 2, sats=None)
    df2 = sp.inject_doppler_offset(df2, "2025-10-27 21:18:00", "2025-10-27 21:19:20", 4, sats=None)
    df2_attack = converter.add_derived_features(df2)

    df2_attack.to_csv("examplefile_spoofed.csv", index=False)