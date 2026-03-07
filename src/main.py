
from rinex_conversion import RinexToCSV_Converter

# instead of the files insert your files
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