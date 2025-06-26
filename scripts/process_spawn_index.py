import pandas as pd

def load_and_process_spawn_data(input_path, output_path):
    # Load raw data
    df = pd.read_csv(input_path)

    # Filter for Strait of Georgia region
    df_sog = df[df["Region"] == "SoG"]

    # Group by year and sum the CombinedSI values
    yearly_index = df_sog.groupby("Year")["CombinedSI"].sum().reset_index()
    yearly_index.columns = ["Year", "Total_CombinedSI"]

    # Save to processed folder
    yearly_index.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    input_file = "../data/raw/Pacific_Herring_Spawn_Index_Data_1951-2021.csv"
    output_file = "../data/processed/sog_spawn_index.csv"
    load_and_process_spawn_data(input_file, output_file)