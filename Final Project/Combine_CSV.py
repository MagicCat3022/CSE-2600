import pandas as pd
import os
from pathlib import Path

def combine_csv_files(file_list, output_file):
    combined_df = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)
    combined_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    Target_folder = r"C:\Users\AHMET\Documents\GitHub\CSE-2600\Final Project\Data\CSV"
    files = list(Path(Target_folder).glob("*.csv"))
    output_file = Target_folder + r"\CSE2600_Final_Data.csv"
    combine_csv_files(files, output_file)