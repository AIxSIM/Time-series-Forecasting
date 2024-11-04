import pandas as pd
import os
from datetime import datetime


def merge_and_sort_by_datetime(folder_path="datasets", file_suffix="1860217400.csv",
                               output_file="merged_sorted_data.csv"):
    # List all files in the folder that end with the specific ID
    files_to_merge = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(file_suffix)]

    # Read and store each file as a DataFrame in a list
    dfs = [pd.read_csv(file) for file in files_to_merge]

    # Concatenate all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)

    # Convert 'datetime' column to datetime format
    merged_df['datetime'] = pd.to_datetime(merged_df['datetime'], format='%Y%m%d%H%M')

    # Sort by the 'datetime' column
    merged_df = merged_df.sort_values(by='datetime').reset_index(drop=True)

    # Save the merged and sorted DataFrame to a CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"File saved to {output_file}")


# Example usage

VDS_ID = '1860217400'
merge_and_sort_by_datetime(folder_path="datasets", file_suffix=VDS_ID+".csv",
                               output_file="datasets/MRT_TF_INFO_"+VDS_ID+".csv")
