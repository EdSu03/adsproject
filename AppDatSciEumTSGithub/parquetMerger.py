import glob

import pandas as pd






def parquetMerger1(year, input_dir="cleaned_data"):

    file_paths = glob.glob(f"{input_dir}/*{year}*.parquet")
    #file_paths = glob.glob(f"cleaned_data/*.parquet")

    dfs = [pd.read_parquet(fp) for fp in file_paths]

    merged_df = pd.concat(dfs, ignore_index=True)

    merged_df.to_parquet(f"data/yellow_taxi_tripdata_{year}_combined.parquet")

def parquetMerger2(years, input_dir="cleaned_data"):
    file_paths = []
    for year in years:
        file_paths.append(glob.glob(f"{input_dir}/*{year}*.parquet"))
    #file_paths = glob.glob(f"cleaned_data/*.parquet")


    dfs = [pd.read_parquet(fp) for fp in file_paths]


    merged_df = pd.concat(dfs, ignore_index=True)


    merged_df.to_parquet(f"yellow_taxi_tripdata_cleaned_combined.parquet")


if __name__ == "__main__":
    #parquetMerger(2024)
    years = [2022, 2023]
    parquetMerger2(years)