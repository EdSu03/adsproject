#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import glob
from tqdm import tqdm  # Progress bar
import re

def clean_parquet_files_yellow_taxi(input_dir="data", output_dir="cleaned_data"):
    """
    This function processes multiple Parquet files from the 'data' folder by applying data cleaning steps.
    Cleaned files are saved to the 'cleaned_data' directory.
    """

    # -------------------------
    # 1. Ensure output directory exists
    # -------------------------
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # 2. Find all Parquet files in the input directory
    # -------------------------
    file_list = glob.glob(os.path.join(input_dir, "yellow_tripdata_2024-*.parquet"))

    if not file_list:
        print("❌ No files found in './data'. Please check the folder and file names!")
        return

    print(f"🔍 Found {len(file_list)} files in '{input_dir}/'. Starting data cleaning...\n")

    # -------------------------
    # 3. Process all files with a single progress bar
    # -------------------------
    with tqdm(total=len(file_list), desc="Cleaning Progress", unit="file") as pbar:

        for file_path in file_list:

            # -------------------------
            # 3.1 Read Parquet file and convert all column names to lowercase
            # -------------------------
            df = pd.read_parquet(file_path, engine="pyarrow")  # Use "fastparquet" if needed
            df.columns = df.columns.str.lower()
                        
            # -------------------------
            # 3.2 Handle missing values
            # -------------------------
            df["passenger_count"] = df["passenger_count"].fillna(1) # Set NaN to default 1
            df["trip_distance"] = df["trip_distance"].fillna(0)
            df.fillna(0, inplace=True)

            # -------------------------
            # 3.3 Remove unnecessary columns
            # -------------------------
            if "store_and_fwd_flag" in df.columns:
                df.drop(columns=["store_and_fwd_flag"], inplace=True)
            if "ratecodeid" in df.columns:
                df.drop(columns=["ratecodeid"], inplace=True)

            # -------------------------
            # 3.4 Convert datetime columns
            # -------------------------
            df["pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
            df["dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
            df.drop(columns=["tpep_pickup_datetime", "tpep_dropoff_datetime"], inplace=True)
            
            # -------------------------
            # 3.4.5 Filter rows with incorrect year or month based on file name
            # -------------------------
            filename = os.path.basename(file_path)
            match = re.search(r"yellow_tripdata_(\d{4})-(\d{2})\.parquet", filename)
            if match:
                expected_year = int(match.group(1))
                expected_month = int(match.group(2))
                df = df[
                    (df["pickup_datetime"].dt.year == expected_year) &
                    (df["pickup_datetime"].dt.month == expected_month)
                ]
            else:
                print(f"Warning: Filename {filename} doesn't match expected format. Date filtering skipped.")
                
            # -------------------------
            # 3.5 Calculate trip duration in minutes
            # -------------------------
            df["trip_duration"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60

            # -------------------------
            # 3.6 Remove extreme values
            # -------------------------
            df = df[df["trip_distance"] >= 0.05] # Remove trips with distance < 0.05 miles
            df = df[df["fare_amount"] > 0]  # Remove negative fare amounts
            df = df[(df["passenger_count"] > 0)]  # Keep passenger count bigger than 0
            df = df[(df["trip_duration"] >= 1) & (df["trip_duration"] <= 180)]  # Keep trip duration (1-180 mins)
            df = df[df['payment_type'].isin([1, 2, 3, 4, 5, 6])] #The possible payment types
            
            # Remove extremely high trip distances
            df = df[df["trip_distance"] <= 50]  # Only keep trips ≤ 50 miles
            
            # Define reasonable upper limits for monetary values
            fare_limit = df["fare_amount"].quantile(0.99)  # 99% quantile
            tip_limit = df["tip_amount"].quantile(0.99)
            total_limit = df["total_amount"].quantile(0.99)

            # Apply filtering
            df = df[
                (df["fare_amount"] <= fare_limit) &
                (df["tip_amount"] <= tip_limit) &
                (df["total_amount"] <= total_limit)
            ]
            
            # -------------------------
            # 3.6.5 Ensure monetary values appart from fare_amountare non-negative 
            # -------------------------
            monetary_columns = [
                'extra', 'mta_tax', 'improvement_surcharge', 'tip_amount',
                'tolls_amount', 'total_amount', 'congestion_surcharge', 'airport_fee'
             ]
            df = df[(df[monetary_columns] >= 0).all(axis=1)]
            # -------------------------
            # 3.7 Drop duplicate rows
            # -------------------------
            df.drop_duplicates(inplace=True)

            # -------------------------
            # 3.8 Rename columns for better readability (CamelCase) and rearrange columns
            # -------------------------
            df.rename(columns={
                "pickup_datetime": "PickupDatetime",
                "dropoff_datetime": "DropoffDatetime",
                "vendorid": "VendorID",
                "trip_duration": "TripDuration",
                "passenger_count": "PassengerCount",
                "trip_distance": "TripDistance",
                "pulocationid": "PULocationID",
                "dolocationid": "DOLocationID",
                "payment_type": "PaymentType",
                "fare_amount": "FareAmount",
                "extra": "ExtraCharges",
                "mta_tax": "MTATax",
                "tip_amount": "TipAmount",
                "tolls_amount": "TollsAmount",
                "improvement_surcharge": "ImprovementSurcharge",
                "total_amount": "TotalAmount",
                "congestion_surcharge": "CongestionSurcharge",
                "airport_fee": "AirportFee"
            }, inplace=True)
            
            df = df[[
                "VendorID", "PickupDatetime", "DropoffDatetime", "TripDuration",
                "PassengerCount", "TripDistance", "PULocationID", "DOLocationID",
                "PaymentType", "FareAmount", "ExtraCharges", "MTATax", "TipAmount",
                "TollsAmount", "ImprovementSurcharge", "TotalAmount",
                "CongestionSurcharge", "AirportFee"
            ]]

            # -------------------------
            # 3.9 Save cleaned file in the output directory
            # -------------------------
            output_filename = "cleaned_" + os.path.basename(file_path)
            output_path = os.path.join(output_dir, os.path.basename(output_filename))
            df.to_parquet(output_path, index=False)

            # Update progress bar
            pbar.update(1)

    print("\n🎉 Yellow Taxi data successfully cleaned and saved in 'cleaned_data/'!")

# -------------------------
# 4. Run the function
# -------------------------
clean_parquet_files_yellow_taxi()


# In[41]:


import os
import pandas as pd
import glob
from tqdm import tqdm  # Progress bar
import re

def clean_parquet_files_green_taxi(input_dir="data", output_dir="cleaned_data"):
    """
    This function processes multiple Green Taxi Parquet files from the 'data' folder by applying data cleaning steps.
    Cleaned files are saved to the 'cleaned_data' directory.
    """

    # -------------------------
    # 1. Ensure output directory exists
    # -------------------------
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # 2. Find all Green Taxi Parquet files in the input directory
    # -------------------------
    file_list = glob.glob(os.path.join(input_dir, "green_tripdata_2024-*.parquet"))

    if not file_list:
        print("❌ No Green Taxi files found in './data'. Please check the folder and file names!")
        return

    print(f"🔍 Found {len(file_list)} Green Taxi files in '{input_dir}/'. Starting data cleaning...\n")

    # -------------------------
    # 3. Process all files with a single progress bar
    # -------------------------
    with tqdm(total=len(file_list), desc="Cleaning Green Taxi Data", unit="file") as pbar:

        for file_path in file_list:

            # -------------------------
            # 3.1 Read Parquet file and convert all column names to lowercase
            # -------------------------
            df = pd.read_parquet(file_path, engine="pyarrow")  
            df.columns = df.columns.str.lower()
                        
            # -------------------------
            # 3.2 Handle missing values
            # -------------------------
            df["passenger_count"] = df["passenger_count"].fillna(1)  # Default to 1
            df["trip_distance"] = df["trip_distance"].fillna(0)
            df.fillna(0, inplace=True)

            # -------------------------
            # 3.3 Remove unnecessary columns and add df["airport_fee"] = 0 to be consistent with yellow taxi
            # -------------------------
            drop_columns = ["store_and_fwd_flag", "ehail_fee", "trip_type"]
            df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)
            
            df["airport_fee"] = 0

            # -------------------------
            # 3.4 Convert datetime columns
            # -------------------------
            df["pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
            df["dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
            df.drop(columns=["lpep_pickup_datetime", "lpep_dropoff_datetime"], inplace=True)
            
            # -------------------------
            # 3.4.5 Filter rows with incorrect year or month based on file name
            # -------------------------
            filename = os.path.basename(file_path)
            match = re.search(r"green_tripdata_(\d{4})-(\d{2})\.parquet", filename)
            if match:
                expected_year = int(match.group(1))
                expected_month = int(match.group(2))
                df = df[
                    (df["pickup_datetime"].dt.year == expected_year) &
                    (df["pickup_datetime"].dt.month == expected_month)
                ]
            else:
                print(f"⚠️ Warning: Filename {filename} doesn't match expected format. Date filtering skipped.")
                
            # -------------------------
            # 3.5 Calculate trip duration in minutes
            # -------------------------
            df["trip_duration"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60

            # -------------------------
            # 3.6 Remove extreme values
            # -------------------------
            df = df[(df["trip_distance"] >= 0.05) & (df["trip_distance"] <= 50)]  # Keep trips within 0.05 - 50 miles
            df = df[df["fare_amount"] > 0]  # Remove negative fares
            df = df[(df["passenger_count"] > 0)]  # Remove invalid passenger counts
            df = df[(df["trip_duration"] >= 1) & (df["trip_duration"] <= 180)]  # Keep trip duration (1-180 mins)
            df = df[df['payment_type'].isin([1, 2, 3, 4, 5, 6])]  # Valid payment types

            # -------------------------
            # 3.6.5 Handle monetary value outliers
            # -------------------------
            fare_limit = df["fare_amount"].quantile(0.99)  
            tip_limit = df["tip_amount"].quantile(0.99)
            total_limit = df["total_amount"].quantile(0.99)

            df = df[
                (df["fare_amount"] <= fare_limit) &
                (df["tip_amount"] <= tip_limit) &
                (df["total_amount"] <= total_limit)
            ]

            monetary_columns = [
                'extra', 'mta_tax', 'improvement_surcharge', 'tip_amount',
                'tolls_amount', 'total_amount', 'congestion_surcharge'
            ]
            df = df[(df[monetary_columns] >= 0).all(axis=1)]

            # -------------------------
            # 3.7 Drop duplicate rows
            # -------------------------
            df.drop_duplicates(inplace=True)

            # -------------------------
            # 3.8 Rename and reorder columns
            # -------------------------
            df.rename(columns={
                "pickup_datetime": "PickupDatetime",
                "dropoff_datetime": "DropoffDatetime",
                "vendorid": "VendorID",
                "trip_duration": "TripDuration",
                "passenger_count": "PassengerCount",
                "trip_distance": "TripDistance",
                "pulocationid": "PULocationID",
                "dolocationid": "DOLocationID",
                "payment_type": "PaymentType",
                "fare_amount": "FareAmount",
                "extra": "ExtraCharges",
                "mta_tax": "MTATax",
                "tip_amount": "TipAmount",
                "tolls_amount": "TollsAmount",
                "improvement_surcharge": "ImprovementSurcharge",
                "total_amount": "TotalAmount",
                "congestion_surcharge": "CongestionSurcharge",
                "airport_fee": "AirportFee"
            }, inplace=True)
            
            df = df[[
                "VendorID", "PickupDatetime", "DropoffDatetime", "TripDuration",
                "PassengerCount", "TripDistance", "PULocationID", "DOLocationID",
                "PaymentType", "FareAmount", "ExtraCharges", "MTATax", "TipAmount",
                "TollsAmount", "ImprovementSurcharge", "TotalAmount",
                "CongestionSurcharge", "AirportFee"
            ]]

            # -------------------------
            # 3.9 Save cleaned file in the output directory
            # -------------------------
            output_filename = "cleaned_" + os.path.basename(file_path)
            output_path = os.path.join(output_dir, os.path.basename(output_filename))
            df.to_parquet(output_path, index=False)

            # Update progress bar
            pbar.update(1)

    print("\n🎉 Green Taxi data successfully cleaned and saved in 'cleaned_data/'!")

# -------------------------
# 4. Run the function
# -------------------------
clean_parquet_files_green_taxi()


# In[ ]:




