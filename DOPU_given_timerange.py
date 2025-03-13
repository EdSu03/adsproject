import os
import pandas as pd
from datetime import datetime, time

#Run this cell to get ONE parquet file of ALL cleaned data stored in the folder all_cleaned_data.

def get_cleaned_df():
    folder_path = "data/"
    
    os.makedirs("all_cleaned_data", exist_ok=True)
    
    # Get all parquet files in the folder
    parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
    
    # Load and concatenate them into a single DataFrame
    dataframes = [pd.read_parquet(os.path.join(folder_path, file)) for file in parquet_files]
    
    # Combine all dataframes
    df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
    
    df.to_parquet("all_cleaned_data/all_cleaned_data.parquet", index=False)

#return all cleaned data file as a dataframe
#df = pd.read_parquet("all_cleaned_data/all_cleaned_data.parquet")

def DOPU_given_timerange(df, zone, start, end, isDropoff=True):
    """
    input: 
        df: dataframe to extract number of dropoffs from
        zone: integer zone id
        start: time of start (pandas time object)
        end: time of end (pandas time object)
        isDropoff: bool. If true, 
        
    returns:
        integer- average number of dropoffs/pickups made in the zone during the time range
    """
    
    
    #given a zone id, a start time, and end time, find the average number of dropoffs/pickups made during this time
    df_zone = df[df["DOLocationID"] == zone].copy() if isDropoff else df[df["PULocationID"] == zone].copy()
    
    #remove date from datetime object
    df_zone["time"] = df_zone["tpep_dropoff_datetime"].dt.time if isDropoff else df_zone["tpep_pickup_datetime"].dt.time
    
    #filter by times within the given time range
    df_filtered = df_zone[(df_zone["time"] >= start) & (df_zone["time"] <= end)]
    
    #sum all dropoff/pickups
    dropoffs_per_day = df_filtered.groupby(df_filtered["tpep_dropoff_datetime"].dt.date).size()
    
    #return mean dropoffs/pickups
    return int(dropoffs_per_day.mean()) if not dropoffs_per_day.empty else 0

def DOPU_given_one_timerange(df, zone, time, timerange, isDropoff=True):
    """
    input: 
        df: dataframe to extract number of dropoffs from
        zone: integer zone id
        time: datetime.time object- the time to check
        timerange: pd.Deltatime object- the timerange to check within
        isDropoff: bool. If true, returns num of dropoffs. If false, returns num of pickups
        
    returns:
        integer- average number of dropoffs/pickups made in the zone during the time range
    """
    
    #given a zone id, a start time, and end time, find the average number of dropoffs/pickups made during this time
    df_zone = df[df["DOLocationID"] == zone].copy() if isDropoff else df[df["PULocationID"] == zone].copy()
    
    ref_date = datetime.combine(datetime.today(), time)
    #remove date from datetime object
    df_zone["time"] = df_zone["tpep_dropoff_datetime"].dt.time if isDropoff else df_zone["tpep_pickup_datetime"].dt.time
    
    start = (ref_date - timerange).time()
        
    end = (ref_date + timerange).time()
    
    #filter by times within the given time range

    df_filtered = df_zone[(df_zone["time"] >= start) & (df_zone["time"] <= end)] if (start < end) else df_zone[(df_zone["time"] >= start) | (df_zone["time"] <= end)]
    
    #sum all dropoff/pickups
    dropoffs_per_day = df_filtered.groupby(df_filtered["tpep_dropoff_datetime"].dt.date).size()
    
    #return mean dropoffs/pickups
    return int(dropoffs_per_day.mean()) if not dropoffs_per_day.empty else 0