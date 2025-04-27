import pandas as pd


import pandas as pd
def clean_yellow_taxi_data(df, year, month=None):


    #datetime columns are in proper format
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    #rmeove rows with invalid datetime
    #df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    #filter by year and optionally by month
    df = df[df['tpep_pickup_datetime'].dt.year == year]
    if month:
        df = df[df['tpep_pickup_datetime'].dt.month == month]


    #df["passenger_count"] = df["passenger_count"].fillna(1) 
    #df["trip_distance"] = df["trip_distance"].fillna(0)
    #df.fillna(0, inplace=True)

    #VendorID (should be 1 or 2)
    df = df[df['VendorID'].isin([1, 2])]

    #Passenger_count (>= 1)
    df = df[df['passenger_count'] > 0]

    #Trip_distance (> 0)
    df = df[df['trip_distance'] > 0]

    #RateCodeID (1-6)
    df = df[df['RatecodeID'].isin([1, 2, 3, 4, 5, 6])]

    #Store_and_fwd_flag ('Y' or 'N')
    df = df[df['store_and_fwd_flag'].isin(['Y', 'N'])]

    #Payment_type (1-6)
    df = df[df['payment_type'].isin([1, 2, 3, 4, 5, 6])]

    #Ensure monetary values are non-negative
    monetary_columns = [
        'fare_amount', 'extra', 'mta_tax', 'improvement_surcharge', 'tip_amount',
        'tolls_amount', 'total_amount', 'congestion_surcharge'
    ]
    df = df[(df[monetary_columns] >= 0).all(axis=1)]

    df.drop_duplicates(inplace=True)

    if month:
        output_filename = f"yellow_taxi_tripdata_{year}_{month}_clean.parquet"
    else:
        output_filename = f"yellow_taxi_tripdata_{year}_clean.parquet"


    df.to_parquet(output_filename)

    print(f"Cleaned file saved as {output_filename}")

if __name__ == '__main__':
    #Read the Parquet file into a DataFrame

    df = pd.read_parquet("yellow_taxi_tripdata_2024_combined.parquet", engine="pyarrow")
    #print(df.columns)
    clean_yellow_taxi_data(df, 2024)
    #print("Minimum date:", df["tpep_pickup_datetime"].min())
    # Apply function to create a new column for half-hour intervals
    #df["pickup_half_hour"] = df["tpep_pickup_datetime"].apply(round_to_half_hour)
    # Count number of pickups for each half-hour interval
    #demand = df.groupby("pickup_half_hour").size().reset_index(name="pickup_count")
    #demand.set_index("pickup_half_hour", inplace=True)
    #vis_ts(demand)