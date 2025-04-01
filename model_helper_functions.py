import pandas as pd
from zone_coords import get_coords_as_dict
import math
from datetime import datetime, timedelta
from dateutil import parser

def necessary_fields(df : pd.DataFrame) -> pd.DataFrame: 
    return df.drop(columns=['VendorID',
                            'PassengerCount',
                            'PaymentType', 
                            'ExtraCharges', 
                            'MTATax', 
                            'TollsAmount', 
                            'ImprovementSurcharge', 
                            'TotalAmount', 
                            'CongestionSurcharge', 
                            'AirportFee']) # Keep Trip Distance and Duration as these may be helpful for the simulation.

def merge_fare_and_tip(df: pd.DataFrame) -> pd.DataFrame: # Add together fare and tip amount into one revenue.
    df_merged = df
    df_merged['TotalRevenue'] = df['FareAmount'] + df['TipAmount']
    return df_merged.drop(columns=['FareAmount', 'TipAmount'])

def centroid_distances(PUZoneID, DOZoneID): # This is really just for use as a distance measure in the mean time 
                                            # until Oliver sorts out the actual shortest paths between zones. 
    lookup_coords = get_coords_as_dict()
    PUx = lookup_coords[PUZoneID]['x']
    PUy = lookup_coords[PUZoneID]['y']
    DOx = lookup_coords[DOZoneID]['x']
    DOy = lookup_coords[DOZoneID]['y']
    return math.sqrt((DOx - PUx)**2 + (DOy - PUy)**2)

def round_time_to_hour(tim): # Round datetime to nearest hour (less useful than to an int since we'll want a numerical value)
    tim = pd.to_datetime(tim)
    if tim.minute >= 30:
        return tim.replace(minute = 0, second = 0, microsecond=0) + timedelta(hours=1)
    else:
        return tim.replace(minute = 0, second = 0, microsecond=0)
    
def round_time_to_int(tim: pd.Timestamp) -> int: # Round hour to an integer for use in XGBoost or Q-Learning
    tim = pd.to_datetime(tim)
    if tim.minute >= 30:
        return int(tim.hour) + 1
    else:
        return int(tim.hour)