# packages
import pandas as pd


def get_df() -> pd.DataFrame:
    ## url for coords: REMEMBER TO REFERENCE IN REPORT!!!
    url = "https://gist.githubusercontent.com/shwina/72d79165ce9605d8f6e3378ae717b16b/raw/84a47bc587c99c6736f38a97f9dcc32ba8f89b05/taxi_zones.csv"
    return pd.read_csv(url)


def get_coords_as_df() -> pd.DataFrame:  # returns a dataframe
    df = get_df()

    # OBJECTID and LocationID are the same, unnamed field also unnecessary:
    df = df.drop(["Unnamed: 0", "OBJECTID"], axis=1)

    # Don't care for Length or Area of zones (but if we do can always remove these lines later):
    df = df.drop(["Shape_Leng", "Shape_Area"], axis=1)

    # Reorder df => new order = LocationID, borough, zone, x, y
    df = df.loc[:, ["LocationID", "borough", "zone", "x", "y"]]
    return df


# Define co-ordinates as a dictionary of string -> float
Coords = dict[str, float]  # For example {'x': 1.234, 'y': 5.678}


def get_coords_as_dict() -> (
    dict[str, Coords]
):  # returns a dictionary to map (x,y)-coordinates to a zone name
    df = get_df()

    # Remove all columns but zone, x and y:
    df = df.drop(
        ["Unnamed: 0", "OBJECTID", "Shape_Leng", "Shape_Area", "LocationID", "borough"],
        axis=1,
    )

    # Set zone to index (necessary for to_dict to recognize zone as the key)
    df = df.set_index('zone')
    
    return df.T.to_dict()

def get_coord_as_csv():
    df = get_df()

    df = df[['LocationID', 'x', 'y']]
    df.columns = ['LocationID', 'Longitude', 'Latitude']

    df.to_csv('zone_coords.csv', index=False)

    df = df.set_index("zone")

    return df.T.to_dict()

