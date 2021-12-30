import pandas as pd
from haversine import haversine

from src.utils.time import robust_hour_of_iso_date


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df


def driver_historical_completed_bookings(df: pd.DataFrame) -> pd.DataFrame:
    #raise NotImplementedError(
    #    f"Show us your feature engineering skills! Suppose that drivers with a good track record are more likely to accept bookings. "
    #    f"Implement a feature that describes the number of historical bookings that each driver has completed."    
    #)
    #here, we need to get updated historical trip completion rate per driver. we need to ensure we dont leak future trip completion
    #data per driver, so we have to calculate this in a cumulative fashion using all the data available uptil the given booking. we 
    #will have to exclude the current trip completion data from the calculation as well. we will also update the config.toml file 
    #to include our new feature. we may also want to explore the exclusion of customer cancelled trips to improve driver ratings
    df.sort_values(by='event_timestamp',inplace = True)
    df['historical_trip_count'] = df.groupby(['driver_id'])['is_completed'].apply(lambda x: x.shift().cumsum()).fillna(0)
    df['temp'] = df.groupby('driver_id')['is_completed'].cumsum()
    df['historical_trip_count'] = df.groupby('driver_id')['temp'].shift().fillna(0)
    df = df.drop(columns=["temp"])
    return df