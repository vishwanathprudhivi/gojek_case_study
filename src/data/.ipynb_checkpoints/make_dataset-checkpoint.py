import pandas as pd

from src.utils.config import load_config
from src.utils.store import AssignmentStore


def main():
    store = AssignmentStore()
    config = load_config()

    booking_df = store.get_raw("booking_log.csv")
    booking_df = clean_booking_df(booking_df)

    participant_df = store.get_raw("participant_log.csv")
    participant_df = clean_participant_df(participant_df)

    dataset = merge_dataset(booking_df, participant_df)
    dataset = create_target(dataset, config["target"])

    store.put_processed("dataset.csv", dataset)


def clean_booking_df(df: pd.DataFrame) -> pd.DataFrame:
    #we need to keep booking status,driver id as a unique column as well
    unique_columns = [
        "order_id",
        "trip_distance",
        "pickup_latitude",
        "pickup_longitude",
        "booking_status",
        "driver_id"
    ]
    df = df.drop_duplicates(subset=unique_columns)
    #keep only booking_status = 'COMPLETED','CUSTOMER_CANCELLED','DRIVER_CANCELLED','DRIVER_FOUND'
    df = df[df.booking_status.isin(['COMPLETED','CUSTOMER_CANCELLED','DRIVER_CANCELLED'])]
    
    #dropping invalid event timestamps
    df = df.dropna(subset = ['event_timestamp'])
    
    #change here to return entire dataset, not just the primary keys
    return df


def clean_participant_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    #keep only participant_status = 'ACCEPTED','IGNORED','REJECTED'
    df = df[df.participant_status.isin(['ACCEPTED','REJECTED'])]
    return df


def merge_dataset(bookings: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    #the purpose of the merge is to ensure we have information on driver location from the customer.
    #also, since we are trying to optimize for trip completions,
    #we need to define our target as trip completed or not, instead of driver accepting ride. this is because even after the
    #driver accepts the ride, a customer may cancel the booking at any time if the driver is too far away. 
    #we need to merge filtered participant data to the filtered bookings (left table) data
    #df = pd.merge(participants, bookings, on="order_id", how="left")
    df = pd.merge(bookings, participants.drop(columns = ['driver_id','event_timestamp','experiment_key']), on="order_id", how="left")
    
    #drop instances where customer cancelled before driver was assigned
    df = df[~((df.booking_status == 'CUSTOMER_CANCELLED') & (df.participant_status.isnull()))]
    
    
    return df


def create_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    #we need to change the way the target is being calculated. we need to predict succesful trip completions,
    #and the reasons why a trip can fail are driver cancel or customer cancel. if there are no available drivers
    #(driver not found), then that is beyond the control of choosing the right driver as there are no drivers to choose from
    df[target_col] = df["booking_status"].apply(lambda x: int(x == "COMPLETED"))
    return df


if __name__ == "__main__":
    main()
