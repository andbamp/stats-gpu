# Set target variable
TARGET_COL = "fare_amount"

# Set feature columns
FEATURE_COLS = [
    "VendorID",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "duration",
    "pickup_hour",
    "pickup_day",
]

# Download data
CONSOLIDATED_DATA_PATH = download_taxi()
