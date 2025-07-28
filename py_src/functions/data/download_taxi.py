def download_taxi(final_path="taxi_data.parquet"):
    """Downloads and processes data if not already present."""

    # Load consolidate data if exists
    if os.path.exists(final_path):
        print(f"Consolidated data file found at '{final_path}'. Skipping processing.")
        return final_path

    print("Consolidated data not found. Starting download and processing...")

    # Download monthly taxi data
    taxi_data_urls = {
        f"yellow_tripdata_2024-{month:02d}.parquet": f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-{month:02d}.parquet"
        for month in range(1, 13)
    }

    for taxi_filename, taxi_url in taxi_data_urls.items():
        if not os.path.exists(taxi_filename):
            print(f"Downloading {taxi_filename}...")
            try:
                response = requests.get(taxi_url)
                response.raise_for_status()
                with open(taxi_filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"{taxi_filename} downloaded successfully.")
            except Exception as e:
                print(f"Error downloading {taxi_filename}: {e}")
        else:
            print(f"{taxi_filename} already exists.")

    # Helper function: Process
    def process_taxi(df, TARGET_COL, FEATURE_COLS):
        """Processes a single taxi DataFrame chunk."""

        # Convert to datetime objects
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

        # Calculate trip duration in minutes
        df["duration"] = (
            df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60

        # Extract features from datetime
        df["pickup_day"] = df["tpep_pickup_datetime"].dt.dayofweek
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour

        # Basic filtering
        df = df[df["fare_amount"].between(2.5, 200)]
        df = df[df["trip_distance"].between(0.1, 100)]
        df = df[df["duration"].between(1, 360)]
        df = df[df["passenger_count"].between(1, 6)]

        # Select features and target
        df = df[FEATURE_COLS + [TARGET_COL]]

        # Drop NA's
        df.dropna(inplace=True)

        return df

    # Helper function: Consolidate
    def consolidate_taxi(
        files,
        TARGET_COL,
        FEATURE_COLS,
        final_path="taxi_data.parquet",
        batch_size=1_000_000,
    ):
        """Reads and processes files in chunks, then saves to disk"""

        writer = None
        schema = None

        if os.path.exists(final_path):
            os.remove(final_path)

        # Read and process files in chunks
        for i, filename in enumerate(files):
            print(f"--- Processing file {i+1}/{len(files)}: {filename} ---")
            parquet_file = pq.ParquetFile(filename)

            for batch_num, batch in enumerate(
                parquet_file.iter_batches(batch_size=batch_size)
            ):
                print(f"  Processing batch {batch_num + 1}...")
                chunk_df = batch.to_pandas()
                processed_chunk = process_taxi(chunk_df, TARGET_COL, FEATURE_COLS)

                if not processed_chunk.empty:
                    table = pa.Table.from_pandas(processed_chunk)
                    if writer is None:
                        print("  Initializing Parquet writer...")
                        schema = table.schema
                        writer = pq.ParquetWriter(final_path, schema)

                    writer.write_table(table.cast(schema, safe=False))
                    del table

                del chunk_df, processed_chunk, batch
                gc.collect()

        if writer:
            writer.close()
            print(f"\nAll files processed and saved to {final_path}")
            return final_path
        else:
            print("\nNo data was written.")
            return None

    # Consolidate data
    taxi_filenames = list(taxi_data_urls.keys())
    taxi_path = consolidate_taxi(taxi_filenames, TARGET_COL, FEATURE_COLS)

    return taxi_path
