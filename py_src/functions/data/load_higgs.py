def load_higgs(data_path, n_samples_train):
    """Loads and prepares the HIGGS dataset for a specific sample size."""
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found at {data_path}. Please run download_higgs first."
        )

    # --- Caching Logic: Define cache path and check for existence ---
    base_name, _ = os.path.splitext(os.path.basename(data_path))
    cache_dir = os.path.dirname(data_path)
    cache_path = os.path.join(cache_dir, f"{base_name}_N{n_samples_train}.pkl")

    if os.path.exists(cache_path):
        print(f"Cache found! Loading pre-processed data from '{cache_path}'...")
        data_dict = joblib.load(cache_path)
        print(
            f"Data loaded from cache. Train shape: {data_dict['X_train'].shape}, Test shape: {data_dict['X_test'].shape}"
        )
        return (
            data_dict["X_train"],
            data_dict["y_train"],
            data_dict["X_test"],
            data_dict["y_test"],
        )

    # --- If cache not found, run the full processing pipeline ---
    print(f"Cache not found for N={n_samples_train}. Starting full data processing...")

    # Load full dataset
    print(f"Loading full dataset from '{data_path}'...")
    try:
        df_higgs = pd.read_csv(data_path, header=None, compression="gzip")
        X_df = df_higgs.iloc[:, 1:]
        y_series = df_higgs.iloc[:, 0].astype(int)
        X_df.columns = [f"feature_{i}" for i in range(X_df.shape[1])]
        del df_higgs
        gc.collect()
    except Exception as e:
        print(f"Error loading or processing CSV file: {e}")
        return None, None, None, None

    print(f"Full dataset loaded. Shape: {X_df.shape}")

    # Split full dataset to maintain a consistent test set
    print("Splitting data into a training pool and a final test set...")
    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X_df,
        y_series,
        test_size=0.2,  # Reserve 20% for testing
        random_state=GLOBAL_PARAMS["RANDOM_STATE"],
        stratify=y_series,
    )
    del X_df, y_series
    gc.collect()

    # Subsample training data from the pool
    print(f"Subsampling training pool to {n_samples_train} instances...")
    X_train, _, y_train, _ = train_test_split(
        X_train_pool,
        y_train_pool,
        train_size=n_samples_train,
        random_state=GLOBAL_PARAMS["RANDOM_STATE"],
        stratify=y_train_pool,
    )
    del X_train_pool, y_train_pool
    gc.collect()

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    del X_train, X_test
    gc.collect()

    # --- Caching Logic: Save the newly created data to disk ---
    print(f"\nSaving processed data to cache at '{cache_path}'...")
    data_to_save = {
        "X_train": X_train_scaled,
        "y_train": y_train.values,
        "X_test": X_test_scaled,
        "y_test": y_test.values,
    }
    joblib.dump(data_to_save, cache_path)
    print("Save complete.")

    print(
        f"\nData prepared. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}"
    )
    return X_train_scaled, y_train.values, X_test_scaled, y_test.values
