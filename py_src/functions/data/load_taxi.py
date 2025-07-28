def load_taxi(data_path, n_samples_train):
    """Loads and prepares a specific number of samples for training."""
    if data_path is None:
        raise FileNotFoundError(
            "Consolidated data path is not available. Run data prep."
        )

    # --- Caching Logic: Define cache path and check for existence ---
    base_name, _ = os.path.splitext(data_path)
    cache_path = f"{base_name}_N{n_samples_train}.pt"

    if os.path.exists(cache_path):
        print(f"Cache found! Loading pre-processed tensors from '{cache_path}'...")
        data_dict = torch.load(cache_path)
        X_train_t = data_dict["X_train"]
        y_train_t = data_dict["y_train"]
        X_test_t = data_dict["X_test"]
        y_test_t = data_dict["y_test"]
        print(
            f"Tensors loaded. Train shape: {X_train_t.shape}, Test shape: {X_test_t.shape}"
        )
        return X_train_t, y_train_t, X_test_t, y_test_t

    # --- If cache not found, run the full processing pipeline ---
    print(
        f"Cache not found for N={n_samples_train}. Starting full data processing pipeline..."
    )

    full_df = pd.read_parquet(data_path)
    print(f"Full data loaded. Shape: {full_df.shape}")

    print(f"\nLoading and preparing data for N_train = {n_samples_train}...")

    # Split full dataset to maintain a consistent test set pool
    train_pool_df, test_pool_df = train_test_split(
        full_df, test_size=0.2, random_state=GLOBAL_PARAMS["RANDOM_STATE"]
    )
    del full_df
    gc.collect()
    print(
        f"Data split. Train pool shape: {train_pool_df.shape}, Test pool shape: {test_pool_df.shape}"
    )

    # Subsample training and test sets
    train_df = train_pool_df.sample(
        n=n_samples_train, random_state=GLOBAL_PARAMS["RANDOM_STATE"]
    )
    test_df = test_pool_df.sample(
        n=GLOBAL_PARAMS["MAX_TEST_SAMPLES"], random_state=GLOBAL_PARAMS["RANDOM_STATE"]
    )
    del train_pool_df, test_pool_df
    gc.collect()

    # Separate features and target
    X_train_df, y_train_df = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test_df, y_test_df = test_df[FEATURE_COLS], test_df[TARGET_COL]
    del train_df, test_df
    gc.collect()

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    del X_train_df, X_test_df
    gc.collect()

    # Convert to PyTorch Tensors
    X_train_t = torch.from_numpy(X_train_scaled).contiguous()
    y_train_t = torch.from_numpy(y_train_df.values).view(-1, 1)
    X_test_t = torch.from_numpy(X_test_scaled).contiguous()
    y_test_t = torch.from_numpy(y_test_df.values).view(-1, 1)
    del X_train_scaled, y_train_df, X_test_scaled, y_test_df
    gc.collect()

    # --- Caching Logic: Save the newly created tensors to disk ---
    print(f"\nSaving processed tensors to cache at '{cache_path}'...")
    data_to_save = {
        "X_train": X_train_t,
        "y_train": y_train_t,
        "X_test": X_test_t,
        "y_test": y_test_t,
    }
    torch.save(data_to_save, cache_path)
    print("Save complete.")

    print(
        f"\nData prepared. Train shape: {X_train_t.shape}, Test shape: {X_test_t.shape}"
    )
    return X_train_t, y_train_t, X_test_t, y_test_t
