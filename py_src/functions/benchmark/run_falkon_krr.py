def run_falkon_krr(data_path, N, M, use_cpu):
    """Runs a full Falkon benchmark and returns a dictionary of results."""
    model_label = "Falkon (CPU)" if use_cpu else "Falkon (GPU)"
    print(f"\n--- Starting Benchmark for: {model_label} ---")

    # Loading
    X_train_orig, y_train_orig, X_test_orig, y_test_orig = load_taxi(data_path, N)
    if use_cpu:
        target_dtype = torch.float64
    else:
        target_dtype = torch.float32

    X_train = X_train_orig.to(dtype=target_dtype)
    del X_train_orig
    y_train = y_train_orig.to(dtype=target_dtype)
    del y_train_orig
    X_test = X_test_orig.to(dtype=target_dtype)
    del X_test_orig
    y_test = y_test_orig.to(dtype=target_dtype)
    del y_test_orig
    gc.collect()

    # Configuration
    falkon_options = falkon.FalkonOptions(
        keops_active="no" if use_cpu else "yes", use_cpu=use_cpu, debug=True
    )

    falkon_params = {
        "kernel": falkon.kernels.GaussianKernel(sigma=GLOBAL_PARAMS["FALKON_SIGMA"]),
        "penalty": GLOBAL_PARAMS["FALKON_PENALTY"],
        "M": M,
        "maxiter": GLOBAL_PARAMS["FALKON_MAXITER"],
        "seed": GLOBAL_PARAMS["RANDOM_STATE"],
        "options": falkon_options,
    }

    # Training
    def train_falkon():
        model = falkon.Falkon(**falkon_params)
        model.fit(X_train, y_train)
        return model

    model, train_time, train_times = timed_execution(train_falkon)

    del X_train, y_train
    gc.collect()

    # Prediction
    def predict_falkon():
        preds = model.predict(X_test)
        return preds

    pred, pred_time, pred_times = timed_execution(predict_falkon)

    # Evaluation
    metrics = evaluate_regression(y_test, pred)

    # Results
    results = {
        "model": model,
        "train_time": train_time,
        "train_times": train_times,
        "pred_time": pred_time,
        "pred_times": pred_times,
        "metrics": metrics,
        "model_label": model_label,
        "status": "Success",
    }

    print(f"--- Benchmark for {model_label} complete. ---")
    return results
