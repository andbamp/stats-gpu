def run_xgboost_bin(data_path, N, base_params, tree_method, device):
    """Runs a full XGBoost benchmark and returns a dictionary of results."""
    model_label = f"XGBoost ({device}, {tree_method})"
    print(f"\n--- Starting Benchmark for: {model_label} ---")

    # Loading
    X_train, y_train, X_test, y_test = load_higgs(data_path, N)

    # Configure XGBoost parameters
    params = base_params.copy()
    params["tree_method"] = tree_method
    if device == "cuda":
        params["device"] = "cuda"
    else:
        params["n_jobs"] = -1

    # Training
    def train_xgb():
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        return model

    model, train_time, train_times = timed_execution(train_xgb)
    print(f"  Training complete: {train_time:.4f}s")
    del X_train, y_train
    gc.collect()

    # Prediction
    def predict_xgb():
        preds = model.predict_proba(X_test)[:, 1]
        return preds

    pred_proba, pred_time, pred_times = timed_execution(predict_xgb)

    # Evaluation
    pred_binary = (pred_proba > 0.5).astype(int)
    metrics = evaluate_classification(y_test, pred_binary, pred_proba)

    # Compile results
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
