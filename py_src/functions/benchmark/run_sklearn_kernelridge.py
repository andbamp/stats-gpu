def run_sklearn_kernelridge(data_path, N):
    """Runs a Scikit-learn KernelRidge benchmark and returns results."""
    model_label = "Scikit-learn KernelRidge (CPU)"

    if N == 0:
        print(f"\n--- Skipping Benchmark for: {model_label} ---")
        return {
            "model": None,
            "train_time": -1,
            "train_times": [],
            "pred_time": -1,
            "pred_times": [],
            "metrics": {},
            "model_label": model_label,
            "status": "MemoryError",
        }

    try:
        print(f"\n--- Starting Benchmark for: {model_label} ---")

        # Loading
        X_train_orig, y_train_orig, X_test_orig, y_test_orig = load_taxi(data_path, N)

        X_train = X_train_orig.cpu().numpy()
        del X_train_orig
        y_train = y_train_orig.cpu().numpy()
        del y_train_orig
        X_test = X_test_orig.cpu().numpy()
        del X_test_orig
        y_test = y_test_orig.cpu().numpy()
        del y_test_orig
        gc.collect()

        # Training
        def train_skl():
            SKL_GAMMA = 1 / (2 * GLOBAL_PARAMS["FALKON_SIGMA"] ** 2)
            model = KernelRidge(kernel="rbf", gamma=SKL_GAMMA)
            model.fit(X_train, y_train)
            return model

        model, train_time, train_times = timed_execution(train_skl)

        del X_train, y_train
        gc.collect()

        # Prediction
        def predict_skl():
            preds = model.predict(X_test)
            return preds

        pred, pred_time, pred_times = timed_execution(predict_skl)

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
    except MemoryError:
        print("!!! Scikit-learn failed due to MemoryError. !!!")
        return {
            "model": None,
            "train_time": -1,
            "train_times": [],
            "pred_time": -1,
            "pred_times": [],
            "metrics": {},
            "model_label": model_label,
            "status": "MemoryError",
        }
    except Exception as e:
        print(f"!!! Scikit-learn failed due to an exception: {e} !!!")
        return {
            "model": None,
            "train_time": -1,
            "train_times": [],
            "pred_time": -1,
            "pred_times": [],
            "metrics": {},
            "model_label": model_label,
            "status": "Error",
        }
