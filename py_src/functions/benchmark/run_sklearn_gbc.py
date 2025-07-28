def run_sklearn_gbc(data_path, N, base_params):
    """Runs a Scikit-learn GradientBoostingClassifier benchmark as a CPU baseline."""
    model_label = "Scikit-learn GBC (CPU)"

    try:
        print(f"\n--- Starting Benchmark for: {model_label} ---")

        # Loading
        X_train, y_train, X_test, y_test = load_higgs(data_path, N)

        # Configuration
        skl_params = {
            "n_estimators": base_params["n_estimators"],
            "learning_rate": base_params["eta"],
            "max_depth": base_params["max_depth"],
            "subsample": base_params["subsample"],
            "max_features": base_params["colsample_bytree"],
            "random_state": base_params["seed"],
        }

        # Training
        def train_skl_gbc():
            model = GradientBoostingClassifier(**skl_params)
            model.fit(X_train, y_train)
            return model

        model, train_time, train_times = timed_execution(train_skl_gbc)

        print(f"  Training complete: {train_time:.4f}s")
        del X_train, y_train
        gc.collect()

        # Prediction
        def predict_skl_gbc():
            preds = model.predict_proba(X_test)[:, 1]
            return preds

        pred_proba, pred_time, pred_times = timed_execution(predict_skl_gbc)

        # Evaluation
        pred_binary = (pred_proba > 0.5).astype(int)
        metrics = evaluate_classification(y_test, pred_binary, pred_proba)

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
        print(f"!!! {model_label} failed for N={N} due to MemoryError. !!!")
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
        print(f"!!! {model_label} failed for N={N} due to an exception: {e} !!!")
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
