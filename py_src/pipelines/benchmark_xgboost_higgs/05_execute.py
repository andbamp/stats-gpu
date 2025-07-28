all_run_results = []
data_path = download_higgs(GLOBAL_PARAMS["DATA_DIR"])
system_info = get_system_info("xgb")

print("Starting experiment runs...")
print("---" * 20)

for i, config in enumerate(EXPERIMENT_CONFIGS):
    n_samples = config["n_samples"]

    print(f"\n>>> RUN {i+1}/{len(EXPERIMENT_CONFIGS)}: N={n_samples} <<<\n")

    base_results = {**system_info, **config}

    # --- Run Benchmarks based on flags ---
    if RUN_CONTROLS["RUN_GPU_HIST"] and config["gpu_hist"]:
        results = run_xgboost_bin(data_path, n_samples, HYPERPARAMS, "hist", "cuda")
        all_run_results.append({**base_results, **results})

    if RUN_CONTROLS["RUN_CPU_HIST"] and config["cpu_hist"]:
        results = run_xgboost_bin(data_path, n_samples, HYPERPARAMS, "hist", "cpu")
        all_run_results.append({**base_results, **results})

    if RUN_CONTROLS["RUN_CPU_APPROX"] and config["cpu_approx"]:
        results = run_xgboost_bin(data_path, n_samples, HYPERPARAMS, "approx", "cpu")
        all_run_results.append({**base_results, **results})

    if RUN_CONTROLS["RUN_CPU_EXACT"] and config["cpu_exact"]:
        results = run_xgboost_bin(data_path, n_samples, HYPERPARAMS, "exact", "cpu")
        all_run_results.append({**base_results, **results})

    if RUN_CONTROLS["RUN_SKLEARN_GBC"] and config["sklearn_gbc"]:
        results = run_sklearn_gbc(data_path, n_samples, HYPERPARAMS)
        all_run_results.append({**base_results, **results})

    gc.collect()
    print(f"\n>>> RUN {i+1}/{len(EXPERIMENT_CONFIGS)} COMPLETE <<<\n\n")


print("---" * 20)
print("All experiment runs complete.")
