all_run_results = []
system_info = get_system_info()

print("Starting experiment runs...")
print("---" * 20)

for i, config in enumerate(EXPERIMENT_CONFIGS):
    n_samples = config["n_samples"]
    m_points = config["m_points"]

    print(f"\n>>> RUN {i+1}/{len(EXPERIMENT_CONFIGS)}: N={n_samples}, M={m_points} <<<")

    # Load and prepare data for the current configuration
    base_results = {**system_info, **config}

    # --- Run Falkon GPU ---
    if RUN_CONTROLS["RUN_FALKON_GPU"] and config["FALKON_GPU"]:
        # Use float32 for GPU for better performance (Tensor Core usage)
        gpu_results = run_falkon_krr(
            CONSOLIDATED_DATA_PATH,
            N=n_samples,
            M=m_points,
            use_cpu=False,
        )
        all_run_results.append(
            {**base_results, "run_type": "Falkon (GPU)", **gpu_results}
        )

    # --- Run Falkon CPU ---
    if RUN_CONTROLS["RUN_FALKON_CPU"] and config["FALKON_CPU"]:
        # Use float64 for CPU because float32 is numerically unstable on CPU
        cpu_results = run_falkon_krr(
            CONSOLIDATED_DATA_PATH,
            N=n_samples,
            M=m_points,
            use_cpu=True,
        )
        all_run_results.append(
            {**base_results, "run_type": "Falkon (CPU)", **cpu_results}
        )

    # --- Run Scikit-learn CPU ---
    if RUN_CONTROLS["RUN_SKLEARN_CPU"] and config["SKLEARN_CPU"]:
        skl_results = run_sklearn_kernelridge(
            CONSOLIDATED_DATA_PATH,
            N=n_samples,
        )
        all_run_results.append(
            {**base_results, "run_type": "Scikit-learn (CPU)", **skl_results}
        )

    # Moving to next Run!
    print(
        f"\n>>> RUN {i+1}/{len(EXPERIMENT_CONFIGS)}: N={n_samples}, M={m_points} COMPLETE <<<\n\n"
    )

print("\n\nAll experiment runs complete.")
print("---" * 20)
