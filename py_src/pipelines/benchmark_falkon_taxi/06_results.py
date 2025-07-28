if all_run_results:
    # --- 1. Common Data Processing and Saving ---
    # Convert list of dictionaries to a DataFrame
    results_df = pd.DataFrame(all_run_results)
    metrics_df = results_df["metrics"].apply(pd.Series)
    results_df = pd.concat([results_df.drop("metrics", axis=1), metrics_df], axis=1)

    # Reorder columns for clarity
    col_order = [
        "run_type",
        "n_samples",
        "m_points",
        "status",
        "train_time",
        "train_times",
        "pred_time",
        "pred_times",
        "RMSE",
        "R2",
        "cpu_model",
        "gpu_model",
        "ram_total_gb",
        "gpu_vram_gb",
        "python_version",
        "torch_version",
        "falkon_version",
    ]
    # Filter to only existing columns to avoid errors
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    # Display results in the notebook
    print("--- Final Results Summary ---")
    display(results_df)

    # Save to CSV
    output_filename = f"{EXPERIMENT_NAME}.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"\nResults successfully saved to '{output_filename}'")

    # --- 2. Conditional Visualization ---
    # Determine which experiment was run based on the results data.
    is_scalability_experiment = len(results_df["n_samples"].unique()) > 1
    is_m_impact_experiment = (
        len(results_df["m_points"].unique()) > 1
        and len(results_df["n_samples"].unique()) == 1
    )

    # Plot for Experiment 1: Scalability in N
    if is_scalability_experiment:
        print("\n--- Visualization for Scalability (N) Benchmark ---")
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(
            data=results_df, x="n_samples", y="train_time", hue="run_type", ax=ax
        )
        ax.set_title("Training Time by Model and Dataset Size", fontsize=16)
        ax.set_ylabel("Average Training Time (s)")
        ax.set_xlabel("Number of Training Samples (N)")
        ax.set_yscale("log")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Plot for Experiment 2: Impact of M
    elif is_m_impact_experiment:
        print("\n--- Visualization for M-Impact Benchmark ---")
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=results_df, x="m_points", y="RMSE", marker="o", ax=ax)
        ax.set_title("Model RMSE vs. Number of Inducing Points (M)", fontsize=16)
        ax.set_ylabel("RMSE (Root Mean Squared Error)")
        ax.set_xlabel("Number of Inducing Points (M)")
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, p: format(int(x), ","))
        )
        plt.tight_layout()
        plt.show()

else:
    print("No results were generated.")
