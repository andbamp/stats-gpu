if all_run_results:
    # --- 1. Data Processing and Saving ---
    results_df = pd.DataFrame(all_run_results)

    if "metrics" in results_df.columns:
        metrics_df = results_df["metrics"].apply(pd.Series)
        results_df = pd.concat([results_df.drop("metrics", axis=1), metrics_df], axis=1)

    # Reorder columns for clarity
    col_order = [
        "model_label",
        "n_samples",
        "status",
        "train_time",
        "train_times",
        "pred_time",
        "pred_times",
        "auc",
        "f1_score",
        "accuracy",
        "cpu_model",
        "gpu_model",
        "ram_total_gb",
        "python_version",
        "xgboost_version",
    ]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    # Display results
    print("--- Final Results Summary ---")
    display(results_df)

    # Save to CSV
    output_filename = f"{EXPERIMENT_NAME}.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"\nResults successfully saved to '{output_filename}'")

    # --- 2. Visualization ---
    is_scalability_experiment = len(results_df["n_samples"].unique()) > 1

    if is_scalability_experiment:
        print("\n--- Visualization for Scalability (N) Benchmark ---")
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))

        plot_df = results_df[results_df["status"] == "Success"].copy()

        sns.barplot(
            data=plot_df, x="n_samples", y="train_time", hue="model_label", ax=ax
        )
        ax.set_title("Training Time by Method and Dataset Size", fontsize=16)
        ax.set_ylabel("Average Training Time (s) - Log Scale")
        ax.set_xlabel("Number of Training Samples (N)")
        ax.set_yscale("log")

        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, p: format(int(x), ","))
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

else:
    print("No results were generated.")
