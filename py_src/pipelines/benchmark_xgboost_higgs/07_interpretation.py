# Check if a GPU model label exists in the final DataFrame
if "XGBoost (cuda, hist)" in results_df["model_label"].unique():
    print("\n--- Starting Interpretation and Plotting ---")

    # Find the specific run for the largest GPU model from the original results list
    final_model_run = None
    max_n = 0
    for res in all_run_results:
        if (
            res.get("model_label") == "XGBoost (cuda, hist)"
            and res.get("n_samples", 0) > max_n
        ):
            max_n = res["n_samples"]
            final_model_run = res

    if final_model_run and final_model_run.get("model"):
        print(f"Found target model: XGBoost (cuda, hist) with N = {max_n:,}")
        model = final_model_run["model"]

        # 1. Reload the corresponding test data
        print("  Reloading test data...")
        _, _, X_test, y_test = load_higgs(data_path, n_samples_train=max_n)
        print(f"  Test data for N={max_n:,} loaded successfully.")

        # --- Plot Generation ---

        # 2. Generate and save the Confusion Matrix plot
        print("  Generating and saving Confusion Matrix plot...")
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Background", "Signal"],
            yticklabels=["Background", "Signal"],
        )
        plt.title("Confusion Matrix", fontsize=16)
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.savefig(
            f"{EXPERIMENT_NAME}_interpretation_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 3. Generate and save the Feature Importance plot
        print("  Generating and saving Feature Importance plot...")
        feature_importances = model.get_booster().get_score(importance_type="gain")
        importance_df = (
            pd.DataFrame(list(feature_importances.items()), columns=["feature", "gain"])
            .sort_values("gain", ascending=False)
            .head(15)
        )

        plt.figure(figsize=(10, 8))
        sns.barplot(x="gain", y="feature", data=importance_df, palette="viridis")
        plt.title("Feature Importance (Gain)", fontsize=16)
        plt.xlabel("Gain")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(f"{EXPERIMENT_NAME}_interpretation_feature_importance.png", dpi=300)
        plt.close()

        # 4. Calculate, generate, and save SHAP plots
        print("  Calculating SHAP values (this may take a while)...")
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # SHAP Bar Plot
        print("  Generating and saving SHAP bar plot...")
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title("SHAP Summary Plot (Mean Absolute Value)", fontsize=16)
        plt.savefig(
            f"{EXPERIMENT_NAME}_interpretation_shap_bar.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # SHAP Dot Plot
        print("  Generating and saving SHAP dot plot...")
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title("SHAP Summary Plot", fontsize=16)
        plt.savefig(
            f"{EXPERIMENT_NAME}_interpretation_shap_dot.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("\n--- All interpretation plots saved successfully! ---")

    else:
        print("Could not find the final model object to generate diagnostics.")

else:
    print(
        "\n--- Skipping Interpretation Block: No 'XGBoost (cuda, hist)' model found. ---"
    )
