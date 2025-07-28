def evaluate_regression(y_true, y_pred):
    """Calculates regression metrics and returns them as a dictionary."""
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    # Calculate standard regression metrics
    mae = mean_absolute_error(y_true_np, y_pred_np)
    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_np, y_pred_np)

    # Report results
    print("Metrics:")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2: {r2:.4f}")

    # Return metrics as a dictionary
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
