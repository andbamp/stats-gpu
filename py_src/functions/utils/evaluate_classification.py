def evaluate_classification(y_true, y_pred_binary, y_pred_proba):
    """Calculates classification metrics from predictions and returns them as a dictionary."""
    # Ensure inputs are NumPy arrays
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true

    # Calculate metrics
    accuracy = accuracy_score(y_true_np, y_pred_binary)
    f1 = f1_score(y_true_np, y_pred_binary)
    auc = roc_auc_score(y_true_np, y_pred_proba)

    # Report results
    print("Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")

    return {"accuracy": accuracy, "f1_score": f1, "auc": auc}
