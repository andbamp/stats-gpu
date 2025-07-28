# --- A. Experiment Naming ---
# Name for the output CSV file.
EXPERIMENT_NAME = "benchmark_xgboost_higgs"

# --- B. Benchmark Selection ---
# Global flags to enable/disable entire benchmark categories.
RUN_CONTROLS = {
    "RUN_GPU_HIST": True,
    "RUN_CPU_HIST": True,
    "RUN_CPU_APPROX": True,
    "RUN_CPU_EXACT": True,
    "RUN_SKLEARN_GBC": True,
}

# --- C. Scalability Run Definitions ---
# Define the list of sample sizes (N) to benchmark against.
N_SAMPLES_LIST = [
    10_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    8_000_000,
]

# Define configurations for the scalability study.
EXPERIMENT_CONFIGS = [
    {
        "n_samples": n,
        "gpu_hist": True,
        "cpu_hist": True if n <= 5_000_000 else False,
        "cpu_approx": True if n <= 500_000 else False,
        "cpu_exact": True if n <= 500_000 else False,
        "sklearn_gbc": True if n <= 100_000 else False,
    }
    for n in N_SAMPLES_LIST
]

# --- D. Global Model and Run Parameters ---
GLOBAL_PARAMS = {
    "N_RUNS": 5,
    "RANDOM_STATE": 6,
    "MAX_TEST_SAMPLES": 250_000,
    "DATA_DIR": "./data",
}

# Base hyperparameters for the XGBoost model
HYPERPARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.05,
    "max_depth": 8,
    "n_estimators": 500,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "seed": GLOBAL_PARAMS["RANDOM_STATE"],
    # gamma, lambda, and alpha will use their robust defaults (0, 1, and 0)
}

# Seed for reproducibility
np.random.seed(GLOBAL_PARAMS["RANDOM_STATE"])
os.makedirs(GLOBAL_PARAMS["DATA_DIR"], exist_ok=True)
