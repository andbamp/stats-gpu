# --- A. Experiment Naming ---
# Name for the output CSV file.
EXPERIMENT_NAME = "benchmark_falkon_taxi_1"

# --- B. Benchmark Selection ---
# Global flags to enable/disable entire benchmark categories.
RUN_CONTROLS = {
    "RUN_FALKON_GPU": True,
    "RUN_FALKON_CPU": True,
    "RUN_SKLEARN_CPU": True,
}

# --- C. Experiment Run Definitions ---
# Experiment Configurations #1
# List of N to benchmark.
n_samples_list = [
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    20_000_000,
    50_000_000,
]

# Hardcores model runs up to specific N.
# Adds M as log(N) * sqrt(N)
EXPERIMENT_CONFIGS_1 = [
    {
        "n_samples": n,
        "m_points": int(np.log(n) * np.sqrt(n)),
        "FALKON_GPU": True if n <= 2_000_000 else False,
        "FALKON_CPU": True if n <= 200_000 else False,
        "SKLEARN_CPU": True if n <= 10_000 else False,
    }
    for n in n_samples_list
]

# Experiment Configurations #2
N_FIXED = 5_000_000
SQRT_N = np.sqrt(N_FIXED)

# Hardcores model runs to a specific N.
# Adds several M to test effect on model performance.
EXPERIMENT_CONFIGS_2 = [
    {
        "n_samples": N_FIXED,
        "m_points": int(m),
        "FALKON_GPU": True,
        "FALKON_CPU": False,
        "SKLEARN_CPU": False,
    }
    for m in np.concatenate(
        (
            np.linspace(start=0.1 * SQRT_N, stop=1.0 * SQRT_N, num=10),
            np.linspace(start=1.0 * SQRT_N, stop=10.0 * SQRT_N, num=10)[1:],
        )
    )
]

# --- D. Global Model and Run Parameters ---
GLOBAL_PARAMS = {
    "N_RUNS": 5,  # Repetitions for averaging time
    "RANDOM_STATE": 6,  # Seed for reproducibility
    "MAX_TEST_SAMPLES": 100_000,  # Fixed size for test set
    "FALKON_SIGMA": 15.0,  # Gaussian kernel width (sigma)
    "FALKON_PENALTY": 1e-6,  # Regularization penalty (lambda)
    "FALKON_MAXITER": 20,  # Max iterations for CG solver
}

EXPERIMENT_CONFIGS = EXPERIMENT_CONFIGS_1

# Seed for reproducibility
np.random.seed(GLOBAL_PARAMS["RANDOM_STATE"])
torch.manual_seed(GLOBAL_PARAMS["RANDOM_STATE"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(GLOBAL_PARAMS["RANDOM_STATE"])
