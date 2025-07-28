# --- Installation ---
import os
import sys

# Check if environment initialized
try:
    import xgboost
    import shap
    ENV_INITIALIZED = True
except ImportError:
    ENV_INITIALIZED = False

if not ENV_INITIALIZED:
    print("Installing dependencies...")
    # Pin versions for reproducibility

    # PyTorch with specific CUDA 12.4 support
    !pip install --quiet torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    # Core compute and ML libraries
    !pip install --quiet \
        numpy==2.0.2 \
        pandas==2.2.2 \
        pyarrow==16.1.0 \
        scikit-learn==1.6.1 \
        xgboost==2.1.4

    # Plotting libraries
    !pip install --quiet \
        matplotlib==3.9.0 \
        seaborn==0.13.2

    # Model explainability, data repository, and system utilities
    !pip install --quiet \
        shap==0.46.0 \
        requests==2.32.3 \
        py-cpuinfo==9.0.0 \
        psutil==6.0.0

    print("Dependencies installed successfully. Please restart the runtime.")
    # Programmatically exit to restart the runtime.
    os._exit(0)
else:
    print("Dependencies already installed.")
