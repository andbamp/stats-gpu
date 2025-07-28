# --- Installation ---
import os
import sys

# Check if environment initialized
try:
    import falkon

    ENV_INITIALIZED = True
except ImportError:
    ENV_INITIALIZED = False

if not ENV_INITIALIZED:
    print("Installing dependencies...")
    # Downgrade to PyTorch for CUDA 12.1
    !pip install --quiet torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    # Downgrade to KeOps 2.2
    !pip install --quiet pykeops==2.2
    # Install Falkon from pre-compiled wheel for the specific Torch+CUDA version
    !pip install --quiet falkon -f https://falkon.dibris.unige.it/torch-2.2.0_cu121.html
    # Install other necessary libraries
    !pip install --quiet pandas pyarrow scikit-learn seaborn matplotlib py-cpuinfo psutil
    # Downgrade to NumPy 1.26.4
    !pip install --quiet numpy==1.26.4
    print("Dependencies installed successfully.")
    # Programmatically exit to restart the runtime.
    os._exit(0)
else:
    print("Dependencies already installed.")
