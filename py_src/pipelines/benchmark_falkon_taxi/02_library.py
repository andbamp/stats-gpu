# --- Library Imports ---

# Modeling and Computation
import falkon
import torch
import pykeops
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Visualization and Utilities
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow as pa
import pyarrow.parquet as pq
import time
import platform
import psutil
import cpuinfo
import requests
import gc
from IPython.display import display

print("Libraries imported successfully.")
get_system_info()
