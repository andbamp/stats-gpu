# --- Library Imports ---

# Core libraries
import xgboost as xgb
import torch
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import GradientBoostingClassifier

# Visualization and Utilities
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

import time
import platform
import psutil
import cpuinfo
import requests
import gc
from IPython.display import display, Markdown

print("Libraries imported successfully.")
get_system_info("xgb")
