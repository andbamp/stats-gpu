def get_system_info(bench="flk"):
    """Gathers and returns key hardware and software information as a dictionary."""
    info = {}
    try:
        # Environment
        info["python_version"] = platform.python_version()
        info["python_executable"] = sys.executable

        # Software
        info["torch_version"] = torch.__version__
        info["numpy_version"] = np.__version__
        info["pandas_version"] = pd.__version__
        info["sklearn_version"] = sklearn.__version__

        # Benchmark-specific libraries
        if bench == "flk":
            info["falkon_version"] = falkon.__version__
            info["pykeops_version"] = pykeops.__version__
        elif bench == "xgb":
            info["xgboost_version"] = xgb.__version__
            info["shap_version"] = shap.__version__

        # Hardware
        info["os"] = f"{platform.system()} {platform.release()}"
        info["cpu_model"] = cpuinfo.get_cpu_info().get("brand_raw", "N/A")
        info["ram_total_gb"] = psutil.virtual_memory().total / (1024**3)

        # GPU Hardware
        if torch.cuda.is_available():
            info["gpu_model"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )
            info["cuda_version"] = torch.version.cuda
        else:
            info["gpu_model"] = "N/A"
            info["gpu_vram_gb"] = 0
            info["cuda_version"] = "N/A"
    except Exception as e:
        print(f"Could not retrieve all system info: {e}")
    return info
