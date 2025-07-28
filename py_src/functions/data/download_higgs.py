def download_higgs(data_dir="."):
    """Downloads the HIGGS dataset if it is not already present."""
    os.makedirs(data_dir, exist_ok=True)
    higgs_filename = os.path.join(data_dir, "HIGGS.csv.gz")

    if os.path.exists(higgs_filename):
        print(f"HIGGS data file found at '{higgs_filename}'. Skipping download.")
        return higgs_filename

    print(f"HIGGS data not found in '{data_dir}'. Starting download...")
    higgs_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    )

    try:
        response = requests.get(higgs_url, stream=True)
        response.raise_for_status()
        with open(higgs_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Download complete. Data saved to '{higgs_filename}'")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the HIGGS dataset: {e}")
        return None

    return higgs_filename
