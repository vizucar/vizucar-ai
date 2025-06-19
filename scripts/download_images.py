import os
import json
import requests
from tqdm import tqdm
from datetime import datetime

DATASET_PATH = "data/raw/vizucar-bdd.json"
OUTPUT_DIR = "data/downloaded"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "download_errors.log")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def clean_name(s):
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")

with open(DATASET_PATH, 'r') as f:
    data = json.load(f)

def log_error(entry, error_msg):
    with open(LOG_FILE, "a") as log_file:
        make = entry.get("make", "unknown")
        model = entry.get("model", "unknown")
        car_class = entry.get("class", "unknown")
        year = entry.get("year", "unknown")
        url = entry.get("image_url", "N/A")

        log_file.write(
            f"[{datetime.now().isoformat()}] {make}/{model}/{year} ({car_class}) | URL: {url} | ERROR: {error_msg}\n"
        )

def download(entry, idx):
    url = entry.get("image_url")
    make = entry.get("make")
    car_class = entry.get("class")
    model = entry.get("model")
    year = entry.get("year")

    if not all([url, make, car_class, model, year]):
        log_error(entry, "Missing required fields")
        return

    folder_rel = os.path.join(clean_name(make), clean_name(car_class))
    filename = f"{clean_name(model)}_{year}_{idx}.jpg"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()

        folder_abs = os.path.join(OUTPUT_DIR, folder_rel)
        os.makedirs(folder_abs, exist_ok=True)

        with open(os.path.join(folder_abs, filename), "wb") as f:
            f.write(r.content)

    except Exception as e:
        log_error(entry, str(e))

for idx, entry in enumerate(tqdm(data, desc="Downloading images")):
    download(entry, idx)