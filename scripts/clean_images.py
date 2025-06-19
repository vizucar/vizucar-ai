import os
from PIL import Image
from tqdm import tqdm
from datetime import datetime

INPUT_DIR = "data/downloaded"
OUTPUT_DIR = "data/processed"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "cleaning_errors.log")
IMAGE_SIZE = (512, 512)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def log_error(image_path, error_msg):
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"[{datetime.now().isoformat()}] {image_path} | ERROR: {error_msg}\n")

for root, dirs, files in os.walk(INPUT_DIR):
    for file in tqdm(files, desc="Cleaning images"):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        relative_path = os.path.relpath(root, INPUT_DIR)
        input_path = os.path.join(root, file)
        output_path = os.path.join(OUTPUT_DIR, relative_path, file)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            img = Image.open(input_path).convert("RGB")
            img = img.resize(IMAGE_SIZE)
            img.save(output_path, "JPEG")
        except Exception as e:
            print(f"Error processing {input_path} → {e}")
            log_error(input_path, str(e))