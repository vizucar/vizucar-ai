import os
import json
from tqdm import tqdm
from datetime import datetime

PROCESSED_DIR = "data/processed"
PROMPT_DIR = "data/prompts"
LOG_DIR = "logs"
OUTPUT_JSON = os.path.join(PROMPT_DIR, "prompts.json")
MISSING_LOG = os.path.join(LOG_DIR, "missing_metadata.log")

def log_missing_metadata(image_path, model, year):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(MISSING_LOG, "a") as log_file:
        log_file.write(f"[{datetime.now().isoformat()}] MISSING metadata in: {image_path} (model='{model}', year='{year}')\n")

def extract_model_year(filename):
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split("_")
    if len(parts) < 3:
        return "", ""
    year = parts[-2]
    try:
        int(year)
    except ValueError:
        return "", ""
    model = "_".join(parts[:-2]).replace("_", " ").strip()
    return model, year

def generate_prompts(make, car_class, filename):
    model, year = extract_model_year(filename)
    if year:
        prompt_precise = f"{make} {model} {year} {car_class}".strip()
    else:
        prompt_precise = f"{make} {model} {car_class}".strip()
    prompt_general = f"{car_class}".strip()
    return model, year, {
        "prompt_precise": prompt_precise,
        "prompt_general": prompt_general
    }

def main():
    data = []
    os.makedirs(PROMPT_DIR, exist_ok=True)

    for make in tqdm(os.listdir(PROCESSED_DIR), desc="Generating prompts (brands)"):
        make_path = os.path.join(PROCESSED_DIR, make)
        if not os.path.isdir(make_path):
            continue

        for car_class in os.listdir(make_path):
            class_path = os.path.join(make_path, car_class)
            if not os.path.isdir(class_path):
                continue

            for filename in os.listdir(class_path):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                model, year, prompts = generate_prompts(make, car_class, filename)
                image_path = os.path.join(make, car_class, filename).replace("\\", "/")

                if not model:
                    log_missing_metadata(image_path, model, year)
                    continue

                if not year:
                    log_missing_metadata(image_path, model, year)

                data.append({
                    "image_path": image_path,
                    **prompts
                })

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Prompt file created → {OUTPUT_JSON} ({len(data)} entries)")
    print(f"🧾 Missing metadata logged → {MISSING_LOG} (if any)")

if __name__ == "__main__":
    main()