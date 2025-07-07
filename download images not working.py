import os
import pandas as pd
import requests
from tqdm import tqdm
from collections import defaultdict

# --- CONFIG ---
NUM_IMAGES_PER_CLASS = 10
SAFE_CLASSES = [
    "Toaster", "Chair", "Laptop", "Traffic sign", "Wine glass", "Toothbrush",
    "Spoon", "Clock", "Backpack", "Pillow", "Stapler", "Microwave oven",
    "Calculator", "Screwdriver", "Drum", "Lantern", "Teapot", "Candle",
    "Toilet paper", "Cupboard", "Luggage and bags", "Ruler", "Whisk",
    "Filing cabinet", "Printer", "Scissors", "Sink", "Desk", "Ladder",
    "Flashlight", "Keyboard", "Monitor", "Trash bin", "Soap dispenser",
    "Dishwasher", "Bathtub", "Towel", "Mirror", "Curtain", "Shelf",
    "Refrigerator", "Hair dryer", "Vacuum cleaner", "Frying pan", "Bowl",
    "Plate", "Mug", "Bottle opener", "Pizza cutter", "Rolling pin", "Spatula"
]

# --- FILE PATHS ---
ANNOTATIONS_CSV = "validation-annotations-bbox.csv"
CLASS_DESCRIPTIONS_CSV = "class-descriptions-boxable.csv"
IMAGE_IDS_CSV = "validation-images-with-rotation.csv"
OUTPUT_DIR = "openimages_subset"

# --- LOAD METADATA ---
print("Loading metadata...")
classes_df = pd.read_csv(CLASS_DESCRIPTIONS_CSV, header=None, names=["LabelName", "ClassName"])
annotations_df = pd.read_csv(ANNOTATIONS_CSV)
image_ids_df = pd.read_csv(IMAGE_IDS_CSV)

# --- MAP CLASS NAMES TO IDs ---
class_map = dict(zip(classes_df.ClassName, classes_df.LabelName))
selected_ids = {class_map[c] for c in SAFE_CLASSES if c in class_map}

# --- FILTER ANNOTATIONS ---
filtered = annotations_df[annotations_df.LabelName.isin(selected_ids)]

# --- GROUP BY CLASS ---
images_by_class = defaultdict(set)
for _, row in filtered.iterrows():
    images_by_class[row.LabelName].add(row.ImageID)

# --- DOWNLOAD IMAGES ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

for class_name in SAFE_CLASSES:
    class_id = class_map.get(class_name)
    if not class_id or class_id not in images_by_class:
        print(f"Skipping {class_name} (not found)")
        continue

    image_ids = list(images_by_class[class_id])[:NUM_IMAGES_PER_CLASS]
    class_dir = os.path.join(OUTPUT_DIR, class_name.replace(" ", "_"))
    os.makedirs(class_dir, exist_ok=True)

    for img_id in tqdm(image_ids, desc=f"Downloading {class_name}"):
        url = f"https://storage.googleapis.com/openimages/validation/{img_id}.jpg"
        out_path = os.path.join(class_dir, f"{img_id}.jpg")
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
        except Exception as e:
            print(f"Failed to download {img_id}: {e}")
