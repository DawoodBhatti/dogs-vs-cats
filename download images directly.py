import os
import pandas as pd
import requests
from tqdm import tqdm

# A hacky workaround created with Copilot while the fiftyone module wasn't working for me...

# --- Configuration ---

CLASS_DESCRIPTIONS = "class-descriptions-boxable.csv"
ANNOTATIONS = "validation-annotations-bbox.csv"
IMAGES_META = "validation-images-with-rotation.csv"
OUTPUT_DIR = "images_unrelated"
MAX_DOWNLOADS = 12500
MAX_FILE_SIZE_BYTES = 1_150_000  # 🔧 Adjustable threshold: max image size in bytes (1.1 MB)

banned_keywords = [
    "cat", "dog", "nose", "canine", "feline", "mammal", "animal", "fur",
    "furry", "pet", "puppy", "kitten", "bed", "kennel", "crate", "collar", "ball",
    "carnivore"
]


# --- Step 0: Count already-downloaded images ---
existing_files = sum(
    1 for root, _, files in os.walk(OUTPUT_DIR)
    for file in files if file.lower().endswith((".jpg", ".jpeg", ".png"))
)
downloaded = existing_files
print(f"🔁 Resuming download: {downloaded} image(s) already present")


# --- Step 1: Load and filter safe classes ---
print("Loading and filtering class labels...")
class_map = pd.read_csv(CLASS_DESCRIPTIONS, header=None, names=["LabelID", "LabelName"])
label_lookup = dict(zip(class_map["LabelID"], class_map["LabelName"]))
id_lookup = dict(zip(class_map["LabelName"], class_map["LabelID"]))

safe_classes = [
    name for name in class_map["LabelName"]
    if not any(bad_kw in name.lower() for bad_kw in banned_keywords)
]
safe_label_ids = {id_lookup[name] for name in safe_classes}
print(f"🧹 Kept {len(safe_classes)} safe classes after filtering banned keywords")


# --- Step 2: Load and filter annotations ---
print("Loading and filtering annotations...")
annotations = pd.read_csv(ANNOTATIONS, usecols=["ImageID", "LabelName"])
filtered = annotations[annotations["LabelName"].isin(safe_label_ids)]

image_classes = filtered.groupby("ImageID")["LabelName"].apply(
    lambda ids: sorted(set(label_lookup[i] for i in ids if i in label_lookup))
).to_dict()
print(f"📸 Matched {len(image_classes)} images with safe classes")


# --- Step 3: Load image metadata and link classes ---
print("Linking image metadata...")
images_meta = pd.read_csv(IMAGES_META, usecols=["ImageID", "OriginalURL"])
images_meta = images_meta[images_meta["ImageID"].isin(image_classes)]
images_meta["ClassLabels"] = images_meta["ImageID"].map(image_classes)


# --- Step 4: Download images ---
print(f"⬇️ Downloading up to {MAX_DOWNLOADS} images into '{OUTPUT_DIR}/<class>/'")

for _, row in tqdm(images_meta.iterrows(), total=len(images_meta)):
    if downloaded >= MAX_DOWNLOADS:
        break

    image_id = row["ImageID"]
    url = row["OriginalURL"]
    labels = row["ClassLabels"]

    label_strs = [label.lower().replace(" ", "-") for label in labels]
    label_str = "_".join(label_strs)
    primary_class = label_strs[0]

    class_dir = os.path.join(OUTPUT_DIR, primary_class)
    os.makedirs(class_dir, exist_ok=True)

    ext = os.path.splitext(url)[1].split("?")[0] or ".jpg"
    filename = f"{label_str}_{image_id}{ext}"
    filepath = os.path.join(class_dir, filename)

    if os.path.exists(filepath):
        continue

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # 🚫 Skip if too large
        if len(response.content) > MAX_FILE_SIZE_BYTES:
            continue

        with open(filepath, "wb") as f:
            f.write(response.content)

        downloaded += 1
        print(f"\n downloaded: {downloaded}/{MAX_DOWNLOADS} images")

    except Exception:
        pass  # Or log the failure if needed

print(f"\n✅ Finished: {downloaded} image(s) saved to '{OUTPUT_DIR}/<class>/'")
