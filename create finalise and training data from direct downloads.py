import os
import shutil
from PIL import Image
from os import makedirs, listdir, path
from shutil import copyfile
from random import seed, random
from tqdm import tqdm

#%% First cell moves downloaded images into single destination (folder)
source_folder = r'C:\Users\Ali B\Documents\Python projects\dogs-vs-cats\images_unrelated'
destination_folder = r'C:\Users\Ali B\Documents\Python projects\dogs-vs-cats\finalize_dogs_vs_cats_vs_neither\neither'

# Add prefix only if path length exceeds 260
def handle_long_path(path):
    norm_path = os.path.normpath(path)
    return r'\\?\\' + norm_path if len(norm_path) > 260 else norm_path

# Ensure destination folder exists
os.makedirs(handle_long_path(destination_folder), exist_ok=True)

# Walk through all subdirectories
for dirpath, _, filenames in os.walk(handle_long_path(source_folder)):
    for file in filenames:
        raw_source = os.path.join(dirpath, file)
        raw_dest = os.path.join(destination_folder, file)

        source_path = handle_long_path(raw_source)
        destination_path = handle_long_path(raw_dest)

        # Ensure unique filenames in destination
        base, extension = os.path.splitext(file)
        counter = 1
        while os.path.exists(destination_path):
            raw_dest = os.path.join(destination_folder, f"{base}_{counter}{extension}")
            destination_path = handle_long_path(raw_dest)
            counter += 1

        # Check if file exists before moving
        if os.path.exists(source_path):
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.move(source_path, destination_path)
        else:
            print(f"Missing: {source_path}")


#%% Second cell compresses images to reduce filesize

# === CONFIGURATION ===
source_folder = r'C:\Users\Ali B\Documents\Python projects\dogs-vs-cats\finalize_dogs_vs_cats_vs_neither\neither'
destination_folder = os.path.join(source_folder, 'compressed')
os.makedirs(destination_folder, exist_ok=True)

# === HELPER: Handle long paths on Windows ===
def handle_long_path(path):
    norm = os.path.normpath(path)
    return r'\\?\\' + norm if len(norm) > 260 else norm

# === COLLECT IMAGE FILES ===
image_files = [
    f for f in os.listdir(source_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

# === COMPRESS IMAGES WITH PROGRESS BAR ===
for file in tqdm(image_files, desc="Compressing images", unit="img"):
    img_path = os.path.join(source_folder, file)
    try:
        img = Image.open(handle_long_path(img_path))

        # Resize only if both new dimensions would stay >= 250px
        new_width = img.width // 2
        new_height = img.height // 2
        if new_width >= 250 and new_height >= 250:
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            # Calculate scale factor that won’t shrink below 250
            scale = min(img.width / 250, img.height / 250, 1.0)
            new_size = (int(img.width / scale), int(img.height / scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save as JPEG with compression
        output_name = os.path.splitext(file)[0] + '.jpg'
        output_path = os.path.join(destination_folder, output_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(handle_long_path(output_path), format='JPEG', quality=70, optimize=True)

    except Exception as e:
        print(f"❌ Failed to compress {file}: {e}")

#%% Third cell: we can manually delete downloaded images as we now have their compressed counterparts


#%% Fourth cell: we create a split of the compressed data to use for any training 
# === LONG PATH HANDLER ===
def handle_long_path(p):
    norm = path.normpath(p)
    return r'\\?\\' + norm if len(norm) > 260 else norm

# === CREATE DIRECTORIES ===
dataset_home = 'dataset_dogs_vs_cats_vs_neither/'
subdirs = ['train/', 'test/']
labeldirs = ['neither/']

for subdir in subdirs:
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(handle_long_path(newdir), exist_ok=True)

# === SPLIT LOGIC ===
seed(1)
val_ratio = 0.25

src_directory = 'finalize_dogs_vs_cats_vs_neither/neither/compressed/'
files = listdir(src_directory)

failures = []

# === COPY WITH SAFETY AND PROGRESS ===
for file in tqdm(files, desc="Splitting dataset", unit="img"):
    try:
        src = path.join(src_directory, file)
        dst_dir = 'test/' if random() < val_ratio else 'train/'
        dst = path.join(dataset_home, dst_dir, 'neither', file)

        copyfile(handle_long_path(src), handle_long_path(dst))
    except Exception as e:
        print(f"❌ Failed to copy {file}: {e}")
        failures.append(file)

# === SUMMARY ===
if failures:
    print(f"\n⚠️  {len(failures)} files failed to copy:")
    for f in failures:
        print(f" - {f}")
        
#%% Fifth cell: we can manually move the files out of the compressed folder into the level where the folder is
#               and delete the compressed folder so that the structure follows cats and dogs...
