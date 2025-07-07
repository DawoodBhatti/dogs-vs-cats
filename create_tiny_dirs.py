from os import makedirs, listdir, path
from shutil import copyfile
from tqdm import tqdm

# creating a tiny version of our dataset to dry run the model for compilation errors,
# deprecated functions etc


# === CONFIGURATION ===
src_root = 'finalize_dogs_vs_cats_vs_neither/'  # parent folder containing 'cats/', 'dogs/', etc.
dst_root = 'tiny_dogs_vs_cats_vs_neither/'
num_files = 5  # how many files to copy from each class

# === Helper to handle long paths on Windows ===
def handle_long_path(p):
    norm = path.normpath(p)
    return r'\\?\\' + norm if len(norm) > 260 else norm

# === Build dataset ===
subfolders = [d for d in listdir(src_root) if path.isdir(path.join(src_root, d))]
failures = []

for label in subfolders:
    src_folder = path.join(src_root, label)
    dst_folder = path.join(dst_root, label)

    # create destination directory
    makedirs(handle_long_path(dst_folder), exist_ok=True)

    # get files from this class
    files = [f for f in listdir(src_folder) if path.isfile(path.join(src_folder, f))]
    files_to_copy = files[:num_files]

    for file in tqdm(files_to_copy, desc=f"Copying from {label}", unit="img"):
        try:
            src = path.join(src_folder, file)
            dst = path.join(dst_folder, file)
            copyfile(handle_long_path(src), handle_long_path(dst))
        except Exception as e:
            print(f"❌ Failed to copy {label}/{file}: {e}")
            failures.append(f"{label}/{file}")

# === Summary ===
if failures:
    print(f"\n⚠️  {len(failures)} files failed to copy:")
    for f in failures:
        print(f" - {f}")
