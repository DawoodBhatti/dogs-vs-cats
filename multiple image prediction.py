import os
import platform
import ctypes
import numpy as np
import csv
from tkinter import filedialog, Tk
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm  # For progress bar

# Load and prepare the image
def load_image(filename):
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32')
    #image net mean centering
    img = img - [123.68, 116.779, 103.939]
    return img

# Predict the class of an image
def classify_image(model, image_path):
    img = load_image(image_path)
    result = model.predict(img)
    class_names = ['cat', 'dog', 'neither']
    predicted_index = np.argmax(result[0])
    confidence = result[0][predicted_index]
    class_type = class_names[predicted_index]
    return class_type, confidence

# Folder selection dialog with flash and beep
def select_folder_dialog():
    root = Tk()
    root.title("Preparing folder dialog...")
    root.geometry("1x1+100+100")
    root.lift()
    root.attributes('-topmost', True)
    root.update()

    if platform.system() == "Windows":
        try:
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            ctypes.windll.user32.FlashWindow(hwnd, True)
        except Exception as e:
            print(f"[DEBUG] FlashWindow failed: {e}")

    try:
        root.bell()
    except:
        pass

    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder of Images")
    root.destroy()
    return folder_path

# Run classifier on all images in a folder and its subfolders
def classify_folder():
    folder_path = select_folder_dialog()
    if not folder_path:
        print("No folder selected.")
        return

    model_path = os.path.join(os.getcwd(), 'final_model', 'model_epoch_17.keras')
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

    # Gather all image paths first
    image_paths = []
    for root_dir, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                full_path = os.path.join(root_dir, filename)
                relative_path = os.path.relpath(full_path, folder_path)
                image_paths.append((full_path, relative_path))

    if not image_paths:
        print("No images found.")
        return

    # Output files
    txt_output = os.path.join(folder_path, "predictions.txt")
    csv_output = os.path.join(folder_path, "predictions.csv")

    # Confidence threshold for flagging
    low_conf_threshold = 0.6

    with open(txt_output, "w", encoding="utf-8") as txt_file, \
         open(csv_output, "w", newline="", encoding="utf-8") as csv_file:

        txt_file.write("relative_path\tprediction\tconfidence\n")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["relative_path", "prediction", "confidence", "flag"])

        for full_path, rel_path in tqdm(image_paths, desc="Classifying images"):
            try:
                label, confidence = classify_image(model, full_path)
                flag = "LOW_CONFIDENCE" if confidence < low_conf_threshold else ""
                txt_file.write(f"{rel_path}\t{label}\t{confidence:.4f}\n")
                csv_writer.writerow([rel_path, label, f"{confidence:.4f}", flag])
                print(f"{rel_path}: {label} ({confidence:.2%}) {flag}")
            except Exception as e:
                print(f"Error processing {rel_path}: {e}")

    print(f"\n✅ Predictions saved to:\n- {txt_output}\n- {csv_output}")

if __name__ == "__main__":
    classify_folder()
