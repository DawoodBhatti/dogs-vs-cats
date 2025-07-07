import os, sys
import tkinter as tk
from tkinter import filedialog, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from os import getcwd, path
import numpy as np
from PIL import Image, ImageTk
from PIL.Image import Resampling


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS  # Set by PyInstaller at runtime
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Load and prepare the image
def load_image(filename):
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32')
	# specify imagenet mean values for centering
    img = img - [123.68, 116.779, 103.939]
    return img


# Predict the class of an image
def classify_image(image_path):
    cwd = getcwd()
    model_path = resource_path("final_model/model_epoch_17.keras")
    model = load_model(model_path)

    img = load_image(image_path)
    result = model.predict(img)

    class_names = ['cat', 'dog', 'neither']
    predicted_index = np.argmax(result[0])
    confidence = result[0][predicted_index]
    class_type = class_names[predicted_index]

    return class_type, confidence


# GUI Application
class DragDropApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        self.title("Image Classifier")
        self.geometry("400x460")
        self.configure(bg="#f9f9f9")
        self.files = []
        self.tk_image = None

        main_frame = tk.Frame(self, bg="#f9f9f9")
        main_frame.pack(expand=True, fill="both", padx=10, pady=10)

        self.drop_zone = tk.Label(
            main_frame,
            text="Drag and drop your image file here",
            relief="groove",
            width=40,
            height=10,
            bg="white",
            fg="gray",
            borderwidth=2
        )
        self.drop_zone.pack(pady=10)
        self.drop_zone.drop_target_register(DND_FILES)
        self.drop_zone.dnd_bind('<<Drop>>', self.handle_drop)

        button_frame = tk.Frame(main_frame, bg="#f9f9f9")
        button_frame.pack(pady=(10, 10))

        upload_btn = ttk.Button(button_frame, text="Click to Upload", command=self.open_file_dialog)
        upload_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = ttk.Button(button_frame, text="Clear Upload", command=self.clear_upload)
        clear_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(
            main_frame,
            text="",
            fg="blue",
            wraplength=350,
            justify="center",
            bg="#f9f9f9"
        )
        self.status_label.pack()

        self.result_label = tk.Label(
            main_frame,
            text="",
            fg="black",
            font=("Arial", 12, "bold"),
            bg="#f9f9f9"
        )
        self.result_label.pack(pady=(10, 0))


    def show_image_preview(self, image_path):
        def _render_preview():
            try:
                box_width = self.drop_zone.winfo_width()
                box_height = self.drop_zone.winfo_height()
    
                # If the widget hasn't rendered yet, retry shortly
                if box_width < 10 or box_height < 10:
                    self.after(100, lambda: self.show_image_preview(image_path))
                    return
    
                # Load the image
                img = Image.open(image_path)
    
                # Resize proportionally to fit the drop zone
                img.thumbnail((box_width, box_height), Resampling.LANCZOS)
    
                # Convert to Tkinter-compatible image
                self.tk_image = ImageTk.PhotoImage(img)
    
                # Update the drop zone
                self.drop_zone.config(
                    image=self.tk_image,
                    text="",
                    compound="center",
                    width=box_width,
                    height=box_height
                )
    
            except Exception as e:
                self.status_label.config(text=f"Error loading image: {e}")
    
        self.after(100, _render_preview)
    

    def handle_drop(self, event):
        paths = self.split_paths(event.data)
        if paths:
            self.files = paths
            self.status_label.config(text="\n".join(self.files))
            self.show_image_preview(paths[0])
            self.run_prediction(paths[0])


    def open_file_dialog(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")]
        )
        if file_paths:
            self.files = file_paths
            self.status_label.config(text="\n".join(self.files))
            self.show_image_preview(file_paths[0])
            self.run_prediction(file_paths[0])


    def clear_upload(self):
        self.files = []
        self.status_label.config(text="")
        self.result_label.config(text="")
        self.drop_zone.config(
            image="",
            text="Drag and drop your image file here",
            compound="center",
            width=40,   # Reset to original width in characters
            height=10   # Reset to original height in text lines
        )


    def run_prediction(self, image_path):
        try:
            label, confidence = classify_image(image_path)
            self.result_label.config(
                text=f"Prediction: {label} ({confidence:.2%} confidence)"
            )
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}")


    def split_paths(self, raw):
        if self.tk.call('tk', 'windowingsystem') == 'win32':
            return self.tk.splitlist(raw)
        return raw.strip().split()


if __name__ == "__main__":
    app = DragDropApp()
    app.mainloop()
