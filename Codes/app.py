import os
import cv2
import joblib
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
from skimage.feature import hog
from playsound import playsound  # ✅ use playsound instead of pygame

# ------------------- Class mapping -------------------
class_mapping = {
    0: 'Alif', 1: 'Bay', 2: 'Tay', 3: 'Say', 4: 'Jeem', 5: 'Hay', 6: 'Khay',
    7: 'Dal', 8: 'Zal', 9: 'Ray', 10: 'Seen', 11: 'Sheen', 12: 'Sad', 13: 'Zad',
    14: 'Tuy', 15: 'Zuad', 16: 'Ain', 17: 'Ghain', 18: 'Fay', 19: 'Laam', 20: 'Meem',
    21: 'Noon', 22: 'Wao', 23: 'Ttay', 24: 'Pay', 25: 'Chay', 26: 'Ddal',
    27: 'Rray', 28: 'Zhe', 29: 'Kaf', 30: 'Gaf', 31: 'Choti Yay', 32: 'Bari Yay'
}

# ------------------- Paths -------------------
MODEL_PATH = os.path.join('models', 'final_svm_model.pkl')
SELECTOR_PATH = os.path.join('models', 'selector.pkl')
SOUND_FOLDER = os.path.join('sounds')

# ------------------- Load model & selector -------------------
try:
    model = joblib.load(MODEL_PATH)
    selector = joblib.load(SELECTOR_PATH)
except Exception as e:
    raise SystemExit(f"Model or selector could not be loaded: {e}")

expected_features = getattr(selector, "n_features_in_", None)
if expected_features is None:
    raise SystemExit("Selector does not expose expected feature size (n_features_in_).")

# ------------------- HOG parameter candidates -------------------
hog_param_candidates = [
    {"orientations": 9, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2)},
    {"orientations": 9, "pixels_per_cell": (8, 8), "cells_per_block": (3, 3)},
    {"orientations": 9, "pixels_per_cell": (4, 4), "cells_per_block": (2, 2)},
    {"orientations": 9, "pixels_per_cell": (16, 16), "cells_per_block": (2, 2)},
    {"orientations": 12, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2)},
]

# ------------------- Helper: preprocess image -------------------
def preprocess_and_match(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image (unsupported format or corrupt file).")

    img_resized = cv2.resize(img, (128, 128))
    _, img_binary = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    for params in hog_param_candidates:
        features = hog(img_binary,
                       orientations=params["orientations"],
                       pixels_per_cell=params["pixels_per_cell"],
                       cells_per_block=params["cells_per_block"],
                       block_norm='L2-Hys',
                       visualize=False)
        features = np.asarray(features).reshape(1, -1)
        if features.shape[1] == expected_features:
            reduced = selector.transform(features)
            return reduced

    flat = img_binary.flatten().reshape(1, -1)
    if flat.shape[1] == expected_features:
        return selector.transform(flat)

    raise ValueError(f"Input features do not match expected ({expected_features}).")

# ------------------- Predict and play sound -------------------
def predict_and_play(img_path):
    processed = preprocess_and_match(img_path)
    pred_class = model.predict(processed)[0]
    pronunciation = class_mapping.get(pred_class, f"Unknown({pred_class})")
    result_label.config(text=f"Prediction: {pronunciation}")

    # Play sound if available
    sound_file = os.path.join(SOUND_FOLDER, f"{pronunciation}.mp3")
    if os.path.exists(sound_file):
        try:
            playsound(sound_file, block=False)  # non-blocking playback
        except Exception as e:
            print(f"⚠️ Could not play sound: {e}")
    else:
        print(f"No sound file found for {pronunciation}")

# ------------------- UI callbacks -------------------
def upload_image():
    file_path = filedialog.askopenfilename(
        title="Select Urdu Letter Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not file_path:
        return
    show_image_preview(file_path)
    try:
        predict_and_play(file_path)
    except Exception as e:
        result_label.config(text=f"⚠️ Error: {str(e)}")

def capture_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        result_label.config(text="⚠️ Cannot access camera.")
        return

    result_label.config(text="Press SPACE to capture, ESC to cancel.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture Urdu Letter", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            result_label.config(text="Camera closed.")
            return
        elif key == 32:  # SPACE
            img_path = "captured_image.png"
            cv2.imwrite(img_path, frame)
            cap.release()
            cv2.destroyAllWindows()
            show_image_preview(img_path)
            try:
                predict_and_play(img_path)
            except Exception as e:
                result_label.config(text=f"⚠️ Error: {str(e)}")
            return

# ------------------- Image preview -------------------
def show_image_preview(img_path, size=(220, 220)):
    pil_img = Image.open(img_path).convert("L")
    pil_img = pil_img.resize(size, Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(pil_img)
    preview_label.config(image=tk_img)
    preview_label.image = tk_img

# ------------------- Tkinter UI -------------------
root = tk.Tk()
root.title("Urdu Letter Recognition (Enhanced UI)")
root.geometry("550x450")
root.configure(bg="#f8f9fa")

title_label = Label(root, text="📷 Urdu Handwritten Letter Recognition", font=("Helvetica", 16, "bold"), bg="#f8f9fa", fg="#343a40")
title_label.pack(pady=15)

frame = Frame(root, bg="#f8f9fa")
frame.pack(pady=10)

upload_btn = Button(frame, text="Upload Image", font=("Helvetica", 12), bg="#007bff", fg="white", width=15, command=upload_image)
upload_btn.grid(row=0, column=0, padx=10, pady=8)

camera_btn = Button(frame, text="Capture from Camera", font=("Helvetica", 12), bg="#28a745", fg="white", width=18, command=capture_from_camera)
camera_btn.grid(row=0, column=1, padx=10, pady=8)

result_label = Label(root, text="Prediction: ", font=("Helvetica", 14), bg="#f8f9fa", fg="#212529")
result_label.pack(pady=10)

preview_label = Label(root, bg="#dee2e6", width=220, height=220)
preview_label.pack(pady=10)

footer = Label(root, text="Developed by Jutt 🔊", font=("Helvetica", 10), bg="#f8f9fa", fg="#6c757d")
footer.pack(side="bottom", pady=10)

root.mainloop()
