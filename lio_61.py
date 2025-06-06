import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import json
import os

# Load your trained model
model = load_model(r"C:\Users\jerom\Downloads\model_inception_61.h5")

# Load class_indices from JSON file
with open(r"C:\Users\jerom\Desktop\internship\class_indices.json", "r") as f:
    class_indices = json.load(f)

# Create reverse mapping
idx_to_class = {v: k for k, v in class_indices.items()}

# Tkinter window setup
root = tk.Tk()
root.title("Animal Classifier")
root.geometry("400x500")

# Image display area
image_label = tk.Label(root)
image_label.pack(pady=10)

# Result label
result_label = tk.Label(root, text="Predicted: ", font=("Arial", 14))
result_label.pack(pady=10)

# Prediction function
def predict_with_model(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    predicted_class_index = int(np.argmax(predictions, axis=1)[0])
    predicted_label = idx_to_class[predicted_class_index]
    
    return predicted_label

# Button action
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img_resized = img.resize((300, 300))
        tk_img = ImageTk.PhotoImage(img_resized)
        image_label.config(image=tk_img)
        image_label.image = tk_img
        
        predicted = predict_with_model(file_path)
        result_label.config(text=f"Predicted: {predicted}")

# Select image button
browse_button = tk.Button(root, text="Select Image", command=open_image)
browse_button.pack(pady=10)

# Run the app
root.mainloop()
