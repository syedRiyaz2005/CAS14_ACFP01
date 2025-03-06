import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# 1. Load the Pre-trained Model (MobileNetV2)
model = MobileNetV2(weights='imagenet')
print("Model loaded successfully.")

# 2. Read and Preprocess the Image using OpenCV
image_path = 'food_image.jpg'  # Replace with your food image file
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image from", image_path)
else:
    # Convert from BGR (OpenCV default) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the required input size for MobileNetV2 (224x224)
    image_resized = cv2.resize(image_rgb, (224, 224))
    
    # Convert image to array and add batch dimension
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Preprocess the image (scaling pixel values as required by MobileNetV2)
    image_preprocessed = preprocess_input(image_array)
    
    
    preds = model.predict(image_preprocessed)
    
    # Decode the predictions into human-readable labels
    results = decode_predictions(preds, top=3)[0]
    
    # 4. Print the Predictions
    print("Predictions:")
    for (imagenetID, label, prob) in results:
        print(f"{label}: {prob * 100:.2f}%")
