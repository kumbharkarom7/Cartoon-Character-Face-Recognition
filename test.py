import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# =========================
# PATHS
# =========================
MODEL_PATH = "model.h5"
TEST_IMAGE_PATH = "test_images/test.jpg"
TRAIN_DIR = "cartoon_dataset/train"

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# =========================
# GET CLASS NAMES
# =========================
class_names = sorted(os.listdir(TRAIN_DIR))
print("Classes:", class_names)

# =========================
# LOAD & PREPROCESS IMAGE
# =========================
img = image.load_img(TEST_IMAGE_PATH, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# =========================
# PREDICTION
# =========================
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions)
confidence = np.max(predictions)

predicted_class = class_names[predicted_index]

print("\n🎯 Prediction Result")
print("-------------------")
print("Character :", predicted_class)
print("Confidence:", round(confidence * 100, 2), "%")
