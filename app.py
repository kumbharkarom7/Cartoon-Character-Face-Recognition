import cv2
import numpy as np
import pickle
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# 1. Load saved classifier + labels
# -----------------------------
with open("cartoon_face_model.pkl", "rb") as f:
    data = pickle.load(f)

classifier = data["classifier"]   # LogisticRegression
labels = data["labels"]           # LabelEncoder

# -----------------------------
# 2. Load MobileNet feature extractor
# -----------------------------
feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

# -----------------------------
# 3. Load and preprocess image
# -----------------------------
image_path = "test.jpg"
img = cv2.imread(image_path)

if img is None:
    raise ValueError("Image not found. Check file name.")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img.astype("float32")
img = np.expand_dims(img, axis=0)

img = preprocess_input(img)   # IMPORTANT (same as training)

# -----------------------------
# 4. Extract features (1280)
# -----------------------------
features = feature_extractor.predict(img)

# -----------------------------
# 5. Predict using classifier
# -----------------------------
pred_index = classifier.predict(features)[0]
predicted_label = labels.inverse_transform([pred_index])[0]

print("Predicted Character:", predicted_label)
