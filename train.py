import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# =========================
# PATHS
# =========================
BASE_DIR = "cartoon_dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")

# =========================
# PARAMETERS
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12   # you can increase later

# =========================
# DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

NUM_CLASSES = train_gen.num_classes
print("Classes found:", train_gen.class_indices)

# =========================
# MODEL (LOAD OR CREATE)
# =========================
MODEL_PATH = "model.h5"

if os.path.exists(MODEL_PATH):
    print("🔁 Found existing model. Loading and continuing training...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("🆕 No existing model found. Creating new model...")

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False  # IMPORTANT for CPU

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

model.summary()

# =========================
# CHECKPOINT (AUTO-SAVE)
# =========================
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =========================
# TRAIN / CONTINUE TRAINING
# =========================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

print("✅ Training finished. Best model saved as model.h5")
