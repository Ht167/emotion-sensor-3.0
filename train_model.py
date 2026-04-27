"""
================================================================================
train_model.py — Fast Training Script
================================================================================
Loads FER2013 from image folders, trains the CNN, saves emotion_model.h5.
Optimized for speed on CPU.
================================================================================
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_structure import build_emotion_model
import cv2


# ──────────────────────────────────────────────────────────────
# Dataset path (downloaded via kagglehub)
# ──────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache", "kagglehub", "datasets", "msambare", "fer2013", "versions", "1"
)

# Emotion labels must match folder names → FER2013 index order
EMOTION_MAP = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}


def load_images_from_folder(base_path):
    """Load all images from emotion subfolders into arrays."""
    images = []
    labels = []
    total = 0

    for emotion_name, label_idx in EMOTION_MAP.items():
        folder = os.path.join(base_path, emotion_name)
        if not os.path.isdir(folder):
            print(f"  [WARN] Missing folder: {folder}")
            continue

        files = os.listdir(folder)
        count = 0
        for filename in files:
            filepath = os.path.join(folder, filename)
            # Read as grayscale
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Resize to 48x48 if needed
            if img.shape != (48, 48):
                img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
            images.append(img)
            labels.append(label_idx)
            count += 1

        total += count
        print(f"  {emotion_name:>10}: {count} images")

    print(f"  {'TOTAL':>10}: {total} images")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def load_fer2013():
    """Load FER2013 from image folder structure."""
    # Check if dataset exists
    if not os.path.isdir(DATASET_PATH):
        print(f"[INFO] Dataset not found at {DATASET_PATH}")
        print("[INFO] Downloading via kagglehub...")
        try:
            import kagglehub
            path = kagglehub.dataset_download("msambare/fer2013")
            print(f"[OK] Downloaded to: {path}")
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            print("Please run: pip install kagglehub && python -c \"import kagglehub; kagglehub.dataset_download('msambare/fer2013')\"")
            raise

    train_path = os.path.join(DATASET_PATH, "train")
    test_path = os.path.join(DATASET_PATH, "test")

    print("\n[1/4] Loading training images...")
    X_train, y_train = load_images_from_folder(train_path)

    print("\n[2/4] Loading test images...")
    X_test, y_test = load_images_from_folder(test_path)

    # Normalize & reshape
    X_train = X_train.reshape(-1, 48, 48, 1) / 255.0
    X_test = X_test.reshape(-1, 48, 48, 1) / 255.0
    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)

    print(f"\n  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def train():
    """Build, train, and save the emotion model."""
    X_train, y_train, X_test, y_test = load_fer2013()

    print("\n[3/4] Building CNN model...")
    model = build_emotion_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    datagen.fit(X_train)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint('emotion_model.h5', monitor='val_accuracy',
                        save_best_only=True, verbose=1)
    ]

    print("\n[4/4] Training... (~15-25 min on CPU)")
    print("=" * 50)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        epochs=30,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Final eval
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy: {accuracy:.2%}")
    print(f"  Test Loss:     {loss:.4f}")
    print(f"\n  Model saved to: emotion_model.h5")
    if os.path.exists('emotion_model.h5'):
        print(f"  File size:      {os.path.getsize('emotion_model.h5') / 1e6:.1f} MB")


if __name__ == "__main__":
    train()
