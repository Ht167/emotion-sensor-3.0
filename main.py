"""
================================================================================
main.py — "The Application"
================================================================================
PURPOSE:
    Real-time emotion detection using webcam feed.
    - Captures video frames via OpenCV.
    - Detects faces using Haar Cascade (processing.py).
    - Classifies emotion using CNN (model_structure.py).
    - Draws bounding boxes and emotion labels on screen.
    - Displays privacy status footer.

USAGE:
    python main.py

CONTROLS:
    Press 'q' to quit the application.
================================================================================
"""

import cv2
import numpy as np
import os
import sys

# ──────────────────────────────────────────────────────────────
# Import our custom modules
# ──────────────────────────────────────────────────────────────
from model_structure import build_emotion_model
from processing import FaceProcessor


# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

# FER2013 emotion labels (index matches CNN output neuron order)
EMOTION_LABELS = [
    'Angry',      # Index 0
    'Disgust',    # Index 1
    'Fear',       # Index 2
    'Happy',      # Index 3
    'Sad',        # Index 4
    'Surprise',   # Index 5
    'Neutral'     # Index 6
]

# Color palette for bounding boxes (BGR format for OpenCV)
EMOTION_COLORS = {
    'Angry':    (0, 0, 255),       # Red
    'Disgust':  (0, 128, 0),       # Dark Green
    'Fear':     (128, 0, 128),     # Purple
    'Happy':    (0, 255, 255),     # Yellow
    'Sad':      (255, 128, 0),     # Blue-ish
    'Surprise': (0, 165, 255),     # Orange
    'Neutral':  (200, 200, 200)    # Light Gray
}

# Path to the pre-trained model weights
MODEL_WEIGHTS_PATH = 'emotion_model.h5'


def load_model():
    """
    Build the CNN architecture and load pre-trained weights.
    If no weights file is found, the model runs with random weights
    (predictions will be meaningless but the pipeline still works).
    """
    print("\n[INFO] Building CNN model architecture...")
    model = build_emotion_model()

    if os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"[INFO] Loading pre-trained weights from: {MODEL_WEIGHTS_PATH}")
        model.load_weights(MODEL_WEIGHTS_PATH)
        print("[INFO] ✅ Weights loaded successfully!")
    else:
        print(f"[WARNING] No weights file found at '{MODEL_WEIGHTS_PATH}'.")
        print("[WARNING] Running with untrained model — predictions will be random.")
        print("[WARNING] Train the model on FER2013 first, then save weights as 'emotion_model.h5'.")

    return model


def draw_footer(frame, text="Privacy Status: Local Processing Only"):
    """
    Draw a semi-transparent privacy status footer at the bottom of the frame.
    All processing happens locally — no data is sent to any server.
    """
    h, w = frame.shape[:2]
    footer_height = 40

    # Create a dark overlay for the footer region
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - footer_height), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw the privacy text centered in the footer
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - (footer_height - text_size[1]) // 2

    # Green lock icon indicator + text
    cv2.putText(frame, ">> " + text + " <<", (text_x - 20, text_y),
                font, font_scale, (0, 220, 0), thickness, cv2.LINE_AA)


def draw_detection(frame, x, y, w, h, emotion, confidence):
    """
    Draw bounding box and emotion label on the frame.
    """
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))

    # Draw bounding box around the face
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Prepare label with confidence percentage
    label = f"{emotion} ({confidence:.0%})"

    # Draw label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0] + 5, y), color, -1)

    # Draw label text (black on colored background)
    cv2.putText(frame, label, (x + 2, y - 5),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def main():
    """
    Main application loop:
    1. Initialize model and face processor
    2. Open webcam
    3. For each frame: detect faces → preprocess → predict → draw
    4. Display result and wait for 'q' to quit
    """
    print("=" * 60)
    print("  EMOTION DETECTION 2.0 — Real-Time CNN Classifier")
    print("=" * 60)

    # ──────────────────────────────────────────────────────
    # Step 1: Initialize components
    # ──────────────────────────────────────────────────────
    model = load_model()
    processor = FaceProcessor()

    # ──────────────────────────────────────────────────────
    # Step 2: Open webcam (device index 0 = default camera)
    # ──────────────────────────────────────────────────────
    print("\n[INFO] Opening webcam (device 0)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check camera connection.")
        sys.exit(1)

    # Set resolution for smoother display
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Webcam opened successfully!")
    print("[INFO] Press 'q' to quit.\n")

    # ──────────────────────────────────────────────────────
    # Step 3: Main processing loop
    # ──────────────────────────────────────────────────────
    while True:
        # Capture a single frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to capture frame. Exiting.")
            break

        # Flip horizontally for mirror effect (more intuitive)
        frame = cv2.flip(frame, 1)

        # ──────────────────────────────────────────────────
        # Detect all faces in the current frame
        # ──────────────────────────────────────────────────
        faces = processor.detect_faces(frame)

        # ──────────────────────────────────────────────────
        # Process each detected face
        # ──────────────────────────────────────────────────
        for (x, y, w, h) in faces:
            # Preprocess: crop → grayscale → 48x48 → normalize → reshape
            processed_face = processor.preprocess_face(frame, x, y, w, h)

            # ──────────────────────────────────────────────
            # CNN PREDICTION (The Softmax Step):
            # model.predict() returns shape (1, 7) — a probability
            # distribution across all 7 emotion classes.
            #
            # Example output:
            #   [[0.02, 0.01, 0.03, 0.82, 0.05, 0.04, 0.03]]
            #     Angry Disg  Fear  Happy Sad  Surp  Neut
            #
            # np.argmax() finds the INDEX of the highest probability.
            # That index maps to our EMOTION_LABELS list.
            # ──────────────────────────────────────────────
            predictions = model.predict(processed_face, verbose=0)
            emotion_index = np.argmax(predictions[0])
            emotion_label = EMOTION_LABELS[emotion_index]
            confidence = predictions[0][emotion_index]

            # Draw bounding box and emotion label on the frame
            draw_detection(frame, x, y, w, h, emotion_label, confidence)

        # ──────────────────────────────────────────────────
        # Draw the privacy footer
        # ──────────────────────────────────────────────────
        draw_footer(frame)

        # ──────────────────────────────────────────────────
        # Display the annotated frame
        # ──────────────────────────────────────────────────
        cv2.imshow('Emotion Detection 2.0', frame)

        # ──────────────────────────────────────────────────
        # Check for 'q' key press to exit
        # waitKey(1) waits 1ms — keeps ~30 FPS loop speed
        # ──────────────────────────────────────────────────
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] 'q' pressed — shutting down.")
            break

    # ──────────────────────────────────────────────────────
    # Step 4: Cleanup
    # ──────────────────────────────────────────────────────
    cap.release()          # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print("[INFO] Application closed. Goodbye!")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
