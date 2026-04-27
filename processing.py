"""
================================================================================
processing.py — "The Logic"
================================================================================
Handles face detection (Haar Cascade) and preprocessing (grayscale, 48x48, normalize).
================================================================================
"""

import cv2
import numpy as np


class FaceProcessor:
    """
    Encapsulates face detection and preprocessing pipeline.
    
    Pipeline: Raw BGR Frame → Haar Cascade Detection → Crop → Grayscale → 
              Resize 48x48 → Normalize [0,1] → Reshape (1,48,48,1)
    """

    def __init__(self):
        """
        Initialize the Haar Cascade face detector.
        
        Haar Cascade works by:
        1. Scanning the image with a sliding window at multiple scales.
        2. Computing "Haar-like features" (intensity differences in adjacent regions).
        3. A cascade of classifiers quickly rejects non-face regions and passes
           potential faces through more stages for confirmation.
        """
        # Load OpenCV's built-in Haar Cascade XML — no manual download needed
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise IOError(f"[ERROR] Failed to load Haar Cascade from: {cascade_path}")

        print("[INFO] Haar Cascade face detector loaded successfully.")

    def detect_faces(self, frame):
        """
        Detect all faces in a video frame.

        Parameters:
            frame (np.ndarray): Raw BGR frame from webcam.

        Returns:
            faces (list of tuples): Each (x, y, w, h) bounding box.
        
        detectMultiScale parameters:
        - scaleFactor=1.3: Shrinks image by 30% at each scale (detects faces at varying distances)
        - minNeighbors=5: Requires 5 overlapping detections (reduces false positives)
        - minSize=(30,30): Ignores regions smaller than 30x30px
        """
        # Haar Cascade operates on grayscale — uses intensity differences, not color
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return faces

    def preprocess_face(self, frame, x, y, w, h):
        """
        Extract, convert, resize, and normalize a detected face region.

        Steps:
        1. CROP — extract face ROI from full frame
        2. GRAYSCALE — BGR to 1 channel (emotion = geometry, not color; 3x less data)
        3. RESIZE 48x48 — CNN input layer expects fixed dimensions
        4. NORMALIZE /255 — scale [0,255] to [0.0,1.0] for stable gradients
        5. RESHAPE (1,48,48,1) — Keras needs 4D: (batch, height, width, channels)

        Parameters:
            frame (np.ndarray): Original BGR frame.
            x, y, w, h (int): Face bounding box coordinates.

        Returns:
            processed (np.ndarray): Shape (1,48,48,1), float32, values in [0,1].
        """
        # Step 1: Crop face region
        face_roi = frame[y:y+h, x:x+w]

        # Step 2: BGR → Grayscale  (Gray = 0.299*R + 0.587*G + 0.114*B)
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Step 3: Resize to 48x48 (INTER_AREA best for downscaling — averages neighborhoods)
        resized_face = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)

        # Step 4: Normalize pixels [0,255] → [0.0,1.0]
        normalized_face = resized_face.astype(np.float32) / 255.0

        # Step 5: Reshape for Keras: (48,48) → (1,48,48,1)
        processed = np.expand_dims(normalized_face, axis=0)    # → (1,48,48)
        processed = np.expand_dims(processed, axis=-1)         # → (1,48,48,1)

        return processed
