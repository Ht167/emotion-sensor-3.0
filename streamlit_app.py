import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# Import your existing project logic
from processing import FaceProcessor
from model_structure import build_emotion_model

# 1. Page Configuration
st.set_page_config(page_title="Emotion Detector 2.0", page_icon="🎭")
st.title("🎭 Real-Time Emotion Detector")
st.markdown("### Team: Himanshul, Badal, & Shivansh")

# 2. Load Model (Cached so it only loads once)
@st.cache_resource
def load_trained_model():
    model = build_emotion_model()
    # Ensure the path is correct for deployment
    base_path = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_path, 'emotion_model.h5')
    model.load_weights(weights_path)
    return model

model = load_trained_model()
processor = FaceProcessor()
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 3. Streamlit Camera Input
img_file = st.camera_input("Smile for the AI!")

if img_file:
    # Convert the uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Detect faces
    faces = processor.detect_faces(frame)
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Process the face using your existing FaceProcessor
            processed_face = processor.preprocess_face(frame, x, y, w, h)
            
            # Prediction
            preds = model.predict(processed_face, verbose=0)
            emotion_idx = np.argmax(preds)
            label = EMOTION_LABELS[emotion_idx]
            confidence = preds[0][emotion_idx] * 100

            # UI Feedback
            st.success(f"Detected Emotion: **{label}** ({confidence:.2f}%)")
            
            # Show the cropped face to the evaluator to prove detection
            cropped = frame[y:y+h, x:x+w]
            st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), caption="Detected Face ROI")
    else:
        st.warning("No face detected. Try adjusting your lighting or distance.")