# 🎭 Emotion Detection 2.0

> Real-Time Facial Emotion Recognition using CNN & OpenCV

A lightweight, CPU-optimized system that detects human emotions from a live webcam feed using a Convolutional Neural Network trained on the FER2013 dataset.

---

## 📋 10 Core Points of the Project

### 1. 📸 Camera Eye
The system accesses your webcam using `cv2.VideoCapture(0)`. Each frame is captured as a BGR (Blue-Green-Red) image matrix — a 3D numpy array of pixel values. The frame is flipped horizontally for a natural mirror effect.

### 2. 👤 Face Finder (Haar Cascade)
OpenCV's **Haar Cascade Frontal Face Default** algorithm scans each frame to locate faces. It uses pre-trained Haar-like features — simple rectangular patterns that detect intensity differences (light/dark regions) characteristic of facial structures like eyes, nose, and mouth.

### 3. 🔲 The Crop
Once a face is detected, the system extracts only the face region (Region of Interest) from the full frame using the bounding box coordinates `(x, y, w, h)`. This isolates the face from background noise.

### 4. ⬛ Grayscale Conversion
The cropped face is converted from 3-channel BGR to 1-channel grayscale. Emotion is expressed through facial **geometry** (muscle positions, shapes), not skin color. This reduces data by 3× while preserving all relevant information.

### 5. 📐 Shrink Step (48×48 Normalization)
The grayscale face is resized to exactly **48×48 pixels** — the input size the CNN was designed for (matching the FER2013 dataset). This ensures every face, regardless of original size or distance from camera, is processed identically.

### 6. 🔢 Pixel Normalization
Raw pixel values `[0-255]` are divided by `255.0` to scale them to `[0.0-1.0]`. Neural networks learn more effectively with small, normalized inputs because it prevents large values from causing unstable gradient updates during training.

### 7. 🧠 Feature Extraction (CNN Convolutional Layers)
The CNN's 4 convolutional blocks automatically learn to detect increasingly complex features:
- **Block 1 (32 filters):** Basic edges and gradients
- **Block 2 (64 filters):** Textures and curves
- **Block 3 (128 filters):** Facial parts (eyes, mouth shapes)
- **Block 4 (256 filters):** Full facial expressions

Each block uses **BatchNormalization** (stabilizes training), **MaxPooling** (reduces dimensions, adds translation invariance), and **Dropout** (prevents overfitting).

### 8. 🎲 The Probability Game (Softmax)
The final layer uses **Softmax activation** to convert raw scores into a probability distribution across 7 emotions:

```
P(class_i) = e^(z_i) / Σ e^(z_j)
```

All 7 probabilities sum to exactly **1.0**. The emotion with the highest probability wins. Example: `[0.02, 0.01, 0.03, 0.85, 0.04, 0.03, 0.02]` → **Happy (85%)**.

### 9. 🏷️ Label & Display
The predicted emotion and confidence percentage are drawn on the video frame with a color-coded bounding box around each detected face. Each emotion has a unique color for instant visual recognition.

### 10. 🔒 Privacy First
A footer reading **"Privacy Status: Local Processing Only"** is displayed on every frame. All computation happens on-device — no images or data are ever sent to any external server. No internet connection is required.

---

## 🗂️ Project Structure

```
emotion detection 2.0/
├── requirements.txt       → Python dependencies
├── model_structure.py     → CNN architecture definition (The Brain)
├── processing.py          → Face detection & preprocessing (The Logic)
├── main.py                → Live webcam application (The App)
├── README.md              → This documentation file
├── Project_Specs.txt      → Technical specifications for report
└── emotion_model.h5       → Pre-trained weights (after training)
```

---

## 🚀 How to Run (Standard PC — CPU Only)

### Prerequisites
- Python 3.8 or higher
- A working webcam

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Add Pre-Trained Weights
Place your trained `emotion_model.h5` file in the project root directory. Without weights, the model will still run but predictions will be random.

### Step 3: Run the Application
```bash
python main.py
```

### Step 4: Use the Application
- Face the webcam — the system will detect your face automatically.
- Your detected emotion and confidence level will be displayed in real-time.
- Press **`q`** to quit.

---

## ⚙️ Technical Stack

| Component         | Technology                  | Purpose                          |
|--------------------|-----------------------------|----------------------------------|
| Language           | Python 3.8+                 | Core programming language        |
| Computer Vision    | OpenCV (opencv-python)      | Webcam, face detection, drawing  |
| Deep Learning      | TensorFlow-CPU (Keras)      | CNN model inference              |
| Numerical Ops      | NumPy                       | Array manipulation               |
| Face Detection     | Haar Cascade                | Real-time frontal face detection |
| Model Architecture | 4-layer CNN                 | 7-class emotion classification   |
| Training Dataset   | FER2013 (35,887 images)     | Foundation for model weights     |

---

## 🎯 Emotion Classes

| Index | Emotion   | Bounding Box Color |
|-------|-----------|--------------------|
| 0     | Angry     | 🔴 Red             |
| 1     | Disgust   | 🟢 Dark Green      |
| 2     | Fear      | 🟣 Purple          |
| 3     | Happy     | 🟡 Yellow          |
| 4     | Sad       | 🔵 Blue            |
| 5     | Surprise  | 🟠 Orange          |
| 6     | Neutral   | ⚪ Light Gray      |

---

## 📝 License

This project is developed for academic purposes.
