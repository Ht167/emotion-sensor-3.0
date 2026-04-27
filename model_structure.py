"""
================================================================================
model_structure.py — "The Brain"
================================================================================
PURPOSE:
    Defines the Convolutional Neural Network (CNN) architecture used for
    7-class facial emotion classification.

MODEL ARCHITECTURE OVERVIEW:
    ┌─────────────────────────────────────────────────────────┐
    │  Input: 48×48 Grayscale Image (1 channel)               │
    │                                                         │
    │  Conv Block 1 → 32 filters  (3×3) + BN + MaxPool + Drop│
    │  Conv Block 2 → 64 filters  (3×3) + BN + MaxPool + Drop│
    │  Conv Block 3 → 128 filters (3×3) + BN + MaxPool + Drop│
    │  Conv Block 4 → 256 filters (3×3) + BN + MaxPool + Drop│
    │                                                         │
    │  Flatten → Dense(256) → Dropout → Dense(7, softmax)     │
    └─────────────────────────────────────────────────────────┘

TRAINING FOUNDATION:
    FER2013 dataset — 35,887 labeled facial images across 7 emotions.
    This file defines the architecture ONLY. Training is done separately.

KEY CONCEPTS FOR PRESENTATION:
    - Feature Extraction: Conv layers learn to detect edges, textures,
      and facial patterns automatically (no manual feature engineering).
    - Softmax: Converts raw scores into a probability distribution
      across all 7 emotion classes (probabilities sum to 1.0).
================================================================================
"""

# ──────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────
from tensorflow.keras.models import Sequential          # Linear stack of layers
from tensorflow.keras.layers import (
    Conv2D,             # 2D Convolution — slides a filter kernel across the image
    MaxPooling2D,       # Downsamples by taking the max value in each patch
    BatchNormalization, # Normalizes activations — stabilizes & speeds up training
    Dropout,            # Randomly disables neurons — prevents overfitting
    Flatten,            # Converts 2D feature maps into a 1D vector
    Dense               # Fully-connected layer — maps features to class scores
)


def build_emotion_model():
    """
    Constructs and returns the CNN model for emotion recognition.

    Returns:
        model (Sequential): A compiled-ready Keras model.
                             Input shape  → (48, 48, 1)   [grayscale image]
                             Output shape → (7,)          [probability per emotion]

    ────────────────────────────────────────────────────────────
    FEATURE EXTRACTION (Convolutional Blocks):
    ────────────────────────────────────────────────────────────
    Each convolutional block performs three operations:

    1. Conv2D:
       - Slides small 3×3 filter windows across the image.
       - Each filter learns to detect a specific pattern
         (edges in early layers, complex shapes in deeper layers).
       - 'relu' activation: keeps positive values, zeroes negatives.
         This introduces non-linearity so the network can learn
         complex, non-linear relationships in facial features.

    2. BatchNormalization:
       - Normalizes the output of each layer to have mean ≈ 0
         and standard deviation ≈ 1.
       - Why? Prevents "internal covariate shift" — keeps gradients
         healthy and allows higher learning rates.

    3. MaxPooling2D (2×2):
       - Reduces spatial dimensions by 50% in each direction.
       - Keeps only the strongest activation in each 2×2 region.
       - Effect: Makes the model translation-invariant (a smile
         shifted a few pixels is still recognized as a smile).

    4. Dropout:
       - Randomly sets a fraction of neurons to zero during training.
       - Forces the network to learn redundant representations,
         preventing over-reliance on any single neuron.
       - Critical for small datasets like FER2013.

    ────────────────────────────────────────────────────────────
    CLASSIFICATION (Dense Layers):
    ────────────────────────────────────────────────────────────
    After feature extraction, the spatial feature maps are flattened
    into a 1D vector and passed through fully-connected layers.

    The FINAL Dense layer uses SOFTMAX activation:
        softmax(z_i) = exp(z_i) / Σ exp(z_j)

    This converts raw logits into a probability distribution:
        - Each output neuron represents one emotion class.
        - All 7 outputs sum to exactly 1.0.
        - The class with the highest probability is the prediction.

    Example output: [0.02, 0.85, 0.03, 0.01, 0.05, 0.02, 0.02]
                     Angry  Happy  Sad  Disgust Fear Surprise Neutral
                     → Prediction: Happy (85% confidence)
    """

    # ──────────────────────────────────────────────────────────
    # Initialize a Sequential model (layers added one after another)
    # ──────────────────────────────────────────────────────────
    model = Sequential(name="EmotionCNN")

    # ══════════════════════════════════════════════════════════
    # BLOCK 1: First Convolutional Block — Edge Detection
    # ══════════════════════════════════════════════════════════
    # Input: 48×48×1 grayscale image
    # This block learns to detect basic edges and gradients.
    # 32 filters means 32 different edge patterns are learned.
    model.add(Conv2D(
        filters=32,                # Number of learnable filters
        kernel_size=(3, 3),        # Each filter is 3×3 pixels
        padding='same',            # Pad input so output size = input size
        activation='relu',         # ReLU: max(0, x) — introduces non-linearity
        input_shape=(48, 48, 1)    # First layer must specify input dimensions
    ))
    model.add(BatchNormalization())    # Normalize activations for stable training
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Downsample: 48×48 → 24×24
    model.add(Dropout(0.25))           # Drop 25% of neurons to prevent overfitting

    # ══════════════════════════════════════════════════════════
    # BLOCK 2: Second Convolutional Block — Texture Detection
    # ══════════════════════════════════════════════════════════
    # Input: 24×24×32 (from Block 1)
    # This block combines edges into textures (wrinkles, curves).
    # 64 filters = double the capacity to learn richer patterns.
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Downsample: 24×24 → 12×12
    model.add(Dropout(0.25))

    # ══════════════════════════════════════════════════════════
    # BLOCK 3: Third Convolutional Block — Shape Detection
    # ══════════════════════════════════════════════════════════
    # Input: 12×12×64 (from Block 2)
    # This block detects higher-level shapes: mouths, eyebrows.
    # 128 filters for more complex pattern combinations.
    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Downsample: 12×12 → 6×6
    model.add(Dropout(0.25))

    # ══════════════════════════════════════════════════════════
    # BLOCK 4: Fourth Convolutional Block — Expression Detection
    # ══════════════════════════════════════════════════════════
    # Input: 6×6×128 (from Block 3)
    # This block combines shapes into full facial expressions.
    # 256 filters for the most abstract, high-level features.
    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Downsample: 6×6 → 3×3
    model.add(Dropout(0.25))

    # ══════════════════════════════════════════════════════════
    # FLATTEN: Convert 2D Feature Maps → 1D Feature Vector
    # ══════════════════════════════════════════════════════════
    # At this point we have 256 feature maps, each 3×3 pixels.
    # Flatten converts this 3D tensor (3, 3, 256) = 2304 values
    # into a single 1D vector of 2304 numbers.
    model.add(Flatten())

    # ══════════════════════════════════════════════════════════
    # DENSE LAYER: Fully-Connected Classification Head
    # ══════════════════════════════════════════════════════════
    # This layer connects every flattened feature to 256 neurons.
    # It learns WHICH combinations of features correspond to
    # which emotions (e.g., raised eyebrows + open mouth = Surprise).
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Higher dropout (50%) before final layer

    # ══════════════════════════════════════════════════════════
    # OUTPUT LAYER: 7-Class Softmax Probability Distribution
    # ══════════════════════════════════════════════════════════
    # 7 neurons — one per emotion class.
    # Softmax activation ensures outputs are valid probabilities
    # (all positive, sum to 1.0).
    #
    # THE SOFTMAX FUNCTION:
    #   For each class i:  P(class_i) = e^(z_i) / Σ e^(z_j)
    #
    #   - e^(z_i) amplifies differences between scores
    #   - Division by the sum normalizes to a probability distribution
    #   - The class with the HIGHEST probability is the prediction
    #
    # Emotion mapping (by FER2013 index):
    #   0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy,
    #   4 = Sad,   5 = Surprise, 6 = Neutral
    model.add(Dense(
        units=7,               # One neuron per emotion class
        activation='softmax'   # Probability distribution output
    ))

    return model


# ──────────────────────────────────────────────────────────────
# STANDALONE EXECUTION: Print model summary for verification
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  EMOTION CNN — Architecture Summary")
    print("=" * 60)
    model = build_emotion_model()
    model.summary()
    print("\n✅ Model built successfully!")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Output classes:   7 (softmax)")
    print(f"   Input shape:      (48, 48, 1) grayscale")
