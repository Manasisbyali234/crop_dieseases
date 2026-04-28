# CNN Layers Documentation — Crop Disease Detection System

## Overview

This document describes the Convolutional Neural Network (CNN) architecture proposed for the
Crop Disease Detection System. The current project uses OpenCV-based color thresholding.
This document covers how CNN layers replace that with a real deep learning pipeline.

---

## Module: `cnn_model.py`

**Framework:** TensorFlow / Keras  
**Input Shape:** 224 × 224 × 3 (RGB image)  
**Output Classes:** 4 — Healthy, Leaf Blight, Powdery Mildew, Rust Disease

---

## CNN Architecture — Layer-by-Layer

### 1. Input Layer

| Property     | Value          |
|--------------|----------------|
| Module       | `keras.Input`  |
| Shape        | (224, 224, 3)  |
| Description  | Accepts a normalized RGB image array |

```python
from tensorflow.keras import Input
Input(shape=(224, 224, 3))
```

---

### 2. Convolutional Block 1 — Low-Level Feature Extraction

#### Layer: Conv2D (32 filters)

| Property     | Value                          |
|--------------|--------------------------------|
| Module       | `tensorflow.keras.layers.Conv2D` |
| Filters      | 32                             |
| Kernel Size  | (3, 3)                         |
| Activation   | ReLU                           |
| Purpose      | Detects basic edges, color gradients, and textures in the leaf image |

```python
from tensorflow.keras import layers
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3))
```

#### Layer: MaxPooling2D

| Property     | Value                              |
|--------------|------------------------------------|
| Module       | `tensorflow.keras.layers.MaxPooling2D` |
| Pool Size    | (2, 2)                             |
| Purpose      | Reduces spatial dimensions by half, retains dominant features |

```python
layers.MaxPooling2D((2, 2))
```

---

### 3. Convolutional Block 2 — Mid-Level Pattern Detection

#### Layer: Conv2D (64 filters)

| Property     | Value                          |
|--------------|--------------------------------|
| Module       | `tensorflow.keras.layers.Conv2D` |
| Filters      | 64                             |
| Kernel Size  | (3, 3)                         |
| Activation   | ReLU                           |
| Purpose      | Detects disease-specific patterns like spots, patches, and lesions |

```python
layers.Conv2D(64, (3, 3), activation='relu')
```

#### Layer: MaxPooling2D

| Property     | Value                              |
|--------------|------------------------------------|
| Module       | `tensorflow.keras.layers.MaxPooling2D` |
| Pool Size    | (2, 2)                             |
| Purpose      | Further reduces spatial size, improves computational efficiency |

```python
layers.MaxPooling2D((2, 2))
```

---

### 4. Convolutional Block 3 — High-Level Disease Feature Learning

#### Layer: Conv2D (128 filters)

| Property     | Value                          |
|--------------|--------------------------------|
| Module       | `tensorflow.keras.layers.Conv2D` |
| Filters      | 128                            |
| Kernel Size  | (3, 3)                         |
| Activation   | ReLU                           |
| Purpose      | Learns complex disease-specific representations (rust pustules, mildew coating, blight necrosis) |

```python
layers.Conv2D(128, (3, 3), activation='relu')
```

#### Layer: MaxPooling2D

| Property     | Value                              |
|--------------|------------------------------------|
| Module       | `tensorflow.keras.layers.MaxPooling2D` |
| Pool Size    | (2, 2)                             |
| Purpose      | Final spatial reduction before classification |

```python
layers.MaxPooling2D((2, 2))
```

---

### 5. Flatten Layer

| Property     | Value                           |
|--------------|---------------------------------|
| Module       | `tensorflow.keras.layers.Flatten` |
| Purpose      | Converts 3D feature maps (height × width × channels) into a 1D vector for the Dense layers |

```python
layers.Flatten()
```

---

### 6. Dense Layer — Fully Connected Classifier

| Property     | Value                         |
|--------------|-------------------------------|
| Module       | `tensorflow.keras.layers.Dense` |
| Units        | 128                           |
| Activation   | ReLU                          |
| Purpose      | Learns non-linear combinations of extracted features to classify disease type |

```python
layers.Dense(128, activation='relu')
```

---

### 7. Dropout Layer — Regularization

| Property     | Value                            |
|--------------|----------------------------------|
| Module       | `tensorflow.keras.layers.Dropout` |
| Rate         | 0.5                              |
| Purpose      | Randomly disables 50% of neurons during training to prevent overfitting |

```python
layers.Dropout(0.5)
```

---

### 8. Output Layer — Softmax Classifier

| Property     | Value                         |
|--------------|-------------------------------|
| Module       | `tensorflow.keras.layers.Dense` |
| Units        | 4                             |
| Activation   | Softmax                       |
| Purpose      | Outputs probability distribution across 4 disease classes |
| Classes      | `[Healthy, Leaf Blight, Powdery Mildew, Rust Disease]` |

```python
layers.Dense(4, activation='softmax')
```

---

## Complete Model Code — `cnn_model.py`

```python
# Module: cnn_model.py
# Framework: TensorFlow >= 2.x / Keras

from tensorflow.keras import models, layers

CLASS_NAMES = ["Healthy", "Leaf Blight", "Powdery Mildew", "Rust Disease"]

def build_cnn():
    model = models.Sequential([
        # Block 1 — Low-level features
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),

        # Block 2 — Mid-level patterns
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Block 3 — High-level disease features
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

---

## Image Preprocessing Module — `preprocess.py`

Used to prepare images before passing to the CNN.

```python
# Module: preprocess.py
# Modules used: numpy, cv2 (opencv-python), tensorflow.keras.utils

import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def preprocess_image(image_path):
    """Resize and normalize image for CNN input."""
    img = cv2.imread(image_path)                        # cv2 — read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # cv2 — BGR to RGB
    img = cv2.resize(img, (224, 224))                   # cv2 — resize to 224x224
    img = img.astype('float32') / 255.0                 # numpy — normalize to [0, 1]
    return np.expand_dims(img, axis=0)                  # numpy — add batch dimension
```

---

## Prediction Module — Integration with `disease_detector.py`

```python
# Replaces analyze_image_features() in disease_detector.py
# Modules used: numpy, cnn_model, preprocess

import numpy as np
from cnn_model import CLASS_NAMES
from preprocess import preprocess_image

def predict_with_cnn(model, image_path):
    img_array = preprocess_image(image_path)            # preprocess — normalize image
    predictions = model.predict(img_array)              # keras model — run inference
    disease_idx = np.argmax(predictions)                # numpy — get highest probability index
    confidence = int(np.max(predictions) * 100)         # numpy — convert to percentage
    disease_name = CLASS_NAMES[disease_idx]
    return disease_name, confidence
```

---

## Layer Summary Table

| # | Layer          | Module                              | Output Shape       | Parameters | Purpose                          |
|---|----------------|-------------------------------------|--------------------|------------|----------------------------------|
| 1 | Input          | `keras.Input`                       | (224, 224, 3)      | 0          | Raw RGB image input              |
| 2 | Conv2D(32)     | `keras.layers.Conv2D`               | (222, 222, 32)     | 896        | Edge & texture detection         |
| 3 | MaxPooling2D   | `keras.layers.MaxPooling2D`         | (111, 111, 32)     | 0          | Spatial downsampling             |
| 4 | Conv2D(64)     | `keras.layers.Conv2D`               | (109, 109, 64)     | 18,496     | Spot & patch pattern detection   |
| 5 | MaxPooling2D   | `keras.layers.MaxPooling2D`         | (54, 54, 64)       | 0          | Spatial downsampling             |
| 6 | Conv2D(128)    | `keras.layers.Conv2D`               | (52, 52, 128)      | 73,856     | Disease-specific feature learning|
| 7 | MaxPooling2D   | `keras.layers.MaxPooling2D`         | (26, 26, 128)      | 0          | Spatial downsampling             |
| 8 | Flatten        | `keras.layers.Flatten`              | (86528,)           | 0          | 3D → 1D conversion               |
| 9 | Dense(128)     | `keras.layers.Dense`                | (128,)             | 11,075,712 | Non-linear classification        |
|10 | Dropout(0.5)   | `keras.layers.Dropout`              | (128,)             | 0          | Overfitting prevention           |
|11 | Dense(4)       | `keras.layers.Dense`                | (4,)               | 516        | Disease class probabilities      |

**Total Trainable Parameters:** ~11,169,476

---

## Required Dependencies

Add to `requirements.txt`:

```
tensorflow>=2.10.0
opencv-python>=4.5.0
Pillow>=8.0.0
numpy>=1.19.0
pandas>=1.3.0
openpyxl>=3.0.0
```

---

## Module Import Reference

| Module                              | Install Command              | Used For                        |
|-------------------------------------|------------------------------|---------------------------------|
| `tensorflow.keras.models`           | `pip install tensorflow`     | Building Sequential model       |
| `tensorflow.keras.layers.Conv2D`    | `pip install tensorflow`     | Convolutional feature extraction|
| `tensorflow.keras.layers.MaxPooling2D` | `pip install tensorflow`  | Spatial downsampling            |
| `tensorflow.keras.layers.Flatten`   | `pip install tensorflow`     | Feature map flattening          |
| `tensorflow.keras.layers.Dense`     | `pip install tensorflow`     | Fully connected classification  |
| `tensorflow.keras.layers.Dropout`   | `pip install tensorflow`     | Regularization                  |
| `cv2` (opencv-python)               | `pip install opencv-python`  | Image reading & preprocessing   |
| `numpy`                             | `pip install numpy`          | Array operations & predictions  |

---

*Document generated for: Crop Disease Detection System v1.2.4*
