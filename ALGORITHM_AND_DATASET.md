# Crop Disease Detection System
# Algorithm & Dataset Documentation

---

## TABLE OF CONTENTS

1. Overview
2. Dataset Description
3. Dataset Feature Schema
4. Dataset Values (Full Table)
5. Algorithm 1 — k-NN Classifier (Image Analysis)
6. Algorithm 2 — CNN (Convolutional Neural Network)
7. Algorithm 3 — CSV Keyword Classifier
8. Confidence Score Calculation
9. How Algorithms Are Selected (Decision Flow)
10. Feature Extraction Process
11. Disease Area Marking Algorithm
12. Crop Validation Algorithm
13. Summary Table

---

## 1. OVERVIEW

The system uses THREE algorithms depending on the situation:

| Situation | Algorithm Used |
|---|---|
| Trained CNN model file exists (`crop_cnn_model.h5`) | CNN (TensorFlow/Keras) |
| No trained model file | k-NN Classifier on built-in HSV dataset |
| CSV / Excel file uploaded | Keyword Classifier + Dataset Confidence |

---

## 2. DATASET DESCRIPTION

### Type
Built-in Reference Dataset — embedded directly in `disease_detector.py` as a NumPy array.
No external file is required. The dataset is always available.

### Purpose
Used by the k-NN classifier to identify crop diseases when no trained CNN model is present.
Also used to compute deterministic confidence scores for CSV analysis.

### Source
HSV (Hue, Saturation, Value) feature profiles derived from representative crop leaf image statistics
for each of the 4 disease classes.

### Size
- 16 samples total
- 4 samples per disease class
- 5 features per sample

### Classes
| Class Index | Class Name |
|---|---|
| 0 | Healthy |
| 1 | Leaf Blight |
| 2 | Powdery Mildew |
| 3 | Rust Disease |

---

## 3. DATASET FEATURE SCHEMA

Each sample in the dataset has 5 features:

| Feature Index | Feature Name | Description | Unit / Range |
|---|---|---|---|
| 0 | hue_mean | Mean Hue value of the image in HSV space | 0 – 180 (OpenCV scale) |
| 1 | sat_mean | Mean Saturation value | 0 – 255 |
| 2 | val_mean | Mean Value (brightness) | 0 – 255 |
| 3 | green_ratio | Fraction of pixels in green HSV range [30–95, 40–255, 40–255] | 0.0 – 1.0 |
| 4 | dark_ratio | Fraction of pixels with Value < 60 (dark/necrotic areas) | 0.0 – 1.0 |

---

## 4. DATASET VALUES (FULL TABLE)

```
Index | Class          | hue_mean | sat_mean | val_mean | green_ratio | dark_ratio
------|----------------|----------|----------|----------|-------------|----------
  0   | Healthy        |   55     |   120    |   160    |    0.55     |   0.05
  1   | Healthy        |   60     |   130    |   155    |    0.58     |   0.04
  2   | Healthy        |   50     |   115    |   165    |    0.52     |   0.06
  3   | Healthy        |   58     |   125    |   158    |    0.56     |   0.05
  4   | Leaf Blight    |   18     |    80    |    60    |    0.10     |   0.35
  5   | Leaf Blight    |   22     |    90    |    55    |    0.08     |   0.40
  6   | Leaf Blight    |   15     |    75    |    65    |    0.12     |   0.32
  7   | Leaf Blight    |   20     |    85    |    58    |    0.09     |   0.38
  8   | Powdery Mildew |   10     |    20    |   230    |    0.15     |   0.02
  9   | Powdery Mildew |   15     |    18    |   240    |    0.12     |   0.02
 10   | Powdery Mildew |   12     |    22    |   235    |    0.14     |   0.02
 11   | Powdery Mildew |    8     |    16    |   245    |    0.10     |   0.01
 12   | Rust Disease   |   20     |   180    |   140    |    0.08     |   0.10
 13   | Rust Disease   |   25     |   190    |   135    |    0.07     |   0.12
 14   | Rust Disease   |   18     |   175    |   145    |    0.09     |   0.09
 15   | Rust Disease   |   28     |   185    |   130    |    0.06     |   0.11
```

### Class Signatures (What Makes Each Class Unique)

| Class | Key Distinguishing Features |
|---|---|
| Healthy | High green_ratio (0.52–0.58), moderate hue (50–60), low dark_ratio |
| Leaf Blight | Low val_mean (55–65), high dark_ratio (0.32–0.40), low green_ratio |
| Powdery Mildew | Very low sat_mean (16–22), very high val_mean (230–245), low dark_ratio |
| Rust Disease | High sat_mean (175–190), orange hue range (18–28), low green_ratio |

---

## 5. ALGORITHM 1 — k-NN CLASSIFIER (IMAGE ANALYSIS FALLBACK)

### Name
k-Nearest Neighbors (k-NN) with k=3, Majority Vote

### Used When
No trained CNN model file (`crop_cnn_model.h5`) is found.

### Input
A crop leaf image (JPG, PNG, BMP, TIFF)

### Output
- Predicted disease class name (string)
- Confidence score (integer, 60–96%)

### Step-by-Step Process

```
Step 1: Feature Extraction
        Read image → Convert BGR to HSV
        Extract 5 features:
          - hue_mean   = mean of H channel
          - sat_mean   = mean of S channel
          - val_mean   = mean of V channel
          - green_ratio = pixels in HSV range [30-95, 40-255, 40-255] / total pixels
          - dark_ratio  = pixels with V < 60 / total pixels

Step 2: Normalisation (Min-Max Scaling)
        For each of the 5 features:
          col_min = minimum value in dataset column
          col_max = maximum value in dataset column
          normalised = (value - col_min) / (col_max - col_min)
        Both the query image features and all 16 dataset samples are normalised.

Step 3: Distance Calculation (Euclidean)
        For each of the 16 dataset samples:
          distance = sqrt( sum( (feat_i - sample_i)^2 ) for i in 0..4 )
        Result: 16 distance values

Step 4: k=3 Nearest Neighbors
        Sort distances ascending → take top 3 closest samples
        Collect their class labels (votes)

Step 5: Majority Vote
        Count votes per class among top 3
        Predicted class = class with most votes

Step 6: Confidence Calculation
        Among the top-3 neighbors that match the predicted class,
        find the minimum distance (closest match).
        confidence = 96 - (min_distance × 60)
        Clamped to range [60, 96]
```

### Example

```
Query image features (normalised): [0.62, 0.48, 0.55, 0.60, 0.08]
Nearest 3 samples: [index 0 (Healthy, d=0.05), index 1 (Healthy, d=0.07), index 3 (Healthy, d=0.09)]
Votes: Healthy=3, others=0
Predicted: Healthy
Confidence: 96 - (0.05 × 60) = 93%
```

---

## 6. ALGORITHM 2 — CNN (CONVOLUTIONAL NEURAL NETWORK)

### Name
Convolutional Neural Network (CNN) — Sequential Architecture

### Used When
Trained model file `crop_cnn_model.h5` exists in the project folder.

### Framework
TensorFlow / Keras

### Architecture

```
Input: 224 × 224 × 3 (RGB image)
  │
  ├── Conv2D(32 filters, 3×3, ReLU)
  ├── MaxPooling2D(2×2)
  │
  ├── Conv2D(64 filters, 3×3, ReLU)
  ├── MaxPooling2D(2×2)
  │
  ├── Conv2D(128 filters, 3×3, ReLU)
  ├── MaxPooling2D(2×2)
  │
  ├── Flatten
  ├── Dense(128, ReLU)
  ├── Dropout(0.5)
  │
  └── Dense(4, Softmax)  ← 4 output classes
```

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Metric | Accuracy |
| Epochs | 10 (default) |
| Batch Size | 32 |
| Input Size | 224 × 224 pixels |
| Train/Val Split | 80% / 20% |

### Data Augmentation (during training)

| Augmentation | Value |
|---|---|
| Rescale | 1/255 (normalize to 0–1) |
| Horizontal Flip | Yes |
| Zoom Range | ±10% |
| Rotation Range | ±15° |

### Training Dataset Folder Structure Required

```
dataset_dir/
├── Healthy/           ← crop leaf images with no disease
├── Leaf Blight/       ← images showing blight symptoms
├── Powdery Mildew/    ← images showing white powdery patches
└── Rust Disease/      ← images showing orange/rust spots
```

### Inference Process

```
Step 1: Read image → Convert BGR to RGB → Resize to 224×224
Step 2: Normalize pixel values: divide by 255.0
Step 3: Add batch dimension: shape becomes (1, 224, 224, 3)
Step 4: model.predict() → outputs 4 probability scores (Softmax)
Step 5: argmax → index of highest probability = predicted class
Step 6: confidence = max probability × 100 (as integer %)
```

### Output
- Class name from CLASS_NAMES[argmax]
- Confidence = max(softmax_output) × 100

---

## 7. ALGORITHM 3 — CSV KEYWORD CLASSIFIER

### Name
Keyword Matching + Dataset-Derived Confidence

### Used When
User uploads a CSV or Excel file.

### Two Modes

#### Mode A — Name Column Detected
Triggered when CSV has a column named: `name`, `crop`, `plant`, `sample`, or `label`

```
For each row:
  text = lowercase value of name column
  Match against keyword lists:
    Class 0 (Healthy):        healthy, normal, good, clean, fresh, green, fine
    Class 1 (Leaf Blight):    blight, spot, lesion, necrosis, brown, black, rot, wilt
    Class 2 (Powdery Mildew): mildew, white, powder, gray, grey, fuzzy, downy, pale
    Class 3 (Rust Disease):   rust, orange, yellow, pustule, chlorosis, streak
  If keyword matched → assign that class
  If no keyword matched → default to Healthy
  Confidence = _conf_for_class(matched_class)  ← from dataset
```

#### Mode B — Disease Indicator Columns Detected
Triggered when CSV has columns containing: `disease`, `infected`, `symptom`, `damage`, `severity`

```
For each row:
  Check all disease-indicator columns:
    If numeric value > 0 → has_disease = True, track severity
    If text value not in ('', none, nan, 0, no, false, healthy) → has_disease = True

  If has_disease:
    Compute _conf_for_class() for classes 1, 2, 3
    Pick class with highest dataset-derived confidence
  Else:
    Assign Healthy, confidence = _conf_for_class(0)
```

### Confidence for CSV (_conf_for_class)

```
For a given class index:
  1. Get all 4 dataset samples for that class
  2. Compute centroid (mean of 4 samples)
  3. Normalise samples and centroid using dataset min/max
  4. Compute Euclidean distance from each sample to centroid
  5. avg_dist = mean of those 4 distances
  6. confidence = 95 - (avg_dist × 50)
  7. Clamped to [70, 95]
```

This means confidence is always the same for a given class — it reflects
how tightly clustered that class is in the dataset (tighter = higher confidence).

### Pre-computed Confidence Values (approximate)

| Class | Approx. Confidence |
|---|---|
| Healthy | ~93% |
| Leaf Blight | ~88% |
| Powdery Mildew | ~92% |
| Rust Disease | ~89% |

---

## 8. CONFIDENCE SCORE CALCULATION — SUMMARY

| Algorithm | Confidence Method |
|---|---|
| CNN | max(softmax output) × 100 |
| k-NN | 96 − (min_distance_to_predicted_class × 60), clamped [60, 96] |
| CSV Keyword | 95 − (avg_intra_class_distance × 50), clamped [70, 95] |

---

## 9. HOW ALGORITHMS ARE SELECTED (DECISION FLOW)

```
Application starts
       │
       ▼
Is TensorFlow installed?
  NO  → k-NN + built-in dataset (always)
  YES →
       Does crop_cnn_model.h5 exist?
         NO  → k-NN + built-in dataset
         YES → Load CNN model
                    │
                    ▼
              Did model load successfully?
                NO  → k-NN + built-in dataset (fallback)
                YES → CNN inference

User uploads CSV?
  → Always uses CSV Keyword Classifier
    (confidence values come from built-in dataset)
```

---

## 10. FEATURE EXTRACTION PROCESS (for k-NN)

```python
# Code reference: disease_detector.py → _extract_features()

img = cv2.imread(image_path)                          # Read image (BGR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)            # Convert to HSV

h = hsv[:,:,0]   # Hue channel
s = hsv[:,:,1]   # Saturation channel
v = hsv[:,:,2]   # Value channel

hue_mean   = mean(h)                                  # Feature 1
sat_mean   = mean(s)                                  # Feature 2
val_mean   = mean(v)                                  # Feature 3

green_mask = pixels where H in [30,95], S in [40,255], V in [40,255]
green_ratio = count(green_mask) / total_pixels        # Feature 4

dark_ratio  = count(pixels where V < 60) / total     # Feature 5
```

---

## 11. DISEASE AREA MARKING ALGORITHM

After disease is predicted, affected regions are highlighted with red bounding boxes.

```
Step 1: Apply HSV mask for predicted disease:
        Leaf Blight    → dark brown [10,40,20]–[30,255,100] + dark [0,0,0]–[180,255,50]
        Powdery Mildew → white/gray [0,0,200]–[180,50,255]
        Rust Disease   → orange [10,80,80]–[25,255,255] + [25,100,100]–[35,255,255]

Step 2: Morphological cleaning:
        MORPH_CLOSE (5×5 kernel) → fill small holes
        MORPH_OPEN  (5×5 kernel) → remove noise

Step 3: Find external contours (cv2.findContours)

Step 4: Sort contours by area (largest first)
        Draw red rectangle (#FF0000, width=4) around top 5 contours
        where contour area > 150 pixels²

Step 5: If no contours found → draw 2 fallback boxes at fixed positions
        (width/4, height/3) and (3*width/4, 2*height/3)
```

---

## 12. CROP VALIDATION ALGORITHM

Before any analysis, the system checks if the uploaded image is actually a crop/plant image.

```
Checks performed:
  green_ratio  = pixels in HSV [30–95, 40–255, 40–255] / total  → must be > 0.15
  veg_ratio    = pixels where G > R+10 AND G > B+10 AND G > 50  → must be > 0.10
  sky_ratio    = pixels in HSV [90–130, 30–255, 150–255]        → must be < 0.40
  skin_ratio   = pixels in HSV [0–20, 30–170, 100–255]          → must be < 0.35
  gray_ratio   = pixels in HSV [0–180, 0–30, 0–255]             → must be < 0.70

Result:
  PASS (is crop) if: green_ratio > 0.15 AND veg_ratio > 0.10
                     AND sky_ratio < 0.40 AND skin_ratio < 0.35
                     AND gray_ratio < 0.70
  FAIL → raises ValueError: "Not a crop image"
```

---

## 13. SUMMARY TABLE

| Component | Detail |
|---|---|
| Primary Algorithm | CNN (TensorFlow/Keras Sequential) |
| Fallback Algorithm | k-NN (k=3, Euclidean distance, majority vote) |
| CSV Algorithm | Keyword Matching + Dataset Confidence |
| Dataset Type | Built-in HSV feature reference dataset |
| Dataset Size | 16 samples × 5 features |
| Dataset Classes | 4 (Healthy, Leaf Blight, Powdery Mildew, Rust Disease) |
| Dataset Features | hue_mean, sat_mean, val_mean, green_ratio, dark_ratio |
| Normalisation | Min-Max scaling per feature column |
| CNN Input Size | 224 × 224 × 3 (RGB) |
| CNN Layers | 3× Conv2D + MaxPool, Dense(128), Dropout(0.5), Dense(4) |
| CNN Optimizer | Adam |
| CNN Loss | Categorical Crossentropy |
| Confidence Range (CNN) | 0 – 100% (from softmax) |
| Confidence Range (k-NN) | 60 – 96% (distance-based) |
| Confidence Range (CSV) | 70 – 95% (dataset centroid-based) |
| Disease Marking | HSV masking + contour detection + bounding boxes |
| Crop Validation | Multi-ratio HSV check (green, vegetation, sky, skin, gray) |

---

*Document Version: 3.0*
*Project: Crop Disease Detection System — CropGuard AI*
*Language: Python 3.7+*
*Last Updated: Current Version*
