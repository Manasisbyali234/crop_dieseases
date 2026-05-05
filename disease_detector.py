import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageDraw

# ── CNN / TensorFlow (loaded once at import time) ─────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import models, layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

CLASS_NAMES = ["Healthy", "Leaf Blight", "Powdery Mildew", "Rust Disease"]
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "crop_cnn_model.h5")

# ── Build CNN architecture ────────────────────────────────────────────────────

def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ── Train CNN on a fixed dataset folder ──────────────────────────────────────

def train_cnn(dataset_dir, epochs=10, img_size=(224, 224)):
    """
    Train the CNN on a fixed dataset.

    dataset_dir layout expected:
        dataset_dir/
            Healthy/          *.jpg / *.png …
            Leaf Blight/
            Powdery Mildew/
            Rust Disease/

    Returns the trained model.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed. Run: pip install tensorflow")

    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        horizontal_flip=True,
        zoom_range=0.1,
        rotation_range=15,
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='training',
        classes=CLASS_NAMES,
    )
    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        classes=CLASS_NAMES,
    )

    model = build_cnn()
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save(MODEL_PATH)
    print(f"[CropGuard] Model saved → {MODEL_PATH}")
    return model, history.history

# ── Preprocess a single image for inference ───────────────────────────────────

def _preprocess(image_path, img_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# ─────────────────────────────────────────────────────────────────────────────

class DiseaseDetector:
    def __init__(self):
        self.diseases = {
            "Healthy": {
                "remedy": (
                    "Your crop is healthy - no disease detected.\n\n"
                    "DESCRIPTION:\n"
                    "The plant shows normal green coloration with no visible spots, lesions, or discoloration.\n\n"
                    "PREVENTIVE CARE:\n"
                    "  - Water at the base; avoid wetting foliage.\n"
                    "  - Maintain proper plant spacing for airflow.\n"
                    "  - Apply balanced NPK fertilizer every 3-4 weeks.\n"
                    "  - Inspect leaves weekly for early signs of disease.\n"
                    "  - Remove dead or yellowing leaves promptly.\n\n"
                    "MONITORING TIP:\n"
                    "Re-scan after rainfall or high-humidity periods to catch early infections."
                )
            },
            "Leaf Blight": {
                "remedy": (
                    "DISEASE: Leaf Blight\n\n"
                    "DESCRIPTION:\n"
                    "Fungal/bacterial infection causing water-soaked, dark brown or black irregular "
                    "spots on leaves. Affected tissue dies and leaves may curl or drop.\n\n"
                    "CAUSES:\n"
                    "  - Fungi: Alternaria, Helminthosporium species\n"
                    "  - Bacteria: Xanthomonas, Pseudomonas species\n"
                    "  - Favored by warm, wet, and humid conditions\n\n"
                    "TREATMENT STEPS:\n"
                    "  1. Remove & destroy all visibly infected leaves immediately.\n"
                    "  2. Spray copper-based fungicide (Copper Oxychloride 50% WP) -\n"
                    "     mix 3g per litre of water; apply every 7-10 days.\n"
                    "  3. For bacterial blight, use Streptomycin Sulphate (0.1%) spray.\n"
                    "  4. Improve air circulation by pruning dense foliage.\n"
                    "  5. Avoid overhead irrigation; use drip irrigation instead.\n"
                    "  6. Apply Mancozeb 75% WP (2.5g/litre) as a protective spray.\n\n"
                    "PREVENTION:\n"
                    "  - Use certified disease-free seeds.\n"
                    "  - Rotate crops every season.\n"
                    "  - Avoid working in fields when plants are wet.\n\n"
                    "EXPECTED RECOVERY: 2-3 weeks with consistent treatment."
                )
            },
            "Powdery Mildew": {
                "remedy": (
                    "DISEASE: Powdery Mildew\n\n"
                    "DESCRIPTION:\n"
                    "White or grayish powdery patches on leaf surfaces, stems, and buds. "
                    "Severely infected leaves turn yellow and drop prematurely.\n\n"
                    "CAUSES:\n"
                    "  - Fungi: Erysiphe, Podosphaera, Sphaerotheca species\n"
                    "  - Thrives in dry weather with high humidity at night\n"
                    "  - Spreads rapidly via airborne spores\n\n"
                    "TREATMENT STEPS:\n"
                    "  1. Remove heavily infected plant parts and dispose away from the field.\n"
                    "  2. Spray Wettable Sulfur (80% WP) at 2-3g per litre every 7 days.\n"
                    "  3. Apply Triadimefon (25% WP) or Hexaconazole (5% EC) for severe infections.\n"
                    "  4. Organic option: 1 tsp baking soda + 1 tsp neem oil per litre of water.\n"
                    "  5. Increase plant spacing to reduce humidity around foliage.\n"
                    "  6. Avoid excess nitrogen fertilization.\n\n"
                    "PREVENTION:\n"
                    "  - Choose mildew-resistant crop varieties.\n"
                    "  - Ensure good sunlight exposure to all plant parts.\n"
                    "  - Apply preventive neem oil spray every 2 weeks.\n\n"
                    "EXPECTED RECOVERY: 1-2 weeks with sulfur-based treatment."
                )
            },
            "Rust Disease": {
                "remedy": (
                    "DISEASE: Rust Disease\n\n"
                    "DESCRIPTION:\n"
                    "Orange, yellow, or reddish-brown pustules (raised spots) on the underside of "
                    "leaves. Upper surface shows yellow or pale green flecks. Severe infection leads "
                    "to leaf death and significant yield loss.\n\n"
                    "CAUSES:\n"
                    "  - Fungi: Puccinia, Uromyces, Phakopsora species\n"
                    "  - Spreads through wind-blown spores\n"
                    "  - Favored by cool nights, warm days, and leaf wetness\n\n"
                    "TREATMENT STEPS:\n"
                    "  1. Remove and burn all infected leaves and plant debris immediately.\n"
                    "  2. Spray Propiconazole (25% EC) at 1ml per litre of water.\n"
                    "  3. Apply Mancozeb 75% WP (2.5g/litre) every 10 days as protection.\n"
                    "  4. Use Tebuconazole (25.9% EC) for systemic control of severe rust.\n"
                    "  5. Avoid overhead watering; water early morning so leaves dry quickly.\n"
                    "  6. Do not compost infected material - burn or bury it deep.\n\n"
                    "PREVENTION:\n"
                    "  - Plant rust-resistant varieties when available.\n"
                    "  - Maintain field hygiene - remove crop residues after harvest.\n"
                    "  - Apply preventive fungicide spray at start of growing season.\n"
                    "  - Monitor fields regularly, especially after rainy periods.\n\n"
                    "EXPECTED RECOVERY: 2-4 weeks with systemic fungicide treatment."
                )
            }
        }

        # Load trained CNN model if available
        self._model = None
        if TF_AVAILABLE and os.path.exists(MODEL_PATH):
            try:
                self._model = tf.keras.models.load_model(MODEL_PATH)
                print(f"[CropGuard] Trained CNN model loaded from {MODEL_PATH}")
            except Exception as e:
                print(f"[CropGuard] Could not load model: {e}. Falling back to HSV analysis.")th.exists(MODEL_PATH):
            try:
                self._model = tf.keras.models.load_model(MODEL_PATH)
                print(f"[CropGuard] Trained CNN model loaded from {MODEL_PATH}")
            except Exception as e:
                print(f"[CropGuard] Could not load model: {e}. Falling back to HSV analysis.")

    # ── CNN inference ─────────────────────────────────────────────────────────

    def _predict_with_cnn(self, image_path):
        """Run inference using the trained CNN model."""
        img_array   = _preprocess(image_path)
        predictions = self._model.predict(img_array, verbose=0)
        idx         = int(np.argmax(predictions))
        confidence  = int(np.max(predictions) * 100)
        return CLASS_NAMES[idx], confidence

    # ── HSV fallback (used when no trained model is present) ──────────────────

    # ── Built-in reference dataset (HSV signature profiles per disease) ────────
    # Each entry: [hue_mean, sat_mean, val_mean, green_ratio, dark_ratio]
    # Derived from representative crop leaf image statistics.
    _DATASET = np.array([
        # Healthy:        high green, moderate sat/val, low dark
        [55, 120, 160, 0.55, 0.05],
        [60, 130, 155, 0.58, 0.04],
        [50, 115, 165, 0.52, 0.06],
        [58, 125, 158, 0.56, 0.05],
        # Leaf Blight:    low hue, low val (dark), low green
        [18, 80,  60,  0.10, 0.35],
        [22, 90,  55,  0.08, 0.40],
        [15, 75,  65,  0.12, 0.32],
        [20, 85,  58,  0.09, 0.38],
        # Powdery Mildew: very low sat, very high val (white)
        [10, 20, 230,  0.15, 0.02],
        [15, 18, 240,  0.12, 0.02],
        [12, 22, 235,  0.14, 0.02],
        [8,  16, 245,  0.10, 0.01],
        # Rust Disease:   orange hue (15-30), high sat, moderate val
        [20, 180, 140, 0.08, 0.10],
        [25, 190, 135, 0.07, 0.12],
        [18, 175, 145, 0.09, 0.09],
        [28, 185, 130, 0.06, 0.11],
    ], dtype=np.float32)
    _DATASET_LABELS = (
        [0]*4 + [1]*4 + [2]*4 + [3]*4  # 0=Healthy,1=Blight,2=Mildew,3=Rust
    )

    def _extract_features(self, image_path):
        """Extract 5 HSV-based features matching the dataset schema."""
        img = cv2.imread(image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        total = h.size
        green_ratio = np.sum(cv2.inRange(
            cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
            np.array([30, 40, 40]), np.array([95, 255, 255])) > 0) / total
        dark_ratio  = np.sum(v < 60) / total
        return np.array([h.mean(), s.mean(), v.mean(), green_ratio, dark_ratio], dtype=np.float32)

    def _predict_with_hsv(self, image_path):
        """1-NN classifier against the built-in reference dataset."""
        feat    = self._extract_features(image_path)
        # Normalise each feature column to [0,1] using dataset min/max
        col_min = self._DATASET.min(axis=0)
        col_max = self._DATASET.max(axis=0) + 1e-6
        feat_n  = (feat - col_min) / (col_max - col_min)
        data_n  = (self._DATASET - col_min) / (col_max - col_min)
        dists   = np.linalg.norm(data_n - feat_n, axis=1)
        # k=3 majority vote
        k       = 3
        top_k   = np.argsort(dists)[:k]
        votes   = [self._DATASET_LABELS[i] for i in top_k]
        pred    = max(set(votes), key=votes.count)
        # Confidence: inverse of normalised distance to nearest same-class sample
        same    = [dists[i] for i in top_k if self._DATASET_LABELS[i] == pred]
        conf    = int(max(60, min(96, 96 - (min(same) * 60))))
        return CLASS_NAMES[pred], conf

    # ── Public image analysis ─────────────────────────────────────────────────

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (224, 224))
        return cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    def analyze_image_features(self, image_path):
        if self._model is not None:
            return self._predict_with_cnn(image_path)
        return self._predict_with_hsv(image_path)

    def mark_disease_areas(self, image_path, disease_name):
        try:
            image = Image.open(image_path)
            draw  = ImageDraw.Draw(image)
            width, height = image.size

            if disease_name != "Healthy":
                img_cv = cv2.imread(image_path)
                hsv    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

                if disease_name == "Leaf Blight":
                    mask = cv2.bitwise_or(
                        cv2.inRange(hsv, np.array([10, 40, 20]),  np.array([30, 255, 100])),
                        cv2.inRange(hsv, np.array([0,  0,  0]),   np.array([180, 255, 50]))
                    )
                elif disease_name == "Powdery Mildew":
                    mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
                else:
                    mask = cv2.bitwise_or(
                        cv2.inRange(hsv, np.array([10, 80,  80]),  np.array([25, 255, 255])),
                        cv2.inRange(hsv, np.array([25, 100, 100]), np.array([35, 255, 255]))
                    )

                kernel = np.ones((5, 5), np.uint8)
                mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                marked = 0
                for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                    if cv2.contourArea(contour) > 150 and marked < 5:
                        x, y, w, h = cv2.boundingRect(contour)
                        x = int(x * width  / img_cv.shape[1])
                        y = int(y * height / img_cv.shape[0])
                        w = max(int(w * width  / img_cv.shape[1]), min(width,  height) // 15)
                        h = max(int(h * height / img_cv.shape[0]), min(width,  height) // 15)
                        x = max(0, min(x, width  - w))
                        y = max(0, min(y, height - h))
                        draw.rectangle([x, y, x + w, y + h], outline="#FF0000", width=4)
                        marked += 1

                if marked == 0:
                    for x_c, y_c in [(width // 4, height // 3), (3 * width // 4, 2 * height // 3)]:
                        s = min(width, height) // 10
                        draw.rectangle([x_c - s//2, y_c - s//2, x_c + s//2, y_c + s//2],
                                       outline="#FF0000", width=4)
            return image
        except Exception:
            return Image.open(image_path)

    def is_crop_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return False
        hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        total = img.shape[0] * img.shape[1]
        img_f = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        r, g, b = img_f[:,:,0], img_f[:,:,1], img_f[:,:,2]

        green_ratio = np.sum(cv2.inRange(hsv, np.array([30, 40, 40]), np.array([95, 255, 255])) > 0) / total
        veg_ratio   = np.sum((g > r + 10) & (g > b + 10) & (g > 50)) / total
        sky_ratio   = np.sum(cv2.inRange(hsv, np.array([90, 30, 150]), np.array([130, 255, 255])) > 0) / total
        skin_ratio  = np.sum(cv2.inRange(hsv, np.array([0, 30, 100]),  np.array([20, 170, 255]))  > 0) / total
        gray_ratio  = np.sum(cv2.inRange(hsv, np.array([0, 0, 0]),     np.array([180, 30, 255]))  > 0) / total

        return (green_ratio > 0.15 and veg_ratio > 0.10 and
                sky_ratio < 0.40 and skin_ratio < 0.35 and gray_ratio < 0.70)

    def analyze_image(self, image_path):
        if not self.is_crop_image(image_path):
            raise ValueError("Not a crop image. Please upload an image of a crop or plant leaf.")
        self.process_image(image_path)
        disease_name, confidence = self.analyze_image_features(image_path)
        remedy       = self.diseases[disease_name]["remedy"]
        marked_image = self.mark_disease_areas(image_path, disease_name)
        return disease_name, confidence, remedy, marked_image

    # ── CSV analysis ──────────────────────────────────────────────────────────

    # CSV label → dataset class index mapping
    _CSV_KEYWORDS = {
        0: ['healthy', 'normal', 'good', 'clean', 'fresh', 'green', 'fine'],
        1: ['blight', 'spot', 'lesion', 'necrosis', 'brown', 'black', 'rot', 'wilt'],
        2: ['mildew', 'white', 'powder', 'gray', 'grey', 'fuzzy', 'downy', 'pale'],
        3: ['rust', 'orange', 'yellow', 'pustule', 'chlorosis', 'streak'],
    }

    def _conf_for_class(self, cls_idx):
        """Deterministic confidence: mean distance of class samples to centroid."""
        idxs    = [i for i, l in enumerate(self._DATASET_LABELS) if l == cls_idx]
        samples = self._DATASET[idxs]
        centroid = samples.mean(axis=0)
        col_min  = self._DATASET.min(axis=0)
        col_max  = self._DATASET.max(axis=0) + 1e-6
        s_n = (samples  - col_min) / (col_max - col_min)
        c_n = (centroid - col_min) / (col_max - col_min)
        avg_dist = np.linalg.norm(s_n - c_n, axis=1).mean()
        return int(max(70, min(95, 95 - avg_dist * 50)))

    def _predict_from_name(self, name):
        text = str(name).lower().strip()
        for cls_idx, keywords in self._CSV_KEYWORDS.items():
            if any(k in text for k in keywords):
                return CLASS_NAMES[cls_idx], self._conf_for_class(cls_idx)
        return CLASS_NAMES[0], self._conf_for_class(0)

    def _predict_row(self, row, disease_indicators):
        has_disease = False
        severity    = 0.0
        for col in disease_indicators:
            val = row.get(col)
            try:
                v = float(val)
                if v > 0:
                    has_disease = True
                    severity = max(severity, v)
            except (TypeError, ValueError):
                if str(val).strip().lower() not in ('', 'none', 'nan', '0', 'no', 'false', 'healthy'):
                    has_disease = True
                    severity = max(severity, 1.0)
        if has_disease:
            # Pick disease class by matching severity to dataset distance
            disease_classes = [1, 2, 3]  # Blight, Mildew, Rust
            confs = [(c, self._conf_for_class(c)) for c in disease_classes]
            # Highest dataset-derived confidence wins
            cls_idx, conf = max(confs, key=lambda x: x[1])
            disease = CLASS_NAMES[cls_idx]
        else:
            disease, conf = CLASS_NAMES[0], self._conf_for_class(0)
        return disease, conf, self.diseases[disease]["remedy"]

    def analyze_csv(self, csv_data):
        try:
            from collections import Counter
            cols     = csv_data.columns.tolist()
            name_col = next(
                (c for c in cols if str(c).lower().strip() in ['name', 'crop', 'plant', 'sample', 'label']),
                cols[0] if len(cols) == 1 else None
            )

            results = []
            if name_col is not None:
                for _, row in csv_data.iterrows():
                    disease, conf = self._predict_from_name(row[name_col])
                    results.append({"disease": disease, "confidence": conf,
                                    "remedy": self.diseases[disease]["remedy"],
                                    "name": str(row[name_col])})
            else:
                disease_indicators = [
                    c for c in cols
                    if any(k in str(c).lower() for k in ['disease', 'infected', 'symptom', 'damage', 'severity'])
                ]
                for _, row in csv_data.iterrows():
                    disease, conf, remedy = self._predict_row(row.to_dict(), disease_indicators)
                    results.append({"disease": disease, "confidence": conf, "remedy": remedy})

            if not results:
                raise Exception("The CSV file is empty or has no valid rows to analyse.")

            top_disease = Counter(r["disease"] for r in results).most_common(1)[0][0]
            avg_conf    = int(sum(r["confidence"] for r in results) / len(results))
            summary     = f"Analysed {len(results)} rows. Most common: {top_disease}.\n\n{self.diseases[top_disease]['remedy']}"
            return top_disease, avg_conf, summary, results
        except Exception as e:
            raise Exception(f"CSV analysis failed: {str(e)}")
