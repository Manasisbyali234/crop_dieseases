# Crop Disease Detection System — Technical Documentation

---

## 1. Project Overview

A desktop GUI application that detects crop diseases from uploaded images or CSV data using color-based image analysis and simulated ML predictions.

- Language: Python 3.7+
- GUI: Tkinter
- Image Processing: OpenCV, Pillow
- Data Handling: Pandas, NumPy

---

## 2. File Structure

```
crop/
├── crop_disease_detector.py   # GUI application (main entry point)
├── disease_detector.py        # Core analysis engine
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
└── DOCUMENTATION.md           # This file
```

---

## 3. Dependencies

| Package         | Version   | Purpose                        |
|----------------|-----------|--------------------------------|
| opencv-python  | >=4.5.0   | Image processing & analysis    |
| Pillow         | >=8.0.0   | Image display & annotation     |
| numpy          | >=1.19.0  | Numerical array operations     |
| pandas         | >=1.3.0   | CSV/Excel data handling        |
| openpyxl       | >=3.0.0   | Excel file support             |

Install all:
```bash
pip install -r requirements.txt
```

---

## 4. Running the Application

```bash
python crop_disease_detector.py
```

The window opens centered at 1280×820 pixels (minimum 1000×700).

---

## 5. Application Architecture

### 5.1 `crop_disease_detector.py` — GUI Layer

**Class: `CropDiseaseDetector`**

Responsible for all UI construction and user interaction.

#### UI Layout

```
┌─────────────────────────────────────────────────────┐
│  Header: Title + AI Badge + Subtitle                │
├──────────────────────────┬──────────────────────────┤
│  Left Panel              │  Right Panel (400px)     │
│  ┌────────────────────┐  │  ┌──────────────────┐   │
│  │ File Upload Card   │  │  │ Analysis Card    │   │
│  │ [Upload Image]     │  │  │ [Start Analysis] │   │
│  │ [Upload CSV]       │  │  └──────────────────┘   │
│  └────────────────────┘  │  ┌──────────────────┐   │
│  ┌────────────────────┐  │  │ Results Card     │   │
│  │ Preview Card       │  │  │ - Disease Name   │   │
│  │ (image / CSV info) │  │  │ - Confidence %   │   │
│  └────────────────────┘  │  │ - Treatment Text │   │
│                          │  └──────────────────┘   │
├─────────────────────────────────────────────────────┤
│  Status Bar                                         │
└─────────────────────────────────────────────────────┘
```

#### Key Methods

| Method | Description |
|--------|-------------|
| `_build_ui()` | Builds header, body, and status bar |
| `upload_image()` | Opens file dialog for image selection |
| `upload_csv()` | Opens file dialog for CSV/Excel selection |
| `detect_disease()` | Disables button, starts progress bar, spawns analysis thread |
| `_run_analysis()` | Runs in background thread; calls `DiseaseDetector` |
| `_finish_image()` | Updates preview with annotated image |
| `_finish_csv()` | Renders scrollable results table |
| `_display_results()` | Updates disease label, confidence badge/bar, remedy text |
| `_reset_results()` | Clears results panel on new file upload |

#### Threading Model

Analysis runs in a `daemon` thread to keep the GUI responsive:
```
Main Thread  →  detect_disease()  →  spawns Thread(_run_analysis)
                                          ↓
                                   DiseaseDetector.analyze_image/csv()
                                          ↓
                                   root.after(0, _finish_*)   ← back to main thread
```

---

### 5.2 `disease_detector.py` — Analysis Engine

**Class: `DiseaseDetector`**

Handles all image processing, feature extraction, disease marking, and CSV analysis.

---

## 6. Disease Detection Process

### 6.1 Image Analysis Pipeline

```
upload_image()
     │
     ▼
process_image()          ← OpenCV: read → resize to 224×224 → grayscale
     │
     ▼
analyze_image_features() ← HSV color analysis + edge detection
     │
     ▼
mark_disease_areas()     ← Draw red bounding boxes on affected regions
     │
     ▼
Return: (disease_name, confidence, remedy, annotated_image)
```

#### Step 1 — `process_image(image_path)`
- Reads image with `cv2.imread()`
- Resizes to 224×224 pixels
- Converts to grayscale
- Returns grayscale NumPy array (used as preprocessing step)

#### Step 2 — `analyze_image_features(image_path)`

Converts image to HSV color space and applies color range masks:

| Disease | HSV Range Detected | Threshold |
|---|---|---|
| Leaf Blight | Dark brown/black (Hue 10–30, low Value) + very dark pixels | >3% of image |
| Powdery Mildew | White/light gray (low Saturation, high Value ≥200) | >2% of image |
| Rust Disease | Orange/yellow-orange (Hue 10–35, high Saturation) | >2% of image |

Confidence scoring:
```python
# Example for Leaf Blight
confidence = min(92, 72 + int(blight_ratio * 400))
```

If no disease threshold is met → returns `"Healthy"` with confidence 88–95%.

#### Step 3 — `mark_disease_areas(image_path, disease_name)`
- Re-applies the same HSV mask for the detected disease
- Uses morphological operations (close + open) to clean the mask
- Finds contours with `cv2.findContours()`
- Draws up to 5 red bounding boxes (`#FF0000`, width=4) around areas > 150px²
- Falls back to 2 fixed-position boxes if no contours found
- Returns annotated PIL Image

---

### 6.2 CSV Analysis Pipeline

```
upload_csv()
     │
     ▼
pandas.read_csv() / read_excel()
     │
     ▼
analyze_csv()
     │
     ├── Detect disease-indicator columns
     │   (columns containing: disease, infected, symptom, damage, severity)
     │
     ├── For each row → _predict_row()
     │       ├── If indicator column has non-zero/non-empty value → random disease
     │       └── Else → Healthy
     │
     ├── Count most common disease (Counter)
     ├── Average confidence across all rows
     └── Return: (top_disease, avg_confidence, summary, all_row_results)
```

Results are displayed as a scrollable table in the preview panel showing:
- Row number, Disease, Confidence %, Status icon (✅ / ⚠️)

---

## 7. Supported Diseases

| Disease | Visual Indicator | Confidence Range | Recovery Time |
|---|---|---|---|
| Healthy | Normal green coloration | 88–95% | N/A |
| Leaf Blight | Dark brown/black spots | 72–92% | 2–3 weeks |
| Powdery Mildew | White powdery patches | 75–90% | 1–2 weeks |
| Rust Disease | Orange/rust pustules | 73–88% | 2–4 weeks |

---

## 8. Results Display

| UI Element | Healthy | Disease Detected |
|---|---|---|
| Disease Label | ✅ No Disease Detected | ⚠️ Disease Name |
| Confidence Badge color | Green (`#22c55e`) | Red (≥80%) / Yellow (<80%) |
| Confidence Bar | Green fill | Red/Yellow fill proportional to % |
| Remedy Text | Preventive care tips | Treatment steps + prevention |

---

## 9. Error Handling

| Scenario | Handling |
|---|---|
| Invalid/corrupt image | `messagebox.showerror()` shown |
| CSV parse failure | `messagebox.showerror()` shown |
| Feature analysis crash | Falls back to `"Healthy", 85` |
| Disease marking failure | Returns original unmodified image |
| Analysis thread exception | `_analysis_error()` shows error dialog |

---

## 10. Limitations & Future Improvements

**Current Limitations:**
- Disease detection is color-threshold based, not a trained ML model
- CSV prediction is rule-based (column name matching + random assignment)
- No persistent storage of results

**Recommended Improvements:**
- Replace `analyze_image_features()` with a trained CNN (TensorFlow/PyTorch)
- Add export results to PDF/CSV
- Add detection history with database (SQLite)
- Support batch image processing
- Build a web interface (Flask/FastAPI + React)
