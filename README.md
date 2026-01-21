# Crop Disease Detection System

A desktop application for detecting crop diseases using AI-ML and image processing techniques. This application uses dummy predictions for demonstration purposes.

## Features

- **GUI-based Interface**: Clean and intuitive Tkinter-based user interface
- **Image Upload**: Support for common image formats (JPG, PNG, BMP, TIFF)
- **Image Preview**: Real-time preview of uploaded crop images
- **Image Processing**: Basic preprocessing including resizing and grayscale conversion
- **Disease Detection**: Simulated ML predictions with confidence levels
- **Treatment Suggestions**: Recommended remedies for detected diseases

## Supported Disease Types

- **Healthy**: Normal, disease-free crops
- **Leaf Blight**: Fungal infection causing leaf spots
- **Powdery Mildew**: White powdery fungal growth
- **Rust Disease**: Orange/brown rust-colored spots

## Installation

1. **Clone or download** this repository
2. **Install Python** (3.7 or higher)
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**:
   ```bash
   python crop_disease_detector.py
   ```

2. **Upload an image**:
   - Click "Upload Crop Image" button
   - Select a crop leaf image from your computer
   - The image will appear in the preview area

3. **Detect disease**:
   - Click "Detect Disease" button
   - View results in the right panel:
     - Disease name
     - Confidence percentage
     - Suggested remedy

## How It Works

### Image Processing
- **Resizing**: Images are resized to 224x224 pixels for consistency
- **Grayscale Conversion**: Color images are converted to grayscale using OpenCV
- **Display Optimization**: Images are thumbnailed for GUI display

### Machine Learning Simulation
- **Dummy Predictions**: Random selection from predefined disease categories
- **Confidence Simulation**: Realistic confidence percentages (70-95%)
- **Randomization**: Each detection provides varied results for demonstration

### User Interface
- **Left Panel**: Image upload and preview
- **Right Panel**: Detection controls and results
- **Error Handling**: Proper validation and user feedback

## Technical Stack

- **Python**: Core programming language
- **Tkinter**: GUI framework
- **OpenCV**: Image processing
- **PIL (Pillow)**: Image handling and display
- **NumPy**: Numerical operations

## File Structure

```
crop/
├── crop_disease_detector.py  # Main application file
├── requirements.txt          # Python dependencies
└── README.md                # Documentation
```

## Future Enhancements

- Integration with real ML models (TensorFlow/PyTorch)
- Database storage for detection history
- Batch processing capabilities
- Advanced image preprocessing
- Web-based interface
- Mobile application support

## Notes

- This is a demonstration application using dummy data
- For production use, replace dummy predictions with trained ML models
- Ensure proper image quality for best results
- The application requires an active display for GUI operation