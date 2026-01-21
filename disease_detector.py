import cv2
import numpy as np
import pandas as pd
import random
from PIL import Image, ImageDraw

class DiseaseDetector:
    def __init__(self):
        # Dummy disease data
        self.diseases = {
            "Healthy": {
                "confidence": random.randint(85, 95),
                "remedy": "Your crop appears healthy! Continue regular care and monitoring."
            },
            "Leaf Blight": {
                "confidence": random.randint(75, 90),
                "remedy": "Apply copper-based fungicide. Remove affected leaves and improve air circulation."
            },
            "Powdery Mildew": {
                "confidence": random.randint(80, 92),
                "remedy": "Use sulfur-based fungicide. Reduce humidity and increase spacing between plants."
            },
            "Rust Disease": {
                "confidence": random.randint(78, 88),
                "remedy": "Apply rust-resistant fungicide. Remove infected plant debris and avoid overhead watering."
            }
        }
    
    def process_image(self, image_path):
        """Basic image processing using OpenCV"""
        try:
            # Read image
            img = cv2.imread(image_path)
            
            # Resize image
            img_resized = cv2.resize(img, (224, 224))
            
            # Convert to grayscale
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            return img_gray
            
        except Exception as e:
            raise Exception(f"Image processing failed: {str(e)}")
    
    def analyze_image_features(self, image_path):
        """Analyze image features to make more accurate predictions"""
        try:
            # Load and process image
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Enhanced feature analysis
            h_mean = np.mean(hsv[:,:,0])
            s_mean = np.mean(hsv[:,:,1])
            v_mean = np.mean(hsv[:,:,2])
            
            # Color variance analysis
            color_variance = np.var(img_rgb.reshape(-1, 3), axis=0).mean()
            
            # Texture analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Disease-specific color detection
            # Check for blight (dark brown/black spots)
            lower_blight = np.array([10, 50, 0])
            upper_blight = np.array([25, 255, 80])
            blight_mask = cv2.inRange(hsv, lower_blight, upper_blight)
            blight_ratio = np.sum(blight_mask > 0) / blight_mask.size
            
            # Check for mildew (white/gray powdery areas)
            lower_mildew = np.array([0, 0, 180])
            upper_mildew = np.array([180, 60, 255])
            mildew_mask = cv2.inRange(hsv, lower_mildew, upper_mildew)
            mildew_ratio = np.sum(mildew_mask > 0) / mildew_mask.size
            
            # Check for rust (orange/rust colored spots)
            lower_rust = np.array([5, 100, 100])
            upper_rust = np.array([20, 255, 255])
            rust_mask = cv2.inRange(hsv, lower_rust, upper_rust)
            rust_ratio = np.sum(rust_mask > 0) / rust_mask.size
            
            # Decision logic based on detected features
            if blight_ratio > 0.05:  # 5% of image shows blight characteristics
                confidence = min(90, 70 + int(blight_ratio * 400))
                return "Leaf Blight", confidence
            elif mildew_ratio > 0.03:  # 3% of image shows mildew characteristics
                confidence = min(92, 75 + int(mildew_ratio * 500))
                return "Powdery Mildew", confidence
            elif rust_ratio > 0.02:  # 2% of image shows rust characteristics
                confidence = min(88, 72 + int(rust_ratio * 600))
                return "Rust Disease", confidence
            elif edge_density < 0.08 and color_variance < 600:  # Smooth, uniform (healthy)
                return "Healthy", random.randint(88, 95)
            else:
                # Analyze overall image health
                if color_variance > 1500 or edge_density > 0.15:
                    # High variance suggests disease
                    diseases = ["Leaf Blight", "Powdery Mildew", "Rust Disease"]
                    disease = random.choice(diseases)
                    return disease, random.randint(75, 85)
                else:
                    return "Healthy", random.randint(85, 92)
                    
        except Exception as e:
            print(f"Feature analysis error: {e}")
            # Fallback to random
            disease = random.choice(list(self.diseases.keys()))
            return disease, random.randint(70, 90)
    
    def mark_disease_areas(self, image_path, disease_name):
        """Mark affected areas with neat red boxes"""
        try:
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            width, height = image.size
            
            if disease_name != "Healthy":
                # Load image for analysis
                img_cv = cv2.imread(image_path)
                hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
                
                # Use red color for all disease markings
                box_color = "#FF0000"  # Bright red
                
                # Enhanced disease detection based on color
                if disease_name == "Leaf Blight":
                    lower_bound = np.array([10, 50, 0])
                    upper_bound = np.array([25, 255, 80])
                elif disease_name == "Powdery Mildew":
                    lower_bound = np.array([0, 0, 180])
                    upper_bound = np.array([180, 60, 255])
                else:  # Rust Disease
                    lower_bound = np.array([5, 100, 100])
                    upper_bound = np.array([20, 255, 255])
                
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                marked_areas = 0
                if contours:
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 200 and marked_areas < 5:
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Scale to original image
                            x = int(x * width / img_cv.shape[1])
                            y = int(y * height / img_cv.shape[0])
                            w = int(w * width / img_cv.shape[1])
                            h = int(h * height / img_cv.shape[0])
                            
                            # Ensure minimum size
                            min_size = min(width, height) // 15
                            w = max(w, min_size)
                            h = max(h, min_size)
                            
                            # Keep within bounds
                            x = max(0, min(x, width - w))
                            y = max(0, min(y, height - h))
                            
                            # Draw neat red rectangle
                            draw.rectangle([x, y, x + w, y + h], outline=box_color, width=4)
                            marked_areas += 1
                
                # Fallback: mark typical disease locations
                if marked_areas == 0:
                    positions = [(width//4, height//3), (3*width//4, 2*height//3), (width//2, height//2)]
                    
                    for x_center, y_center in positions[:2]:  # Mark 2 areas
                        box_size = min(width, height) // 10
                        x = x_center - box_size//2
                        y = y_center - box_size//2
                        draw.rectangle([x, y, x + box_size, y + box_size], outline=box_color, width=4)
            
            return image
            
        except Exception as e:
            print(f"Error marking disease areas: {str(e)}")
            return Image.open(image_path)
    
    def analyze_image(self, image_path):
        """Analyze image and return disease prediction"""
        # Process image
        processed_image = self.process_image(image_path)
        
        # Use feature-based prediction
        disease_name, confidence = self.analyze_image_features(image_path)
        
        # Get remedy from disease data
        remedy = self.diseases[disease_name]["remedy"]
        
        # Mark disease areas on image
        marked_image = self.mark_disease_areas(image_path, disease_name)
        
        return disease_name, confidence, remedy, marked_image
    
    def analyze_csv(self, csv_data):
        """Analyze CSV data and return disease prediction"""
        try:
            # Simulate CSV-based prediction
            disease_name = random.choice(list(self.diseases.keys()))
            disease_info = self.diseases[disease_name]
            
            # Simulate confidence based on data quality
            rows, cols = csv_data.shape
            base_confidence = min(90, 60 + (cols * 2))  # More columns = higher confidence
            confidence = base_confidence + random.randint(-10, 10)
            confidence = max(60, min(95, confidence))
            
            # Custom remedy for CSV analysis
            csv_remedy = f"Based on {rows} data points with {cols} features: {disease_info['remedy']}"
            
            return disease_name, confidence, csv_remedy
            
        except Exception as e:
            raise Exception(f"CSV analysis failed: {str(e)}")